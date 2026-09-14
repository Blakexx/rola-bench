"""Box-side runner for the similarity fleet job (FLEET_RUN_CMD -> this module).

DECOUPLED PROBE ARCHITECTURE. Each run_id is "<cell>::<probe>": the cell identifies the trained
model (its MQAR experiment config + shipped checkpoint), the probe selects WHICH measurement to
compute on it. The fleet dedups by full run_id, so each (cell, probe) is tracked independently and
a probe added later dispatches only its own "<cell>::<probe>" ids.

Legacy un-suffixed run_ids ("<cell>") are treated as "<cell>::rank" so the existing rank rows
remain valid (no re-run). Checkpoints are looked up by the CELL (suffix stripped), so all probes
of one cell share its single shipped .pt.

Probes (registry below):
  rank     - rank + sharpness DISTRIBUTION of the FINAL token-mixing matrix over the cell's own MQAR test
             slices (both pre/post causal mask). EXPENSIVE (per-slice SVD). Any model family with a
             faithful extractor: attention, GDN, GLA, Based, linear attention. RoLA has none until its
             effective matrix is defined from the route factors (carded).
"""
import base64
import json
import math
import os
import sys
import traceback
import zlib

import torch

from rola_bench.similarity import evaluator as E

RESULTS = os.environ.get("FLEET_RESULTS", "/workspace/res.jsonl")
PAYLOAD_DIR = os.environ.get("FLEET_PAYLOAD_DIR", "/workspace/payloads")


def _b64_f16(arr):
    """ndarray -> base64(zlib(float16 bytes)). Compact binary that rides INSIDE the jsonl row, so heavy
    2D artifacts (the [nc x nc] joint) return through the generic fleet json-pull like any other result."""
    import numpy as np
    return base64.b64encode(zlib.compress(arr.astype(np.float16).tobytes())).decode("ascii")


def _unb64_f16(s, shape):
    import numpy as np
    return np.frombuffer(zlib.decompress(base64.b64decode(s)), dtype=np.float16).astype(np.float32).reshape(shape)


RANK_FRAC = float(os.environ.get("RANK_FRAC", "0.25"))     # fraction of each test slice's examples to eval
MEM_BUDGET = float(os.environ.get("RANK_MEM_BUDGET", "3e9"))  # bytes for the [chunk,H,L,L] W tensor (4090-safe)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
# per-(kernel) metric -> rank_stats key whose per-slice `_dist` we aggregate across chunks.
_METRICS = ("eff_rank", "stable_rank", "rank_1e-02", "sharp_diffuseness", "sharp_eff_key_frac", "sharp_top1")


def _agg(vals):
    """Summarize a concatenated per-slice distribution -> mean + percentiles (compact)."""
    if not vals:
        return None
    s = sorted(vals)
    n = len(s)

    def q(p):
        return s[min(n - 1, int(p * (n - 1) + 0.5))]

    return {"mean": round(sum(s) / n, 3), "p10": round(q(0.1), 3), "p50": round(q(0.5), 3),
            "p90": round(q(0.9), 3), "min": round(s[0], 3), "max": round(s[-1], 3), "n": n}


def _chunk_size(L, n_heads):
    """Largest example-chunk whose [chunk, H, L, L] fp32 W fits MEM_BUDGET. n_heads is the model's
    ACTUAL head count (probed per cell): heads-shape baselines reach H~140, so a hardcoded H=8 would
    under-budget by ~17x and OOM the heavy slices."""
    per = max(1, n_heads * L * L * 4)
    return max(1, min(64, int(MEM_BUDGET // per)))


def _append(row):
    with open(RESULTS, "a") as fh:
        fh.write(json.dumps(row) + "\n")
        fh.flush()


def _parse_runid(rid):
    """'<cell>::<probe>' -> (cell, probe). Legacy un-suffixed -> ('<rid>', 'rank')."""
    if "::" in rid:
        cell, probe = rid.rsplit("::", 1)
        return cell, probe
    return rid, "rank"


def _ckpt_path(rid):
    """Find the cell's checkpoint. Cloud ships it under PAYLOAD_DIR/<run_id>; locally the files are
    named PAYLOAD_DIR/<cell>.pt. Try all three so the SAME runner works in both."""
    cell = _parse_runid(rid)[0]
    for cand in (rid, cell, cell + ".pt"):
        p = os.path.join(PAYLOAD_DIR, cand)
        if os.path.exists(p):
            return p
    return None


def _configs_by_runid():
    """run_id -> TrainConfig over every MQAR experiment spec, every tier, so a stored cell's model can be rebuilt."""
    import glob

    from rola_bench.mqar.build_configs import build_configs, load_spec

    os.environ["GRID_TIERS"] = "all"
    by_id = {}
    for path in sorted(glob.glob(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mqar",
                                              "experiments", "*.yaml"))):
        by_id.update({c.run_id: c for c in build_configs(load_spec(path))[0]})
    return by_id


def _build_model(cfg):
    from zoology.model import LanguageModel
    return LanguageModel(cfg.model)


def _pick_extractor_builder(model):
    """Return (family, builder(layer_index, causal)->extractor) by matching the rebuilt model's
    layer-0 mixer against each family predicate. None if no faithful extractor applies."""
    mods = list(model.modules())
    checks = [
        ("mha", E._is_mha, E.build_softmax_attention_extractor),
        ("gdn", E._is_gdn, E.build_gdn_extractor),
        ("gla", E._is_gla, E.build_gla_extractor),
        ("based", E._is_based, E.build_based_extractor),
        ("fla_linear", E._is_fla_linear_attn, E.build_fla_linear_attn_extractor),
    ]
    for fam, pred, builder in checks:
        if any(pred(m) for m in mods):
            return fam, builder
    return None, None


# ------------------------------------------------------------------------------------------------
# PROBES. Each: probe(run_id, cfg, model, vocab) -> bool ok; appends its own row(s) carrying a
# `probe` field. Setup (build model + load ckpt) is done ONCE by main() and shared across the probe.
# ------------------------------------------------------------------------------------------------

def _probe_rank(run_id, cfg, model, vocab):
    """Rank/sharpness DISTRIBUTION over the real MQAR test set (per kv-density slice), at RANK_FRAC
    sampling, memory-chunked. One row per (run_id, causal). EXPENSIVE (per-slice SVD)."""
    TEST_CONFIGS = cfg.data.test_configs                            # the cell's own test slices
    family, builder = _pick_extractor_builder(model)
    if builder is None:
        _append({"run_id": run_id, "ok": False, "probe": "rank", "error": "no faithful extractor for this family"})
        return False
    wrote_ok = True
    for causal in (False, True):
        try:
            extractor = builder(layer_index=0, causal=causal)
            _sc0 = min(TEST_CONFIGS, key=lambda c: c.input_seq_len)
            with torch.no_grad():
                _probe = E.make_mqar_eval(vocab, _sc0.input_seq_len, _sc0.num_kv_pairs, 1, seed=1234, device=DEV)
                n_heads = int(extractor(model, _probe).shape[1])
            per_slice = {}
            for sc in TEST_CONFIGS:                                   # one entry per kv-density slice
                L, kv = sc.input_seq_len, sc.num_kv_pairs
                n = max(1, math.ceil(RANK_FRAC * sc.num_examples))
                batch = E.make_mqar_eval(vocab, L, kv, n, seed=1234, device=DEV)  # [n, L] real MQAR
                cs = _chunk_size(L, n_heads)
                acc = {m: [] for m in _METRICS}
                with torch.no_grad():
                    for i in range(0, n, cs):
                        attn = extractor(model, batch[i:i + cs])      # [b,H,L,L]
                        st = E.rank_stats(attn, max_b=cs, max_l=L, emit_spectrum=False)
                        for m in _METRICS:
                            acc[m].extend(st.get(m + "_dist", []))
                        del attn
                        if DEV == "cuda":
                            torch.cuda.empty_cache()
                per_slice[f"kv{kv}"] = {"L": L, "n_examples": n, **{m: _agg(acc[m]) for m in _METRICS}}
            _append({"run_id": run_id, "ok": True, "probe": "rank", "family": family, "causal": causal,
                     "frac": RANK_FRAC, "per_slice": per_slice})
        except Exception as e:  # noqa: BLE001
            wrote_ok = False
            _append({"run_id": run_id, "ok": False, "probe": "rank", "family": family, "causal": causal,
                     "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1500:]})
    return wrote_ok


PROBES = {
    "rank": _probe_rank,        # per-slice SVD of the token-mixing matrix (the families with an extractor)
}


def main():
    run_ids = [r for r in os.environ.get("RUN_IDS", "").split(",") if r]
    by_id = _configs_by_runid()
    for rid in run_ids:
        cell, probe = _parse_runid(rid)
        cfg = by_id.get(cell)
        if cfg is None:
            _append({"run_id": rid, "ok": False, "error": "cell is in no MQAR experiment spec"})
            print(f"[sim] {rid}: no config", flush=True)
            continue
        if probe not in PROBES:
            _append({"run_id": rid, "ok": False, "error": f"unknown probe '{probe}'; have {list(PROBES)}"})
            print(f"[sim] {rid}: unknown probe", flush=True)
            continue
        ck = _ckpt_path(rid)
        if ck is None:
            _append({"run_id": rid, "ok": False, "probe": probe, "error": "checkpoint payload missing"})
            print(f"[sim] {rid}: no ckpt", flush=True)
            continue
        try:
            model = _build_model(cfg).to(DEV)
            sd = E._load_state_dict(ck, map_location=DEV, state_dict_key="model")
            model.load_state_dict(sd, strict=True)
            model.eval()
            print(f"[sim] {rid} probe={probe} on {DEV} (frac={RANK_FRAC})", flush=True)
            PROBES[probe](rid, cfg, model, cfg.model.vocab_size)
            print(f"[sim] {rid} done", flush=True)
        except Exception as e:  # noqa: BLE001 -- never abort the batch on one cell
            _append({"run_id": rid, "ok": False, "probe": probe,
                     "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1500:]})
            print(f"[sim] {rid} FAILED: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
