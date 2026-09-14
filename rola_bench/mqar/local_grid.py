#!/usr/bin/env python3
"""LOCAL overnight dispatcher for an MQAR-family grid — one tier, sequentially, on this box's card.

    GRID_TIERS=decoupling python -m rola_bench.mqar.local_grid --config graph_grid --dry-run
    GRID_TIERS=decoupling flock /tmp/rola_gpu.lock \
        python -m rola_bench.mqar.local_grid --config graph_grid

`python -m rola_bench.fleet` is the FLEET orchestrator: it rents boxes and never trains locally. This is its local
counterpart for a grid that is small enough to run on the developer card overnight
([[local-small-rented-big]]: local stays small BY DESIGN — tiers, gates, micro-probes; the big
anchors and the publishable LM arms go to rented A100-class boxes). It reuses the fleet's own
machinery rather than reimplementing it: the spec expands through `build_configs`, each cell trains
through `run.run_one` (the same subprocess, the same stdout parsing, the same result
record), and resume is by `run_id`.

THE THREE LOCAL-BOX RULES, all enforced here rather than remembered:

  * CUDA IS EXCLUSIVE ON THIS BOX. Always run under `flock /tmp/rola_gpu.lock`. If this process can
    take that lock itself, nobody was holding it, which means the flock prefix was forgotten — the
    dispatcher refuses and prints the command it wanted. (It does not TAKE the lock: `flock(1)` holds
    it on its own descriptor, so a second acquisition here would deadlock against the very prefix it
    is checking for.)
  * MEMORY IS SHARED WITH AN INTERACTIVE SESSION. Every training subprocess runs under
    `torch.cuda.set_per_process_memory_fraction(--mem-fraction)`, default 0.65, applied inside the
    child (`run.run_one` reads `ROLA_GPU_MEM_FRACTION`). A cell that would have swallowed
    the card OOMs itself instead of taking the session down with it.
  * ONE CELL AT A TIME. Cells are strictly sequential — never two training processes on the card.

FOOTPRINT, printed per cell before it launches and totalled by `--dry-run`. The estimate is a stated
sum of terms, not a fudge factor (each is named in `footprint`), and every run records BOTH the
estimate and the child's own `torch.cuda.max_memory_allocated()`, so the estimator is falsifiable
rather than decorative. MEASURED on the 2026-08-02 smoke: measured/estimated in [0.90, 1.12] across
the six conditions, i.e. it is a planning number good to ~15%, and the per-process cap above is what
actually protects the box.

WHEN THE TIER FINISHES, `rola_bench.mqar.analysis.graph` turns its rows into the headline statistic
(measured collapse threshold N* vs the solved chi of each condition's graph).
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import sys
import time
from pathlib import Path

from rola_results import Store, checkout, key

REPO = Path(__file__).resolve().parents[2]


def local_store(config: str):
    """The local grid's store (`rola_results` at `mqar/<config>`, beside the fleet's rows) and its semantics: the cell
    and the code that trains it -- this checkout and the rola engine's, each commit and tracked diff."""
    import importlib.util

    code = {}
    for name, path in (("rola_bench", REPO), ("rola", Path(importlib.util.find_spec("rola").origin).resolve().parents[1])):
        here = checkout(path)
        code[name] = {"git_sha": here["git_sha"], "diff_sha256": here["diff_sha256"]}

    def semantics(cell: str) -> dict:
        return {"bench": "mqar", "config": config, "cell": cell, "local": code}

    return Store(f"mqar/{config}"), semantics


GPU_LOCK = "/tmp/rola_gpu.lock"

#: Bytes per parameter held on the card during training: the fp32 weight, its gradient, and Adam's
#: two moments.
BYTES_PER_PARAM = 4 * 4
#: Live copies of the [batch, L, vocab] logit tensor at the loss: the logits themselves, the fp32
#: cast cross-entropy makes, and its gradient. This term dominates every cell in the MQAR family
#: (vocab 8192 x L 128 x batch 128 is ~537 MB per copy), which is why it is spelled out.
LOGIT_COPIES = 3
#: Live copies of the [batch, heads, L, L] attention score tensor, for the ATTENTION arms only. It is
#: quadratic in L and linear in the HEAD COUNT, so the narrow rungs of a d_qk ladder are the
#: expensive ones (d_qk = 2 at d_model = 128 is 64 heads). MEASURED: the smoke's
#: `mha-dqk2 @ L=512, batch 64` cell needs 4 GiB for one copy and OOMs under the 0.65 cap — this term
#: exists so `--dry-run` says so before the night starts, instead of the tier discovering it.
ATTN_SCORE_COPIES = 2
#: Saved activations per token per layer, in units of d_model floats. Measured-by-calibration, not
#: derived: the hybrid block's short conv, projections, routing logits and residuals all contribute,
#: and the constant is the one term in `footprint` that is fitted rather than counted.
ACT_FLOATS_PER_TOKEN_PER_LAYER = 64


def _refuse_without_the_gpu_lock():
    """Refuse unless someone else already holds `/tmp/rola_gpu.lock` — i.e. unless we are running
    under the `flock` prefix. Two training processes on this card is the failure this prevents."""
    fh = open(GPU_LOCK, "w")  # noqa: SIM115 -- closed on both branches below, after the lock probe
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return                     # held by our own flock(1) prefix: correct
    fcntl.flock(fh, fcntl.LOCK_UN)
    fh.close()
    sys.exit(
        f"REFUSING: nothing holds {GPU_LOCK}, so this was not launched under the lock. CUDA is "
        f"exclusive on this box. Re-run as:\n\n    GRID_TIERS=$TIER flock {GPU_LOCK} "
        f"{sys.executable} -m rola_bench.mqar.local_grid --config <spec>\n\n"
        "(use --dry-run to inspect the cell list without the lock).")


def _n_params(cfg) -> int:
    """Trainable parameters of the cell, COUNTED off the constructed model rather than estimated.

    The model is built on CPU; at this grid's scale (d_model 128, 2 layers) that is a fraction of a
    second and it is the only way the embedding/head term is right for a given vocabulary."""
    from zoology.model import LanguageModel
    model = LanguageModel(config=cfg.model)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def footprint(cfg, n_params: int) -> dict:
    """Estimated peak device bytes for one cell, as a sum of NAMED terms.

    `params`     weights + grads + Adam moments.
    `logits`     the [batch, L, vocab] tensor and its live copies at the loss — the dominant term.
    `activations` saved activations, `ACT_FLOATS_PER_TOKEN_PER_LAYER` per token per layer.
    `state`      the recurrent state carried across chunks, batch x H x N x d_v per layer, read off
                 the cell's realized kernel config (never a formula — [[realized state]] rule).
    """
    from rola_bench.mqar import realized_state
    batch = cfg.data.batch_size[0] if isinstance(cfg.data.batch_size, (list, tuple)) \
        else cfg.data.batch_size
    seq = max(c.input_seq_len for c in cfg.data.train_configs)
    d_model, layers = cfg.model.d_model, cfg.model.n_layers
    kernel = cfg.model.sequence_mixer.kwargs["configs"][-1]
    content, overhead = realized_state.state_floats(kernel, d_model)
    heads = kernel["kwargs"].get("num_heads", 0) if kernel["name"].endswith("MHA") else 0
    terms = {
        "params": n_params * BYTES_PER_PARAM,
        "logits": batch * seq * cfg.model.vocab_size * 4 * LOGIT_COPIES,
        "activations": batch * seq * d_model * layers * ACT_FLOATS_PER_TOKEN_PER_LAYER * 4,
        "state": (0 if content is None else (content + overhead) * batch * layers * 4),
        "attn_scores": batch * heads * seq * seq * 4 * ATTN_SCORE_COPIES,
    }
    terms["total"] = sum(terms.values())
    return terms


def _smoke_cells(configs, envs, epochs=2, train_examples=512, test_examples=128):
    """One TOY cell per data condition: the smallest state rung of the first arm, shrunk to a couple
    of epochs. It goes through the same `run_one` the grid uses, so it exercises the whole path the
    grid does — data build, stamp, model construction, train, eval, slice metrics, result emit — and
    a break in any of them fails here rather than three cells into an overnight tier.

    One cell PER CONDITION, because the conditions are what this file adds: the four constructions
    have genuinely different episode geometry (a block layout vs an interleaved lifetime schedule),
    and a smoke that only covered one of them would not have tested the others at all.
    """
    import copy
    picked, out_c, out_e = set(), [], []
    for cfg, env in zip(configs, envs, strict=True):
        cond = cfg.run_id.split("_s")[-1].partition("-")[2]     # the "-{cond}" run_id suffix
        # one cell per (condition, ARM FAMILY): the attention reference reaches the model through a
        # different mixer, so a smoke that only covered the routed arms would leave half the grid's
        # cells unexercised.
        key = (cond, "mha" if cfg.run_id.startswith("grid-mha") else "routed")
        if key in picked:
            continue
        picked.add(key)
        c = copy.deepcopy(cfg)
        c.max_epochs = epochs
        for seg in c.data.train_configs:
            seg.num_examples = train_examples
        for seg in c.data.test_configs:
            seg.num_examples = test_examples
        c.data.cache_dir = None            # toy segments must never poison the grid's segment cache
        c.run_id = f"smoke-{c.run_id}"
        out_c.append(c)
        out_e.append(dict(env))
    return out_c, out_e


def _steps(cfg) -> int:
    examples = sum(c.num_examples for c in cfg.data.train_configs)
    batch = cfg.data.batch_size[0] if isinstance(cfg.data.batch_size, (list, tuple)) \
        else cfg.data.batch_size
    return math.ceil(examples / batch) * cfg.max_epochs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="experiment spec (experiments/<name>.yaml)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the cell list, per-cell footprint and the wall-clock estimate; "
                         "train nothing and take no lock")
    ap.add_argument("--mem-fraction", type=float, default=0.65,
                    help="per-training-process cap on this card (default 0.65)")
    ap.add_argument("--sec-per-step", type=float, default=None,
                    help="wall-clock model for --dry-run: seconds per optimizer step at L=128 on "
                         "this box (default: the measured value in SEC_PER_STEP_L128)")
    ap.add_argument("--limit", type=int, default=None, help="run only the first N pending cells")
    ap.add_argument("--smoke", action="store_true",
                    help="ONE toy cell per data condition (2 epochs, 512 train examples, the "
                         "smallest N) through the real runner — the end-to-end pipeline gate")
    a = ap.parse_args(argv)

    from rola_bench.mqar.run import load_configs, run_one

    tier = os.environ.get("GRID_TIERS", "all")
    configs, envs = load_configs(a.config)
    if a.smoke:
        configs, envs = _smoke_cells(configs, envs)
    store, semantics = local_store(a.config)
    #: DONE FOR THIS CODE: a cell is complete when the current checkouts (this repository and the rola engine it
    #: trains) have an ok sample, the local counterpart of the fleet's current image (`rola_bench.fleet.jobs`).
    done = {c.run_id for c in configs if Store.complete(store.get(key(semantics(c.run_id))))}

    pending = [(i, c, e) for i, (c, e) in enumerate(zip(configs, envs, strict=True)) if c.run_id not in done]
    if a.limit:
        pending = pending[:a.limit]

    sec_per_step = a.sec_per_step if a.sec_per_step is not None else SEC_PER_STEP_L128
    print(f"spec={a.config} tier={tier}: {len(configs)} cells, {len(done)} done, "
          f"{len(pending)} to run -> {store.location}", flush=True)
    total_bytes, total_sec = 0, 0.0
    plan = []
    for i, cfg, _env in pending:
        fp = footprint(cfg, _n_params(cfg))
        steps = _steps(cfg)
        seq = max(c.input_seq_len for c in cfg.data.train_configs)
        est = steps * sec_per_step * seq / 128.0        # per-step cost is ~linear in L at this size
        plan.append((i, cfg, fp, steps, est))
        total_bytes = max(total_bytes, fp["total"])
        total_sec += est
        print(f"  {cfg.run_id:64s} peak~{fp['total']/2**30:5.2f}GiB "
              f"(logits {fp['logits']/2**30:.2f} + act {fp['activations']/2**30:.2f} + "
              f"state {fp['state']/2**20:.0f}MiB + attn {fp['attn_scores']/2**30:.2f} + "
              f"params {fp['params']/2**20:.0f}MiB) "
              f"{steps} steps ~{est/60:.0f}min", flush=True)
    print(f"\nTIER TOTAL: {len(plan)} cells, worst-cell peak ~{total_bytes/2**30:.2f} GiB, "
          f"estimated wall clock ~{total_sec/3600:.1f} h "
          f"(at {sec_per_step*1000:.0f} ms/step at L=128)", flush=True)
    if a.dry_run:
        return 0

    _refuse_without_the_gpu_lock()
    env_cap = f"{a.mem_fraction}"
    os.environ["ROLA_GPU_MEM_FRACTION"] = env_cap
    print(f"per-process memory cap: {a.mem_fraction} of the card; cells run SEQUENTIALLY", flush=True)
    bad = []
    for n, (i, cfg, fp, _, est) in enumerate(plan, 1):
        print(f"\n[{n}/{len(plan)}] {cfg.run_id} | peak~{fp['total']/2**30:.2f}GiB | "
              f"~{est/60:.0f}min", flush=True)
        t0 = time.time()
        peak_file = Path(f"/tmp/_rla_peak_{i}.json")
        peak_file.unlink(missing_ok=True)
        os.environ["ROLA_PEAK_MEM_PATH"] = str(peak_file)
        r = run_one(i, cfg.run_id, cfg, dict(envs[i]))
        r["footprint_estimate_bytes"] = fp
        r["wall_clock_estimate_s"] = est
        r["peak_bytes"] = (json.loads(peak_file.read_text())["peak_bytes"]
                           if peak_file.exists() else None)
        if r["peak_bytes"]:
            print(f"  peak {r['peak_bytes']/2**30:.2f}GiB measured vs "
                  f"{fp['total']/2**30:.2f}GiB estimated "
                  f"({r['peak_bytes']/fp['total']:.2f}x)", flush=True)
        if r.get("ok"):
            store.put(semantics(cfg.run_id), output=r, wall_s=time.time() - t0, provenance={"tier": tier})
        else:
            store.put(semantics(cfg.run_id), error=str(r.get("error") or "the cell reported ok=false"),
                      wall_s=time.time() - t0, provenance={"tier": tier}, row=r)
        ok = bool(r.get("ok"))
        if not ok:
            bad.append(cfg.run_id)
        print(f"  -> {'OK' if ok else 'FAIL'} max_acc={r.get('max_acc', 0):.3f} "
              f"t={time.time()-t0:.0f}s (est {est:.0f}s)", flush=True)
    if bad:
        print(f"\n{len(bad)} FAILED cell(s): {bad}", flush=True)
    return 1 if bad else 0


#: Seconds per optimizer step at L=128, batch 64, on the local card (RTX 3080 Ti). A planning number
#: for `--dry-run` only; override with `--sec-per-step`. Cells at other L are scaled linearly in L,
#: which is what the logit and attention-free mixer terms both do at this size.
#:
#: RE-PRICED 2026-08-15 on the STANDALONE CUDA ENGINE (campaign-dispatch B4, b2-3_build.md §3). The
#: previous 0.139 was measured 2026-08-02 against the fp64 oracle, which is what every routed cell
#: trained on before the repoint — a different instrument, not a slower box. The B4 smoke measured
#: 7.7-8.2 ms/step at L=64 and 18.0-20.8 ms/step at L=256 across all four wirings, both rungs; the
#: linear-in-L model those two brackets is 11.5 ms at L=128, and that interpolation is stated rather
#: than hidden because no routed cell runs AT L=128 on this grid.
SEC_PER_STEP_L128 = 0.0115


if __name__ == "__main__":
    sys.exit(main())
