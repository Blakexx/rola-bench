"""The scaling benchmark: RoLA's layers against attention and the FLA baselines, layer against layer, over L.

Arms (`SCALE_ARMS`): `attn` (causal SDPA attention, the O(L^2) reference with a growing KV cache), the FLA baseline
layers from fla's (`gla`, `gdn`, `deltanet`, `gsa`, each at its canonical geometry), and the named RoLA cells
(`rola_bench.models.rola.NAMED`), each built through fla at N states per head and value width d_v. Every arm is its
canonical `nn.Module` at d_model = DM, timed as `module(x)`.

Axes:
  * bench   train (forward and backward) | prefill (forward only) | decode (one token into a seeded state)
  * L       sequence length, the scaling axis
  * N       RoLA states per head; a cell is kept only while RoLA's realized state is under attention's KV cache
  * d_v     RoLA's per-head value width

Within one (bench, L, N, d_v) cell the arms run round-robin, so drift charges each alike; memory is measured in a
separate one-arm pass. A cell an arm cannot run is a sentinel row, not a stop: `UNBUILT` when the library refuses it
(rola refusing a configuration outside its built envelope, or a pass it has not built), `OOM` when the device cannot
hold it, `ERR` otherwise. With `SCALE_PREFLIGHT_ARMS=1` each arm is first run once in a child process, so a crash that
would take the process down becomes that arm's row. RoLA's recurrent state is read off the built layer.

Run:  python -m rola_bench.perf.scaling [--benches train,prefill,decode] [--lens 1024,4096,16384]
          [--ns 64,256,1024] [--dvs 64] [--out DIR]
"""
import argparse
import contextlib
import gc
import json
import os
import subprocess
import sys
import tempfile
import traceback
import warnings
from functools import cache
from pathlib import Path

import torch

from rola_bench.models import rola as cells

from . import _common as C

DM = 512
DECODE_STEPS = 100
DECODE_WARMUP = 20
SEED_PREFILL = 256       # tokens that seed a recurrent state before the timed decode step
STATE_DTYPE = torch.float32

BASELINE_GEOM = {
    "gla": {"num_heads": 4, "expand_k": 2.0, "expand_v": 8.0, "gate_logit_normalizer": 16},
    "gdn": {"num_heads": 8, "head_dim": 256, "expand_v": 2.0},
    "deltanet": {"num_heads": 8, "expand_k": 1.0, "expand_v": 1.0},
    "gsa": {"num_heads": 4, "expand_k": 1.0, "expand_v": 1.0, "num_slots": 64},
}
BASELINE_CLASSES = {"gla": "GatedLinearAttention", "gdn": "GatedDeltaNet", "deltanet": "DeltaNet",
                    "gsa": "GatedSlotAttention"}
ROLA_ARMS = tuple(cells.NAMED)


def _infeasible_excs():
    oor = ()
    for m in ("triton.runtime.errors", "triton.compiler.errors"):
        with contextlib.suppress(Exception):
            oor = (__import__(m, fromlist=["OutOfResources"]).OutOfResources,)
            break
    return (torch.cuda.OutOfMemoryError, NotImplementedError) + oor


_INFEASIBLE = _infeasible_excs()


def _skip_label(exc) -> str:
    return "UNBUILT" if isinstance(exc, NotImplementedError) else "OOM"


def _env_flag(name, default=False):
    v = os.environ.get(name)
    return default if v is None else v not in ("0", "", "false", "False", "no", "No")


# ------------------------------------------------------------------ attention's backend
def _sdpa_flags():
    return {"flash": bool(torch.backends.cuda.flash_sdp_enabled()),
            "efficient": bool(torch.backends.cuda.mem_efficient_sdp_enabled()),
            "math": bool(torch.backends.cuda.math_sdp_enabled()),
            "cudnn": bool(getattr(torch.backends.cuda, "cudnn_sdp_enabled", lambda: False)())}


def _sdpa_backend_label():
    forced = os.environ.get("SCALE_ATTN_BACKEND", "").strip().lower()
    return f"forced:{forced}" if forced else "auto:" + "+".join(k for k, ok in _sdpa_flags().items() if ok)


def _sdpa_context():
    forced = os.environ.get("SCALE_ATTN_BACKEND", "").strip().lower()
    if not forced:
        return contextlib.nullcontext()
    from torch.nn.attention import SDPBackend, sdpa_kernel
    aliases = {"flash": SDPBackend.FLASH_ATTENTION, "efficient": SDPBackend.EFFICIENT_ATTENTION,
               "mem_efficient": SDPBackend.EFFICIENT_ATTENTION, "math": SDPBackend.MATH,
               "cudnn": SDPBackend.CUDNN_ATTENTION}
    if forced not in aliases:
        raise ValueError(f"unknown SCALE_ATTN_BACKEND={forced!r}; expected one of {sorted(aliases)}")
    return sdpa_kernel(aliases[forced])


# ------------------------------------------------------------------ the arms
class _CausalMHA(torch.nn.Module):
    """Causal multi-head attention through SDPA; `cache=(k, v)` attends a single token to a grown KV cache."""

    def __init__(self, d_model, n_heads):
        super().__init__()
        self.nh, self.hd = n_heads, d_model // n_heads
        self.qkv = torch.nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = torch.nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, cache=None):
        B, L, D = x.shape
        q, k, v = (t.view(B, L, self.nh, self.hd).transpose(1, 2) for t in self.qkv(x).chunk(3, dim=-1))
        if cache is not None:
            k, v = torch.cat([cache[0], k], dim=2), torch.cat([cache[1], v], dim=2)
        o = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=(L > 1))
        return self.proj(o.transpose(1, 2).reshape(B, L, D))


def build_baseline(kind):
    import fla.layers as layers

    return getattr(layers, BASELINE_CLASSES[kind])(hidden_size=DM, use_short_conv=False, layer_idx=0,
                                                  **BASELINE_GEOM[kind]).to(C.DEV, C.DT)


def build_rola(kind, n, dv):
    """A named RoLA cell at N states per head (uniform widths) and value width d_v, as an fla layer."""
    return cells.layer(cells.named(kind, n), hidden_size=DM, num_heads=C.H, head_v_dim=dv).to(C.DEV, C.DT)


@cache
def rola_state_floats(n, dv) -> int:
    """RoLA's recurrent floats per sequence at (N, d_v), read off a layer built on CPU (every named cell at one N and
    d_v carries the same state)."""
    content, overhead = cells.state_floats(cells.layer(cells.named(ROLA_ARMS[0], n), hidden_size=DM,
                                                       num_heads=C.H, head_v_dim=dv))
    return content + overhead


def _mib(n_elems, dtype):
    return round(n_elems * torch.empty((), dtype=dtype).element_size() / 2 ** 20, 1)


def _attn_cache_mib(L):
    return _mib(C.B * C.H * L * (DM // C.H) * 2, C.DT)


def state_x_pct(n, dv, L):
    """RoLA's state as a percent of attention's KV cache at L."""
    rola_bytes = rola_state_floats(n, dv) * torch.empty((), dtype=STATE_DTYPE).element_size()
    return round(100.0 * rola_bytes / (2 * L * DM * torch.empty((), dtype=C.DT).element_size()), 1)


def arm_kinds():
    kinds = tuple(x.strip() for x in os.environ.get("SCALE_ARMS", "attn,rola-base-rla,gla,gdn").split(",") if x.strip())
    allowed = {"attn", *BASELINE_GEOM, *ROLA_ARMS}
    unknown = sorted(set(kinds) - allowed)
    if unknown:
        raise ValueError(f"unknown SCALE_ARMS entries {unknown}; expected a subset of {sorted(allowed)}")
    return kinds


def arm_specs(include_graphed=False):
    """The cell's arms in order: attention, then the recurrent arms, each with a CUDA-graph decode variant when asked."""
    idx = 0
    for kind in arm_kinds():
        yield {"idx": idx, "kind": kind, "arm": kind, "graphed": False}
        idx += 1
        if include_graphed and kind != "attn":
            yield {"idx": idx, "kind": kind, "arm": f"{kind}-graphed", "graphed": True}
            idx += 1


def build_arm(spec, n, dv):
    if spec["kind"] == "attn":
        return _CausalMHA(DM, C.H).to(C.DEV, C.DT)
    if spec["kind"] in BASELINE_GEOM:
        return build_baseline(spec["kind"])
    return build_rola(spec["kind"], n, dv)


# ------------------------------------------------------------------ one sample
def _fwd(module, x):
    o = module(x)
    return o[0] if isinstance(o, tuple) else o


def _train_rep(module, x):
    """One forward and backward -> (fwd_ms, fwd_bwd_ms, peak_mib, act_mib); act_mib is what the forward retained."""
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    e0, e1, e2 = (torch.cuda.Event(enable_timing=True) for _ in range(3))
    leaf = x.detach().requires_grad_()
    probe = C.ActivationProbe()
    with probe:
        e0.record()
        o = _fwd(module, leaf)
        e1.record()
    torch.cuda.synchronize()
    o.float().sum().backward()
    e2.record()
    torch.cuda.synchronize()
    module.zero_grad(set_to_none=True)
    return e0.elapsed_time(e1), e0.elapsed_time(e2), (torch.cuda.max_memory_allocated() - base) / 2 ** 20, probe.mib


def _prefill_rep(module, x):
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    _fwd(module, x)
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1), (torch.cuda.max_memory_allocated() - base) / 2 ** 20


def _sample(rec):
    bench, spec, module, x = rec["bench"], rec["spec"], rec["module"], rec["x"]
    context = _sdpa_context() if spec["kind"] == "attn" else contextlib.nullcontext()
    with context:
        if bench == "train":
            return _train_rep(module, x)
        with torch.inference_mode():
            return _prefill_rep(module, x)


def _prepare(bench, L, n, dv, spec):
    seed = 10_000 + L + dv * 31 + n * 17 + spec["idx"]
    torch.manual_seed(seed)
    module = build_arm(spec, n, dv)
    torch.manual_seed(seed + 1)
    return {"bench": bench, "spec": spec, "module": module, "x": torch.randn(C.B, L, DM, device=C.DEV, dtype=C.DT)}


def _decode_cache_sample(module, L, graphed=False):
    """One decode step into a state seeded by a bounded prefill, through FLA's cache contract."""
    from fla.models.utils import Cache

    module.eval()
    x_pre = torch.randn(C.B, min(max(1, L), SEED_PREFILL), DM, device=C.DEV, dtype=C.DT)
    x1 = torch.randn(C.B, 1, DM, device=C.DEV, dtype=C.DT)
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    with torch.inference_mode():
        past = Cache()
        module(x_pre, past_key_values=past, use_cache=True)
        if len(past) == 0:
            raise RuntimeError("decode: the prefill seeded no cache, so the timed step would run cold")
        step = lambda: module(x1, past_key_values=past, use_cache=True)  # noqa: E731
        if graphed:
            static_in = x1.clone()
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    module(static_in, past_key_values=past, use_cache=True)
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                module(static_in, past_key_values=past, use_cache=True)

            def step():
                static_in.copy_(x1)
                graph.replay()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        t0.record()
        step()
        t1.record()
        torch.cuda.synchronize()
    return t0.elapsed_time(t1), round((torch.cuda.max_memory_allocated() - base) / 2 ** 20, 1)


def _decode_attn_sample(L):
    """One query token through attention over an L-long KV cache."""
    m = _CausalMHA(DM, C.H).to(C.DEV, C.DT)
    x1 = torch.randn(C.B, 1, DM, device=C.DEV, dtype=C.DT)
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    kc, vc = (torch.randn(C.B, C.H, L, DM // C.H, device=C.DEV, dtype=C.DT) for _ in range(2))
    with torch.inference_mode(), _sdpa_context():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        t0.record()
        m(x1, cache=(kc, vc))
        t1.record()
        torch.cuda.synchronize()
    return t0.elapsed_time(t1), round((torch.cuda.max_memory_allocated() - base) / 2 ** 20, 1)


def _decode_sample(L, n, dv, spec):
    """(ms, state_mib, peak_mib) for one decode sample of an arm."""
    torch.manual_seed(20_000 + L + dv * 31 + n * 17 + spec["idx"])
    if spec["kind"] == "attn":
        ms, peak = _decode_attn_sample(L)
        return ms, _attn_cache_mib(L), peak
    module = build_arm(spec, n, dv)
    ms, peak = _decode_cache_sample(module, L, graphed=spec["graphed"])
    state = _mib(C.B * rola_state_floats(n, dv), STATE_DTYPE) if spec["kind"] in ROLA_ARMS else None
    return ms, state, peak


# ------------------------------------------------------------------ rows
def _row(**kw):
    base = {"bench": None, "L": None, "X_pct": None, "n": None, "dv": None, "arm": None, "fwd_ms": None,
                "fwd_iqr_ms": None, "fb_ms": None, "fb_iqr_ms": None, "peak_mib": None, "act_mib": None,
                "fwd_iqr_pct": None, "fb_iqr_pct": None, "decode_ms": None,
                "decode_iqr_ms": None, "decode_iqr_pct": None, "fwd_samples_ms": None, "fb_samples_ms": None,
                "decode_samples_ms": None, "peak_samples_mib": None, "act_samples_mib": None, "state_mib": None,
                "attn_backend": None, "sdpa_flags": None}
    base.update(kw)
    return base


def _attn_fields(spec):
    attn = spec["kind"] == "attn"
    return {"attn_backend": _sdpa_backend_label() if attn else None, "sdpa_flags": str(_sdpa_flags()) if attn else None}


def _error_row(rows, bench, cell, spec, label, exc=None, detail=None):
    L, x_pct, n, dv = cell
    row = _row(bench=bench, L=L, X_pct=x_pct, n=n, dv=dv, arm=spec["arm"])
    row["decode_ms" if bench == "decode" else "fwd_ms"] = label
    if bench == "train":
        row["fb_ms"] = label
    if exc is not None:
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["trace"] = traceback.format_exc()
    if detail:
        row.update(detail)
    rows.append(row)
    print(f"{label} {spec['arm']} {bench} L{L} N{n} dv{dv}: {row.get('error')}", flush=True)
    return row


def _guarded(rows, bench, cell, spec, fn):
    """fn() or None, a failure written as the arm's sentinel row."""
    try:
        return fn()
    except _INFEASIBLE as e:
        _error_row(rows, bench, cell, spec, _skip_label(e), e)
    except Exception as e:  # noqa: BLE001 -- a cell's failure is its row; the sweep continues
        _error_row(rows, bench, cell, spec, "ERR", e)
    finally:
        with contextlib.suppress(Exception):
            torch.cuda.empty_cache()
    return None


def _summarize(rows, bench, cell, spec, recs):
    L, x_pct, n, dv = cell
    row = _row(bench=bench, L=L, X_pct=x_pct, n=n, dv=dv, arm=spec["arm"], **_attn_fields(spec))
    if bench == "train":
        f, fb, pk, ac = zip(*recs, strict=True)
        fs, fbs = C.sample_stats(f), C.sample_stats(fb)
        row.update(fwd_ms=fs["median"], fwd_iqr_ms=fs["iqr"], fwd_iqr_pct=fs["iqr_pct"], fb_ms=fbs["median"],
                   fb_iqr_ms=fbs["iqr"], fb_iqr_pct=fbs["iqr_pct"], peak_mib=round(max(pk)), act_mib=round(max(ac)),
                   fwd_samples_ms=C.samples_json(f), fb_samples_ms=C.samples_json(fb),
                   peak_samples_mib=C.samples_json(pk, digits=1), act_samples_mib=C.samples_json(ac, digits=1))
    elif bench == "prefill":
        f, pk = zip(*recs, strict=True)
        fs = C.sample_stats(f)
        row.update(fwd_ms=fs["median"], fwd_iqr_ms=fs["iqr"], fwd_iqr_pct=fs["iqr_pct"], peak_mib=round(max(pk)),
                   fwd_samples_ms=C.samples_json(f), peak_samples_mib=C.samples_json(pk, digits=1))
    else:
        ms, state, pk = zip(*recs, strict=True)
        ds = C.sample_stats(ms, digits=4)
        row.update(decode_ms=ds["median"], decode_iqr_ms=ds["iqr"], decode_iqr_pct=ds["iqr_pct"],
                   decode_samples_ms=C.samples_json(ms, digits=4), state_mib=state[0], peak_mib=round(max(pk)),
                   peak_samples_mib=C.samples_json(pk, digits=1))
    rows.append(row)
    print(row, flush=True)


# ------------------------------------------------------------------ preflight: one arm in a child process
def _preflight(bench, cell, spec):
    """{'ok': True} or the failure a child process met running the arm once."""
    if not _env_flag("SCALE_PREFLIGHT_ARMS"):
        return {"ok": True}
    L, _x_pct, n, dv = cell
    fd, out = tempfile.mkstemp(prefix="rola-preflight-", suffix=".json")
    os.close(fd)
    cmd = [sys.executable, "-m", "rola_bench.perf.scaling", "--preflight", json.dumps(
        {"out": out, "bench": bench, "L": L, "n": n, "dv": dv, "spec": spec})]
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, check=False,
                              timeout=int(os.environ.get("SCALE_PREFLIGHT_TIMEOUT", "900")),
                              env={**os.environ, "PYTHONUNBUFFERED": "1"})
        try:
            data = json.loads(Path(out).read_text())
        except (OSError, ValueError) as e:
            data = {"ok": False, "label": "ERR", "error": f"preflight wrote no result: {e}"}
        if proc.returncode and data.get("ok"):
            data = {"ok": False, "label": "ERR", "error": f"preflight exited {proc.returncode}"}
        if not data.get("ok"):
            data["stderr"] = proc.stderr[-4000:]
        return data
    except subprocess.TimeoutExpired as e:
        return {"ok": False, "label": "ERR", "error": f"preflight timed out after {e.timeout}s"}
    finally:
        Path(out).unlink(missing_ok=True)


def _preflight_child(job: dict) -> None:
    bench, L, n, dv, spec = job["bench"], job["L"], job["n"], job["dv"], job["spec"]
    try:
        if bench == "decode":
            _decode_sample(L, n, dv, spec)
        else:
            _sample(_prepare(bench, L, n, dv, spec))
        data = {"ok": True}
    except _INFEASIBLE as e:
        data = {"ok": False, "label": _skip_label(e), "error": f"{type(e).__name__}: {e}"}
    except Exception as e:  # noqa: BLE001 -- the child reports whatever stopped the arm
        data = {"ok": False, "label": "ERR", "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()}
    Path(job["out"]).write_text(json.dumps(data))


# ------------------------------------------------------------------ a cell, round-robin
def timed_cell(bench, cell, rows):
    """Every arm of one (bench, L, N, d_v) cell, interleaved; memory re-measured one arm at a time."""
    L, _x_pct, n, dv = cell
    decode = bench == "decode"
    active = []
    for spec in arm_specs(include_graphed=decode and _env_flag("SCALE_DECODE_GRAPHED")):
        flight = _preflight(bench, cell, spec)
        if not flight.get("ok"):
            _error_row(rows, bench, cell, spec, flight.get("label", "ERR"),
                       detail={k: v for k, v in flight.items() if k in ("error", "trace", "stderr")})
            continue
        prepared = True if decode else _guarded(rows, bench, cell, spec, lambda s=spec: _prepare(bench, L, n, dv, s))
        if prepared is not None:
            active.append({"spec": spec, "prepared": prepared, "data": []})

    warmup, reps = (DECODE_WARMUP, DECODE_STEPS) if decode else (C.WARMUP, C.REPS)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*format mismatch.*")
        for r in range(warmup + reps):
            survivors = []
            for rec in C.rotated(active, r):
                fn = ((lambda s=rec["spec"]: _decode_sample(L, n, dv, s)) if decode
                      else (lambda p=rec["prepared"]: _sample(p)))
                sample = _guarded(rows, bench, cell, rec["spec"], fn)
                if sample is None:
                    continue
                if r >= warmup:
                    rec["data"].append(sample)
                survivors.append(rec)
            active = survivors
            if not active:
                break

    for rec in active:
        if not rec["data"]:
            continue
        if not decode:
            spec = rec["spec"]
            isolated = _guarded(rows, bench, cell, spec, lambda s=spec: _isolated_sample(bench, L, n, dv, s))
            if isolated is None:
                continue
            memory = isolated[2:] if bench == "train" else isolated[1:]
            rec["data"] = [sample[:len(sample) - len(memory)] + memory for sample in rec["data"]]
        _summarize(rows, bench, cell, rec["spec"], rec["data"])
    del active
    gc.collect()


def _isolated_sample(bench, L, n, dv, spec):
    """A second warm sample of one arm alone, for its memory."""
    rec = _prepare(bench, L, n, dv, spec)
    _sample(rec)
    return _sample(rec)


def sweep(benches, lens, ns, dvs, rows):
    for bench in benches:
        for L in lens:
            for dv in dvs:
                for n in ns:
                    x_pct = state_x_pct(n, dv, L)
                    if x_pct < 100.0:
                        timed_cell(bench, (L, x_pct, n, dv), rows)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preflight", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--benches", default="train,prefill,decode")
    ap.add_argument("--lens", default=os.environ.get("SCALE_LENS", "1024,4096,16384,65536"))
    ap.add_argument("--ns", default=os.environ.get("SCALE_NS", "64,256,1024"), help="RoLA states per head")
    ap.add_argument("--dvs", default=os.environ.get("SCALE_DVS", "64"), help="RoLA per-head value widths")
    ap.add_argument("--reps", type=int, default=C.REPS)
    ap.add_argument("--out", default=str(C.RESULTS_DIR))
    a = ap.parse_args()
    C.require_cuda()
    if a.preflight:
        _preflight_child(json.loads(a.preflight))
        return
    C.REPS = a.reps
    ints = lambda s: [int(v) for v in s.split(",") if v.strip()]  # noqa: E731
    benches = [b for b in a.benches.split(",") if b.strip()]
    print(f"# scaling | {C.gpu_name()} | B={C.B} H={C.H} d_model={DM} bf16 | benches={benches} | L={a.lens} "
          f"N={a.ns} d_v={a.dvs} | arms={arm_kinds()}", flush=True)
    rows = []
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        sweep(benches, ints(a.lens), ints(a.ns), ints(a.dvs), rows)
    finally:
        if rows:
            C.write_csv(out / "scaling.csv", rows)
            print(f"\nWROTE {out}/scaling.csv ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
