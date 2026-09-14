"""Box-side runner for the perf fleet job (FLEET_RUN_CMD -> this module).

Reads RUN_IDS (comma list of bench modules) from the env, runs each WHOLE bench on this box's GPU
preserving the in-process round-robin, and appends one ndjson summary row per run_id to
$FLEET_RESULTS — {"run_id", "ok", "gpu", <bench rows>}. The dispatcher's done_ids keys off that ok
row; the full per-cell table lives inside it (and in the CSVs the benches also write for the paper).

Mirrors mqar/run.py's contract: never abort the batch on one cell's failure; write an
ok=False row with the error so the dispatcher records it rather than silently dropping it.
"""
import csv
import json
import os
import sys
import traceback
from pathlib import Path

RESULTS = os.environ.get("FLEET_RESULTS", "/workspace/res.jsonl")
REPS = int(os.environ.get("PERF_REPS", "15"))


def _append(row):
    Path(RESULTS).parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS, "a") as fh:
        fh.write(json.dumps(row) + "\n")
        fh.flush()


def _read_csv(path):
    if not Path(path).exists():
        return []
    with open(path) as fh:
        return list(csv.DictReader(fh))


def run_scaling(out_dir, which):
    """One bench type of the scaling sweep (`which` in train, prefill, decode) as its own run_id, the axes from the
    box environment (SCALE_*)."""
    from rola_bench.perf import _common as C
    from rola_bench.perf import scaling as sc

    C.REPS = REPS
    default_lens = {"train": "1024,4096,16384,65536", "prefill": "1024,4096,16384,65536,131072",
                    "decode": "1024,4096,16384,65536,262144,1048576"}
    ints = lambda v: [int(x) for x in v.split(",") if x.strip()]  # noqa: E731
    lens = ints(os.environ.get(f"SCALE_LENS_{which.upper()}") or os.environ.get("SCALE_LENS") or default_lens[which])
    rows = sc.sweep([which], lens, ints(os.environ.get("SCALE_NS", "64,256,1024")),
                    ints(os.environ.get("SCALE_DVS", "64")), [])
    if rows:
        C.write_csv(Path(out_dir) / f"scaling_{which}.csv", rows)
    return {"scaling": rows, "bench": which}


BENCHES = {
    "perf-scaling-train": lambda o: run_scaling(o, "train"),
    "perf-scaling-prefill": lambda o: run_scaling(o, "prefill"),
    "perf-scaling-decode": lambda o: run_scaling(o, "decode"),
}


def _has_successful_measurement(data):
    rows = data.get("scaling") if isinstance(data, dict) else None
    if rows is None:
        return True
    timing_fields = ("fwd_ms", "fb_ms", "decode_ms")
    return any(
        any(r.get(k) not in (None, "ERR", "OOM", "UNBUILT") for k in timing_fields)
        for r in rows
    )


def main():
    import torch

    import rola_bench.perf._common as C
    run_ids = [r for r in os.environ.get("RUN_IDS", "").split(",") if r]
    # Capture CUDA status EXPLICITLY. Previously require_cuda() raised SystemExit on a failed init
    # (e.g. driver/runtime mismatch -> "named symbol not found"), which the per-bench except didn't
    # catch -> run_perf exited with EMPTY results and no recorded error (the A100 mystery). Now a
    # CUDA-init failure becomes a VISIBLE, synced failure row carrying the actual error string.
    cuda_ok = torch.cuda.is_available()
    cuda_err = ""
    if not cuda_ok:
        try:
            torch.cuda.init()
        except Exception as e:  # noqa: BLE001
            cuda_err = f"{type(e).__name__}: {e}"
    gpu = torch.cuda.get_device_name() if cuda_ok else "NO_CUDA"
    out_dir = str(C.RESULTS_DIR)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for rid in run_ids:
        if not cuda_ok:
            _append({"run_id": rid, "ok": False, "gpu": gpu,
                     "error": f"cuda unavailable: {cuda_err or 'is_available()=False'}"})
            print(f"[perf] {rid} FAILED: cuda unavailable ({cuda_err})", flush=True)
            continue
        fn = BENCHES.get(rid)
        if fn is None:
            _append({"run_id": rid, "ok": False, "gpu": gpu, "error": "unknown perf bench id"})
            print(f"[perf] unknown bench {rid!r}", flush=True)
            continue
        try:
            print(f"[perf] running {rid} on {gpu} (reps={REPS})", flush=True)
            data = fn(out_dir)
            ok = _has_successful_measurement(data)
            row = {"run_id": rid, "ok": ok, "gpu": gpu, "reps": REPS, **data}
            if not ok:
                row["error"] = "no successful timing rows"
            _append(row)
            print(f"[perf] {rid} {'OK' if ok else 'FAILED: no successful timing rows'}", flush=True)
        except Exception as e:  # noqa: BLE001 -- one bench's failure must not drop the rest
            _append({"run_id": rid, "ok": False, "gpu": gpu,
                     "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()})
            print(f"[perf] {rid} FAILED: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
