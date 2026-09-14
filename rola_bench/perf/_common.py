"""The perf timing protocol: round-robin interleaving, CUDA-event timing with warmup excluded, median and IQR over the
kept samples, and honest activation accounting (bytes retained for backward, deduped by storage).

Every round cycles all arms of a cell before the next, rotating their order, so clock and thermal drift charge each arm
alike and the ratios between arms are drift-free. Run on an idle GPU.
"""
import csv
import json
import os
import statistics
import tempfile
from pathlib import Path

os.environ.setdefault("HF_HOME", "/tmp/rola-bench-hf")
os.environ.setdefault("HF_HUB_CACHE", os.path.join(os.environ["HF_HOME"], "hub"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.environ["HF_HUB_CACHE"])

import torch

# Benchmark shapes. DEFAULT = the canonical LM-training shape (the headline efficiency table).
# These are module globals so the arms (chunked.mk / mk_gdn / ...) read them at call time; the
# per-cell loop overrides them via set_shape() to time at each MQAR cell's exact dims (task #98 —
# cost-vs-accuracy), with no arm-signature changes. nc is passed explicitly (not a global).
B = int(os.environ.get("PERF_B", "8"))      # batch; drop it (e.g. 2) so a long L fits
H = 8
DEV, DT = "cuda", torch.bfloat16

REPS, WARMUP = 15, 4

#: where a run writes its CSVs; a recorded result is the fleet row (`rola_results` at perf/<config>)
RESULTS_DIR = Path(tempfile.gettempdir()) / "rola-bench-perf"


def median(xs):
    return statistics.median(xs)


def iqr(xs):
    """Inter-quartile range; 0 for <2 samples (statistics.quantiles needs >=2)."""
    if len(xs) < 2:
        return 0.0
    q = statistics.quantiles(xs, n=4)
    return q[2] - q[0]


def _round_float(x, digits=3):
    return round(float(x), digits)


def sample_stats(xs, digits=3):
    """Robust timing summary for benchmark samples.

    Median/IQR remain the headline fields. The percentage fields make noisy cells easy to filter before
    plotting, while raw samples remain available for paired/round-robin analysis.
    """
    vals = [float(x) for x in xs]
    if not vals:
        return {
            "n": 0,
            "median": None,
            "iqr": None,
            "iqr_pct": None,
            "mean": None,
            "stdev": None,
            "cv_pct": None,
            "min": None,
            "max": None,
            "samples": [],
        }
    med = median(vals)
    spread = iqr(vals)
    mean = statistics.fmean(vals)
    stdev = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    denom = abs(med) if med else 0.0
    return {
        "n": len(vals),
        "median": _round_float(med, digits),
        "iqr": _round_float(spread, digits),
        "iqr_pct": _round_float(100.0 * spread / denom, 2) if denom else 0.0,
        "mean": _round_float(mean, digits),
        "stdev": _round_float(stdev, digits),
        "cv_pct": _round_float(100.0 * stdev / abs(mean), 2) if mean else 0.0,
        "min": _round_float(min(vals), digits),
        "max": _round_float(max(vals), digits),
        "samples": [_round_float(v, digits) for v in vals],
    }


def samples_json(xs, digits=3):
    return json.dumps([_round_float(x, digits) for x in xs], separators=(",", ":"))


def rotated(items, round_idx):
    """Rotate a list for position-fair round-robin timing within a cell."""
    if not items:
        return items
    k = round_idx % len(items)
    return items[k:] + items[:k]


def gpu_name():
    return torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"


def require_cuda():
    if not torch.cuda.is_available():
        raise SystemExit("perf benchmarks require a CUDA GPU (torch.cuda.is_available() is False)")


class ActivationProbe:
    """Context manager recording bytes retained for backward (deduped by storage)."""

    def __init__(self):
        self.saved = {}

    def __enter__(self):
        def pack(x):
            if x.is_cuda:
                self.saved[x.untyped_storage().data_ptr()] = x.untyped_storage().nbytes()
            return x
        self._hook = torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x)
        self._hook.__enter__()
        return self

    def __exit__(self, *a):
        return self._hook.__exit__(*a)

    @property
    def mib(self):
        return sum(self.saved.values()) / 2 ** 20


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    return path
