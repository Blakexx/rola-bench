"""ROLA-BENCH'S MEASUREMENT REGISTRY: the libraries rola is compared against, as units of rola-devtools' measurement
service (`rola_devtools.measure`), on the central cells.

    flash    causal attention through torch's forced flash backend (`attention.py`), a timed arm on every central QKV
             cell; the service gives it a memory node on each

A unit's identity is its code (the unit and every rola-bench file it imports) and the environment it measures on (the
GPU and driver, torch); the cell's record is the service's to key. The composer (`__main__.py`) runs this registry in a
rola target's venv, beside each rola checkout's `benchmarks/registry.py`.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from functools import cache
from importlib import metadata
from pathlib import Path

from rola_devtools.build import identity as ident
from rola_devtools.measure import Arm, Registration, Timed

ROOT = Path(__file__).resolve().parents[2]


@cache
def environment_key() -> str:
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,compute_cap", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=120).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        smi = ""
    return hashlib.sha256(json.dumps({"gpu": smi, "torch": metadata.version("torch")}).encode()).hexdigest()


@cache
def code_key() -> str:
    return ident.code(ROOT, "rola_bench/measure/registry.py", (".",), ("rola_bench/measure/attention.py",))


class Flash(Arm):
    location = "bench/memory"

    def accepts(self, cell) -> str | None:
        from rola_devtools.cells.qkv import QKVCell

        return None if isinstance(cell, QKVCell) else f"{cell.name}: attention takes a QKV cell"

    def identity(self, cell) -> dict:
        return {"code": code_key(), "environment": environment_key()}

    def setup(self, cell, ws: Path) -> Timed:
        from .attention import flash

        arm = flash(cell)
        return Timed(call=arm.call, built=arm.cell, instrument=arm.instrument)


def registry() -> list[Registration]:
    return [Registration("flash", f"{__name__}:Flash")]
