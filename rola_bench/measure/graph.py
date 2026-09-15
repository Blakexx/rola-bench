"""ROLA-BENCH'S MEASUREMENT GRAPH: the libraries rola is compared against, as units of `rola_devtools.graph`.

    time.flash@<cell>     causal attention through torch's forced flash backend, timed for a session (`attention.py`)
    memory.flash@<cell>   the same call alone: the device's peak allocated and reserved bytes over its calls

The cells are this package's `registry.json` (`cells.py:qkv`). A unit's identity is its code (the unit and every
rola-bench file it imports, with the registry), the cell's parameters and the environment it measures on (the GPU and
driver, torch). The composer (`compose.py`) runs this graph in a rola target's venv, beside rola's own graph.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from functools import cache
from importlib import metadata
from pathlib import Path

from rola_devtools.graph import Node, Refusal, Timed, Unit
from rola_devtools.graph import identity as ident

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = Path(__file__).with_name("registry.json")
#: the calls a memory node's peak is taken over, after one warm call (rola's graph takes the same)
MEMORY_CALLS = 5


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
    return ident.code(ROOT, "rola_bench/measure/graph.py", (".",),
                      ("rola_bench/measure/attention.py", "rola_bench/measure/registry.json"))


@cache
def _registry():
    """This package's cells alone: its groups name rola's cells, which only a composer holding a rola checkout resolves."""
    from rola_devtools.cells import Registry

    doc = json.loads(REGISTRY.read_text())
    return Registry({c["name"]: {"name": c["name"], "data": c.get("data", doc["data"]),
                                 "params": {k: v for k, v in c.items() if k not in ("name", "data")}}
                     for c in doc["cells"]}, {})


def _flash(cell: str):
    from rola_devtools.cells import build

    from .attention import arms

    try:
        builders = arms(build(_registry().cell(cell)))
    except TypeError as why:
        raise Refusal(f"{cell}: {why}") from why
    return builders["flash"]()


class Flash(Unit):
    location = "bench/time"
    timed = True

    def __init__(self, cell: str) -> None:
        self.cell = cell

    def identity(self) -> dict:
        return {"code": code_key(), "environment": environment_key(), "cell": _registry().cell(self.cell)["params"]}

    def setup(self, ws: Path):
        arm = _flash(self.cell)
        return Timed(call=arm.call, built=arm.cell, instrument=arm.instrument)

    def post(self, ws: Path) -> dict:
        return {"calls": len(json.loads((ws / "samples.json").read_text())["ms"])}


class FlashMemory(Unit):
    location = "bench/memory"
    repeatable = True

    def __init__(self, cell: str) -> None:
        self.cell = cell

    def identity(self) -> dict:
        return {"code": code_key(), "environment": environment_key(), "cell": _registry().cell(self.cell)["params"],
                "calls": MEMORY_CALLS}

    def setup(self, ws: Path):
        import torch

        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        arm = _flash(self.cell)
        arm.call()
        return arm, before

    def execute(self, prepared, ws: Path) -> None:
        import torch

        arm, before = prepared
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        for _ in range(MEMORY_CALLS):
            arm.call()
        torch.cuda.synchronize()
        (ws / "memory.json").write_text(json.dumps({
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "allocated_after_bytes": torch.cuda.memory_allocated(), "allocated_before_build_bytes": before,
            "built": arm.cell}))

    def post(self, ws: Path) -> dict:
        return json.loads((ws / "memory.json").read_text())


def graph() -> list[Node]:
    here = f"{__name__}:"
    nodes = []
    for cell in _registry().cells:
        nodes.append(Node(f"time.flash@{cell}", here + "Flash", {"cell": cell}))
        nodes.append(Node(f"memory.flash@{cell}", here + "FlashMemory", {"cell": cell}))
    return nodes
