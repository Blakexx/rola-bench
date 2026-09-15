"""THE COMPOSER: rola checkouts and rola-bench's references measured as one build over their graphs.

A TARGET is a rola checkout and the venv that runs it (`worktree:PATH[,venv:PATH][,label:NAME]`). Each is an INSTANCE of
rola's own graph (`benchmarks/graph.py`), described and run in its own directory under its own venv; rola-bench's graph
(`graph.py`, the attention reference) is one more instance, run in the first target's venv. Nothing here defines a
measurement: every node is its owner's, keyed as its owner keys it, so a node this composer runs and the same node run
from its checkout share a record.

What the composer adds is the relation between instances:
- a GROUP (a `points` entry of `registry.json`: cells by runner, what they hold equal) selects what launches together;
- a SESSION per group and subject interleaves the timed arms of that subject on the group's rola cells in every target,
  and on a carry subject the attention reference on the group's attention cells, the target (role `subject`) the
  reference instance the pairings divide by; the group, its claim and each instance's role are the session's relation;
- the selection: the subject target's instruments (`carry.*`), every instance's memory rows, every session's arms.
"""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path

from rola_devtools.graph import Env
from rola_devtools.graph.engine import Instance, Session

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = Path(__file__).with_name("registry.json")
#: the groups the kernel's gate reads first; `--groups gate` selects them
GATE_GROUPS = ("L65536-N65536-dv64", "L1024-N65536-dv64")
#: the rola subjects a session is composed for, and those the attention reference rides
SUBJECTS = ("carry_forward", "prefill_op", "entmax_solve", "decode_step")
ATTENTION_SUBJECTS = ("carry_forward", "prefill_op")
BENCH = "bench"


@dataclass(frozen=True)
class Target:
    worktree: Path
    venv: Path
    label: str

    @property
    def python(self) -> str:
        return str(self.venv / "bin" / "python")


def parse(spec: str) -> Target:
    fields = dict(part.split(":", 1) for part in spec.split(","))
    if "worktree" not in fields:
        raise SystemExit(f"a target needs worktree:PATH -- got {spec!r}")
    worktree = Path(fields["worktree"]).expanduser().resolve()
    venv = Path(fields.get("venv") or worktree.parent / f"venv-{worktree.name}").expanduser()
    if not (venv / "bin" / "python").exists():
        raise SystemExit(f"{worktree.name}: no venv at {venv} (name one with venv:PATH)")
    label = fields.get("label", worktree.name)
    if label == BENCH:
        raise SystemExit(f"the label {BENCH!r} is rola-bench's own instance")
    return Target(worktree, venv, label)


def rola_env(target: Target) -> Env:
    wt = target.worktree
    return Env(target.label, target.python, str(wt), "benchmarks.graph:graph",
               {"PYTHONPATH": f"{wt}:{wt}/benchmarks:{wt}/tools"})


def bench_env(python: str) -> Env:
    return Env(BENCH, python, str(ROOT), "rola_bench.measure.graph:graph", {"PYTHONPATH": str(ROOT)})


def groups(target: Target, names: str) -> list[dict]:
    """The selected groups, resolved against the target checkout's cells and this package's registry."""
    from rola_devtools.cells import Registry

    cells = target.worktree / "benchmarks" / "cells"
    registry = Registry.load([cells / "carry_cells.json", cells / "layer_cells.json", REGISTRY])
    wanted = sorted(registry.points) if names == "all" else list(GATE_GROUPS if names == "gate" else names.split(","))
    unknown = sorted(set(wanted) - set(registry.points))
    if unknown:
        raise SystemExit(f"no group {unknown} in {REGISTRY.name}")
    return [registry.point(name) for name in wanted]


def compose(instances: list[Instance], selected: list[dict], *, nodes: str = "all", cells: str = "all",
            rounds: int = 8, reps: int = 11, warmup: int = 10) -> tuple[list[Session], set[str]]:
    """`(sessions, selection)` over the instances: the first is the subject target, then its references, then
    rola-bench's. `nodes` is `all` or comma-separated name prefixes (`carry.phases`, `memory`, `time`); `cells` narrows
    the groups' cells."""
    rola = [inst for inst in instances if inst.env.label != BENCH]
    bench = next((inst for inst in instances if inst.env.label == BENCH), None)
    names = {inst.env.label: {n["name"] for n in inst.nodes} for inst in instances}
    prefixes = None if nodes == "all" else tuple(nodes.split(","))
    narrow = None if cells == "all" else set(cells.split(","))

    def chosen(name: str) -> bool:
        return prefixes is None or any(name == p or name.startswith(p + ".") or name.startswith(p + "@") for p in prefixes)

    def cells_of(group: dict, runner: str) -> list[str]:
        return [c["name"] for c in group["runners"].get(runner, []) if narrow is None or c["name"] in narrow]

    roles = {inst.env.label: "subject" if i == 0 else "reference" for i, inst in enumerate(rola)}
    if bench is not None:
        roles[BENCH] = "attention"
    sessions, selection = [], set()
    group_cells = {c for group in selected for c in cells_of(group, "rola")}
    attention_cells = {c for group in selected for c in cells_of(group, "attention")}
    subject = rola[0]
    for name in names[subject.env.label]:
        cell = name.split("@", 1)[1] if "@" in name else None
        if name.startswith("carry.") and chosen(name) and (cell is None or cell in group_cells or cell == "arm0"):
            selection.add(subject.qualified(name))
    for inst in rola:
        selection |= {inst.qualified(f"memory.{s}@{c}") for s in SUBJECTS for c in group_cells
                      if chosen(f"memory.{s}@{c}") and f"memory.{s}@{c}" in names[inst.env.label]}
    if bench is not None:
        selection |= {bench.qualified(f"memory.flash@{c}") for c in attention_cells
                      if chosen(f"memory.flash@{c}") and f"memory.flash@{c}" in names[BENCH]}
    for group in selected:
        for subj in SUBJECTS:
            members = [inst.qualified(f"time.{subj}@{c}") for inst in rola for c in cells_of(group, "rola")
                       if f"time.{subj}@{c}" in names[inst.env.label] and chosen(f"time.{subj}@{c}")]
            if not members:
                continue
            if bench is not None and subj in ATTENTION_SUBJECTS:
                members += [bench.qualified(f"time.flash@{c}") for c in cells_of(group, "attention")
                            if f"time.flash@{c}" in names[BENCH]]
            relation = {"group": group["name"], "holds": group["holds"], "equal": group["equal"],
                        "cells": {runner: [c["name"] for c in cs] for runner, cs in group["runners"].items()},
                        "roles": {label: role for label, role in roles.items()
                                  if any(m.startswith(label + ":") for m in members)}}
            sessions.append(Session(f"{subj}@{group['name']}", "bench/session", tuple(members),
                                    reference=subject.env.label, rounds=rounds, reps=reps, warmup=warmup,
                                    relation=relation))
            selection |= set(members)
    return sessions, selection


def hold(target: Target):
    """The subject checkout's own device and clock hold (`benchmarks/graph.py`'s `hold`)."""
    spec = importlib.util.spec_from_file_location("rola_target_graph", target.worktree / "benchmarks" / "graph.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.hold
