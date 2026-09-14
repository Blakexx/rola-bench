"""THE MODULES: what the suite measures on a rola checkout, as nodes for the engine.

Every module runs the target checkout's own instrument through its JSON command line (rola's `tools/*.py --json`) and
stores the instrument's raw output. A module declares the instrument file and the data files it reads; its nodes' keys
take the target's binary, that instrument's code key, the parameters and the environment. Nothing relational is stored:
a TIMING SESSION records every arm it interleaved -- the binary under test and the references the run was given, each
with its declared role -- and ratios between arms are the reader's.

    carry.sass        the SASS signatures of the built carry arms          (per target)
    carry.registers   peak live registers by region, arm 0 with line info  (per target)
    carry.phases      the phase clock: cycles a warp a window, per warp    (per carry cell)
    carry.counters    the profiler's pipe and resource counters, one launch (per carry cell)
    carry.census      every stall sample by component, reason, source line (per carry cell)
    carry.timeline    the pipes over one launch, PM sampling               (per carry cell)
    timing.session    every arm's interleaved launch times for a subject   (per subject x call count x cell it applies to)
"""
from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from .engine import Node
from .target import Lane, Target, binary_key, cells, environment_key, instrument, instrument_key, sh, subjects

#: the carry cells the kernel's gate reads first; `--cells gate` selects them
GATE_CELLS = ("nl64k-alt-k4", "flagship-dense", "flagship-alt-k4", "flagship-cohort-k4")
KERNEL_SOURCE = ("csrc/rola/src", "build/generated/carry_parts.inc")


@dataclass
class Options:
    references: list[Target]
    cells: str = "all"
    subjects: str = "all"
    reps: int = 5
    warmup: int = 2
    rounds: int = 3


def _select(available: tuple[str, ...], choice: str) -> list[str]:
    if choice == "all":
        return list(available)
    if choice == "gate":
        return [c for c in GATE_CELLS if c in available]
    wanted = choice.split(",")
    unknown = sorted(set(wanted) - set(available))
    if unknown:
        raise SystemExit(f"not in the target's registry: {unknown}")
    return wanted


def _identity(t: Target, entry: str, data: tuple[str, ...] = (), **params) -> dict:
    return {"binary": binary_key(t), "instrument": {entry: instrument_key(t, entry, data)},
            "environment": environment_key(t), "params": params}


def _meta(t: Target) -> dict:
    head = sh(["git", "rev-parse", "HEAD"], t.worktree)[1].strip()
    dirty = bool(sh(["git", "status", "--porcelain", "--untracked-files=no"], t.worktree)[1].strip())
    return {"label": t.label, "worktree": t.worktree.name, "git_sha": head, "dirty": dirty}


def carry_nodes(t: Target, opt: Options, wanted: set[str]) -> list[Node]:
    meta = _meta(t)
    so = str(next((t.worktree / "rola").glob("_C*.so")))
    nodes: list[Node] = []

    def add(module: str, unit: str, identity: dict, args: list[str], repeatable: bool, timeout: int = 3600) -> None:
        if module in wanted:
            nodes.append(Node(module, unit, identity, lambda _deps, dest, a=args: instrument(t, a, dest, timeout),
                              repeatable=repeatable, meta=meta))

    add("carry.sass", "", _identity(t, "tools/sass_gate.py"), ["tools/sass_gate.py", so], False, 900)
    add("carry.registers", "arm0", _identity(t, "tools/life_ranges.py", KERNEL_SOURCE, arm=0),
        ["tools/life_ranges.py", "--arm", "0", "--source", "csrc/rola/src/carry/carry_kernel.cuh"], False, 1800)
    for cell in _select(cells(t, "carry"), opt.cells):
        sched = t.schedule_for(cell)
        add("carry.phases", cell, _identity(t, "tools/phase_ledger.py", ("benchmarks/cells/carry_cells.json",), launches=1),
            ["tools/phase_ledger.py", cell, "--launches", "1"], True)
        add("carry.counters", cell, _identity(t, "tools/pipe_counters.py", ("benchmarks/cells/carry_cells.json",),
                                              schedule=sched), ["tools/pipe_counters.py", cell, "--schedule", sched], True)
        add("carry.census", cell, _identity(t, "tools/stall_census.py", ("benchmarks/cells/carry_cells.json",
                                                                         "tools/budgets/carry.json",
                                                                         "csrc/rola/src/carry/carry_kernel.cuh"),
                                            schedule=sched), ["tools/stall_census.py", cell, "--schedule", sched], True)
        if "carry.timeline" in wanted:
            identity = _identity(t, "tools/pipe_timeline.py", ("benchmarks/cells/carry_cells.json",))
            nodes.append(Node("carry.timeline", cell, identity, lambda _deps, dest, c=cell: _timeline(t, c, dest),
                              repeatable=True, meta=meta))
    return nodes


def _timeline(t: Target, cell: str, dest: Path) -> None:
    """pipe_timeline writes `<stem>.json` beside its capture; the capture itself is scratch, and the suite's record keeps
    the JSON (`--no-record`: the tool stores nothing of its own)."""
    with tempfile.TemporaryDirectory(prefix="rola_suite_tl_") as tmp:
        rc, text = sh([t.python, "tools/pipe_timeline.py", "--cell", cell, "--out", f"{tmp}/tl", "--no-record"], t.worktree,
                      3600)
        out = Path(tmp) / "tl.json"
        if not out.exists():
            raise RuntimeError(f"pipe_timeline exited {rc} without output:\n{text[-1500:]}")
        dest.write_text(out.read_text())


def timing_nodes(t: Target, opt: Options, wanted: set[str]) -> list[Node]:
    """One interleaved session per unit (a subject at a call count) and cell: the target (role `subject`) and every
    reference (role `reference`), each through its own checkout's probe worker and its own spelling of the unit, timed in
    one invocation under the clock lock."""
    if "timing.session" not in wanted:
        return []
    arms = [("subject", t)] + [("reference", r) for r in opt.references]
    roster = subjects(t)
    names = sorted({subject for subject, _calls in roster})
    chosen = names if opt.subjects == "all" else opt.subjects.split(",")
    nodes = []
    for subject in chosen:
        if subject not in names:
            raise SystemExit(f"{subject} is not a bench subject of {t.label}: {names}")
        for calls in sorted(n for s, n in roster if s == subject):
            applies = roster[subject, calls].cells
            carry = set(applies) <= set(cells(t, "carry"))
            for cell in _select(applies, opt.cells) if carry else list(applies):
                lanes = [(role, a, subjects(a).get((subject, calls))) for role, a in arms]
                identity = {"arms": [{"role": role, "binary": binary_key(a),
                                      "instrument": instrument_key(a, "tools/probe_cells.py",
                                                                   ("benchmarks/cells/carry_cells.json",)),
                                      "schedule": a.schedule_for(cell) if "carry" in subject else None,
                                      "lane": [lane.bench, lane.calls] if lane else None} for role, a, lane in lanes],
                            "environment": environment_key(t),
                            "params": {"reps": opt.reps, "warmup": opt.warmup, "rounds": opt.rounds}}
                meta = {"arms": [{"role": role, **_meta(a)} for role, a in arms]}
                unit = f"{subject}@{cell}" + (f"@calls={calls}" if calls != 1 else "")
                nodes.append(Node("timing.session", unit, identity, partial(_session, t, lanes, subject, calls, cell, opt),
                                  repeatable=True, meta=meta))
    return nodes


def _session(t: Target, lanes: list[tuple[str, Target, Lane | None]], subject: str, calls: int, cell: str, opt: Options,
             _deps: dict, dest: Path) -> None:
    missing = [a.label for _role, a, lane in lanes if lane is None or cell not in lane.cells]
    if missing:
        raise RuntimeError(f"{missing} carry no {subject} at {calls} call(s) on {cell}")
    specs = []
    for _role, a, lane in lanes:
        spec = a.spec() + f",bench:{lane.bench}" + (f",calls:{lane.calls}" if lane.calls != 1 else "")
        spec += f",schedule:{a.schedule_for(cell)}" if "carry" in subject else ""
        specs += ["--binary", spec]
    with tempfile.TemporaryDirectory(prefix="rola_suite_session_") as tmp:
        rc, text = sh([t.python, "tools/probe_cells.py", "--bench", lanes[0][2].bench, "--cells", cell, "--reps",
                       str(opt.reps), "--warmup", str(opt.warmup), "--rounds", str(opt.rounds), "--no-record", "--out",
                       f"{tmp}/rows.jsonl", *specs], t.worktree, 7200)
        rows_path = Path(tmp) / "rows.jsonl"
        if not rows_path.exists():
            raise RuntimeError(f"probe_cells exited {rc} without rows:\n{text[-1500:]}")
        rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
    by_label = {r["binary"]: r for r in rows}
    doc = {"subject": subject, "calls": calls, "cell": cell, "session": rows[0].get("session") if rows else None,
           "arms": [{"role": role, "label": a.label, **{k: v for k, v in by_label.get(a.label, {}).items()
                                                        if k not in ("worktree", "venv")}} for role, a, _lane in lanes]}
    errors = [a["label"] for a in doc["arms"] if "error" in a or "median_of_round_medians_ms" not in a]
    dest.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    if errors:
        raise RuntimeError(f"arm(s) {errors} measured nothing: {[a.get('error') for a in doc['arms']]}")


def nodes_for(t: Target, opt: Options, modules: str) -> list[Node]:
    known = ("carry.sass", "carry.registers", "carry.phases", "carry.counters", "carry.census", "carry.timeline",
             "timing.session")
    wanted = {m for m in known if modules == "all" or any(m == x or m.startswith(x + ".") for x in modules.split(","))}
    if not wanted:
        raise SystemExit(f"no module matches {modules!r}; known: {known}")
    return carry_nodes(t, opt, wanted) + timing_nodes(t, opt, wanted)
