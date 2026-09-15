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
    timing.session    every arm's interleaved launch times for a subject   (per point x subject x call count)

What runs is chosen by POINT (`registry.json`, `rola_devtools.cells`): a point groups cells by runner, the carry
instruments take the rola cells of the selected points that the target's rola runner accepts, and a session times
every arm on the point's cells the arm's runner accepts.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from .engine import Node
from .target import (
    SUITE_REGISTRY,
    Target,
    accepted,
    binary_key,
    environment_key,
    extension,
    instrument,
    instrument_key,
    registry,
    sh,
)

#: the points the kernel's gate reads first; `--points gate` selects them
GATE_POINTS = ("L65536-N65536-dv64", "L1024-N65536-dv64")
KERNEL_SOURCE = ("csrc/rola/src", "build/generated/carry_parts.inc")
CELL_FILES = ("benchmarks/cells/carry_cells.json", "benchmarks/cells/layer_cells.json")


@dataclass
class Options:
    references: list[Target]
    points: str = "all"
    cells: str = "all"
    subjects: str = "all"
    reps: int = 11
    warmup: int = 10
    rounds: int = 8


def points(t: Target, opt: Options) -> list[dict]:
    """The selected points, resolved against the target's registry and narrowed to `--cells`; a point left without a rola
    cell is dropped."""
    reg = registry(t)
    names = sorted(reg.points) if opt.points == "all" else list(GATE_POINTS if opt.points == "gate" else opt.points.split(","))
    unknown = sorted(set(names) - set(reg.points))
    if unknown:
        raise SystemExit(f"no point {unknown} in the registry ({SUITE_REGISTRY.name})")
    wanted = None if opt.cells == "all" else set(opt.cells.split(","))
    out = []
    for name in names:
        point = reg.point(name)
        runners = {r: [c for c in cells if wanted is None or c["name"] in wanted or r != "rola"]
                   for r, cells in point["runners"].items()}
        if runners.get("rola"):
            out.append({**point, "runners": runners})
    if wanted is not None:
        missing = sorted(wanted - {c["name"] for p in out for c in p["runners"]["rola"]})
        if missing:
            raise SystemExit(f"--cells {missing} are not rola cells of the selected points")
    return out


def rola_cells(t: Target, opt: Options) -> dict[str, dict]:
    """Every rola cell of the selected points, by name, with what the target's runner makes of it."""
    return accepted(t, t, tuple(sorted({c["name"] for p in points(t, opt) for c in p["runners"]["rola"]})))


def _identity(t: Target, entry: str, data: tuple[str, ...] = (), **params) -> dict:
    return {"binary": binary_key(t), "instrument": {entry: instrument_key(t, entry, data)},
            "environment": environment_key(t), "params": params}


def _meta(t: Target) -> dict:
    head = sh(["git", "rev-parse", "HEAD"], t.worktree)[1].strip()
    dirty = bool(sh(["git", "status", "--porcelain", "--untracked-files=no"], t.worktree)[1].strip())
    return {"label": t.label, "worktree": t.worktree.name, "git_sha": head, "dirty": dirty}


def carry_nodes(t: Target, opt: Options, wanted: set[str]) -> list[Node]:
    meta = _meta(t)
    so = str(extension(t))
    nodes: list[Node] = []

    def add(module: str, unit: str, identity: dict, args: list[str], repeatable: bool, timeout: int = 3600) -> None:
        if module in wanted:
            nodes.append(Node(module, unit, identity, lambda _deps, dest, a=args: instrument(t, a, dest, timeout),
                              repeatable=repeatable, meta=meta))

    add("carry.sass", "", _identity(t, "tools/sass_gate.py"), ["tools/sass_gate.py", so], False, 900)
    add("carry.registers", "arm0", _identity(t, "tools/life_ranges.py", KERNEL_SOURCE, arm=0),
        ["tools/life_ranges.py", "--arm", "0", "--source", "csrc/rola/src/carry/carry_kernel.cuh"], False, 1800)
    registry_cells = registry(t).cells
    carry = sorted(name for name, v in rola_cells(t, opt).items()
                   if "arms" in v and registry_cells[name]["data"].endswith(":carry_cell"))
    for cell in carry:
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


#: THE ATTENTION REFERENCE rides every session of these subjects on a point that sends the attention runner a cell
ATTENTION_SUBJECTS = ("carry_forward", "prefill_op")
SUITE_ROOT = Path(__file__).resolve().parents[2]


def arm_name(subject: str, calls: int, schedule: str) -> str:
    """A rola arm's name in its checkout's `bench.provider` grammar: the subject and its non-default dials."""
    return subject + (f"@calls={calls}" if calls != 1 else "") + (f"@schedule={schedule}" if schedule != "first" else "")


def _units(arms: list[str]) -> set[tuple[str, int]]:
    """The (subject, call count) units among a cell's arm names, at the default state arm."""
    out = set()
    for name in arms:
        subject, *dials = name.split("@")
        fields = dict(d.split("=", 1) for d in dials)
        if "state" not in fields:
            out.add((subject, int(fields.get("calls", 1))))
    return out


def timing_nodes(t: Target, opt: Options, wanted: set[str]) -> list[Node]:
    """One interleaved session per point, unit (a subject at a call count) and carry order: the target (role `subject`),
    every reference (role `reference`) and, for a carry subject on a point with an attention cell, the attention reference,
    each an arm of rola's `tools/compare.py` run from the target on the point's cells every rola arm accepts."""
    if "timing.session" not in wanted:
        return []
    labelled = [("subject", t)] + [("reference", r) for r in opt.references]
    selected = points(t, opt)
    every = tuple(sorted({c["name"] for p in selected for c in p["runners"]["rola"]}))
    takes = {a.label: accepted(a, t, every) for _role, a in labelled}
    nodes = []
    for point in selected:
        names = tuple(c["name"] for c in point["runners"]["rola"])
        units = sorted(set().union(*(_units(takes[t.label][n].get("arms", [])) for n in names)))
        for subject, calls in units:
            if opt.subjects != "all" and subject not in opt.subjects.split(","):
                continue
            #: one session per carry order: an arm has one name for every cell it runs on, so cells whose orders differ
            #: for any arm (a DENSE/SPARSE target schedule) are separate sessions
            groups: dict[tuple[str, ...], list[str]] = {}
            for name in names:
                orders = tuple(a.schedule_for(name) if "carry" in subject else "first" for _role, a in labelled)
                if all(arm_name(subject, calls, order) in takes[a.label][name].get("arms", [])
                       for (_role, a), order in zip(labelled, orders, strict=True)):
                    groups.setdefault(orders, []).append(name)
            for orders, cells in sorted(groups.items()):
                attention = subject in ATTENTION_SUBJECTS and "attention" in point["runners"]
                session_cells = cells + ([c["name"] for c in point["runners"]["attention"]] if attention else [])
                resolved = {r: [c for c in cs if c["name"] in session_cells] for r, cs in point["runners"].items()}
                identity = {"point": {**point, "runners": {r: cs for r, cs in resolved.items() if cs}},
                            "arms": [{"role": role, "binary": binary_key(a),
                                      "instrument": instrument_key(a, "tools/compare.py",
                                                                   ("benchmarks/bench/provider.py", *CELL_FILES)),
                                      "arm": arm_name(subject, calls, order)}
                                     for (role, a), order in zip(labelled, orders, strict=True)],
                            "attention": (_digest(Path(__file__).with_name("attention.py"))
                                          + _digest(Path(__file__).with_name("cells.py"))) if attention else None,
                            "environment": environment_key(t),
                            "params": {"reps": opt.reps, "warmup": opt.warmup, "rounds": opt.rounds}}
                meta = {"arms": [{"role": role, **_meta(a)} for role, a in labelled]}
                unit = f"{subject}@{point['name']}" + (f"@calls={calls}" if calls != 1 else "") + \
                    ("" if set(orders) == {"first"} else "@schedule=" + "+".join(dict.fromkeys(orders)))
                nodes.append(Node("timing.session", unit, identity,
                                  partial(_session, t, labelled, subject, calls, orders, point["name"], session_cells,
                                          attention, opt), repeatable=True, meta=meta))
    return nodes


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _session(t: Target, labelled: list[tuple[str, Target]], subject: str, calls: int, orders: tuple[str, ...], point: str,
             cells: list[str], attention: bool, opt: Options, _deps: dict, dest: Path) -> None:
    argv = [t.python, "tools/compare.py", "--point", point, "--registry", str(SUITE_REGISTRY), "--cells", ",".join(cells),
            "--reference", t.label, "--reps", str(opt.reps), "--warmup", str(opt.warmup), "--rounds", str(opt.rounds)]
    for (_role, a), order in zip(labelled, orders, strict=True):
        argv += ["--arm", f"label:{a.label},arm:{arm_name(subject, calls, order)},worktree:{a.worktree},venv:{a.venv}"]
    if attention:
        argv += ["--foreign", f"label:attention,runner:attention,provider:rola_bench.measure.attention:arms,arm:flash,"
                              f"python:{t.python},cwd:{SUITE_ROOT}"]
    with tempfile.TemporaryDirectory(prefix="rola_suite_session_") as tmp:
        rc, text = sh([*argv, "--out", f"{tmp}/result.json"], t.worktree, 7200)
        out = Path(tmp) / "result.json"
        if not out.exists():
            raise RuntimeError(f"compare exited {rc} without a result:\n{text[-1500:]}")
        result = json.loads(out.read_text())
    roles = {a.label: role for role, a in labelled} | ({"attention": "attention"} if attention else {})
    doc = {"point": point, "subject": subject, "calls": calls, "roles": roles, "result": result}
    dest.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")


def nodes_for(t: Target, opt: Options, modules: str) -> list[Node]:
    known = ("carry.sass", "carry.registers", "carry.phases", "carry.counters", "carry.census", "carry.timeline",
             "timing.session")
    wanted = {m for m in known if modules == "all" or any(m == x or m.startswith(x + ".") for x in modules.split(","))}
    if not wanted:
        raise SystemExit(f"no module matches {modules!r}; known: {known}")
    for name, verdict in sorted(rola_cells(t, opt).items()):
        if "refused" in verdict:
            print(f"{t.label} refuses {name}: {verdict['refused']}", flush=True)
    return carry_nodes(t, opt, wanted) + timing_nodes(t, opt, wanted)
