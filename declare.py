"""ROLA-BENCH'S ROOT: rola checkouts and the libraries they are compared against, measured as one declared build.

    python -m rola_devtools.build plan declare.py:suite --arg target=worktree:PATH --arg groups=gate
    python -m rola_devtools.build run  declare.py:suite --arg target=worktree:PATH[,venv:PATH][,label:NAME] \\
        [--arg references='worktree:PATH,label:master;worktree:PATH'] [--arg groups=all|gate|a,b] [--arg cells=all|a,b] \\
        [--arg skip_cells=a,b] [--arg parts=instruments,memory,null,sessions] [--arg instruments=all|sass,phases,...] \\
        [--arg attention=yes|no] [--arg rounds=8] [--arg reps=11] [--arg warmup=10] [--arg store_root=DIR]

Run it with a python that has rola-devtools and rola-results (the target's venv does). A TARGET is a rola checkout and
the venv that runs it; `references` adds others, `;`-separated. Each checkout's own `declare.py` is loaded by path and
called under its label's scope with the selected cells and one shared timing server: its build, machine facts and
timing registrations, and for the target alone its instruments (`parts`, `instruments`). A checkout without a
`declare.py` predates the declaration API and is not compared. Every central cell is a NODE the checkouts share
(`rola_devtools.cells.declare`), and a target that runs on cells takes those nodes as its data inputs. rola-bench's own
entry is the attention reference
(`rola_bench/measure/attention.py`'s `flash`), registered on the groups' QKV cells in the target's venv.

For each selected group (`rola_bench/measure/groups.py`) and each arm set it times together, one SESSION
(`measure_timing`) over every checkout's registration of those arms on the group's cells, the target's clock reader
proving the clock; one MEMORY pass over every registration on every selected cell; a NULL GATE over the target's
`carry_forward` on each group's first RoLA cell, its entries timed against copies of themselves in second workers, so a
comparison across checkouts' workers is read beside what the gate found; a store for each instrument
(`rola/<instrument>`), each session (`timing/session`), the memory pass (`timing/memory`) and the null gate
(`timing/null`); and the server's stop, which runs whatever failed. Which checkout is the reference for a ratio is
chosen when the records are read (`rola_results`), never here.
"""
from __future__ import annotations

from pathlib import Path

from rola_devtools.build.declare import Env, load
from rola_devtools.cells.declare import cells as cell_nodes
from rola_devtools.store import store
from rola_devtools.timing.declare import (
    measure_memory,
    measure_null_gate,
    measure_timing,
    register_timing,
    start_timing_server,
    stop_timing_server,
)

HERE = Path(__file__).resolve().parent
BENCH = "bench"
PARTS = ("instruments", "memory", "null", "sessions")


def _cells_of(registration) -> set:
    """The cells a registration registers on: its data inputs are the cell NODES, each naming its cell."""
    return {node.params["cell"] for node in registration.inputs}


def checkout(spec: str) -> dict:
    """`worktree:PATH[,venv:PATH][,label:NAME]`: a rola checkout, the venv that runs it (default: `venv-<name>` beside
    it) and its label (default: its directory's name)."""
    fields = dict(part.split(":", 1) for part in spec.split(","))
    if "worktree" not in fields:
        raise SystemExit(f"a checkout needs worktree:PATH -- got {spec!r}")
    worktree = Path(fields["worktree"]).expanduser().resolve()
    venv = Path(fields.get("venv") or worktree.parent / f"venv-{worktree.name}").expanduser()
    if not (venv / "bin" / "python").exists():
        raise SystemExit(f"{worktree.name}: no venv at {venv} (name one with venv:PATH)")
    if not (worktree / "declare.py").is_file():
        raise SystemExit(f"{worktree}: no declare.py -- a checkout from before the declaration API is not compared")
    label = fields.get("label", worktree.name)
    if label == BENCH:
        raise SystemExit(f"the label {BENCH!r} is rola-bench's own entry")
    return {"worktree": worktree, "python": str(venv / "bin" / "python"), "label": label}


def root(g, target: str, references: str = "", groups: str = "all", cells: str = "all", skip_cells: str = "",
         parts: str = ",".join(PARTS), instruments: str = "all", attention: str = "yes", rounds: str = "8",
         reps: str = "11", warmup: str = "10", store_root: str = "") -> dict:
    from rola_devtools.cells import central

    chosen_parts = set(filter(None, parts.split(",")))
    if chosen_parts - set(PARTS):
        raise SystemExit(f"parts takes {PARTS}, got {sorted(chosen_parts - set(PARTS))}")
    checkouts = [checkout(target), *(checkout(spec) for spec in filter(None, references.split(";")))]
    labels = [c["label"] for c in checkouts]
    if len(set(labels)) != len(labels):
        raise SystemExit(f"checkout labels must differ, got {labels}")

    group_file = load(HERE / "rola_bench" / "measure" / "groups.py")
    registry = central()
    skip = set(filter(None, skip_cells.split(",")))
    narrowed = None if cells == "all" else set(cells.split(","))
    selected = []
    for group in group_file["select"](groups):
        group_file["check"](group, registry)
        kept = tuple(c for c in group.cells if c not in skip and (narrowed is None or c in narrowed))
        if kept:
            selected.append((group, kept))
    names = list(dict.fromkeys(c for _group, kept in selected for c in kept))

    timing = bool(chosen_parts & {"memory", "null", "sessions"})
    server = start_timing_server(g) if timing else None
    root_dir = store_root or None
    entries: dict[str, list] = {}
    stores, clock, null_entry = [], None, None
    for i, c in enumerate(checkouts):
        declared = load(c["worktree"] / "declare.py")
        subject = i == 0
        if subject and "instruments" in chosen_parts:
            which = {} if instruments == "all" else {"instruments": [n for n in instruments.split(",") if n]}
        else:
            which = {"instruments": ()}
        out = declared["declare"](g.scoped(c["label"]), declared["checkout"](c["worktree"], python=c["python"],
                                                                             label=c["label"]),
                                  cells=names, timing=server, **which)
        for arm, registration in out["entries"].items():
            entries.setdefault(arm, []).append(registration)
        for name, instrument in out["instruments"].items():
            stores.append(store(g, f"{c['label']}/store/{name}", source=instrument, location=f"rola/{name}",
                                cache=instrument.cache, root=root_dir))
        clock = clock or out["clock"]
        null_entry = null_entry or out["entries"].get("carry_forward")
    qkv = [c for c in names if registry.cell(c)["data"].split(":")[0].rsplit(".", 1)[-1] == "qkv"]
    if timing and attention == "yes" and qkv:
        bench = Env(BENCH, checkouts[0]["python"], str(HERE), {"PYTHONPATH": str(HERE)})
        entries["flash"] = [register_timing(g, f"{BENCH}/flash", server=server, env=bench,
                                            executor="rola_bench.measure.attention:flash", cells=cell_nodes(g, qkv),
                                            code={"entry": "rola_bench/measure/attention.py", "roots": ["."]})]

    measured = []
    if "sessions" in chosen_parts:
        for group, kept in selected:
            for arms in group.together:
                members = [r for arm in arms for r in entries.get(arm, ()) if _cells_of(r) & set(kept)]
                if not members:
                    continue
                name = f"{group.name}/{'+'.join(arms)}"
                session = measure_timing(g, f"session/{name}", server=server, entries=members, clock=clock, cells=kept,
                                         rounds=int(rounds), reps=int(reps), warmup=int(warmup))
                measured.append(session)
                stores.append(store(g, f"store/session/{name}", source=session, location="timing/session", root=root_dir))
    on_gate = set() if null_entry is None else _cells_of(null_entry)
    null_cells = [next(c for c in kept if c in on_gate) for _group, kept in selected if set(kept) & on_gate]
    if "null" in chosen_parts and null_cells:
        gate = measure_null_gate(g, "null", server=server, entry=null_entry, clock=clock, cells=null_cells,
                                 rounds=int(rounds), reps=int(reps), warmup=int(warmup))
        measured.append(gate)
        stores.append(store(g, "store/null", source=gate, location="timing/null", root=root_dir))
    if "memory" in chosen_parts and entries:
        memory = measure_memory(g, "memory", server=server, entries=[r for rs in entries.values() for r in rs], cells=names)
        measured.append(memory)
        stores.append(store(g, "store/memory", source=memory, location="timing/memory", root=root_dir))
    terminal = [*stores, stop_timing_server(g, server=server, after=[*measured, *stores])] if timing else stores
    return {"suite": g.group("suite", terminal)}
