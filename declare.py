"""ROLA-BENCH'S ROOT: rola checkouts and the libraries they are compared against, measured as one declared build.

    python -m rola_devtools.build plan declare.py:suite --arg target=worktree:PATH --only 'session/attention/*'
    python -m rola_devtools.build run  declare.py:suite --arg target=worktree:PATH[,venv:PATH][,label:NAME] \\
        [--arg references='worktree:PATH,label:master;worktree:PATH'] [--arg rounds=8] [--arg reps=11] \\
        [--arg warmup=10] [--arg store_root=DIR] [--only GLOB]... [--skip GLOB]...
    python -m rola_devtools.build run  declare.py:diffs --arg target=... --arg references=...   # the dual run

Run it with a python that has rola-devtools and rola-results (the target's venv does). A TARGET is a rola checkout and
the venv that runs it; `references` adds others, `;`-separated. Each checkout's own `declare.py` is loaded by path and
called under its label's scope with one shared timing server: its build, machine facts, timing registrations and its
diff SIDES, and for the target alone its instruments and its own kernel-vs-oracle diff. THE ROOT DECLARES EVERYTHING
and takes no selector: which of it a build runs is pruned by label at the CLI (`--only`, `--skip`), which knows nothing
of cells, groups or parts. A checkout without a
`declare.py` predates the declaration API and is not compared. Every central cell is a NODE the checkouts share
(`rola_devtools.cells.declare`), and a target that runs on cells takes those nodes as its data inputs. rola-bench's own
entry is the attention reference
(`rola_bench/measure/attention.py`'s `flash`), registered on the groups' QKV cells in the target's venv.

For each surface the target exposes (its `SURFACES`) and each reference, one DIFF of the two checkouts' sides under the
rule the target states for that surface -- the oracle pair's dual run and kernel-vs-kernel conformance, stored at
`diff/<surface>`. For each group (`rola_bench/measure/groups.py`) and each arm set it times together, one SESSION
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


#: THE RULES A SURFACE IS COMPARED UNDER ACROSS CHECKOUTS: the strategy and its parameters, as a function of the facts
#: the target states about the surface's numerics (its `NUMERICS`). The oracle is BIT-IDENTICAL: both checkouts run fp64
#: on identical inputs, so a rewrite that only nearly reproduces the old one has changed the definition. The producer is
#: SUPPORT-EQUAL: its declared purpose was to replace a bisection with a closed form, so agreement to 1e-13 is the result
#: and the support set is the claim with teeth. A kernel is PER SLOT at twice the reassociation its checkout measured,
#: with the quantities it states exact held exact -- bit-identity is not expected of a kernel whose fan-in reassociates.
def _kernel_rule(facts: dict) -> tuple:
    per_quantity = {q: {"clauses": [[2 * f["rtol"], 2 * f["atol"]]]} for q, f in facts.get("reassociation", {}).items()}
    per_quantity.update({q: {"clauses": [[0.0, 0.0]]} for q in facts.get("exact", ())})
    return "per-slot", {"per_quantity": per_quantity}


RULES = {
    "oracle": lambda _facts: ("bit-identical", {}),
    "producer": lambda _facts: ("support-equal", {"rtol": 0.0, "atol": 1e-13}),
    "carry-kernel": _kernel_rule,
}


def root(g, target: str, references: str = "", rounds: str = "8", reps: str = "11", warmup: str = "10",
         store_root: str = "") -> dict:
    """EVERY target the suite has over these checkouts; a build that wants fewer prunes by label (`--only`, `--skip`)."""
    from rola_devtools.cells import central
    from rola_devtools.diff import diff

    checkouts = [checkout(target), *(checkout(spec) for spec in filter(None, references.split(";")))]
    labels = [c["label"] for c in checkouts]
    if len(set(labels)) != len(labels):
        raise SystemExit(f"checkout labels must differ, got {labels}")

    group_file = load(HERE / "rola_bench" / "measure" / "groups.py")
    registry = central()
    groups = group_file["select"]("all")
    for group in groups:
        group_file["check"](group, registry)
    names = list(dict.fromkeys(c for group in groups for c in group.cells))

    server = start_timing_server(g)
    root_dir = store_root or None
    entries: dict[str, list] = {}
    stores, clock, null_entry, declared_by = [], None, None, {}
    for i, c in enumerate(checkouts):
        declared = load(c["worktree"] / "declare.py")
        #: the TARGET declares its instruments and its own kernel-vs-oracle diff; a reference is what it is compared
        #: against, so it declares its sides and its timing arms and nothing it would measure about itself
        which = {} if i == 0 else {"instruments": ()}
        out = declared["declare"](g.scoped(c["label"]), declared["checkout"](c["worktree"], python=c["python"],
                                                                             label=c["label"]), timing=server, **which)
        declared_by[c["label"]] = (declared, out)
        for arm, registration in out["entries"].items():
            entries.setdefault(arm, []).append(registration)
        for name, instrument in out["instruments"].items():
            stores.append(store(g, f"{c['label']}/store/{name}", source=instrument, location=f"rola/{name}",
                                cache=instrument.cache, root=root_dir))
        for name, verdict in out["diffs"].items():
            stores.append(store(g, f"{c['label']}/store/{name}", source=verdict, location=f"rola/{name}", root=root_dir))
        clock = clock or out["clock"]
        null_entry = null_entry or out["entries"].get("carry_forward")

    #: THE CROSS-CHECKOUT DIFFS: each surface the target exposes, against the same surface in each reference, under
    #: THIS ROOT'S rule for it (`RULES`) -- the suite owns how checkouts are compared, as it owns how they are timed
    subject_decl, subject = declared_by[labels[0]]
    diffs = []
    for label in labels[1:]:
        _decl, other = declared_by[label]
        for surface in subject_decl["SURFACES"]:
            left, right = subject["sides"][surface], other["sides"].get(surface)
            if right is None or surface not in RULES:
                continue
            strategy, params = RULES[surface](subject_decl.get("NUMERICS", {}).get(surface, {}))
            verdict = diff(g, f"diff/{surface}/{label}", left=left, right=right, strategy=strategy, params=params,
                           minimum=len(left.inputs))
            kept = store(g, f"store/diff/{surface}/{label}", source=verdict, location=f"diff/{surface}", root=root_dir)
            stores.append(kept)
            diffs.append(kept)  #: the group runs the diff AND files its verdict

    qkv = [c for c in names if registry.cell(c)["data"].split(":")[0].rsplit(".", 1)[-1] == "qkv"]
    if qkv:
        bench = Env(BENCH, checkouts[0]["python"], str(HERE), {"PYTHONPATH": str(HERE)})
        code = {"entry": "rola_bench/measure/attention.py", "roots": ["."]}
        entries["flash"] = [register_timing(g, f"{BENCH}/flash", server=server, env=bench,
                                            executor="rola_bench.measure.attention:flash", cells=cell_nodes(g, qkv),
                                            code=code)]
        #: THE INCREMENTAL CELL's arms: each (T, projection) one registration of the reference with its parameters,
        #: on every cell of the groups that name it
        arms: dict[str, tuple[dict, list]] = {}
        for group in groups:
            for arm, params in group.flash_arms:
                on = arms.setdefault(arm, (params, []))[1]
                on += [c for c in group.cells if c in qkv and c not in on]
        for arm, (params, on) in arms.items():
            entries[arm] = [register_timing(g, f"{BENCH}/{arm}", server=server, env=bench,
                                            executor="rola_bench.measure.attention:flash", cells=cell_nodes(g, on),
                                            params=params, code=code)]

    measured = []
    for group in groups:
        for arms in group.together:
            members = [r for arm in arms for r in entries.get(arm, ()) if _cells_of(r) & set(group.cells)]
            if not members:
                continue
            name = f"{group.name}/{'+'.join(arms)}"
            session = measure_timing(g, f"session/{name}", server=server, entries=members, clock=clock,
                                     cells=group.cells, rounds=int(rounds), reps=int(reps), warmup=int(warmup))
            measured.append(session)
            stores.append(store(g, f"store/session/{name}", source=session, location="timing/session", root=root_dir))
    on_gate = set() if null_entry is None else _cells_of(null_entry)
    null_cells = [next(c for c in group.cells if c in on_gate) for group in groups if set(group.cells) & on_gate]
    if null_cells:
        gate = measure_null_gate(g, "null", server=server, entry=null_entry, clock=clock, cells=null_cells,
                                 rounds=int(rounds), reps=int(reps), warmup=int(warmup))
        measured.append(gate)
        stores.append(store(g, "store/null", source=gate, location="timing/null", root=root_dir))
    memory = measure_memory(g, "memory", server=server, entries=[r for rs in entries.values() for r in rs], cells=names)
    measured.append(memory)
    stores.append(store(g, "store/memory", source=memory, location="timing/memory", root=root_dir))
    stop = stop_timing_server(g, server=server, after=[*measured, *stores])
    return {"suite": g.group("suite", [*stores, stop]), "diffs": g.group("diffs", diffs)}
