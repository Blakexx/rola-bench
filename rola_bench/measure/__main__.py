"""`python -m rola_bench.measure plan|run|show|verdict` -- see rola_bench/measure/README.md."""
from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path

from rola_results import ROOT, Store, checkout

#: what a run composes; `--only` narrows it
PARTS = ("instruments", "memory", "sessions")


def main() -> int:
    #: a SIGTERM unwinds like an interrupt, so the workers in flight are stopped with their trees
    signal.signal(signal.SIGTERM, lambda signum, _frame: sys.exit(128 + signum))
    ap = argparse.ArgumentParser(prog="python -m rola_bench.measure", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, help_ in (("run", "build every target, then run every selected node and session not yet stored"),
                        ("plan", "list every selected node and session and whether it is stored")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("--target", required=True, help="worktree:PATH[,venv:PATH][,label:NAME]: the subject checkout")
        p.add_argument("--reference", action="append", default=[], help="a checkout timed beside the target (repeatable)")
        p.add_argument("--groups", default="all", help="all, gate, or a comma list of groups (groups.json)")
        p.add_argument("--only", default=",".join(PARTS), help=f"a comma list of {', '.join(PARTS)}")
        p.add_argument("--units", default="all", help="all, or a comma list of the subject's instruments (carry.phases, ...)")
        p.add_argument("--cells", default="all", help="all, or a comma list narrowing the groups' cells")
        p.add_argument("--skip-cells", default="", help="a comma list of cells to leave out (a cell known to hang)")
        p.add_argument("--no-attention", action="store_true", help="leave rola-bench's attention reference out")
        p.add_argument("--reps", type=int, default=11, help="odd: a round's median is one of its samples")
        p.add_argument("--warmup", type=int, default=10, help="at least the driver's floor of 10")
        p.add_argument("--rounds", type=int, default=8,
                       help="at least 8: below it the paired test cannot call any difference significant")
        p.add_argument("--store-root", type=Path, default=ROOT, help="the rola-results backend (default: its records/)")
        if name == "run":
            p.add_argument("--repeat", action="store_true", help="add a sample to repeatable nodes and every session")
            p.add_argument("--force", action="store_true", help="run every selected node again")
    p = sub.add_parser("show", help="print the records of a location (rola/carry.phases, bench/session, ...)")
    p.add_argument("location")
    p = sub.add_parser("verdict", help="each timing unit's newest session judged against its reference (rola_results)")
    p.add_argument("--cell")
    p.add_argument("--subject")
    p.add_argument("--baseline", help="the reference label (default: a session's first reference)")
    a = ap.parse_args()

    if a.cmd == "verdict":
        from rola_results.verdict import table, verdicts

        print(table(verdicts(cell=a.cell, subject=a.subject, baseline=a.baseline)))
        return 0

    if a.cmd == "show":
        for rec in Store(a.location).records():
            ok = [s for s in rec["samples"] if s["ok"]]
            last = rec["samples"][-1]["provenance"]
            print(f"{rec['key'][:12]} {len(ok)} ok / {len(rec['samples']) - len(ok)} failed  "
                  f"{last.get('label', '') or last.get('checkout', '')} {(last.get('git_sha') or '')[:7]}")
        return 0

    from rola_devtools.measure.service import Service, outcomes_line

    from .groups import load, select, sessions
    from .targets import instances, parse

    only = set(a.only.split(","))
    if only - set(PARTS):
        raise SystemExit(f"--only takes {PARTS}, got {sorted(only - set(PARTS))}")
    targets = [parse(a.target), *(parse(r) for r in a.reference)]
    insts = instances(targets, attention=not a.no_attention)
    chosen = select(load(), a.groups)
    skip = frozenset(filter(None, a.skip_cells.split(",")))
    if a.cells != "all":
        skip |= {c for g in chosen for c in g.cells} - set(a.cells.split(","))
    roles = {i.label: i.role for i in insts}
    planned = [s for g in chosen for s in sessions(g, roles, reference=targets[0].label, skip=skip, rounds=a.rounds,
                                                   reps=a.reps, warmup=a.warmup)] if "sessions" in only else []
    cells = sorted({c for g in chosen for c in g.cells} - skip)
    timed = {arm for g in chosen for arms in g.together for arm in arms}
    units = None if a.units == "all" else set(a.units.split(","))

    def wanted(instance, unit) -> bool:
        if unit["kind"] == "instrument":
            return "instruments" in only and instance.role == "subject" and (units is None or unit["name"] in units)
        return unit["kind"] == "arm" and unit["name"] in timed

    def store(location):
        return Store(location, a.store_root)

    dry = a.cmd == "plan"
    with Service(insts, provenance=lambda instance: {"label": instance.label, **checkout(instance.cwd)},
                 log=lambda line: print(line, flush=True)) as service:
        built = service.build(store, dry=dry)
        unbuilt = [o for o in built if o.status not in ("complete", "ran")]
        if unbuilt:
            print(f"measure: the builds are {outcomes_line(built)}; the units on cells are described once they are built")
            return 0 if all(o.status == "pending" for o in unbuilt) else 1
        nodes = service.nodes(cells, sessions=planned, select=wanted, memory="memory" in only)
        outcomes = service.run(nodes, store, repeat=getattr(a, "repeat", False), force=getattr(a, "force", False),
                               dry=dry)
    print(f"measure: {outcomes_line(outcomes)}; records under {a.store_root}")
    return 1 if any(o.status in ("failed", "blocked") for o in outcomes) else 0


if __name__ == "__main__":
    sys.exit(main())
