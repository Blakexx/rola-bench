"""`python -m rola_bench.measure plan|run|show` -- see rola_bench/measure/README.md."""
from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path

from rola_results import ROOT, Store, checkout

from .compose import bench_env, compose, groups, hold, parse, rola_env


def main() -> int:
    #: a SIGTERM unwinds like an interrupt, so the workers in flight are stopped with their trees
    signal.signal(signal.SIGTERM, lambda signum, _frame: sys.exit(128 + signum))
    ap = argparse.ArgumentParser(prog="python -m rola_bench.measure", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, help_ in (("run", "run every selected node and session not yet stored"),
                        ("plan", "list every selected node and session and whether it is stored")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("--target", required=True, help="worktree:PATH[,venv:PATH][,label:NAME]: the subject checkout")
        p.add_argument("--reference", action="append", default=[], help="a checkout timed beside the target (repeatable)")
        p.add_argument("--groups", default="all", help="all, gate, or a comma list of groups (registry.json)")
        p.add_argument("--nodes", default="all", help="all, or comma-separated node prefixes (carry.phases, memory, time)")
        p.add_argument("--cells", default="all", help="all, or a comma list narrowing the groups' cells")
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
    a = ap.parse_args()

    if a.cmd == "show":
        for rec in Store(a.location).records():
            ok = [s for s in rec["samples"] if s["ok"]]
            last = rec["samples"][-1]["provenance"]
            print(f"{rec['key'][:12]} {len(ok)} ok / {len(rec['samples']) - len(ok)} failed  "
                  f"{last.get('label', '') or last.get('checkout', '')} {(last.get('git_sha') or '')[:7]}")
        return 0

    from rola_devtools.graph.engine import load, run

    target = parse(a.target)
    references = [parse(r) for r in a.reference]
    labels = [t.label for t in (target, *references)]
    if len(set(labels)) != len(labels):
        raise SystemExit(f"target and reference labels must differ, got {labels}")
    envs = [rola_env(t) for t in (target, *references)] + ([] if a.no_attention else [bench_env(target.python)])
    instances = [load(env) for env in envs]
    sessions, selection = compose(instances, groups(target, a.groups), nodes=a.nodes, cells=a.cells, rounds=a.rounds,
                                  reps=a.reps, warmup=a.warmup)
    outcomes = run(instances, lambda location: Store(location, a.store_root), sessions, hold=hold(target), select=selection,
                   repeat=getattr(a, "repeat", False), force=getattr(a, "force", False), dry=a.cmd == "plan",
                   provenance=lambda env: {"label": env.label, **checkout(env.cwd)},
                   log=lambda line: print(line, flush=True))
    counts: dict[str, int] = {}
    for outcome in outcomes:
        counts[outcome.status] = counts.get(outcome.status, 0) + 1
    print("measure: " + ", ".join(f"{n} {status}" for status, n in sorted(counts.items())) + f"; records under {a.store_root}")
    return 1 if counts.get("failed") or counts.get("blocked") else 0


if __name__ == "__main__":
    sys.exit(main())
