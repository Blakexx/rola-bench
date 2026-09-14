"""`python -m rola_bench.measure run|plan|show` -- see rola_bench/measure/README.md."""
from __future__ import annotations

import argparse
import sys

from rola_results import Store

from . import engine
from .modules import Options, nodes_for
from .target import parse


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m rola_bench.measure", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, help_ in (("run", "run every incomplete node"), ("plan", "list every node and whether it is complete")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("--target", required=True, help="worktree:PATH[,venv:PATH][,label:NAME][,schedule:first|DENSE/SPARSE]")
        p.add_argument("--reference", action="append", default=[], help="an arm timed against the target (repeatable)")
        p.add_argument("--modules", default="all", help="all, or a comma list of modules or prefixes (carry, timing.session)")
        p.add_argument("--cells", default="all", help="all, gate, or a comma list of carry cells")
        p.add_argument("--subjects", default="all", help="all, or a comma list of bench subjects")
        p.add_argument("--reps", type=int, default=11, help="odd: a round's median is one of its samples")
        p.add_argument("--warmup", type=int, default=10, help="at least the driver's floor of 10")
        p.add_argument("--rounds", type=int, default=3)
        if name == "run":
            p.add_argument("--repeat", action="store_true", help="add a sample to complete nodes of repeatable modules")
            p.add_argument("--force", action="store_true", help="run every selected node again")
    p = sub.add_parser("show", help="print the records of a module")
    p.add_argument("module")
    a = ap.parse_args()

    if a.cmd == "show":
        for rec in Store(f"suite/{a.module}").records():
            ok = [s for s in rec["samples"] if s["ok"]]
            last = rec["samples"][-1]["provenance"]
            print(f"{rec['key'][:12]} {rec['semantics']['unit'] or '-':40s} {len(ok)} ok / {len(rec['samples']) - len(ok)} "
                  f"failed  {last.get('label', '')} {(last.get('git_sha') or '')[:7]}")
        return 0

    target = parse(a.target)
    opt = Options([parse(r) for r in a.reference], a.cells, a.subjects, a.reps, a.warmup, a.rounds)
    nodes = nodes_for(target, opt, a.modules)
    outcomes = engine.run(nodes, dry=a.cmd == "plan", repeat=getattr(a, "repeat", False),
                          force=getattr(a, "force", False), log=lambda line: print(line, flush=True))
    counts: dict[str, int] = {}
    for o in outcomes:
        counts[o.status] = counts.get(o.status, 0) + 1
    print("suite: " + ", ".join(f"{n} {s}" for s, n in sorted(counts.items())) + f"; records under {Store('suite').dir}")
    return 1 if counts.get("failed") or counts.get("blocked") else 0


if __name__ == "__main__":
    sys.exit(main())
