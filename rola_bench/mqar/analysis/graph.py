#!/usr/bin/env python3
"""THE HEADLINE STATISTIC of the graph grid: do measured collapse thresholds track the PREDICTED chi?

    python -m rola_bench.mqar.analysis.graph --config graph_grid [--criterion 0.5] [--json out.json]

The grid sweeps `N` (addressable RoLA states per head) against a data distribution whose
co-occurrence graph has a solved chromatic number `chi` (`zoology.data.graph_chi`). The theory says
recall collapses when `N` falls below `chi` — not below the item count, not below the key-vocabulary
size. This module turns each (condition, arm) accuracy-vs-N curve into ONE number, the interpolated
`N*` at which accuracy crosses a criterion, and reports it against that condition's predicted chi.

SYMBOLS. `chi` = predicted demand (the solved chromatic number of the realized union graph, an
interval when the solver could not close it). `N` = states per head. `N*` = the interpolated
crossing. `ratio = N* / chi` is the statistic: the CLAIM is that it is ~1 and, crucially, CONSTANT
across conditions whose item counts and vocabularies differ by 8x — a constant that is not 1 still
supports the claim (it is a per-arm efficiency), a ratio that MOVES with the item count instead does
not.

HOW THE PREDICTION IS OBTAINED. Not from the knob, and not from the run record: the run records
carry only what `run.parse_stdout` extracts, whose slice regex is written in
`num_kv_pairs`, and the generator's stdout stamp is absent whenever a segment came from the on-disk
cache. So this module REBUILDS each condition's episodes from the spec's own protocol knobs, at the
segment size the cell trained on, and solves them — the same code path, the same seed discipline,
therefore the same graph. `reuse` makes this non-optional: its realized chi grows with the number of
sampled sequences, so a prediction taken at any other sample size would be a different number.

INTERPOLATION. Accuracy is read at each N on the cell's ladder and the crossing is interpolated
LINEARLY IN log2(N), because the ladder is geometric (16, 32, 64, ...) and a linear-in-N
interpolation would place the threshold according to the spacing of the rungs rather than the shape
of the curve. A curve that never crosses reports `None` on the correct side (`below` / `above`), and
those cells are counted and named rather than dropped — a condition where every arm stays under the
criterion is a result about the condition, not missing data.

THE ATTENTION REFERENCE is reported separately, as an INSTRUMENT FLOOR, and never as a point on a
capacity curve: softmax attention's state is a growing KV cache, so it has no rung on the N axis at
all. Its number answers a prior question — is this family solvable by the backbone at this length? —
and a family whose attention score is far from ceiling is one where a routed arm's failure measures
generator difficulty rather than demand. The table says so in that case.

TWO INSTRUMENTS, ONE ANALYSIS (2026-08-03). The same machinery reads the PRESENCE-RECALL grid
(`zoology.data.presence_recall`, v1's theory-confirmation instrument) because that generator stamps
its demand in the same `graph_demand_*` vocabulary and its cells carry the same run_id shape. What
differs is the METRIC'S FLOOR, and it is not cosmetic: presence-recall's answer is one bit, so a
label-free predictor already scores `TASKS['presence_recall'].chance = 0.5` and a collapse threshold read at
a criterion of 0.5 would be the threshold of guessing. The task table below therefore carries a
chance line and a default criterion PER TASK (presence: chance 0.5, criterion 0.75 = halfway from
chance to ceiling), both printed with every table so no number is ever quoted without the line it is
measured against. The task is taken from the spec's own `data.protocol.kind`, never assumed.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from itertools import pairwise
from pathlib import Path
from typing import NamedTuple

REPO = Path(__file__).resolve().parents[3]
_EXP_DIR = Path(__file__).resolve().parents[1] / "experiments"


class Task(NamedTuple):
    """What a generator's accuracy MEANS: the floor it is read against and where a threshold sits.

    `chance` is the accuracy of the best label-free predictor. `criterion` is the default collapse
    threshold, placed halfway from `chance` to 1 so it is a statement about the model and not about
    guessing. `builder` names the module whose `build_episodes` re-solves the condition's graph."""

    chance: float
    criterion: float
    builder: str


#: Keyed on `data.protocol.kind`. `graph_recall`'s target is a value token out of a vocabulary of
#: thousands, so its label-free floor is ~0 and the historical 0.5 criterion is a real statement;
#: `presence_recall`'s target is one bit, so 0.5 is what guessing scores.
TASKS = {
    "graph_recall": Task(chance=0.0, criterion=0.5, builder="zoology.data.graph_recall"),
    "presence_recall": Task(chance=0.5, criterion=0.75, builder="zoology.data.presence_recall"),
}


def _load_rows(location: str):
    """Every completed cell stored at `location` (`rola_results`), the newest sample per run_id."""
    from rola_results import outputs

    rows = {}
    for _record, _sample, r in sorted(outputs(location), key=lambda x: x[1]["utc"]):
        if r.get("ok") and r.get("run_id"):
            rows[r["run_id"]] = r
    return list(rows.values())


_RUNID = re.compile(r"grid-routed-(?P<tag>[^-]+(?:-mass)?)-nc(?P<nc>\d+)_L(?P<L>\d+)_"
                    r"st\d+_lr[\d.e+-]+_s(?P<seed>\d+)-(?P<cond>.+)$")
#: The attention REFERENCE column. It has no `nc` because softmax attention has no rung on the state
#: axis (a growing KV cache is not a capacity setting), which is exactly why it is reported as the
#: instrument floor and never as a point on an N curve.
_ATTN_RUNID = re.compile(r"grid-mha-dqk(?P<dqk>\d+)_L(?P<L>\d+)_lr[\d.e+-]+_s(?P<seed>\d+)-"
                         r"(?P<cond>.+)$")


def parse_run_id(run_id: str):
    """`(arm_tag, N, L, seed, condition)` — the run_id IS the key of a result row, by house rule.

    `N` is None for the attention reference, which is the signal that the row belongs in the floor
    table rather than on a capacity curve."""
    m = _RUNID.match(run_id)
    if m:
        return (m["tag"], int(m["nc"]), int(m["L"]), int(m["seed"]), m["cond"])
    m = _ATTN_RUNID.match(run_id)
    if m:
        return (f"mha-dqk{m['dqk']}", None, int(m["L"]), int(m["seed"]), m["cond"])
    return None


def task_of(spec) -> Task:
    """The spec's task, from its own `data.protocol.kind`. Refuses an unknown kind by name: a
    silently defaulted chance line would let a presence threshold be read at the guessing floor."""
    kind = (spec["data"].get("protocol") or {}).get("kind")
    if kind not in TASKS:
        raise ValueError(
            f"data.protocol.kind={kind!r} has no entry in `TASKS`, so this module does not know the "
            f"chance line its accuracies are measured against. Known: {sorted(TASKS)}.")
    return TASKS[kind]


def predicted_chi(spec, cond_knobs, seq_len, num_examples, seed=0):
    """Solve the condition's realized graph at the size the cell actually trained on.

    Returns `{lower, upper, exact, method, nodes, edges}`. Rebuilt rather than read back: see the
    module docstring. The episode builder is the one the spec's own protocol names, so a presence
    cell is re-solved by the presence generator and a graph cell by the graph one — the same code
    path and the same seed discipline as the training segment, therefore the same graph."""
    import importlib

    from zoology.data.graph_recall import demand_bounds

    proto = spec["data"]["protocol"]
    build_episodes = importlib.import_module(task_of(spec).builder).build_episodes
    knobs = {k: v for k, v in proto.items()
             if k not in ("kind", "train_examples", "test_examples", "test_seq_lens")}
    knobs.update({k: v for k, v in cond_knobs.items() if k != "tag"})
    ep = build_episodes(vocab_size=spec["backbone"]["vocab"], num_examples=num_examples,
                        input_seq_len=seq_len, seed=seed, **knobs)
    return demand_bounds(ep.groups)


def crossing(points, criterion):
    """Interpolated N* where accuracy first reaches `criterion`, linear in log2(N).

    `points` is [(N, accuracy)]. Returns `(N*, note)` where `note` is None on a real crossing,
    `'below'` if the curve never reaches the criterion, `'above'` if it starts above it (the
    threshold is below the ladder's floor and this grid cannot see it)."""
    pts = sorted(points)
    if not pts:
        return None, "empty"
    if pts[0][1] >= criterion:
        return float(pts[0][0]), "above"
    for (n0, a0), (n1, a1) in pairwise(pts):
        if a1 >= criterion:
            if a1 == a0:
                return float(n1), None
            t = (criterion - a0) / (a1 - a0)
            return float(2 ** (math.log2(n0) + t * (math.log2(n1) - math.log2(n0)))), None
    return None, "below"


def analyse(config: str, criterion: float | None = None, location: str | None = None):
    from rola_bench.mqar.build_configs import load_spec

    spec = load_spec(str(_EXP_DIR / f"{config}.yaml"))
    task = task_of(spec)
    criterion = task.criterion if criterion is None else criterion
    if criterion <= task.chance:
        raise ValueError(
            f"criterion {criterion} is at or below this task's chance line ({task.chance}): the "
            "crossing would be the threshold of GUESSING, not of recall. Raise the criterion.")
    rows = _load_rows(location or f"mqar/{config}")
    conds = {c["tag"]: c for c in spec.get("conditions", [])}
    proto = spec["data"]["protocol"]

    curves: dict[tuple, dict[int, list]] = {}
    lengths: dict[str, int] = {}
    floor: dict[str, list] = {}
    for r in rows:
        key = parse_run_id(r["run_id"])
        if key is None:
            continue
        tag, nc, L, _seed, cond = key
        acc = r.get("best_overall") or r.get("max_acc") or 0.0
        lengths[cond] = L
        if nc is None:
            # attention: the FLOOR is its widest head (capacity-adequate by construction); the
            # narrower rungs of the d_qk ladder are its own capacity axis and are handled below.
            floor.setdefault(cond, {}).setdefault(int(tag.split("dqk")[1]), []).append(acc)
            continue
        curves.setdefault((cond, tag), {}).setdefault(nc, []).append(acc)

    attn: dict[str, dict[int, list]] = {}
    for r in rows:
        key = parse_run_id(r["run_id"])
        if key is None or key[1] is not None:
            continue
        tag, _nc, _L, _seed, cond = key
        attn.setdefault(cond, {}).setdefault(int(tag.split("dqk")[1]), []).append(
            r.get("best_overall") or r.get("max_acc") or 0.0)

    out, chi_cache = [], {}
    for (cond, tag), by_nc in sorted(curves.items()):
        if cond not in chi_cache:
            chi_cache[cond] = predicted_chi(spec, conds.get(cond, {}), lengths[cond],
                                            proto["train_examples"])
        chi = chi_cache[cond]
        pts = [(nc, sum(v) / len(v)) for nc, v in by_nc.items()]      # mean over seeds
        n_star, note = crossing(pts, criterion)
        centre = (chi["lower"] + chi["upper"]) / 2
        out.append({"condition": cond, "arm": tag, "chi_lower": chi["lower"], "chi_upper": chi["upper"],
                        "chi_exact": chi["exact"], "chi_method": chi["method"], "nodes": chi["nodes"],
                        "edges": chi["edges"], "n_star": n_star, "note": note,
                        "ratio": (n_star / centre if n_star else None),
                        "curve": sorted(pts), "seeds": len(next(iter(by_nc.values()))) if by_nc else 0,
                        "attn_reference": _floor_of(floor.get(cond))})

    # --- attention's OWN capacity axis: d_qk* against log2(chi), whose slope is 1/c ----------------
    attn_rows = []
    for cond, by_dqk in sorted(attn.items()):
        if cond not in chi_cache:
            chi_cache[cond] = predicted_chi(spec, conds.get(cond, {}), lengths[cond],
                                            proto["train_examples"])
        chi = chi_cache[cond]
        pts = [(d, sum(v) / len(v)) for d, v in by_dqk.items()]
        # LINEAR in d_qk, not in log(d_qk): the predicted law is d_qk* = log(chi)/c, so d_qk is
        # already the log-scaled coordinate and interpolating it geometrically would double-log it.
        d_star, note = _crossing_linear(pts, criterion)
        attn_rows.append({"condition": cond, "chi_lower": chi["lower"], "chi_upper": chi["upper"],
                              "log2_chi": math.log2(max(chi["lower"], 1)), "d_qk_star": d_star, "note": note,
                              "curve": sorted(pts)})
    return spec, out, {k: _floor_of(v) for k, v in floor.items()}, attn_rows


def _floor_of(by_dqk):
    """The instrument floor: attention's accuracy at its WIDEST head, i.e. its capacity-adequate
    cell. Reading the floor off a narrow rung of the d_qk ladder would report a capacity result as a
    solvability one."""
    if not by_dqk:
        return None
    vals = by_dqk[max(by_dqk)]
    return sum(vals) / len(vals)


def _crossing_linear(points, criterion):
    """`crossing`, interpolated linearly in the coordinate itself rather than in log2 of it."""
    pts = sorted(points)
    if not pts:
        return None, "empty"
    if pts[0][1] >= criterion:
        return float(pts[0][0]), "above"
    for (x0, a0), (x1, a1) in pairwise(pts):
        if a1 >= criterion:
            if a1 == a0:
                return float(x1), None
            return float(x0 + (criterion - a0) / (a1 - a0) * (x1 - x0)), None
    return None, "below"


def fit_packing_constant(attn_rows):
    """Least-squares `d_qk* = slope * log2(chi) + intercept` over the attention rows that crossed.

    `slope = 1/c` for the packing law "one head addresses ~exp(c * d_qk) items", so `c = 1/slope`.
    Refuses to report a slope from fewer than three distinct demands: two points define a line
    through themselves and would be a ratio dressed as a fit."""
    pts = [(r["log2_chi"], r["d_qk_star"]) for r in attn_rows if r["d_qk_star"] and not r["note"]]
    distinct = {round(x, 3) for x, _ in pts}
    if len(distinct) < 3:
        return None, f"only {len(distinct)} distinct demand(s) crossed; a slope needs >= 3"
    n = len(pts)
    sx = sum(x for x, _ in pts)
    sy = sum(y for _, y in pts)
    sxx = sum(x * x for x, _ in pts)
    sxy = sum(x * y for x, y in pts)
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-12:
        return None, "degenerate design matrix"
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    return (slope, intercept), None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="graph_grid")
    ap.add_argument("--criterion", type=float, default=None,
                    help="accuracy the collapse threshold is defined at (default: the TASK's own, "
                         "0.5 for graph_recall and 0.75 for presence_recall, which is halfway from "
                         "that task's 0.5 chance line to the ceiling)")
    ap.add_argument("--location", default=None, help="the rola_results location (default: mqar/<config>)")
    ap.add_argument("--json", default=None, help="also write the rows here")
    a = ap.parse_args(argv)

    spec, rows, floor, attn_rows = analyse(a.config, a.criterion, a.location)
    task = task_of(spec)
    crit = task.criterion if a.criterion is None else a.criterion
    if not rows and not floor:
        print(f"no completed cells found for {a.config} — nothing to analyse")
        return 1

    # THE LINE EVERY NUMBER BELOW IS READ AGAINST, printed before any of them. A presence-recall
    # accuracy of 0.55 is noise off a coin flip; a graph-recall accuracy of 0.55 is most of the task.
    print(f"TASK {spec['data']['protocol']['kind']}: chance line {task.chance:.2f}, collapse "
          f"criterion {crit:.2f}"
          + (" (task default)" if a.criterion is None else " (--criterion)"))
    if floor:
        print("INSTRUMENT FLOOR — attention at its widest head (a solvability check, not a ranking):")
        for cond, acc in sorted(floor.items()):
            verdict = "" if acc >= 0.9 else ("  <-- the family may be too hard for the backbone at "
                                             "this L; a routed failure here is not evidence about N")
            print(f"  {cond:22s} attention accuracy {acc:.3f}{verdict}")
        print()
    print(f"{'condition':22s} {'arm':10s} {'chi':>9s} {'method':22s} {'|V|':>5s} "
          f"{'N*':>8s} {'N*/chi':>7s}  note")
    for r in rows:
        chi = str(r["chi_lower"]) if r["chi_exact"] else f"{r['chi_lower']}-{r['chi_upper']}"
        n_star = "{:.1f}".format(r["n_star"]) if r["n_star"] else "-"
        ratio = "{:.2f}".format(r["ratio"]) if r["ratio"] else "-"
        print(f"{r['condition']:22s} {r['arm']:10s} {chi:>9s} {r['chi_method']:22s} "
              f"{r['nodes']:5d} {n_star:>8s} {ratio:>7s}  {r['note'] or ''}")
    got = [r["ratio"] for r in rows if r["ratio"] and not r["note"]]
    if got:
        gm = math.exp(sum(math.log(x) for x in got) / len(got))
        spread = max(got) / min(got)
        print(f"\nHEADLINE (criterion {crit}, chance {task.chance}): N*/chi over {len(got)} "
              f"(condition, arm) cells "
              f"= {gm:.2f}x geometric mean, spread {spread:.2f}x across cells.")
        print("The claim is the CONSTANCY of this ratio across conditions whose item counts and key "
              "vocabularies differ; a constant != 1 is a per-arm efficiency, a ratio that tracks the "
              "item count instead is a falsification.")
    if attn_rows:
        print("\nAXIS 2 — ATTENTION's own capacity law: a head packs ~exp(c*d_qk) addresses, so its "
              "threshold is predicted LINEAR IN log2(chi).")
        print(f"{'condition':22s} {'chi':>6s} {'log2 chi':>9s} {'d_qk*':>7s}  note")
        for r in attn_rows:
            d = "{:.2f}".format(r["d_qk_star"]) if r["d_qk_star"] else "-"
            print(f"{r['condition']:22s} {r['chi_lower']:6d} {r['log2_chi']:9.2f} {d:>7s}  "
                  f"{r['note'] or ''}")
        fit, why = fit_packing_constant(attn_rows)
        if fit:
            slope, intercept = fit
            c = (1.0 / slope) if slope else float("inf")
            print(f"FIT: d_qk* = {slope:.2f} * log2(chi) + {intercept:.2f}  =>  packing constant "
                  f"c = 1/slope = {c:.3f} (per bit of demand)")
        else:
            print(f"FIT: not reported — {why}")

    missing = [r for r in rows if r["note"]]
    if missing:
        print(f"\n{len(missing)} cell(s) with no crossing on the ladder (reported, not dropped): "
              + ", ".join(f"{r['condition']}/{r['arm']}({r['note']})" for r in missing))
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=2))
        print(f"\nrows -> {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
