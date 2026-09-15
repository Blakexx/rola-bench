"""THE GROUPS: which central cells launch together, the claim they hold, and which arms are timed in one session.

A GROUP is a selection of central cells (`rola_devtools.cells`) and its relation: `holds`, the claim in words; `equal`,
the parameters every cell of the group carries with one value; and `together`, the arm sets each timed in one session
over the group's cells, in every checkout a run measures. `check` holds a group to its claim against the central
registry. The root (`declare.py`) loads this file by path.

    attention(tokens, states, dv, rola_cells)   RoLA's carry cells of one length, state count N and value width beside
                                                the attention cell of that length and width; `carry_forward` and
                                                `prefill_op` each timed beside `flash`
    layer(construction, cell)                   a layer input under one of RoLA's constructions; its solve and its
                                                decode step each timed alone

`GATE` names the groups the kernel's gate reads first. The capacity-fair comparison is at N = L (rola's
docs/measurement.md); a group's claim says whether its RoLA cells' state count equals their tokens.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Group:
    name: str
    holds: str
    equal: tuple[str, ...]
    cells: tuple[str, ...]
    together: tuple[tuple[str, ...], ...]
    states: int | None = None


def attention(tokens: int, states: int, dv: int, rola_cells: tuple[str, ...]) -> Group:
    fair = "N = L, capacity-fair" if states == tokens else f"N {'>' if states > tokens else '<'} L, not capacity-fair"
    return Group(f"L{tokens}-N{states}-dv{dv}",
                 f"{tokens} tokens and value width {dv}, bf16, attention one head of width {dv}; RoLA's state "
                 f"N = {states} ({fair})",
                 ("tokens", "dv"), (*rola_cells, f"qkv-L{tokens}-dv{dv}"),
                 (("carry_forward", "flash"), ("prefill_op", "flash")), states)


def layer(construction: str, cell: str) -> Group:
    return Group(f"layer-{construction}", f"the layer input {cell} under RoLA's construction {construction}", (), (cell,),
                 ((f"entmax_solve@layer={construction}",), (f"decode_step@layer={construction}",)))


GROUPS = {g.name: g for g in (
    attention(16, 256, 64, ("identity-passthrough-dense", "identity-passthrough-paged", "single-deposit")),
    attention(32, 256, 64, ("one-box-tiny",)),
    attention(64, 256, 64, ("one-box-short",)),
    attention(128, 4096, 64, ("flat-small-carried",)),
    attention(256, 4096, 64, ("flat-small-dense", "flat-small-idle-resident", "flat-small-alt-k4", "flat-small-alt-k16",
                              "flat-small-struct50", "flat-small-struct25", "deep3-dense", "deep3-alt-k4")),
    attention(256, 65536, 64, ("deep4-dense", "deep4-alt-k2")),
    attention(1024, 65536, 32, ("flagship-dv32-alt-k4",)),
    attention(1024, 256, 64, ("one-box-multiwindow", "one-box-alt-k4")),
    attention(1024, 4096, 64, ("flat-small-multiwindow",)),
    attention(1024, 65536, 64, ("flagship-dense", "flagship-alt-k4", "flagship-alt-k16", "flagship-cohort-k4",
                                "flagship-cohort-k16", "flagship-both-k4", "flagship-paged", "flagship-struct50",
                                "flagship-struct25")),
    attention(1024, 65536, 128, ("flagship-dv128-alt-k4",)),
    attention(1792, 65536, 64, ("flagship-partial-window",)),
    attention(4096, 65536, 64, ("flagship-both-k4-L4096", "flagship-alt-k4-L4096", "flagship-cohort-k4-L4096")),
    attention(16384, 16384, 64, ("nl16k-dense", "nl16k-alt-k4")),
    attention(16384, 65536, 64, ("flagship-dense-L16384", "flagship-alt-k16-L16384", "flagship-alt-k4-L16384",
                                 "flagship-cohort-k16-L16384")),
    attention(65536, 65536, 64, ("nl64k-dense", "nl64k-alt-k4")),
    layer("chunk-p73-pinned", "layer-B2-T2048-H8-h512-dv64-bf16-s0"),
    layer("chunk-sparse-gain8", "layer-B2-T512-H8-h512-dv64-bf16-s1"),
    layer("chunk-deadblocks-T64", "layer-B2-T64-H8-h512-dv64-bf16-s5"),
    layer("chunk-dense-D2", "layer-B2-T512-H8-h512-dv64-bf16-s2"),
    layer("chunk-dense-NL4096", "layer-B2-T4096-H8-h512-dv64-bf16-s6"),
    layer("chunk-bwd-smalln", "layer-B2-T2048-H8-h512-dv64-bf16-s7"),
    layer("chunk-bwd-dense-N64", "layer-B2-T2048-H8-h512-dv64-bf16-s8"),
    layer("chunk-bwd-splitk", "layer-B1-T512-H2-h512-dv64-bf16-s9"),
    layer("chunk-deep-D4-w8", "layer-B2-T512-H8-h512-dv64-bf16-s3"),
    layer("chunk-decode-w16", "layer-B2-T128-H2-h128-dv64-fp32-s4-dec8"),
    layer("chunk-decode-la-w256", "layer-B2-T128-H2-h128-dv64-fp32-s10-dec8"),
)}
GATE = ("L65536-N65536-dv64", "L1024-N65536-dv64")


def check(group: Group, registry) -> None:
    """Refuses a group whose cells the registry lacks or whose cells break its claim: an `equal` parameter with two
    values, a RoLA cell whose state count is not the group's N, an attention cell that is not one bf16 head."""
    missing = sorted(set(group.cells) - set(registry.cells))
    if missing:
        raise KeyError(f"group {group.name} names cells the central registry does not hold: {missing}")
    params = {c: registry.cell(c)["params"] for c in group.cells}
    for field in group.equal:
        values = {c: p.get(field) for c, p in params.items()}
        if None in values.values() or len({json.dumps(v, sort_keys=True) for v in values.values()}) != 1:
            raise ValueError(f"group {group.name} holds {field} equal, but its cells carry {values}")
    if group.states is None:
        return
    for c, p in params.items():
        if "widths" in p and math.prod(p["widths"]) != group.states:
            raise ValueError(f"group {group.name} claims N = {group.states}, but {c} has {math.prod(p['widths'])} states")
        if "heads" in p and (p["heads"], p["dtype"]) != (1, "bfloat16"):
            raise ValueError(f"group {group.name} claims one bf16 attention head, but {c} is {p['heads']} x {p['dtype']}")


def select(names: str) -> list[Group]:
    """A comma list of group names, `gate` (the groups of `GATE`) and `all`."""
    wanted = list(dict.fromkeys(n for name in names.split(",") for n in
                                (GROUPS if name == "all" else GATE if name == "gate" else (name,))))
    unknown = sorted(set(wanted) - set(GROUPS))
    if unknown:
        raise SystemExit(f"no group {unknown}; the groups are {sorted(GROUPS)}")
    return [GROUPS[name] for name in wanted]
