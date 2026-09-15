"""THE GROUPS: which central cells launch together, what they hold, and which arms are timed in one session.

A group (`groups.json`) is a selection of central cells (`rola_devtools.cells`) and its relation: `holds`, the claim in
words; `equal`, the parameters every cell of the group carries with one value, checked when the file loads; and
`together`, the arm sets each timed in one session. `sessions` turns a group into the service's sessions
(`rola_devtools.measure.service.Session`): one per arm set, its members every instance's arms of the set on the group's
cells each accepts, its relation the group's claim and each instance's role. `GATE_GROUPS` are the groups the kernel's
gate reads first.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

GROUPS = Path(__file__).with_name("groups.json")
#: the groups the kernel's gate reads first; `--groups gate` selects them
GATE_GROUPS = ("L65536-N65536-dv64", "L1024-N65536-dv64")
#: where every session's record lives
SESSION_LOCATION = "bench/session"


@dataclass(frozen=True)
class Group:
    name: str
    holds: str
    equal: tuple[str, ...]
    cells: tuple[str, ...]
    together: tuple[tuple[str, ...], ...]


def load(path: Path = GROUPS, registry=None) -> dict[str, Group]:
    """Every group of `path`, its cells checked against the central registry and its `equal` claim against their
    parameters."""
    if registry is None:
        from rola_devtools.cells import central

        registry = central()
    doc = json.loads(Path(path).read_text())
    if doc.get("schema") != 1:
        raise ValueError(f"{path}: schema {doc.get('schema')!r}; this reader reads 1")
    groups: dict[str, Group] = {}
    for g in doc["groups"]:
        group = Group(g["name"], g["holds"], tuple(g["equal"]), tuple(g["cells"]), tuple(tuple(t) for t in g["together"]))
        if group.name in groups:
            raise ValueError(f"{path}: the group {group.name} is named twice")
        missing = sorted(set(group.cells) - set(registry.cells))
        if missing:
            raise KeyError(f"group {group.name} names cells the central registry does not hold: {missing}")
        if not group.cells or not group.together or any(not arms for arms in group.together):
            raise ValueError(f"group {group.name} selects no cell or times an empty arm set")
        for field in group.equal:
            values = {c: registry.cell(c)["params"].get(field) for c in group.cells}
            if None in values.values() or len({json.dumps(v, sort_keys=True) for v in values.values()}) != 1:
                raise ValueError(f"group {group.name} holds {field} equal, but its cells carry {values}")
        groups[group.name] = group
    return groups


def select(groups: dict[str, Group], names: str) -> list[Group]:
    """`all`, `gate`, or a comma list of group names."""
    wanted = list(groups) if names == "all" else list(GATE_GROUPS if names == "gate" else names.split(","))
    unknown = sorted(set(wanted) - set(groups))
    if unknown:
        raise SystemExit(f"no group {unknown} in {GROUPS.name}")
    return [groups[name] for name in wanted]


def sessions(group: Group, roles: dict[str, str], *, reference: str, skip: frozenset[str] = frozenset(),
             rounds: int, reps: int, warmup: int) -> list:
    """The group's sessions: one per arm set, on the group's cells less `skip`, its ratios divided by `reference`'s arms."""
    from rola_devtools.measure.service import Session

    cells = tuple(c for c in group.cells if c not in skip)
    relation = {"group": group.name, "holds": group.holds, "equal": list(group.equal), "cells": list(group.cells),
                "roles": roles}
    return [Session(f"{'+'.join(arms)}@{group.name}", SESSION_LOCATION, arms, cells, reference=reference,
                    relation=relation, rounds=rounds, reps=reps, warmup=warmup)
            for arms in group.together if cells]
