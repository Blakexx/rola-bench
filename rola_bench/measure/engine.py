"""THE ENGINE: a DAG of measurement nodes, each keyed by what its result depends on, each result kept.

A NODE is one action on one unit: a module (what it runs and on what), its unit (a cell, a subject at a cell), and the
nodes it depends on. Its KEY is sha256 over the node's semantics -- the module, the unit, the identity of every input the
module declares (a binary, an instrument's code, the environment) -- and, for each dependency, that node's key and the
digest of its output. So a change anywhere re-keys exactly the nodes downstream of it, and a revert finds the old key's
result again.

A module's results are stored through `rola_results` at the location `suite/<module>`: a RECORD per key holding the
semantics that made it and every SAMPLE -- each time the node ran under that key, its raw output or its failure, when, how
long, and the checkouts it ran on. A key with a successful sample is COMPLETE and does not run again; a key whose samples
all failed is retried; `repeat` adds a sample to a complete key of a repeatable module. Nothing derived from two records
-- a ratio, a delta between commits -- is ever stored: that is the reader's.
"""
from __future__ import annotations

import hashlib
import tempfile
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from rola_results import ROOT, Store, key


@dataclass(frozen=True)
class Node:
    """One action on one unit. `identity` is every input the action depends on besides its dependency nodes, already
    reduced to hashes (a binary's sha256, an instrument's code key, the environment key); `run(outputs, dest)` writes the
    raw output to `dest`, reading its dependencies' outputs from `outputs`."""

    module: str
    unit: str
    identity: dict
    run: Callable[[dict[str, Path], Path], None]
    deps: tuple[str, ...] = ()
    repeatable: bool = False
    meta: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return f"{self.module}@{self.unit}" if self.unit else self.module


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def order(nodes: list[Node]) -> list[Node]:
    """Dependencies first; refuses a dependency that is not a node of the graph, and a cycle."""
    by_name = {n.name: n for n in nodes}
    done: list[Node] = []
    state: dict[str, str] = {}

    def visit(n: Node) -> None:
        if state.get(n.name) == "done":
            return
        if state.get(n.name) == "open":
            raise SystemExit(f"a dependency cycle through {n.name}")
        state[n.name] = "open"
        for d in n.deps:
            if d not in by_name:
                raise SystemExit(f"{n.name} depends on {d}, which is not in the graph")
            visit(by_name[d])
        state[n.name] = "done"
        done.append(n)

    for n in nodes:
        visit(n)
    return done


@dataclass
class Outcome:
    name: str
    key: str
    status: str
    wall_s: float = 0.0
    samples: int = 0
    error: str = ""


def key_of(node: Node, dep_keys: dict[str, tuple[str, str]]) -> tuple[str, dict]:
    semantics = {"module": node.module, "unit": node.unit, "identity": node.identity,
                 "deps": {d: {"key": k, "output": o} for d, (k, o) in sorted(dep_keys.items())}}
    return key(semantics), semantics


def run(nodes: list[Node], root: Path = ROOT, *, repeat: bool = False, force: bool = False, dry: bool = False,
        log: Callable[[str], None] = print) -> list[Outcome]:
    """Walk the graph in dependency order; run what is not complete (or, with `repeat`, what is repeatable; with `force`,
    everything); keep every result. A node whose dependency failed or was not run is BLOCKED, and stays unrecorded."""
    outputs: dict[str, Path] = {}
    dep_keys: dict[str, tuple[str, str]] = {}
    outcomes: list[Outcome] = []
    pending: set[str] = set()
    for node in order(nodes):
        missing = [d for d in node.deps if d not in outputs]
        if dry and missing and all(d in pending for d in missing):
            pending.add(node.name)
            outcomes.append(Outcome(node.name, "", "pending"))
            log(f"pending   {node.name} (after {', '.join(missing)})")
            continue
        if missing:
            outcomes.append(Outcome(node.name, "", "blocked", error=f"not run: {', '.join(missing)}"))
            log(f"blocked   {node.name} (no result for {', '.join(missing)})")
            continue

        store = Store(f"suite/{node.module}", root)
        k, semantics = key_of(node, {d: dep_keys[d] for d in node.deps})
        record = store.get(k)
        complete = Store.complete(record)
        ok = sum(s["ok"] for s in record["samples"]) if record else 0
        if complete:
            outputs[node.name] = store.output(record)
            dep_keys[node.name] = (k, digest(outputs[node.name]))
        if complete and not force and not (repeat and node.repeatable):
            outcomes.append(Outcome(node.name, k, "complete", samples=ok))
            log(f"complete  {node.name} [{k[:12]}] {ok} sample(s)")
            continue
        if dry:
            pending.add(node.name)
            outcomes.append(Outcome(node.name, k, "pending"))
            log(f"pending   {node.name} [{k[:12]}]" + (" (failed before)" if record and not complete else ""))
            continue

        started = time.time()
        with tempfile.TemporaryDirectory(prefix="rola_suite_node_") as tmp:
            dest = Path(tmp) / "out.json"
            try:
                node.run({d: outputs[d] for d in node.deps}, dest)
                if not dest.exists():
                    raise RuntimeError("the action wrote no output")
            except (Exception, SystemExit) as ex:
                wall = time.time() - started
                store.put(semantics, error=f"{type(ex).__name__}: {ex}", wall_s=wall, provenance=node.meta,
                          trace=traceback.format_exc()[-3000:])
                outcomes.append(Outcome(node.name, k, "failed", wall, ok, str(ex)[-300:]))
                log(f"FAILED    {node.name} [{k[:12]}] {wall:.1f}s: {str(ex)[-200:]}")
                continue
            wall = time.time() - started
            sample = store.put(semantics, output_file=dest, wall_s=wall, provenance=node.meta)
        outputs[node.name] = store.dir / sample["output"]
        dep_keys[node.name] = (k, digest(outputs[node.name]))
        outcomes.append(Outcome(node.name, k, "ran", wall, ok + 1))
        log(f"ran       {node.name} [{k[:12]}] {wall:.1f}s" + (f", sample {ok + 1}" if node.repeatable else ""))
    return outcomes
