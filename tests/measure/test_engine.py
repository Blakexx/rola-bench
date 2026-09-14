"""The engine's contract, on fake nodes: keys compose down the graph, complete keys do not run, a revert finds its old
result, failures are kept and retried, dependents of a failure are blocked, repeats accumulate, force re-runs.
`python -m unittest rola_bench.measure.test_engine`"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rola_results import Store

from rola_bench.measure import engine


class Graph:
    """a -> b (b reads a's output); c independent. Each action counts its runs and writes its identity."""

    def __init__(self):
        self.runs: dict[str, int] = {}
        self.code = {"a": "1", "b": "1", "c": "1"}
        self.fail: set[str] = set()

    def action(self, name):
        def run(deps, dest):
            self.runs[name] = self.runs.get(name, 0) + 1
            if name in self.fail:
                raise RuntimeError(f"{name} failed")
            upstream = {d: json.loads(p.read_text()) for d, p in deps.items()}
            dest.write_text(json.dumps({"node": name, "code": self.code[name], "upstream": upstream, "n": self.runs[name]}))
        return run

    def nodes(self):
        return [engine.Node("m.a", "", {"code": self.code["a"]}, self.action("a"), repeatable=True),
                engine.Node("m.b", "", {"code": self.code["b"]}, self.action("b"), deps=("m.a",)),
                engine.Node("m.c", "cell", {"code": self.code["c"]}, self.action("c"))]


class EngineTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.g = Graph()

    def tearDown(self):
        self.tmp.cleanup()

    def run_graph(self, **kw):
        return {o.name: o for o in engine.run(self.g.nodes(), self.root, log=lambda _m: None, **kw)}

    def test_cold_then_complete(self):
        first = self.run_graph()
        self.assertEqual({o.status for o in first.values()}, {"ran"})
        again = self.run_graph()
        self.assertEqual({o.status for o in again.values()}, {"complete"})
        self.assertEqual(self.g.runs, {"a": 1, "b": 1, "c": 1})

    def test_upstream_change_rekeys_downstream_only(self):
        self.run_graph()
        self.g.code["a"] = "2"
        out = self.run_graph()
        self.assertEqual((out["m.a"].status, out["m.b"].status, out["m.c@cell"].status), ("ran", "ran", "complete"))

    def test_revert_finds_the_old_key(self):
        self.run_graph()
        self.g.code["b"] = "2"
        self.run_graph()
        self.g.code["b"] = "1"
        out = self.run_graph()
        self.assertEqual(out["m.b"].status, "complete")
        self.assertEqual(self.g.runs["b"], 2)

    def test_failure_is_kept_blocks_dependents_and_is_retried(self):
        self.g.fail.add("a")
        out = self.run_graph()
        self.assertEqual((out["m.a"].status, out["m.b"].status, out["m.c@cell"].status), ("failed", "blocked", "ran"))
        record = Store("suite/m.a", self.root).get(out["m.a"].key)
        self.assertFalse(Store.complete(record))
        self.assertIn("a failed", record["samples"][-1]["error"])
        self.g.fail.clear()
        out = self.run_graph()
        self.assertEqual((out["m.a"].status, out["m.b"].status, out["m.c@cell"].status), ("ran", "ran", "complete"))
        record = Store("suite/m.a", self.root).get(out["m.a"].key)
        self.assertEqual([s["ok"] for s in record["samples"]], [False, True])

    def test_repeat_accumulates_samples_under_one_key(self):
        first = self.run_graph()
        out = self.run_graph(repeat=True)
        self.assertEqual(out["m.a"].key, first["m.a"].key)
        self.assertEqual(out["m.a"].samples, 2)
        self.assertEqual(out["m.c@cell"].status, "complete")
        store = Store("suite/m.a", self.root)
        record = store.get(out["m.a"].key)
        outputs = [json.loads(store.output(record, s["n"]).read_text())["n"] for s in record["samples"]]
        self.assertEqual(outputs, [1, 2])

    def test_dry_run_changes_nothing(self):
        out = self.run_graph(dry=True)
        self.assertEqual({o.status for o in out.values()}, {"pending"})
        self.assertEqual(self.g.runs, {})

    def test_force_runs_everything(self):
        self.run_graph()
        out = self.run_graph(force=True)
        self.assertEqual({o.status for o in out.values()}, {"ran"})

    def test_a_dependency_outside_the_graph_is_refused(self):
        with self.assertRaises(SystemExit):
            engine.order([engine.Node("m.x", "", {}, lambda d, p: None, deps=("m.missing",))])


if __name__ == "__main__":
    unittest.main()
