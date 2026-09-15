"""The suite's root (`declare.py`) and groups: every group holds its claim against the central cells and a group breaking
it is refused; the root loads each checkout's own declarations, gives the target alone its instruments, and declares one
session per group and arm set over every checkout's registration on the group's cells beside the attention reference,
one memory pass, a null gate on the target's carry arm, a store for each result and a server stop that runs after
everything; a checkout without declarations is not compared. No GPU: nothing is built.
`python -m unittest tests.measure.test_suite`"""
from __future__ import annotations

import sys
import tempfile
import textwrap
import unittest
from dataclasses import replace
from pathlib import Path

from rola_devtools.build.declare import Graph, load
from rola_devtools.cells import central

ROOT = Path(__file__).resolve().parents[2]
GROUPS = load(ROOT / "rola_bench" / "measure" / "groups.py")
SUITE = load(ROOT / "declare.py")

#: a checkout's declarations as rola's `declare.py` shapes them: a build, its instruments, and on the carry cells a
#: `carry_forward` registration and a clock reader
FAKE = textwrap.dedent('''
    from rola_devtools.build.declare import Env
    from rola_devtools.timing.declare import register_clock_reader, register_timing


    def checkout(path, *, python, label):
        return Env(label, python, str(path))


    def declare(g, env, *, cells, timing=None, instruments=("sass", "phases")):
        binary = g.node("binary", executor="fake:build", env=env)
        out = {"binary": binary, "entries": {}, "clock": None,
               "instruments": {n: g.node(n, executor="fake:tool", env=env, deps={"binary": binary}) for n in instruments}}
        if timing is None:
            return out
        carry = [c for c in cells if not c.startswith(("qkv-", "layer-"))]
        out["entries"]["carry_forward"] = register_timing(g, "carry_forward", server=timing, env=env,
                                                          executor="fake:timed", cells=carry, deps={"binary": binary})
        out["clock"] = register_clock_reader(g, "clock", server=timing, env=env, executor="fake:clock")
        return out
''')


class Groups(unittest.TestCase):
    def test_every_group_holds_its_claim_and_the_gate_is_among_them(self):
        registry = central()
        for group in GROUPS["GROUPS"].values():
            GROUPS["check"](group, registry)
        self.assertEqual([g.name for g in GROUPS["select"]("gate")], list(GROUPS["GATE"]))
        self.assertEqual(len(GROUPS["select"]("all,gate")), len(GROUPS["GROUPS"]))
        with self.assertRaises(SystemExit):
            GROUPS["select"]("no-such-group")

    def test_a_group_breaking_its_claim_is_refused(self):
        registry, flagship = central(), GROUPS["GROUPS"]["L1024-N65536-dv64"]
        with self.assertRaisesRegex(ValueError, "holds tokens equal"):
            GROUPS["check"](replace(flagship, cells=(*flagship.cells, "qkv-L256-dv64")), registry)
        with self.assertRaisesRegex(ValueError, "claims N = 4096"):
            GROUPS["check"](replace(flagship, states=4096), registry)
        with self.assertRaises(KeyError):
            GROUPS["check"](replace(flagship, cells=("no-such-cell",)), registry)


class Root(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name)
        for name in ("tip", "base"):
            (base / name).mkdir()
            (base / name / "declare.py").write_text(FAKE)
            (base / f"venv-{name}" / "bin").mkdir(parents=True)
            (base / f"venv-{name}" / "bin" / "python").symlink_to(sys.executable)
        self.base = base

    def root(self, **args):
        g = Graph()
        public = SUITE["root"](g, target=f"worktree:{self.base / 'tip'}", **args)
        return public, g.targets

    def test_a_group_is_one_session_per_arm_set_over_every_checkouts_registration_and_the_reference(self):
        public, targets = self.root(references=f"worktree:{self.base / 'base'},label:master", groups="gate")
        flagship = GROUPS["GROUPS"]["L1024-N65536-dv64"]
        session = targets["session/L1024-N65536-dv64/carry_forward+flash"]
        members = {d.label for role, d in session.deps.items() if role.startswith("r")}
        self.assertEqual(members, {"tip/carry_forward", "master/carry_forward", "bench/flash"})
        self.assertEqual(session.params["cells"], list(flagship.cells))
        self.assertEqual(session.deps["clock"].label, "tip/clock")
        self.assertEqual(session.holds, {"gpu": "all", "clock": 1})
        self.assertIn("tip/phases", targets)
        self.assertNotIn("master/phases", targets)
        stored = {t.label: t.deps["source"].label for t in targets.values() if (t.executor or "").endswith("store:put")}
        self.assertEqual({k: stored[k] for k in ("tip/store/phases", "store/memory")},
                         {"tip/store/phases": "tip/phases", "store/memory": "memory"})
        gate = targets["null"]
        self.assertEqual((gate.deps["r0"].label, gate.params["cells"]),
                         ("tip/carry_forward", ["nl64k-dense", "flagship-dense"]))
        stop = targets["timing-server-stop"]
        self.assertTrue(stop.always_run)
        self.assertIn(targets["store/memory"], stop.deps.values())
        self.assertEqual(set(public), {"suite"})

    def test_the_parts_and_cells_narrow_what_is_declared(self):
        _public, targets = self.root(groups="gate", parts="instruments", cells="flagship-dense")
        self.assertNotIn("timing-server", targets)
        self.assertIn("tip/phases", targets)
        self.assertFalse(any(label.startswith("session/") for label in targets))
        _public, targets = self.root(groups="gate", skip_cells="nl64k-dense,nl64k-alt-k4,qkv-L65536-dv64", parts="sessions")
        self.assertEqual(sorted(label for label in targets if label.startswith("session/")),
                         ["session/L1024-N65536-dv64/carry_forward+flash", "session/L1024-N65536-dv64/prefill_op+flash"])

    def test_a_checkout_without_declarations_or_with_a_taken_label_is_refused(self):
        (self.base / "base" / "declare.py").unlink()
        with self.assertRaisesRegex(SystemExit, "before the declaration API"):
            self.root(references=f"worktree:{self.base / 'base'}")
        with self.assertRaisesRegex(SystemExit, "rola-bench's own entry"):
            SUITE["root"](Graph(), target=f"worktree:{self.base / 'tip'},label:bench")


if __name__ == "__main__":
    unittest.main()
