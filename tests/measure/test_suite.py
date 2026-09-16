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
    from rola_devtools.cells.declare import cells as cell_nodes
    from rola_devtools.timing.declare import register_clock_reader, register_timing


    def checkout(path, *, python, label):
        return Env(label, python, str(path))


    SURFACES = {"oracle": ("fake:side", "carry", "oracle", {"host_cpu": "all"}, "bit-identical", {})}


    def declare(g, env, *, timing=None, instruments=("sass", "phases")):
        from rola_devtools.cells import central
        from rola_devtools.diff import side

        cells = sorted(central().cells)
        carry = [c for c in cells if not c.startswith(("qkv-", "layer-", "producer-"))]
        binary = g.node("binary", executor="fake:build", env=env)
        oracle = [c for c in carry if central().cell(c)["params"].get("tier") == "oracle"]
        out = {"binary": binary, "entries": {}, "clock": None,
               "instruments": {n: g.node(n, executor="fake:tool", env=env, deps={"binary": binary}) for n in instruments},
               "sides": {"oracle": side(g, "side/oracle", env=env, executor="fake:side", cells=cell_nodes(g, oracle))},
               "diffs": {}}
        if timing is None:
            return out
        out["entries"]["carry_forward"] = register_timing(g, "carry_forward", server=timing, env=env,
                                                          executor="fake:timed", cells=cell_nodes(g, carry),
                                                          deps={"binary": binary})
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
        public, targets = self.root(references=f"worktree:{self.base / 'base'},label:master")
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
        #: the null gate takes the FIRST carry cell of every group the target's carry_forward registers on: one
        #: cell a group, so a worker's bias is read once per group's shape and never per cell
        with_carry = [g for g in GROUPS["GROUPS"].values() if any(not c.startswith(("qkv-", "layer-")) for c in g.cells)]
        self.assertEqual(gate.deps["r0"].label, "tip/carry_forward")
        self.assertEqual(gate.params["cells"], [next(c for c in g.cells if not c.startswith(("qkv-", "layer-")))
                                                for g in with_carry])
        stop = targets["timing-server-stop"]
        self.assertTrue(stop.always_run)
        self.assertIn(targets["store/memory"], stop.deps.values())
        self.assertEqual(set(public), {"suite", "diffs"})

    def test_the_root_takes_no_selector_and_a_build_prunes_it_by_label(self):
        """The root declares EVERYTHING; `--only`/`--skip` at the CLI is the one way to run less of it."""
        import inspect

        from rola_devtools.build.scheduler import select

        self.assertFalse({"cells", "skip_cells", "parts", "groups", "instruments"}
                         & set(inspect.signature(SUITE["root"]).parameters))
        public, targets = self.root()
        #: a session per group arm set that has a member -- every group, not a gate subset; the double registers
        #: only `carry_forward` and `flash`, so the arm sets naming other arms declare nothing
        sessions = {label for label in targets if label.startswith("session/")}
        expected = {f"session/{g.name}/{'+'.join(arms)}" for g in GROUPS["GROUPS"].values() for arms in g.together
                    if set(arms) & {"carry_forward", "flash"}}
        self.assertEqual(sessions, expected)
        only = select([public["suite"]], only=["tip/phases"])
        self.assertEqual([t.label for t in only], ["tip/binary", "tip/phases"])

    def test_each_surface_the_target_exposes_is_diffed_against_each_reference_under_its_own_rule(self):
        public, targets = self.root(references=f"worktree:{self.base / 'base'},label:master")
        verdict = targets["diff/oracle/master"]
        self.assertEqual((verdict.deps["left"].label, verdict.deps["right"].label), ("tip/side/oracle", "master/side/oracle"))
        self.assertEqual((verdict.params["strategy"], verdict.params["minimum"]),
                         ("bit-identical", len(targets["tip/side/oracle"].inputs)))
        self.assertEqual(targets["store/diff/oracle/master"].params["location"], "diff/oracle")
        self.assertEqual([t.label for t in public["diffs"].deps.values()], ["diff/oracle/master"])
        self.assertNotIn("diff/oracle/tip", targets)

    def test_a_checkout_without_declarations_or_with_a_taken_label_is_refused(self):
        (self.base / "base" / "declare.py").unlink()
        with self.assertRaisesRegex(SystemExit, "before the declaration API"):
            self.root(references=f"worktree:{self.base / 'base'}")
        with self.assertRaisesRegex(SystemExit, "rola-bench's own entry"):
            SUITE["root"](Graph(), target=f"worktree:{self.base / 'tip'},label:bench")


if __name__ == "__main__":
    unittest.main()
