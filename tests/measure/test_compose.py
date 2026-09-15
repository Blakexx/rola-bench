"""The composer relates instances without defining a measurement: a group's sessions interleave every target's arms on
the group's rola cells and, on a carry subject, the attention reference; the subject target's instruments and every
instance's memory rows are selected; nothing an instance does not declare is named.
`python -m unittest tests.measure.test_compose`"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rola_devtools.graph import Env
from rola_devtools.graph.engine import Instance

from rola_bench.measure.compose import BENCH, compose, parse


def _instance(label: str, names: list[str]) -> Instance:
    return Instance(Env(label, "python", ".", "g:graph"), tuple({"name": n, "timed": n.startswith("time.")} for n in names))


def _rola(label: str, cells: list[str]) -> Instance:
    names = ["carry.sass", "carry.registers@arm0"]
    for cell in cells:
        names += [f"carry.phases@{cell}", f"time.carry_forward@{cell}", f"memory.carry_forward@{cell}"]
    return _instance(label, names)


GROUP = {"name": "L1024", "holds": "1024 tokens", "equal": ["tokens"],
         "runners": {"rola": [{"name": "dense"}, {"name": "alt"}, {"name": "oracle-only"}],
                     "attention": [{"name": "attn-L1024"}]}}


class Compose(unittest.TestCase):
    def setUp(self):
        self.tip, self.base = _rola("tip", ["dense", "alt"]), _rola("base", ["dense", "alt"])
        self.bench = _instance(BENCH, ["time.flash@attn-L1024", "memory.flash@attn-L1024"])

    def test_a_group_is_one_session_per_subject_with_the_attention_reference(self):
        sessions, selection = compose([self.tip, self.base, self.bench], [GROUP])
        (session,) = sessions
        self.assertEqual(session.name, "carry_forward@L1024")
        self.assertEqual(session.members, ("tip:time.carry_forward@dense", "tip:time.carry_forward@alt",
                                           "base:time.carry_forward@dense", "base:time.carry_forward@alt",
                                           "bench:time.flash@attn-L1024"))
        self.assertEqual(session.reference, "tip")
        self.assertEqual(session.relation["roles"], {"tip": "subject", "base": "reference", "bench": "attention"})
        self.assertLessEqual(set(session.members), selection)

    def test_instruments_are_the_subjects_and_memory_is_everyones(self):
        _sessions, selection = compose([self.tip, self.base, self.bench], [GROUP])
        self.assertIn("tip:carry.phases@alt", selection)
        self.assertIn("tip:carry.sass", selection)
        self.assertNotIn("base:carry.phases@alt", selection)
        self.assertLessEqual({"tip:memory.carry_forward@dense", "base:memory.carry_forward@dense",
                              "bench:memory.flash@attn-L1024"}, selection)

    def test_nodes_and_cells_narrow_what_is_selected(self):
        sessions, selection = compose([self.tip, self.bench], [GROUP], nodes="memory", cells="alt,attn-L1024")
        self.assertEqual(sessions, [])
        self.assertEqual(selection, {"tip:memory.carry_forward@alt", "bench:memory.flash@attn-L1024"})

    def test_a_target_needs_a_worktree_and_a_venv(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "venv" / "bin").mkdir(parents=True)
            (Path(tmp) / "venv" / "bin" / "python").write_text("")
            target = parse(f"worktree:{tmp},venv:{tmp}/venv,label:tip")
            self.assertEqual((target.label, target.python), ("tip", f"{tmp}/venv/bin/python"))
            with self.assertRaises(SystemExit):
                parse(f"worktree:{tmp},venv:{tmp}/venv,label:{BENCH}")
        with self.assertRaises(SystemExit):
            parse("venv:/nowhere")


if __name__ == "__main__":
    unittest.main()
