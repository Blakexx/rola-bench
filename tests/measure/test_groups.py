"""The suite's groups and targets: every group selects central cells whose `equal` claim holds, a group becomes one
session per arm set on its cells less the skipped ones with its relation recorded, and a target needs a worktree and a
venv and may not take rola-bench's own label. No GPU. `python -m unittest tests.measure.test_groups`"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rola_devtools.cells import central

from rola_bench.measure.groups import GATE_GROUPS, SESSION_LOCATION, load, select, sessions
from rola_bench.measure.targets import BENCH, instances, parse


class Groups(unittest.TestCase):
    def test_every_group_loads_against_the_central_cells_and_the_gate_is_among_them(self):
        groups = load()
        self.assertLessEqual(set(GATE_GROUPS), set(groups))
        cells = central().cells
        for group in groups.values():
            self.assertLessEqual(set(group.cells), set(cells), group.name)

    def test_a_group_whose_cells_do_not_hold_its_claim_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "groups.json"
            path.write_text(json.dumps({"schema": 1, "groups": [
                {"name": "mixed", "holds": "", "equal": ["tokens"], "cells": ["flagship-dense", "qkv-L256-dv64"],
                 "together": [["carry_forward", "flash"]]}]}))
            with self.assertRaisesRegex(ValueError, "holds tokens equal"):
                load(path)
            path.write_text(json.dumps({"schema": 1, "groups": [
                {"name": "absent", "holds": "", "equal": [], "cells": ["no-such-cell"], "together": [["x"]]}]}))
            with self.assertRaises(KeyError):
                load(path)

    def test_a_group_is_one_session_per_arm_set_with_its_relation(self):
        group = load()["L1024-N65536-dv64"]
        roles = {"tip": "subject", "base": "reference", BENCH: "library"}
        out = sessions(group, roles, reference="tip", skip=frozenset({"flagship-dense"}), rounds=8, reps=11, warmup=10)
        self.assertEqual([s.name for s in out], ["carry_forward+flash@L1024-N65536-dv64",
                                                 "prefill_op+flash@L1024-N65536-dv64"])
        first = out[0]
        self.assertEqual((first.location, first.reference, first.arms), (SESSION_LOCATION, "tip", ("carry_forward", "flash")))
        self.assertNotIn("flagship-dense", first.cells)
        self.assertIn("qkv-L1024-dv64", first.cells)
        self.assertEqual((first.relation["group"], first.relation["roles"]), ("L1024-N65536-dv64", roles))

    def test_select_takes_all_the_gate_or_names_and_refuses_an_unknown_group(self):
        groups = load()
        self.assertEqual([g.name for g in select(groups, "gate")], list(GATE_GROUPS))
        with self.assertRaises(SystemExit):
            select(groups, "no-such-group")


class Targets(unittest.TestCase):
    def test_a_target_needs_a_worktree_and_a_venv_and_the_bench_label_is_taken(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "venv" / "bin").mkdir(parents=True)
            (Path(tmp) / "venv" / "bin" / "python").write_text("")
            target = parse(f"worktree:{tmp},venv:{tmp}/venv,label:tip")
            self.assertEqual((target.label, target.python), ("tip", f"{tmp}/venv/bin/python"))
            with self.assertRaises(SystemExit):
                parse(f"worktree:{tmp},venv:{tmp}/venv,label:{BENCH}")
            insts = instances([target, parse(f"worktree:{tmp},venv:{tmp}/venv,label:base")], attention=True)
            self.assertEqual([(i.label, i.role, i.registry) for i in insts],
                             [("tip", "subject", "benchmarks.registry:registry"),
                              ("base", "reference", "benchmarks.registry:registry"),
                              (BENCH, "library", "rola_bench.measure.registry:registry")])
        with self.assertRaises(SystemExit):
            parse("venv:/nowhere")


if __name__ == "__main__":
    unittest.main()
