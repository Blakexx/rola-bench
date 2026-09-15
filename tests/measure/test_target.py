"""A target's binary is found where its checkout built it: in rola's binary plugin, or in `rola/` for a checkout from
before the plugin; never a part-harness driver beside it.
`python -m unittest tests.measure.test_target`"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rola_bench.measure.target import Target, extension


def _checkout(root: Path, *files: str) -> Target:
    for name in files:
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(name.encode())
    return Target(worktree=root, venv=root / "venv", label="t")


class ExtensionTest(unittest.TestCase):
    def test_the_plugin_holds_the_binary(self):
        with tempfile.TemporaryDirectory() as tmp:
            t = _checkout(Path(tmp), "rola_cu13/_C.abi3.so", "rola_cu13/_C_parts.abi3.so")
            self.assertEqual(extension(t), Path(tmp) / "rola_cu13/_C.abi3.so")

    def test_a_checkout_from_before_the_plugin(self):
        with tempfile.TemporaryDirectory() as tmp:
            t = _checkout(Path(tmp), "rola/_C.cpython-311-x86_64-linux-gnu.so")
            self.assertEqual(extension(t), Path(tmp) / "rola/_C.cpython-311-x86_64-linux-gnu.so")

    def test_a_driver_alone_is_no_binary(self):
        with tempfile.TemporaryDirectory() as tmp:
            t = _checkout(Path(tmp), "rola_cu13/_C_parts.abi3.so")
            with self.assertRaises(SystemExit):
                extension(t)


if __name__ == "__main__":
    unittest.main()
