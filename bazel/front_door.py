#!/usr/bin/env python3
"""rola-bench's front door stub: bazelisk runs `bazel/bazel` (this file, the wrapper `.bazeliskrc` names) in place of the
Bazel it downloaded, and it runs rola-devtools' front door (`rola_devtools.graph.front_door`) for this checkout, from
the machine's rola-devtools checkout -- `workspace.devtools` in the dev config's workspace section, else the sibling
`rola-devtools` beside this checkout (a worktree's beside its worktrees folder). Standard library only: it runs before
any environment exists."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SIBLINGS = (ROOT.parent if ROOT.parent.name == "worktrees" else ROOT).parent


def devtools() -> Path:
    pointer = os.environ.get("ROLA_DEV_CONFIG")
    section = (Path(pointer).expanduser() if pointer else Path.home() / ".config" / "rola") / "workspace.json"
    value = json.loads(section.read_text()).get("devtools") if section.is_file() else None
    return Path(value).expanduser() if value else SIBLINGS / "rola-devtools"


sys.path.insert(0, str(devtools()))
from rola_devtools.graph import front_door  # noqa: E402 -- the path insert above must precede it

if __name__ == "__main__":
    sys.exit(front_door.main(sys.argv[1:], ROOT))
