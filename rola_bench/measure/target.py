"""TARGETS: the rola checkouts a run measures, and the identities their nodes are keyed by.

A TARGET is a rola checkout and the venv that runs it (`worktree:PATH[,venv:PATH][,label:NAME][,schedule:POLICY]`).
Everything a node runs is that checkout's OWN instrument, invoked by its JSON command line from its own directory under
its own venv, so a target measures itself by its own definitions -- the way the probe runs each binary's own worker.

The identities:
- a BINARY is the sha256 of the checkout's built extension (`rola/_C*.so`);
- an INSTRUMENT is the sha256 of the instrument file and every repository file it imports, transitively (found by
  walking the imports, so a dependency cannot be forgotten), plus the data files and directories the module names (the cell
  registry, a budget, the kernel source a compile reads);
- the ENVIRONMENT is the GPU and driver, the CUDA toolkit's assembler, torch, the Nsight Compute CLI and the SM clock the
  host locks -- read from the machine, never from paths, so moving a tool does not re-key and upgrading one does.
The machine's settings come from the target's own `tools/dev.py get` (the dev config).
"""
from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import tempfile
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

from rola_devtools.process import run

#: where a checkout's repository-local imports resolve
IMPORT_ROOTS = ("tools", "benchmarks", ".")


@dataclass(frozen=True)
class Target:
    worktree: Path
    venv: Path
    label: str
    schedule: str = "first"
    extra: dict = field(default_factory=dict, compare=False, hash=False)

    @property
    def python(self) -> str:
        return str(self.venv / "bin" / "python")

    def schedule_for(self, cell: str) -> str:
        """A carry order policy per cell: `first`, or `dense/sparse` (e.g. `box/sparse-g32`) split on the cell's name."""
        if "/" not in self.schedule:
            return self.schedule
        dense, sparse = self.schedule.split("/", 1)
        return dense if "dense" in cell or "struct" in cell else sparse

    def spec(self) -> str:
        """The probe's `--binary` spec for this target (without its schedule)."""
        return f"worktree:{self.worktree},venv:{self.venv},label:{self.label}"


def parse(spec: str) -> Target:
    fields = dict(part.split(":", 1) for part in spec.split(","))
    if "worktree" not in fields:
        raise SystemExit(f"a target needs worktree:PATH -- got {spec!r}")
    worktree = Path(fields["worktree"]).expanduser().resolve()
    venv = Path(fields.get("venv") or _venv_of(worktree)).expanduser()
    return Target(worktree, venv, fields.get("label", worktree.name), fields.get("schedule", "first"))


def _venv_of(worktree: Path) -> str:
    sibling = worktree.parent / f"venv-{worktree.name}"
    if not (sibling / "bin" / "python").exists():
        raise SystemExit(f"no venv given for {worktree} and no sibling {sibling}")
    return str(sibling)


def sh(cmd: list[str], cwd: Path, timeout: int = 3600) -> tuple[int, str]:
    """The command's return code and output. A timeout, an interrupt or the suite's own exit stops its whole process tree
    (`rola_devtools.process`): a profiled unit's measured process otherwise outlives it on the device."""
    done = run(cmd, cwd=cwd, timeout=timeout)
    return done.returncode, done.stdout + done.stderr


@cache
def config(target: Target, key: str) -> str:
    rc, out = sh([target.python, "tools/dev.py", "get", key], target.worktree, 600)
    if rc:
        raise SystemExit(f"{target.label}: tools/dev.py get {key} failed: {out[-400:]}")
    return out.strip().splitlines()[-1]


@cache
def binary_key(target: Target) -> str:
    sos = sorted((target.worktree / "rola").glob("_C*.so"))
    if not sos:
        raise SystemExit(f"{target.label}: no built extension under {target.worktree}/rola")
    return hashlib.sha256(sos[0].read_bytes()).hexdigest()


@cache
def instrument_key(target: Target, entry: str, data: tuple[str, ...] = ()) -> str:
    """sha256 over the instrument's import closure inside the checkout and the named data files and directories, path and
    bytes each."""
    root = target.worktree
    seen: set[Path] = set()
    todo = [root / entry]
    while todo:
        path = todo.pop()
        if path in seen or not path.exists():
            continue
        seen.add(path)
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                names = [node.module]
            else:
                continue
            for name in names:
                rel = name.replace(".", "/")
                for base in IMPORT_ROOTS:
                    for cand in (root / base / f"{rel}.py", root / base / rel / "__init__.py"):
                        if cand.exists():
                            todo.append(cand)
    files = sorted(seen)
    for d in data:
        path = root / d
        files += sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
    h = hashlib.sha256()
    for path in files:
        h.update(str(path.relative_to(root)).encode() + b"\0" + path.read_bytes() + b"\0")
    return h.hexdigest()


@cache
def environment(target: Target) -> dict:
    """The machine facts a number depends on, read from the machine."""
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,compute_cap", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    cuda = config(target, "cuda-home")
    ptxas = sh([f"{cuda}/bin/ptxas", "--version"], target.worktree, 120)[1].split()
    torch = sh([target.python, "-c", "import torch; print(torch.__version__, torch.version.cuda)"], target.worktree, 600)[1]
    ncu = sh([config(target, "toolchain.ncu"), "--version"], target.worktree, 120)[1]
    return {"gpu": smi, "ptxas": next((w for w in ptxas if w.startswith("V")), " ".join(ptxas[-2:])),
            "torch": torch.strip().splitlines()[-1],
            "ncu": next((line.split("Version ")[1].split()[0] for line in ncu.splitlines() if "Version " in line), ""),
            "clock_ghz": config(target, "clock.ghz")}


def environment_key(target: Target) -> str:
    return hashlib.sha256(json.dumps(environment(target), sort_keys=True).encode()).hexdigest()


def instrument(target: Target, args: list[str], dest: Path, timeout: int = 3600) -> None:
    """Run `python <args> --json <tmp>` in the target checkout and move its JSON to `dest`; a non-zero exit with no JSON
    raises with the output's tail."""
    with tempfile.TemporaryDirectory(prefix="rola_suite_") as tmp:
        out = Path(tmp) / "out.json"
        rc, text = sh([target.python, *args, "--json", str(out)], target.worktree, timeout)
        if not out.exists():
            raise RuntimeError(f"{args[0]} exited {rc} without output:\n{text[-1500:]}")
        dest.write_text(out.read_text())


#: THE SUITE'S REGISTRY: the attention reference's cells and the points that group cells by runner (`rola_devtools.cells`)
SUITE_REGISTRY = Path(__file__).with_name("registry.json")


@cache
def registry(target: Target):
    """The target checkout's cells (`benchmarks/cells`) with the suite's cells and points: what a run's points resolve to."""
    from rola_devtools.cells import Registry

    cells = target.worktree / "benchmarks" / "cells"
    return Registry.load([cells / "carry_cells.json", cells / "layer_cells.json", SUITE_REGISTRY])


def rola_runner(target: Target, arm: str = ""):
    """A checkout's rola runner as the driver runs it: its own `bench.provider` under its own venv, from its own tree."""
    from rola_devtools.interleave import ArmSpec

    return ArmSpec(target.label, "bench.provider:arms", arm, python=target.python, cwd=str(target.worktree),
                   env={"PYTHONPATH": f"{target.worktree}:{target.worktree}/benchmarks"}, runner="rola")


@cache
def accepted(target: Target, source: Target, names: tuple[str, ...]) -> dict[str, dict]:
    """What `target`'s rola runner makes of the rola cells `names` as `source`'s registry defines them (a session sends
    every arm the target's cells): `{name: {"arms": [...]}}`, or `{name: {"refused": why}}`."""
    from rola_devtools.interleave import accepts

    reg = registry(source)
    return accepts(rola_runner(target), [reg.cell(name) for name in names])
