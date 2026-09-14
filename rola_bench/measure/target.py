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
    done = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
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


#: THE CELLS A TARGET CAN RUN: its registry's, less every carry cell whose arm (D, DV, warps_per_cta) its binary does not
#: carry -- the binary's own answer (`rola.ops.carry.arms()`), never the source tree's.
_REGISTRY = """
import json, sys
sys.path.insert(0, 'benchmarks')
from benchmarks.cells.registry import CELLS
from rola.ops.carry import arms
carried = {tuple(arm) for arm in arms()}
runnable = {name: (kind, spec) for name, (kind, spec) in CELLS.items()
            if kind != 'carry' or (len(spec.widths), spec.dv, spec.warps_per_cta) in carried}
"""


@cache
def cells(target: Target, kind: str) -> tuple[str, ...]:
    """The registry's cells of `kind` (carry or layer) the target runs; the cells its binary cannot carry are named once."""
    code = _REGISTRY + (f"print(json.dumps([sorted(n for n, (k, _) in runnable.items() if k == {kind!r}), "
                        f"sorted(n for n, (k, _) in CELLS.items() if k == {kind!r} and n not in runnable), sorted(carried)]))")
    rc, out = sh([target.python, "-c", code], target.worktree, 600)
    if rc:
        raise SystemExit(f"{target.label}: could not read the cell registry: {out[-400:]}")
    runnable, uncarried, carried = json.loads(out.strip().splitlines()[-1])
    if uncarried:
        print(f"{target.label}: the binary carries {[tuple(a) for a in carried]}; left out, no arm for: "
              f"{', '.join(uncarried)}", flush=True)
    return tuple(runnable)


@dataclass(frozen=True)
class Lane:
    """How one checkout spells a timing unit to its probe: its own bench name, the call count its worker takes, and the
    cells `bench.subjects.applicable` admits there."""

    bench: str
    calls: int
    cells: tuple[str, ...]


#: A ROSTER FROM BEFORE ROLA'S CALL COUNT (`bench.subjects.Subject.calls`) names a multi-call unit as a bench of its own.
#: An entry retires once no reference predates the count.
PRE_COUNT_BENCHES = {"prefill_op_chunked": ("prefill_op", 4)}

_ROSTER = _REGISTRY + """
from bench.subjects import SUBJECTS, applicable
counts = sorted({n for s in SUBJECTS.values() for n in getattr(s, 'calls', (1,))})
roster = {}
for cell, (kind, spec) in sorted(runnable.items()):
    for n in counts:
        for name in applicable(spec, kind, *([n] if n != 1 else [])):
            roster.setdefault(f'{name}@{n}', []).append(cell)
print(json.dumps(roster))
"""


@cache
def subjects(target: Target) -> dict[tuple[str, int], Lane]:
    """The bench roster the target defines: each (subject, call count) unit with the lane that runs it there."""
    rc, out = sh([target.python, "-c", _ROSTER], target.worktree, 600)
    if rc:
        raise SystemExit(f"{target.label}: could not read the bench roster: {out[-400:]}")
    raw = json.loads(out.strip().splitlines()[-1])
    lanes: dict[tuple[str, int], Lane] = {}
    for key, admitted in raw.items():
        bench, n = key.rsplit("@", 1)
        if bench not in PRE_COUNT_BENCHES:
            lanes[bench, int(n)] = Lane(bench, int(n), tuple(admitted))
    for bench, unit in PRE_COUNT_BENCHES.items():
        if f"{bench}@1" in raw:
            lanes.setdefault(unit, Lane(bench, 1, tuple(raw[f"{bench}@1"])))
    return lanes
