"""THE INSTANCES a run measures: rola checkouts (targets) and rola-bench itself.

A TARGET is a rola checkout and the venv that runs it (`worktree:PATH[,venv:PATH][,label:NAME]`); its instance runs the
checkout's own registry (`benchmarks/registry.py`) in its directory under its venv. The first target is the run's
SUBJECT, each other a REFERENCE; rola-bench's own registry (the libraries rola is compared against) is one more instance,
labelled `bench`, in the subject's venv, with the role `library`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH = "bench"


@dataclass(frozen=True)
class Target:
    worktree: Path
    venv: Path
    label: str

    @property
    def python(self) -> str:
        return str(self.venv / "bin" / "python")


def parse(spec: str) -> Target:
    fields = dict(part.split(":", 1) for part in spec.split(","))
    if "worktree" not in fields:
        raise SystemExit(f"a target needs worktree:PATH -- got {spec!r}")
    worktree = Path(fields["worktree"]).expanduser().resolve()
    venv = Path(fields.get("venv") or worktree.parent / f"venv-{worktree.name}").expanduser()
    if not (venv / "bin" / "python").exists():
        raise SystemExit(f"{worktree.name}: no venv at {venv} (name one with venv:PATH)")
    label = fields.get("label", worktree.name)
    if label == BENCH:
        raise SystemExit(f"the label {BENCH!r} is rola-bench's own instance")
    return Target(worktree, venv, label)


def instances(targets: list[Target], *, attention: bool) -> list:
    from rola_devtools.measure.service import Instance

    labels = [t.label for t in targets]
    if len(set(labels)) != len(labels):
        raise SystemExit(f"target and reference labels must differ, got {labels}")
    out = [Instance(t.label, t.python, str(t.worktree), "benchmarks.registry:registry",
                    {"PYTHONPATH": f"{t.worktree}:{t.worktree}/benchmarks:{t.worktree}/tools"},
                    "subject" if i == 0 else "reference") for i, t in enumerate(targets)]
    if attention:
        out.append(Instance(BENCH, targets[0].python, str(ROOT), "rola_bench.measure.registry:registry",
                            {"PYTHONPATH": str(ROOT)}, "library"))
    return out
