"""rola-bench's rules: rola-devtools' generic rules, each handed this workspace's environment (`bench_env`) and tracked
tree, and the runs on its two environments (`bench_python`, `research_python`)."""

load("@bazel_skylib//lib:shell.bzl", "shell")
load("@bench_env//:flags.bzl", "VENV")
load("@bench_tree//:tree.bzl", "TRACKED")
load("@rola_devtools//bazel:rules.bzl", _tree_part = "tree_part", _tree_test = "tree_test")
load("@research_env//:flags.bzl", RESEARCH_VENV = "VENV")

def tree_test(**kwargs):
    """A check over the tree's tracked files, the bench environment's tools on PATH."""
    _tree_test(venv = VENV, **kwargs)

def tree_part(parts = {}):
    """This package's part of the tree: `tracked` and `tracked_python`."""
    _tree_part(TRACKED, parts)

_PYTHON_RUN = """\
#!/bin/bash
# an environment's interpreter, in the checkout (rola-bench's python_run){rola_note}
set -euo pipefail
checkout="${{BUILD_WORKSPACE_DIRECTORY:?run through bazel run}}"
{rola}cd "$checkout"
export PYTHONPATH="$checkout{pythonpath}"
exec {venv}/bin/python {module} "$@"
"""

#: the run's first argument: the rola checkout it imports, its library built (`bazel run //:library` there)
_ROLA = """rola="${1:?the rola checkout this run imports, its library built (bazel run //:library there)}"
shift
rola="$(cd "${BUILD_WORKING_DIRECTORY:-.}" && cd "$rola" && pwd)"
"""

def _python_run_impl(ctx):
    script = ctx.actions.declare_file(ctx.label.name + ".sh")
    ctx.actions.write(script, _PYTHON_RUN.format(
        rola_note = ": its first argument the rola checkout it imports" if ctx.attr.rola else "",
        rola = _ROLA if ctx.attr.rola else "",
        pythonpath = ":$rola" if ctx.attr.rola else "",
        venv = shell.quote(ctx.attr.venv),
        module = ("-m " + shell.quote(ctx.attr.module)) if ctx.attr.module else "",
    ), is_executable = True)
    return [DefaultInfo(executable = script)]

python_run = rule(
    implementation = _python_run_impl,
    executable = True,
    attrs = {
        "venv": attr.string(mandatory = True, doc = "the environment's store venv"),
        "module": attr.string(doc = "the module it runs; none runs the interpreter on the run's arguments"),
        "rola": attr.bool(doc = "the run imports rola: its first argument is the rola checkout, on the path after this " +
                                "one (both hold a `declare.py`)"),
    },
    doc = "A run on an environment's interpreter in the checkout (`bazel run`; its arguments follow `--`), the checkout " +
          "importable, and a rola checkout after it where the run imports rola.",
)

def bench_python(**kwargs):
    """A run on the bench environment (the fork's stack)."""
    python_run(venv = VENV, **kwargs)

def research_python(**kwargs):
    """A run on the research environment (the baselines' stack)."""
    python_run(venv = RESEARCH_VENV, **kwargs)
