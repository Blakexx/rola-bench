# `rola_bench.measure` — rola checkouts measured by their own instruments, every result kept

The local measurement tier of rola-bench. A run takes the checkouts to measure, builds a graph of measurement nodes on
them, runs what is not already stored, and keeps every result through `rola_results` at the location `suite/<module>`
(rola's dev config names the rola-results checkout as `store.root`; `tools/dev.py init` makes it importable). Standard
library and `rola_results` only: it runs from any rola venv or the dev container without fleet, zoology or transformers. Rented, citable runs stay on the fleet tier (`python -m rola_bench.fleet`, `rola_bench/perf/README.md`);
this tier is the engineering gate.

    python -m rola_bench.measure plan --target worktree:PATH --reference worktree:PATH,label:master,schedule:box/sparse-g32 --points gate
    python -m rola_bench.measure run  --target ... [--reference ...] [--modules carry,timing] [--points all|gate|a,b]
                                    [--cells all|a,b] [--subjects all|a,b] [--repeat] [--force]
    python -m rola_bench.measure show carry.phases

## Targets and arms

A target is `worktree:PATH[,venv:PATH][,label:NAME][,schedule:POLICY]`. The venv defaults to the sibling `venv-<name>`.
The schedule is the carry family's order policy the target's binary takes: `first`, or `DENSE/SPARSE` split on the
cell's name (master: `box/sparse-g32`).

Every node runs the target checkout's OWN instrument (rola's `tools/*.py --json`) from its own directory under its own
venv. A checkout is measured by its own definitions, the way rola's probe runs each binary's own worker.

A timing run's arms are explicit. The `--target` is the `subject`, and each `--reference` is a `reference` timed in the
same session: rola's `tools/compare.py`, run from the target, which drives rola-devtools' interleaving driver. Every
rola arm is built by its own checkout's `bench.provider` under its own venv and named by its subject and dials
(`carry_forward@schedule=identity`); every arm is warmed past the driver's floor of 10 launches, then called once per
rep in a fresh random order, under the GPU lock and the clock lock. A session is one point: every rola arm runs on each
of the point's rola cells it accepts, and a carry subject's session also times the attention reference (`attention.py`:
torch's forced flash backend) on the point's attention cell. Latency is only comparable within one session, so the
session records the point with its cells and claim, every row's cell, what its runner built, raw samples in the order
taken, and the clock; the paired ratios it carries are to the target's row on the same cell, within that session.

## Points, cells and runners

What a run measures is chosen by POINT (`registry.json`, read with rola-devtools' `rola_devtools.cells`). A cell is a
data provider and its parameters: rola's cells are the target checkout's own registries (`benchmarks/cells`), and this
file adds the attention reference's (`cells.py:qkv`: tokens, value width, heads, dtype). A point groups cells by runner
-- `rola` (the checkout's `bench.provider`) and `attention` (`attention.py`) -- and states what it holds equal: every
carry point holds the tokens and the value width and names the RoLA state count (`L1024-N65536-dv64`; capacity-fair where
N = L), and every layer cell is a point of its own (`layer-<cell>`). `--points` selects points (`gate` is `GATE_POINTS`)
and `--cells` narrows their rola cells.

Before planning, the suite asks each rola checkout's runner what it makes of every selected rola cell (the driver's
`accepts`): the arms it runs there, or a refusal by name -- a carry arm its binary does not carry, an iteration build
lacking a shipped arm. Refusals are printed; the carry instruments take the target's accepted carry cells; a session
takes the point's cells every rola arm accepts, one session per carry order when an arm's schedule splits the cells.

A run with a reference ends with each timing unit's VERDICT, and `python -m rola_bench.measure verdict` prints it again
from the store without running anything: rola-results' `verdict` query (the reference arm's recent sessions of the unit
as the baseline, the target commit's sessions as its runs, the last session's paired rounds) judged by rola-devtools'
three gates. A regression needs the effect over the baseline's derived threshold, a significant paired test, and a
second run over the line, so a unit that reads slow once says `flagged_not_confirmed` until `--repeat` runs it again.
Sessions run 8 rounds by default, the floor below which the paired test cannot call anything significant.

## Keys, completion, repeats

A node is one module on one unit (a cell, or a subject at a cell). Its key is sha256 over:
- the module and unit;
- the target's BINARY: sha256 of its built extension;
- the INSTRUMENT: its file and every repository file it imports, found by walking the imports, plus the data files and
  directories the module names (the cell registry, a budget, the kernel source a compile reads);
- the ENVIRONMENT, read from the machine: GPU and driver, ptxas, torch, Nsight Compute, the SM clock target;
- the module's parameters;
- for each node it depends on, that node's key and output digest.

A change anywhere re-keys exactly the nodes that depend on it, and a revert finds the old key's result again.

- A key with a successful sample is complete and does not run.
- A failure is a sample too, kept with its error, and a key with no successful sample runs again. Nodes depending on a
  failure are blocked, not recorded.
- `--repeat` adds a sample to complete nodes of repeatable modules (every measurement, not the static SASS or register
  reads).
- `--force` runs every selected node again.

## Records

A record (`rola_results`, `records/suite/<module>/<key>.json`) holds the semantics that made the key and every sample:
its output (`<key>.<n>.out.json`, the instrument's raw JSON) or its error, when it ran, how long it took, and its
provenance (the target's label, commit and dirtiness; a timing session's every arm). Nothing relational is stored: no baseline ratio, no
delta between commits.

## Modules

| module | unit | instrument (rola) | raw output |
|---|---|---|---|
| `carry.sass` | the target | `tools/sass_gate.py` | SASS signatures of every carry kernel function, per cubin |
| `carry.registers` | arm 0 | `tools/life_ranges.py --arm 0` | peak live registers, by region |
| `carry.phases` | carry cell | `tools/phase_ledger.py` | cycles a warp a window per phase, per warp |
| `carry.counters` | carry cell | `tools/pipe_counters.py` | the profiler's pipe and resource counters, launch totals |
| `carry.census` | carry cell | `tools/stall_census.py` | stall samples by component and reason, phase and wavefront census |
| `carry.timeline` | carry cell | `tools/pipe_timeline.py` | the pipes' PM-sampled series over one launch |
| `timing.session` | subject @ point [@ calls=N] | `tools/compare.py` | every row's cell and raw samples in order, round medians, paired ratios within each cell, clock |

`--points gate` is the kernel's gate points (`GATE_POINTS`: the N = L 64K point and the flagship point). Timing covers
every subject the target's runner offers on a point's cells, at every call count (`bench.subjects.Subject.calls`: the
sequence as N carried calls). A reference must be a checkout whose cells are data providers and whose `bench.provider` is
a runner (rola from ce9eaf1 on).

## Tests

`python -m unittest rola_bench.measure.test_engine` checks the engine's contract on fake nodes, with no GPU.
