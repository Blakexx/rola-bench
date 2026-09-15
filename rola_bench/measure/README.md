# `rola_bench.measure` — rola checkouts and the libraries they are compared against, measured as one declared build

The local measurement tier of rola-bench. The build is rola-devtools' declared build system (`rola_devtools.build`),
its sessions are its timing system's (`rola_devtools.timing`), and every input is a central cell
(`rola_devtools.cells`). The root is `declare.py` at the repository's root: it loads each rola checkout's own
`declare.py` by path, composes them with this package's attention reference and groups, and declares the sessions, the
memory pass and the stores. This package defines no measurement of rola's.

    python -m rola_devtools.build plan declare.py:suite --arg target=worktree:PATH --arg groups=gate
    python -m rola_devtools.build run  declare.py:suite --arg target=worktree:PATH[,venv:PATH][,label:NAME] \
        [--arg references='worktree:PATH,label:master;worktree:PATH'] [--arg groups=all|gate|a,b] \
        [--arg cells=all|a,b] [--arg skip_cells=a,b] [--arg parts=instruments,memory,null,sessions] \
        [--arg instruments=all|sass,phases,...] [--arg attention=yes|no] [--arg rounds=8] [--arg reps=11] \
        [--arg warmup=10] [--arg store_root=DIR] [--force]

Run it from this repository with a python that has rola-devtools and rola-results (a rola checkout's venv does).

## Checkouts

A checkout is a rola worktree and the venv that runs it (`venv-<name>` beside it unless `venv:` names one); its label
scopes its targets (`tip/binary`) and defaults to its directory's name. The `target` is the first checkout, the
`references` the others. Each checkout's `declare.py` declares, in that checkout's venv and directory: its build
(cached while its binary stands), its machine facts, its timing registrations on the selected cells (`carry_forward` and
`prefill_op` on carry cells, `entmax_solve@layer=C` and `decode_step@layer=C` on layer cells under each construction)
and its clock reader; the target alone also declares its instruments (`sass`, `registers`, and per carry cell `phases`,
`counters`, `census`, `timeline`, `roofline`). A worktree without a `declare.py` predates the declaration API and is not
compared.

`attention.py` is rola-bench's own entry: `flash`, causal attention through torch's forced flash backend on every QKV
cell of the selected groups, registered in the target's venv.

## Groups and sessions (`groups.py`)

A GROUP is a selection of central cells and its relation, as code: what it holds (`L1024-N65536-dv64`: 1024 tokens,
value width 64, RoLA's state N = 65536, attention one bf16 head), the parameters its cells share, and the arm sets it
times together. `check` refuses a group whose cells break its claim. `gate` selects `GATE`.

For each selected group and arm set the root declares one SESSION (`measure_timing`): every checkout's registration of
those arms, and the attention entry, on the group's cells. Every entry is built before any call (a barrier); then the
warmup, then rounds of reps, each rep every entry once in a fresh random order, under the GPU lock and the clock lock
with the target's clock reader proving the clock before and after, an untimed reset before every call. An entry that
cannot run on a cell (a binary without the arm, a setup that does not fit) is recorded as that member's failure and the
session goes on; a session that cannot run at all fails the build, and the server's stop still runs. One MEMORY pass
(`measure_memory`) takes each entry on each cell alone. One NULL GATE (`measure_null_gate`) times the target's
`carry_forward` on each group's first RoLA cell against copies of itself in second workers: where the per-rep ratios put
one outside their interquartile range, a worker's bias is found, and a ratio across checkouts' workers is read beside it.

## Records

Instruments at `rola/<instrument>`, sessions at `timing/session`, the memory pass at `timing/memory`, the null gate at
`timing/null`, through `rola_results`. A store target appends a run-stamped sample to the record its source's semantics
key -- the executor, its parameters, the cells' records and the draw, and every dependency's key and output -- so a
session of the same
checkouts' binaries on the same cells adds a sample to one record, and a session run from rola's own root with the same
entries shares it. A session's samples are raw and ordered (round, rep, position); which checkout is the reference for a
ratio is chosen when the records are read, within a session.

## Tests

`python -m unittest tests.measure.test_suite` checks the groups against the central cells and the root's declarations
on fake checkouts, with no GPU; rola-devtools' `tests/test_declared_build.py` and `tests/test_timing.py` check the build
and timing systems.
