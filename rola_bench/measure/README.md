# `rola_bench.measure` — rola checkouts and their references, measured as one build over their owners' graphs

The local measurement tier of rola-bench. A measurement is a UNIT its owner declares in a GRAPH
(`rola_devtools.graph`): rola's are in each rola checkout's `benchmarks/graph.py`, rola-bench's (the attention reference)
in `graph.py` here. This package composes them: it loads rola's graph once per checkout it measures and its own graph
beside them, relates their nodes (which run together, in what role), and runs what the store does not hold, through
`rola_results`. It defines no measurement of its own, so a node it runs and the same node run from its owner's checkout
(`python -m rola_devtools.graph run benchmarks.graph:graph`) share one record.

    python -m rola_bench.measure plan --target worktree:PATH[,venv:PATH][,label:NAME] [--reference ...] --groups gate
    python -m rola_bench.measure run  --target ... [--reference ...] [--groups all|gate|a,b] [--nodes all|carry,memory,time]
                                    [--cells all|a,b] [--no-attention] [--repeat] [--force] [--store-root DIR]
    python -m rola_bench.measure show rola/carry.phases

Run it with the target's venv python (the hold reads the device's clock through the target's rola).

## Instances

A target is a rola checkout and the venv that runs it; `--reference` adds another. Each is an instance of its own
`benchmarks/graph.py`, described and run in its own directory under its own venv: its instruments (`carry.sass`,
`carry.registers@arm0`, and per probe cell `carry.phases`, `carry.counters`, `carry.census`, `carry.timeline`), a timed
arm per subject and cell (`time.<subject>@<cell>`) and that arm's peak memory alone (`memory.<subject>@<cell>`).
rola-bench's own instance (label `bench`, run in the target's venv) holds `time.flash@<cell>` and `memory.flash@<cell>`:
causal attention through torch's forced flash backend on this package's `registry.json` cells.

A node's key is its owner's: its unit, parameters, identity (the binary's sha256, its code and every checkout file the code
imports, the environment -- GPU, driver, assembler, torch, Nsight Compute, the locked clock) and each dependency's key.
Labels and paths are not in it.

## Groups, sessions, selection

A GROUP is a `points` entry of `registry.json`: cells by runner and what they hold equal (`L1024-N65536-dv64`: 1024
tokens, value width 64, RoLA's state N = 65536). `--groups gate` is `GATE_GROUPS`. The composer (`compose.py`) turns the
selected groups into:

- one SESSION per group and subject: the timed arms of that subject on the group's rola cells in every rola instance,
  and, for a carry subject (`carry_forward`, `prefill_op`), the attention reference on the group's attention cells. Every
  member sets up before any call (a barrier), then the members are interleaved in a fresh random order each rep under
  the target's GPU and clock hold, and the session's record keeps every member's samples, its post, and its ratios to
  the target's node of the same name. Its relation -- the group, its claim, each instance's role (`subject`,
  `reference`, `attention`) -- is recorded with it and never keyed;
- a SELECTION: the target's instruments on the groups' cells, and every instance's memory rows.

`--nodes` narrows by node prefix, `--cells` by cell. A unit that cannot run what it was given (an arm the binary lacks,
a cell the kernel does not take) refuses in setup and the refusal is stored; a crash or a timeout fails its node or its
whole session, and a node depending on a failure is blocked.

## Records

Instruments at `rola/<instrument>`, memory at `rola/memory` and `bench/memory`, sessions at `bench/session` (a lone
timed arm run from its checkout lands at its own location, `rola/time`). A record holds the semantics its key hashes and
every sample: its output or error, when, how long, and its provenance (each instance's label and checkout). Ratios are
stored only within a session, where they were measured; everything across sessions is a reading of the records.

## Tests

`python -m unittest tests.measure.test_compose` checks the composition on fake instances, with no GPU;
`rola_devtools.graph`'s tests check the engine.
