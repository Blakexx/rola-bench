# `rola_bench.measure` — rola checkouts and the libraries they are compared against, measured as one build

The local measurement tier of rola-bench, over rola-devtools' measurement service (`rola_devtools.measure`) and its
central cells (`rola_devtools.cells`). A measurement is a UNIT its owner registers: rola's in each rola checkout's
`benchmarks/registry.py`, rola-bench's (the attention reference) in `registry.py` here. This package defines no
measurement of rola's: it names the instances a run measures, the groups of cells that launch together, and which arms
are timed in one session, and the service builds, composes and runs what the store does not hold, through
`rola_results`. A node run here and the same node run from its owner's checkout
(`python -m rola_devtools.measure run benchmarks.registry:registry ...`) share one record.

    python -m rola_bench.measure plan --target worktree:PATH[,venv:PATH][,label:NAME] [--reference ...] --groups gate
    python -m rola_bench.measure run  --target ... [--reference ...] [--groups all|gate|a,b]
                                    [--only instruments,memory,sessions] [--units carry.phases,...] [--cells all|a,b]
                                    [--skip-cells a,b] [--no-attention] [--repeat] [--force] [--store-root DIR]
    python -m rola_bench.measure show rola/carry.phases
    python -m rola_bench.measure verdict [--cell C] [--subject S] [--baseline LABEL]

Run it with the target's venv python.

## Instances (`targets.py`)

A target is a rola checkout and the venv that runs it; `--reference` adds another. Each is an instance of its own
`benchmarks/registry.py`, served by one worker in its own directory under its own venv for the whole run: its build, its
clock reader, its instruments (`carry.sass`, `carry.registers`, and per carry cell its binary carries `carry.phases`,
`carry.counters`, `carry.census`, `carry.timeline`) and its bench subjects as timed arms (`carry_forward`, `prefill_op`,
`entmax_solve@layer=C`, `decode_step@layer=C`). The first target is the SUBJECT, the others REFERENCES. rola-bench's own
instance (label `bench`, in the subject's venv, role `library`) registers `flash`: causal attention through torch's
forced flash backend on every central QKV cell (`attention.py`).

The service runs every instance's build first -- a rola arm accepts a cell from its binary's own arm tables -- then
describes each unit on the groups' cells: a unit that does not accept a cell makes no node there. A node's key is its
owner's unit, parameters and identity (the binary's sha256, its code and every checkout file the code imports, the
environment), the cell's record and the digest of the central draw, and each dependency's key and output. Labels, roles
and paths are not in it.

## Groups, sessions, selection (`groups.py`, `groups.json`)

A GROUP is a selection of central cells and its relation: what it holds (`L1024-N65536-dv64`: 1024 tokens, value width
64, RoLA's state N = 65536), the parameters its cells share (checked when the file loads), and `together`, the arm sets
each timed in one session. `--groups gate` is `GATE_GROUPS`. A run composes, for the selected groups:

- one SESSION per group and arm set: every instance's arms of the set on the group's cells each accepts -- rola's
  `carry_forward` on the group's carry cells in every checkout beside `flash` on its QKV cell. Every member sets up
  before any call (a barrier), then warmup, then interleaved rounds in a fresh random order each rep under the GPU lock
  with the host's clock proven, and the session's record keeps every member's samples, its post and its ratios to the
  subject's arm on the same cell. Its relation -- the group, its claim, each instance's role -- is recorded, never keyed;
- the subject's INSTRUMENTS on the groups' cells (`--units` narrows them), and a MEMORY node for every instance's timed
  arm on each cell: the arm alone, torch's allocator peak plus what the arm holds outside the allocator.

`--only` keeps some of the three, `--cells` narrows the cells, `--skip-cells` leaves cells out. A refusal (a setup that
does not fit the device) is stored; a crash or a timeout fails its node or its whole session, and a node depending on a
failure is blocked.

## Records

Builds at `rola/build`, instruments at `rola/<instrument>`, memory at `rola/memory` and `bench/memory`, sessions at
`bench/session`. A record holds the semantics its key hashes and every sample: its output or error, when, how long, and
its provenance (each instance's label and checkout). Ratios are stored only within a session, where they were measured;
everything across sessions is a reading of the records.

## Tests

`python -m unittest tests.measure.test_groups` checks the groups against the central cells and the targets, with no GPU;
rola-devtools' `tests/test_measure.py` checks the service on a fake registry.
