# rola-bench

RoLA's benchmarks and its local measurement suite. The benchmarks (MQAR recall, language modeling, layer
performance, the similarity audit) are declarative specs that the generic [`fleet`](https://github.com/Blakexx/fleet-dev)
package runs on rented GPUs. The measurement suite measures rola checkouts with their own instruments on this machine.
Every result goes through `rola_results`.

rola-bench is a library consumer like any other. It builds RoLA only through fla (our fork of it), whose RoLA layer and HF model
are a thin wrapper over rola. It never reaches into rola's kernels:

```
rola-bench ──▶ fla           the fork's rola branch: the RoLA layer and HF model (over rola), and the baseline layers
           ──▶ zoology       MQAR's model, data and baselines; its RoLAMixer wraps fla's RoLA
           ──▶ fleet         GPU orchestration
           ──▶ rola-results  the result store
           ──▶ rola          (through fla; the measurement suite measures rola checkouts directly)
```

## Layout

```
rola_bench/
  models/rola.py        the RoLA arms: wirings, cells, named cells, built through fla; state read off the layer
  fleet/                python -m rola_bench.fleet (the launcher), jobs.py (Job factory + result sink),
                        spec.py (spec loader), verify.py (the box boot gate)
  mqar/                 job.py, run.py (box side), smoke.py, build_configs.py, local_grid.py (this machine's card),
                        realized_state.py, analysis/graph.py, experiments/*.yaml
  lm/                   job.py, run.py -> train_lm.py, eval_lm.py, arms.py, smoke.py, experiments/default.yaml
  perf/                 job.py, run.py, scaling.py, smoke.py, experiments/paper_v1.yaml (protocol: perf/README.md)
  similarity/           job.py, run.py, evaluator.py, experiments/main.yaml
  measure/              the local measurement suite (measure/README.md)
tests/<area>/           one directory per area above
docker/                 the fleet images
```

## The RoLA arms (`rola_bench/models/rola.py`)

A **wiring** names each routing level's duties, outermost first: read and write density (`dense` or `sparse`),
whether the duties share a projection (`tied`), and a sparse duty's entmax order (`alpha`). A **cell** is a wiring at a
state count N per head, with every level's width (the uniform factorization b**D = N unless widths are given), a decay
source (`None`, or `{source: constant | learned, ...}`) and the router pins. The canonical MQAR slate is
`rola-arm1-densread-sparsewrite`, `rola-arm2-union`, `rola-arm3-levelsplit` and the one-level control `rola-d1-dense`.
The LM and perf benches select the named cells (`rola-base-rla`, `rola-base-massdecay-global`, ...).

Every bench builds a cell one way: `mixer_config` for zoology, `layer` for a bare fla layer, or `config_kwargs`
for `RoLAConfig`. It reads a built cell's state back off the layer (`state_floats`), never from a formula. Whatever
rola refuses, every build refuses with rola's own error. Today that is prefill while its kernel is rebuilt and
training before its native backward, so every RoLA arm of every bench fails until rola builds those passes. The
baseline arms run.

## The benchmarks

An experiment is `<bench> <config>`: `rola_bench/<bench>/experiments/<config>.yaml` is its single source of truth (the
grid, the hyperparameters, and a `fleet:` block of dispatch knobs). The bench's `job.py` expands it into cell ids and
a box-side command, and `rola_bench.fleet.jobs.make_job` assembles the `fleet.Job`.

| Bench | Measures | Specs |
|---|---|---|
| `mqar` | Recall accuracy at matched realized state: RoLA's wirings against zoology's baselines across N, L, learning rates and seeds. | `canonical`, `law_grid`, `graph_grid`, `presence_grid`, `cue_consistency`, `ood_recall`, `separation`, `dmodel64`, `router_bias`; `seed_confirm.py` (re-runs the winners at extra seeds) |
| `lm` | Language modeling: the named RoLA cells against GatedDeltaNet, GLA and attention on one backbone, matched on recurrent state (the job refuses a spec whose bounded arms differ). | `default` |
| `perf` | Layer latency and memory: train, prefill and decode against sequence length, N and d_v, arms round-robin. | `paper_v1` |
| `similarity` | The token-mixing matrix of the best checkpoint per cell: rank and sharpness on its own test slices. RoLA has no extractor until its V3 one is built; its cells are skipped. | `main` |

```bash
python -m rola_bench.fleet mqar canonical --boxes 6
python -m rola_bench.fleet lm default
python -m rola_bench.fleet perf paper_v1 --boxes 1
python -m rola_bench.fleet similarity main
```

The launcher runs CPU-only: it builds the job and coordinates the boxes over HTTP, and every run happens on the rented
GPUs. It reads this machine's environment:

| Variable | Meaning |
|---|---|
| `ROLA_BENCH_CKPTS` | where pulled MQAR checkpoints live (required by jobs that pull checkpoints; no default) |
| `ROLA_LM_CKPTS` | where LM checkpoints live, per config (required by `lm`; no default) |
| `ROLA_BENCH_RUNTIME` | fleet operational state (bad hosts, host state); default `runtime/` here, ignored by git |
| `ROLA_BENCH_IMAGE`, `OFFER_QUERY`, `FLEET_PROVIDER` | override the spec's image, offer query and provider |

## Local runs

Each bench has a smoke that builds every arm on this machine's GPU and runs one step:

```bash
python -m rola_bench.mqar.smoke
python -m rola_bench.lm.smoke
flock /tmp/rola_gpu.lock python -m rola_bench.perf.smoke
```

An MQAR-family grid small enough for this card runs one tier at a time through the fleet's own cell runner, and the
graph and presence grids are read by their analysis:

```bash
GRID_TIERS=decoupling python -m rola_bench.mqar.local_grid --config graph_grid --dry-run
python -m rola_bench.mqar.analysis.graph --config graph_grid
```

The measurement suite measures rola checkouts (SASS, registers, phases, pipe counters and timelines, interleaved timing
sessions) as a graph of nodes keyed by what each result depends on. It runs only what is not stored and keeps every
sample. It needs only the standard library and `rola_results`. See
[`rola_bench/measure/README.md`](rola_bench/measure/README.md):

```bash
python -m rola_bench.measure run --target worktree:<rola checkout> --reference worktree:<baseline>,label:master
```

Local numbers are engineering gates. Citable numbers come from rented runs.

## Results

Every result is stored through `rola_results`, in the rola-results repository: the fleet benches at `<bench>/<config>`,
the local MQAR grid beside the fleet's records at `mqar/<config>` (keyed by its checkouts instead of an image), and the
measurement suite at `suite/<module>`. A fleet job plugs the store into
fleet's `store_result` sink and `done_ids`. Each pulled row is a sample of the record keyed by (bench, config, cell,
image), stored once however often it is pulled again. A cell is done when the current image has an ok sample, so an
image bump runs the grid again. Failures are samples too.

## Tests and the gate

Tests run per area. The GPU-heavy suites are never run unscoped:

```bash
python -m pytest tests/models
python -m pytest tests/mqar
python -m unittest tests.measure.test_engine
```

The commit gate (`.githooks/pre-commit` -> `.pre-commit-config.yaml`) runs ruff with rola's rule set and its E30x spacing family
over `rola_bench` and `tests`, the measurement engine's contract tests, and the public mirror's declaration check.

## Publishing

This repository is developed in the private `rola-bench-dev` and published to the public `rola-bench`: a push to
`master` runs `.github/workflows/mirror.yml`, which publishes the declared files as one snapshot commit
([`.github/mirror/README.md`](.github/mirror/README.md)).

## Images

| Image | Dockerfile | Benchmarks |
|---|---|---|
| `blakeresearch/rola-bench` | `docker/mqar.Dockerfile` | MQAR, perf, similarity |
| `blakeresearch/rola-lm` | `docker/lm.Dockerfile` | LM |

Both build `FROM blakeresearch/fleet-base` and set `FLEET_RUN_CMD` and `FLEET_VERIFY_CMD`
(`python -m rola_bench.fleet.verify`). `docker/build.sh <mqar|lm> [--push] [--prune]` stages clean sources from the
checkouts named by `ROLA_FLA_SRC` and `ROLA_ZOO_SRC`. The images do not install rola yet, and fla does not import
without it, so no current image builds. The fleet base's torch is a CUDA 12.8 build, and rola's manifests are
ratified under CUDA 12.4.
