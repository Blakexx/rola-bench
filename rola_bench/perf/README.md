# The perf benchmark

`scaling.py` times RoLA's layers against attention and the FLA baselines, layer against layer at d_model 512:

- **arms** (`SCALE_ARMS`): `attn` (causal SDPA, a growing KV cache), `gla`, `gdn`, `deltanet`, `gsa` (fla's
  layers at their canonical geometry), and the named RoLA cells of `rola_bench.models.rola` built through fla;
- **axes**: sequence length `L` (`SCALE_LENS[_TRAIN|_PREFILL|_DECODE]`), RoLA's states per head `N` (`SCALE_NS`) and
  value width `d_v` (`SCALE_DVS`); a cell is kept while RoLA's state, read off the built layer, is under attention's
  KV cache;
- **benches**: `train` (forward and backward), `prefill` (forward), `decode` (one token into a state seeded by a
  256-token prefill, through FLA's cache contract; `SCALE_DECODE_GRAPHED=1` adds a CUDA-graph replay of each
  recurrent arm), each its own fleet run_id `perf-scaling-{train,prefill,decode}`.

```bash
python -m rola_bench.fleet perf paper_v1          # the fleet run (experiments/paper_v1.yaml)
python -m rola_bench.perf.scaling --benches prefill --lens 4096 --ns 256   # one local sweep
python -m rola_bench.perf.smoke                   # every arm built, one step each
```

## Protocol

- Within a cell the arms run round-robin, their order rotating each round, so drift charges each alike.
- Timing is CUDA events with warmup excluded; rows report the median, the IQR and the raw samples.
- Latency comes from the interleaved pass; peak and retained-activation memory from a separate pass with the arm alone.
- A cell an arm cannot run is a row, not a stop: `UNBUILT` (the library refuses it: rola outside its built envelope or
  before a pass it has built), `OOM`, or `ERR`. `SCALE_PREFLIGHT_ARMS=1` runs each arm once in a child process first,
  so a crash is the arm's row instead of the sweep's end.
- Results: the fleet row per run_id, stored through `rola_results` at `perf/<config>`; the CSVs a run writes are
  scratch (`RESULTS_DIR`).
