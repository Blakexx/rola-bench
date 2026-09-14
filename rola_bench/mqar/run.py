"""The MQAR runner: one zoology training subprocess per cell of an experiment spec, one result row per cell.

Usage:
  python run.py --config canonical               # every cell of experiments/canonical.yaml
  python run.py --config canonical --shard 0/4   # one shard

Per-run JSONL fields:
  run_id, idx, ok, returncode, elapsed,
  max_acc, grok_ep, epochs_run,
  state_floats, state_content_floats, mixer (the cell's recurrent state read off its mixer built on CPU, and
  that mixer's config: never a formula),
  n_params (total trainable, parsed from Zoology stdout if present),
  slice_accs (per-difficulty kv accuracy from slice_keys=["num_kv_pairs"]),
  valid_acc_curve, env, stderr_tail.
"""
import argparse
import importlib
import json
import os
import pickle
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

ZOO = Path(__file__).resolve().parent
PYTHON = sys.executable


_EXP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments")


def load_configs(name):
    """Load (configs, configs_envs) for an experiment. `name` is a declarative spec — a basename
    resolving to experiments/<name>.yaml (or a path), expanded by the generic generator
    (rola_bench.mqar.build_configs). A bare name with no matching .yaml falls back to importing a
    legacy experiment module (back-compat)."""
    spec = name if name.endswith((".yaml", ".yml")) else os.path.join(_EXP_DIR, f"{name}.yaml")
    if os.path.exists(spec):
        from rola_bench.mqar.build_configs import build_configs, load_spec
        return build_configs(load_spec(spec))
    mod = importlib.import_module(f"rola_bench.mqar.experiments.{name}")
    return mod.configs, mod.configs_envs


def load_configs_and_envs():
    """Shard-runner entry point. Uses RLA_CONFIG env var (default: the canonical spec)."""
    return load_configs(os.environ.get('RLA_CONFIG', 'canonical'))


_JSON_DECODER = json.JSONDecoder()


def _decode_json_at(text, start):
    r"""Decode ONE JSON object beginning at `text[start]`, however deeply it nests.

    Replaces `r'TAG (\{.*?\})'`. The non-greedy regex stopped at the FIRST `}`, so it silently
    truncated any record containing a nested object — which is every RoLA record since the mixer's
    provenance grew a `config` block with a `level_routing_configs` list of dicts. The symptom was a
    parse warning and `state_floats = None` on every routed row, i.e. the capacity coordinate
    missing from exactly the arms the grid is about."""
    return _JSON_DECODER.raw_decode(text, start)[0]


def cell_state(cfg) -> dict:
    """The cell's realized recurrent state per layer and its mixer's config: the mixer the grid built (the hybrid's
    second module, `common.wrap_hybrid`) constructed on CPU and read back (`realized_state`)."""
    from rola_bench.mqar import realized_state

    kernel = cfg.model.sequence_mixer.kwargs["configs"][1]
    content, overhead = realized_state.state_floats(kernel, cfg.model.d_model)
    total = None if content is None else content + overhead
    return {"state_floats": total, "state_content_floats": content, "mixer": kernel}


def parse_stdout(stdout):
    # Full per-epoch curve. Each COMPLETED validation epoch prints one line
    # ("Valid Epoch E/T: 100%|...") that contains the overall accuracy AND every
    # per-kv-slice accuracy together — i.e. all from the SAME checkpoint. We
    # capture the whole [epoch x slice] matrix so any aggregation (final-epoch,
    # best-checkpoint-by-overall, max-over-epochs) is derivable downstream and
    # per-slice numbers stay coherent (same checkpoint), rather than taking an
    # independent per-slice max from possibly-different epochs.
    epoch_curve = []  # list of {epoch, overall, slices:{kv:acc}}, one per completed epoch
    for m in re.finditer(r'Valid Epoch (\d+)/\d+: 100%\|([^\n\r]*)', stdout):
        epoch = int(m.group(1))
        blob = m.group(2)
        mo = re.search(r'valid/accuracy=([\d.]+)', blob)
        if not mo:
            continue
        overall = float(mo.group(1))
        if not (0 <= overall <= 1):
            continue
        slices = {}
        for ms in re.finditer(r'valid/num_kv_pairs/accuracy-(\d+)[^\d.]+([\d.]+)', blob):
            kv, acc = int(ms.group(1)), float(ms.group(2))
            if 0 <= acc <= 1:
                slices[kv] = acc
        # Keep the last reading for each epoch index (Lightning reprints the
        # final 100% line; dedupe by overwriting).
        epoch_curve.append({"epoch": epoch, "overall": overall, "slices": slices})
    # Dedupe by epoch (keep last occurrence — the completed line).
    _by_ep = {}
    for e in epoch_curve:
        _by_ep[e["epoch"]] = e
    epoch_curve = [_by_ep[k] for k in sorted(_by_ep)]

    pairs = [(e["epoch"], e["overall"]) for e in epoch_curve]
    max_acc = max((a for _, a in pairs), default=0.0)
    grok_ep = next((e for e, a in pairs if a >= 0.99), None)
    epochs_run = max((e for e, _ in pairs), default=0)
    valid_acc_curve = pairs

    # Back-compat: independent per-slice max over epochs (the old field).
    slice_accs = {}
    for e in epoch_curve:
        for kv, acc in e["slices"].items():
            slice_accs[kv] = max(slice_accs.get(kv, 0), acc)
    # Coherent single-checkpoint slice breakdowns (the rigorous views):
    #   final_epoch      : last completed epoch's slices
    #   best_checkpoint  : slices from the epoch with the highest overall acc
    slice_accs_final = epoch_curve[-1]["slices"] if epoch_curve else {}
    best_ep = max(epoch_curve, key=lambda e: e["overall"], default=None) if epoch_curve else None
    slice_accs_best = best_ep["slices"] if best_ep else {}
    best_overall = best_ep["overall"] if best_ep else 0.0
    final_overall = epoch_curve[-1]["overall"] if epoch_curve else 0.0
    best_ep_idx = best_ep["epoch"] if best_ep else None

    # MODEL_STATS_JSON line emitted by zoology/logger.py.
    n_params = None
    model_state_total = None
    m = re.search(r'MODEL_STATS_JSON ', stdout)
    if m:
        try:
            ms = _decode_json_at(stdout, m.end())
            n_params = ms.get('num_parameters')
            model_state_total = ms.get('state_size')
        except Exception as e:
            print(f"WARN: MODEL_STATS_JSON parse failed: {e}", flush=True)

    # FLOPS_JSON line (also from logger.py) — forward FLOPs at training batch
    # × seq_len, with disk-backed caching per unique cell config.
    # The flops_by_op dict is a nested dict so the older non-greedy regex won't
    # match cleanly; we use a more permissive single-line greedy match.
    fwd_flops = None
    flops_batch = None
    flops_seq = None
    flops_cache_hit = None
    flops_by_op = None
    flops_error = None
    m = re.search(r'^FLOPS_JSON (\{.*\})\s*$', stdout, re.MULTILINE)
    if m:
        try:
            fs = json.loads(m.group(1))
            fwd_flops = fs.get('forward_flops')
            flops_batch = fs.get('batch')
            flops_seq = fs.get('seq_len')
            flops_cache_hit = fs.get('cache_hit')
            flops_by_op = fs.get('flops_by_op')
            flops_error = fs.get('error')
        except Exception as e:
            flops_error = f"runner json parse failed: {e}; raw: {m.group(1)[:300]}"
    else:
        flops_error = "FLOPS_JSON not found in stdout"

    # PEAKINESS_JSON: end-of-training router-weight stds + softmax entropy on
    # a real test batch. Captures whether RoLA chunks are getting discriminative
    # routing or collapsing to uniform/single-chunk.
    peakiness = None
    m = re.search(r'PEAKINESS_JSON (\{.*?\})\s*$', stdout, re.MULTILINE)
    if m:
        try:
            peakiness = json.loads(m.group(1))
        except Exception as e:
            print(f"WARN: PEAKINESS_JSON parse failed: {e}", flush=True)

    # RANK_JSON: realized effective-attention rank (num_rank, eff_rank) emitted per
    # eval forward per kernel layer (ROLA_MEASURE_RANK=1). Group by epoch, mean over
    # eval batches + layers → rank_curve; rank_final = the last (best-trained) epoch.
    rank_recs = []
    for mm in re.finditer(r'RANK_JSON (\{.*?\})\s*$', stdout, re.MULTILINE):
        try:
            rank_recs.append(json.loads(mm.group(1)))
        except Exception as e:
            print(f"WARN: RANK_JSON parse failed: {e}", flush=True)
    rank_curve, rank_final = [], None
    if rank_recs:
        # The diagnostic now emits multi-tolerance numerical rank (rank_1e-01..rank_1e-04)
        # + eff_rank + sv_ratio_128/256, measured PER (epoch, seq-len) — each MQAR slice has
        # its own L (kv=1024 → longest). Group by (epoch, seqlen) so we don't average across
        # slices; carry every rank_* / sv_ratio_* / eff_rank key present (tolerant of missing).
        ALL_KEYS = {k for r in rank_recs for k in r}
        DIST_KEYS = sorted(k for k in ALL_KEYS if k.endswith('_dist'))
        SPEC_KEYS = [k for k in ('spec_idx', 'spec_p10', 'spec_p50', 'spec_p90') if k in ALL_KEYS]
        RANK_KEYS = sorted(k for k in ALL_KEYS
                           if (k.startswith('rank_') or k.startswith('mrank_') or k.startswith('sv_ratio_')
                               or k in ('eff_rank', 'meff_rank', 'pr_rank', 'mpr_rank',
                                        'stable_rank', 'mstable_rank', 'n_slices'))
                           and not k.endswith('_dist'))
        by = {}
        for r in rank_recs:
            by.setdefault((r.get('epoch', -1), r.get('seqlen', r.get('seq_len', -1))), []).append(r)

        def mean(recs, key):
            vals = [x[key] for x in recs if key in x]
            return round(sum(vals) / len(vals), 3) if vals else None
        for (ep, sl) in sorted(by):
            recs = by[(ep, sl)]
            row = {'epoch': ep, 'seqlen': sl, 'n': len(recs)}
            row.update({k: mean(recs, k) for k in RANK_KEYS})
            # the paper's rank argument is distributional: per-slice (sequence × head)
            # values concatenated across the recs (= kernel layers) at this (epoch, L),
            # sorted — NOT averaged, a mean can report a rank no slice realizes.
            for k in DIST_KEYS:
                vals = [v for x in recs for v in x.get(k, [])]
                if vals:
                    row[k] = sorted(vals)
            # spectrum quantiles kept per layer (quantiles don't aggregate by mean)
            specs = [{sk: x[sk] for sk in SPEC_KEYS} for x in recs if 'spec_idx' in x]
            if specs:
                row['specs'] = specs
            rank_curve.append(row)
        # rank_final: the hardest slice (longest seq-len) at the last epoch — that's where
        # the task demands the most rank and where rank should exceed d_model under routing.
        last_ep = max(r['epoch'] for r in rank_curve)
        cand = [r for r in rank_curve if r['epoch'] == last_ep]
        rank_final = max(cand, key=lambda r: r.get('seqlen') or -1)
        rank_final = {**rank_final, 'nc_dqk': rank_recs[0].get('nc_dqk'),
                      'd_model': rank_recs[0].get('d_model'), 'nc': rank_recs[0].get('nc')}

    return {
        "max_acc": max_acc, "grok_ep": grok_ep, "epochs_run": epochs_run,
        "valid_acc_curve": valid_acc_curve, "slice_accs": slice_accs,
        # Rigorous + flexible reporting: full per-epoch x per-slice matrix plus
        # the coherent single-checkpoint views derived from it.
        "epoch_curve": epoch_curve,
        "slice_accs_final": slice_accs_final,
        "slice_accs_best": slice_accs_best,
        "final_overall": final_overall,
        "best_overall": best_overall,
        "best_ep_idx": best_ep_idx,
        "model_state_total": model_state_total,
        "n_params": n_params,
        "fwd_flops": fwd_flops,
        "flops_by_op": flops_by_op,
        "flops_batch": flops_batch,
        "flops_seq": flops_seq,
        "flops_cache_hit": flops_cache_hit,
        "flops_error": flops_error,
        "peakiness": peakiness,
        "rank_curve": rank_curve,
        "rank_final": rank_final,
    }


def run_one(idx, run_id, cfg, env_overrides, results_file=None):
    tmp_path = Path(f"/tmp/_rla_single_{idx}.py")
    # ROLA_GPU_MEM_FRACTION caps the TRAINING process's share of the card. It is applied in the
    # CHILD because that is where the CUDA allocator lives — the parent here is a pure orchestrator
    # and (on the local box) deliberately holds no CUDA context. `zoology.launch` imports this file
    # to read `configs`, so the cap is in force before the first allocation. Set by the local grid
    # dispatcher (`rola_bench.mqar.local_grid`), which shares the card with an interactive session; unset on the
    # fleet, where a box owns its GPU outright.
    #
    # ROLA_PEAK_MEM_PATH, on the same seam, has the child write its own
    # `torch.cuda.max_memory_allocated()` to a file at exit. Measuring the peak in the process that
    # allocated it is the only way to get a number that can falsify a footprint ESTIMATE; sampling
    # `nvidia-smi` from the parent measures the allocator's reservation, plus whatever else is on the
    # card. Absent env var => not written, and no caller may assume the file exists.
    frac = os.environ.get("ROLA_GPU_MEM_FRACTION")
    peak_path = os.environ.get("ROLA_PEAK_MEM_PATH")
    cap = ("import torch\n"
           + (f"torch.cuda.set_per_process_memory_fraction({float(frac)})\n" if frac else "")
           + (f"import atexit, json as _j\n"
              f"atexit.register(lambda: open({peak_path!r}, 'w').write(_j.dumps("
              f"{{'peak_bytes': torch.cuda.max_memory_allocated()}})))\n" if peak_path else "")
           ) if (frac or peak_path) else ""
    tmp_path.write_text(
        cap +
        "import pickle\n"
        f"with open('/tmp/_rla_cfg_{idx}.pkl', 'rb') as f:\n"
        "    configs = [pickle.load(f)]\n"
    )
    with open(f"/tmp/_rla_cfg_{idx}.pkl", "wb") as f:
        pickle.dump(cfg, f)

    env = os.environ.copy()
    # Best-checkpoint save: if a dir is set, tell the cell where to write its best (by valid
    # acc) model state_dict, named by run_id. Enables post-hoc rank eval / re-eval without
    # re-training. Pulled off the box alongside results.
    ckpt_dir = os.environ.get("SAVE_BEST_CKPT_DIR")
    if ckpt_dir:
        env["BEST_CKPT_PATH"] = f"{ckpt_dir}/{run_id}.pt"
    env["WANDB_MODE"] = "offline"
    # Disable wandb's stdout interception so our FLOPS_JSON / MODEL_STATS_JSON
    # lines reliably reach the subprocess's captured stdout.
    # Without this, wandb's console-capture eats some of our prints (notably
    # FLOPS_JSON), leaving the runner unable to parse FLOPs.
    env["WANDB_CONSOLE"] = "off"
    # Clear any stale MQAR_* env from previous runs (recipe/router knobs).
    for k in list(env):
        if k.startswith("MQAR_ROUTER_STD") or k.startswith("MQAR_CURR"):
            del env[k]
    env.update(env_overrides)

    t0 = time.time()
    proc = None
    # Forward SIGTERM (Vertex spot preemption hits PID 1 = shard_runner, which
    # imports + calls this synchronously, so the handler installed here runs
    # in that same process). Without forwarding, the training subprocess never
    # receives SIGTERM and its checkpoint handler never fires.

    def _forward_sigterm(signum, frame):
        if proc is not None and proc.poll() is None:
            print(f"\n[run_one] forwarding SIGTERM → subprocess pid={proc.pid}", flush=True)
            try:
                proc.send_signal(signal.SIGTERM)
            except Exception as e:
                print(f"[run_one] SIGTERM forward failed: {e}", flush=True)
    old_handler = signal.signal(signal.SIGTERM, _forward_sigterm)
    try:
        # Popen + tee: stream subprocess stdout to parent's stdout in real time
        # (so Cloud Logging / local terminal see Lightning progress, epoch
        # metrics, etc. as they happen) AND buffer for post-hoc parsing of
        # FLOPS_JSON, MODEL_STATS_JSON, valid/accuracy lines.
        proc = subprocess.Popen(
            [PYTHON, "-m", "zoology.launch", str(tmp_path)],
            cwd=str(ZOO), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,  # line-buffered
        )
        buffer = []
        try:
            for line in proc.stdout:
                print(line, end="", flush=True)  # tee → parent stdout
                buffer.append(line)
                # Cheap safety net for the 12000s timeout: check elapsed every line
                if time.time() - t0 > 12000:
                    proc.kill()
                    raise subprocess.TimeoutExpired(proc.args, 12000)
        finally:
            proc.wait()
        # NOTE: on TimeoutExpired the buffered stdout still holds every COMPLETED eval line
        # (the 2026-06-10 SSE run lost a finished-to-epoch-30 result this way) — callers must
        # parse `buffer` even when the run is killed; see the except handler below.
        full_stdout = "".join(buffer)
        elapsed = time.time() - t0
        ok = proc.returncode == 0
        parsed = parse_stdout(full_stdout)
        return {
            "run_id": run_id, "idx": idx, "ok": ok, "elapsed": elapsed,
            "returncode": proc.returncode, "env": env_overrides,
            **cell_state(cfg), **parsed,
            "stderr_tail": full_stdout[-2000:] if not ok else "",
        }
    except subprocess.TimeoutExpired:
        # Parse what completed before the kill — the buffer holds every finished eval line
        # (a timed-out run that reached epoch 30 still has citable best-checkpoint metrics).
        parsed = parse_stdout("".join(buffer)) if buffer else {}
        return {"run_id": run_id, "idx": idx, "ok": False, "elapsed": 12000, "error": "timeout",
                "env": env_overrides, **cell_state(cfg), **parsed}
    finally:
        signal.signal(signal.SIGTERM, old_handler)
        try:
            Path(f"/tmp/_rla_cfg_{idx}.pkl").unlink(missing_ok=True)
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="canonical",
                    help="experiment spec basename (experiments/<name>.yaml) or legacy module name")
    ap.add_argument("--results", default=None,
                    help="output jsonl path (default: <config>_results.jsonl in cwd)")
    ap.add_argument("--shard", default=None,
                    help="N/M to process only idx where idx %% M == N")
    args = ap.parse_args()

    configs, envs = load_configs(args.config)
    results_file = Path(args.results) if args.results else ZOO / f"{args.config}_results.jsonl"
    print(f"Loaded {len(configs)} {args.config} configs", flush=True)
    print(f"Results → {results_file}", flush=True)

    # Explicit work-list: RUN_IDS env (comma-separated run_ids) is the AUTHORITATIVE list of what
    # this box runs — the dispatcher owns the queue and pushes exactly these. NO silent fallback:
    # if RUN_IDS is set but empty/garbage we abort rather than quietly training the whole grid.
    raw_ids = os.environ.get("RUN_IDS")
    if raw_ids is not None:
        only_ids = {x for x in raw_ids.split(",") if x}
        if not only_ids:
            print("FATAL: RUN_IDS is set but empty — refusing to run (no fallback).", flush=True)
            sys.exit(2)
        print(f"RUN_IDS work-list: {len(only_ids)} run_ids", flush=True)
    else:
        only_ids = None  # local/legacy: --shard or run-all

    shard_n, shard_m = None, None
    if args.shard and only_ids is None:
        n, m = args.shard.split("/")
        shard_n, shard_m = int(n), int(m)
        print(f"Shard {shard_n}/{shard_m} — processing idx where idx %% {shard_m} == {shard_n}", flush=True)

    # Resume by RUN_ID (not idx): stable across config changes (e.g. adding LRs shifts indices,
    # so idx-resume would skip the wrong cells). A cell is done iff its run_id has an ok row.
    completed = set()
    if results_file.exists():
        with open(results_file) as fh:
            lines = fh.readlines()
        for line in lines:
            try:
                r = json.loads(line)
                if r.get("ok"):
                    completed.add(r.get("run_id"))
            except Exception:
                pass
    if completed:
        print(f"Resuming: skipping {len(completed)} done (by run_id)", flush=True)

    with open(results_file, "a") as fout:
        for i, (cfg, env_o) in enumerate(zip(configs, envs, strict=True)):
            run_id = cfg.run_id
            if only_ids is not None:
                if run_id not in only_ids:
                    continue
            elif shard_m is not None and i % shard_m != shard_n:
                continue
            if run_id in completed:
                continue
            print(f"\n[{i+1}/{len(configs)}] {run_id} | env: {env_o}", flush=True)
            result = run_one(i, run_id, cfg, env_o, results_file)
            fout.write(json.dumps(result) + "\n")
            fout.flush()
            status = "OK" if result.get("ok") else f"FAIL ({result.get('error', result.get('returncode'))})"
            elapsed = result.get('elapsed', 0)
            epochs_run = result.get('epochs_run', 0) or 0
            n_ep = epochs_run + 1
            s_per_ep = elapsed / max(n_ep, 1)
            fwd = result.get('fwd_flops')
            fwd_str = (f"{fwd/1e9:.1f}G" if isinstance(fwd, (int, float)) and fwd else "n/a")
            cache_str = "cached" if result.get('flops_cache_hit') else "fresh"
            print(f"  → {status}, max_acc={result.get('max_acc', 0):.3f}, "
                  f"grok_ep={result.get('grok_ep')}, "
                  f"state={result.get('state_floats')}, "
                  f"params={result.get('n_params')}, "
                  f"fwd_flops={fwd_str} ({cache_str}), "
                  f"t={elapsed:.0f}s ({s_per_ep:.0f}s/ep × {n_ep}ep)", flush=True)


if __name__ == "__main__":
    main()
