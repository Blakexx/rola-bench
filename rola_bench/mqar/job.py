"""MQAR jobs, config-driven. `mqar_job(<config>)` builds a fleet.Job for ANY MQAR spec
(canonical / law_grid / dmodel64 / router_bias / seed_confirm / ...) — no per-experiment *_job.py.

Launch knobs live WITH the experiment: a `fleet:` block in experiments/<config>.yaml, or a `FLEET`
dict in a module-config (seed_confirm). Everything else (name, results dir, done_ids, run command)
is derived from the config name by rola_bench.fleet.jobs.make_job.

  python -m rola_bench.fleet mqar canonical        # tiers via fleet.env.GRID_TIERS
  python -m rola_bench.fleet mqar law_grid
  python -m rola_bench.fleet mqar seed_confirm
"""
import importlib
import os

from rola_bench.fleet.jobs import make_job
from rola_bench.mqar.build_configs import load_spec
from rola_bench.mqar.run import _EXP_DIR, load_configs


def _fleet_knobs(config):
    spec_path = os.path.join(_EXP_DIR, f"{config}.yaml")
    if os.path.exists(spec_path):
        return dict(load_spec(spec_path).get("fleet", {}))
    try:                                                  # module-config (e.g. seed_confirm)
        mod = importlib.import_module(f"rola_bench.mqar.experiments.{config}")
        return dict(getattr(mod, "FLEET", {}))
    except Exception:
        return {}


def _filtered_ids(config, tiers):
    if tiers:
        os.environ["GRID_TIERS"] = tiers                  # manager-side queue must match the box's tiers
    ids = [c.run_id for c in load_configs(config)[0]]
    # Optional manager-side run_id filters (run_id = grid-<family>-<shape>-...): FAMILIES / SHAPES /
    # EXCLUDE_FAMILIES / EXCLUDE_SHAPES env (comma lists) — e.g. a 4090-safe subset of canonical.

    def part(r, i):
        return r.split("-")[i] if len(r.split("-")) > i else None

    for env, idx, keep in (("FAMILIES", 1, True), ("SHAPES", 2, True),
                           ("EXCLUDE_FAMILIES", 1, False), ("EXCLUDE_SHAPES", 2, False)):
        v = os.environ.get(env)
        if v:
            s = set(v.split(","))
            ids = [r for r in ids if (part(r, idx) in s) == keep]
    return ids


def mqar_job(config):
    knobs = _fleet_knobs(config)
    tiers = knobs.get("env", {}).get("GRID_TIERS")
    run_cmd = f"python run.py --config {config} --results $FLEET_RESULTS"
    return make_job("mqar", config, lambda: _filtered_ids(config, tiers), run_cmd, knobs)
