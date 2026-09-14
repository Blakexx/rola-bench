#!/usr/bin/env python3
"""Launch a RoLA benchmark on a fleet of rented GPUs.

    python -m rola_bench.fleet mqar canonical --boxes 6     # the canonical MQAR grid
    python -m rola_bench.fleet mqar law_grid                # any MQAR spec, by name
    python -m rola_bench.fleet mqar seed_confirm            # seed-confirmation
    python -m rola_bench.fleet lm default                   # the other benchmarks: lm, perf, similarity

This is the suite's orchestration entrypoint. Each benchmark exposes a job BUILDER `f(config)->Job`
(rola_bench.<bench>.job); we hand the built Job to the generic `fleet` Dispatcher, which owns
provisioning, HTTP control, result/checkpoint sync, and idle-free teardown. Per-experiment `*_job.py`
files are gone — an experiment is `<bench> <config>`, with launch knobs in the spec's `fleet:` block.
"""
import argparse
import importlib
import os

# The dispatcher is a pure orchestrator — it builds job configs and HTTP-coordinates remote boxes;
# ALL training/benchmarking runs on the cloud GPUs, never locally. But importing the job configs
# pulls in torch/rola, which would grab a ~1GB local CUDA context for nothing. Hide the GPU so the
# dispatcher stays CPU-only (verified: work_list still builds). Overridable if ever needed.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
from fleet import Dispatcher, provider, semantics_of

# benchmark -> (module, builder) where builder is f(config) -> fleet.Job
JOBS = {
    "mqar": ("rola_bench.mqar.job", "mqar_job"),
    "lm":   ("rola_bench.lm.job", "lm_job"),
    "perf": ("rola_bench.perf.job", "perf_job"),     # kernel-efficiency (whole-bench cells, no ckpt)
    "similarity": ("rola_bench.similarity.job", "sim_job"),  # post-hoc similarity-matrix eval, best-per-cell ckpts
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("benchmark", choices=sorted(JOBS))
    ap.add_argument("config", help="experiment config name (e.g. canonical, law_grid, seed_confirm)")
    ap.add_argument("--boxes", type=int, default=6)
    ap.add_argument("--batch", type=int, default=None, help="override run_ids per box batch")
    # Which GPU provider to rent from. The provider's DECLARED semantics (fleet.semantics) gate the
    # orchestration — e.g. a spec with no_kill: true is refused by name on a provider whose stop()
    # keeps billing — and also select which offer-query dialect the spec must be written in.
    ap.add_argument("--provider", default=os.environ.get("FLEET_PROVIDER", "vast"),
                    help="fleet provider seam (vast, cloudrift)")
    a = ap.parse_args()
    os.environ["FLEET_PROVIDER"] = a.provider          # job builders read the same choice
    prov = provider(a.provider)
    mod, attr = JOBS[a.benchmark]
    builder = getattr(importlib.import_module(mod), attr)
    job = builder(a.config) if callable(builder) else builder   # callable => config-driven builder
    if a.batch:
        job.batch = a.batch
    print(f"provider: {semantics_of(prov).name}")
    Dispatcher(job, boxes=a.boxes, provider=prov).run()


if __name__ == "__main__":
    main()
