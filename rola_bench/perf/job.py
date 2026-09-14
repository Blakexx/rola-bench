"""The perf job builder: each bench of an experiment spec is one scaling run_id (`perf-scaling-{train,prefill,decode}`),
its arms interleaved inside the box-side runner (`rola_bench.perf.run`). Results: `rola_results` at perf/<config>.

    python -m rola_bench.fleet perf paper_v1
"""
from rola_bench.fleet.jobs import make_job
from rola_bench.fleet.spec import env_from, load

BENCHES = ("scaling-train", "scaling-prefill", "scaling-decode")


def perf_job(config="paper_v1"):
    spec = load("perf", config)
    unknown = sorted(set(spec["benches"]) - set(BENCHES))
    if unknown:
        raise ValueError(f"perf spec {config!r} names bench(es) {unknown}; expected a subset of {BENCHES}")
    fleet = dict(spec.get("fleet", {}))
    fleet["env"] = {"PERF_CONFIG": config, **env_from("PERF", {k: spec[k] for k in ("reps",) if k in spec}),
                    **fleet.get("env", {})}
    return make_job("perf", config, lambda: [f"perf-{bench}" for bench in spec["benches"]],
                    "python -u -m rola_bench.perf.run", fleet)
