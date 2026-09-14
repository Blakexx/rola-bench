"""Global config-driven job factory — one builder for every benchmark.

A fleet `Job` is almost entirely boilerplate; the only real per-experiment knobs are the work-list
(run_ids), the box-side run command, and a few fleet settings. This assembles the Job from those,
so benchmarks don't each hand-roll a `*_job.py`.

Results are stored through `rola_results` at `<bench>/<config>`: each row the dispatcher pulls is a sample of the
record keyed by (bench, config, cell, image), stored once however often it is re-pulled, and a cell is done when the
CURRENT image's record has an ok sample -- an image bump runs the grid again (`fleet_store`).

`fleet` (a dict, usually a spec's `fleet:` block) supplies simple knobs: image / offer_query / batch
/ per_batch_timeout / idle_timeout / env, plus any other plain fleet.Job field (no_kill,
pull_in_progress, live_pull_every, post_timeout, ...). `**job_kwargs` is for bench callables that
can't live in a yaml (payload_for) or path overrides (ckpt_dir, state_path).

OFFER QUERIES ARE DIALECT-BOUND. `offer_query` is written in the ACTIVE provider's dialect (declared
as `offer_query_dialect` in fleet.semantics) — vast's filter language and CloudRift's key=value
catalogue selector are not interchangeable, and CloudRift's parser IGNORES unknown keys, so a vast
query silently degrades to "any 1-GPU SKU" instead of erroring. So a spec may give `offer_query` as
either a plain string (taken to be `vast_filter`, which is what every existing spec is written in)
or a `{dialect: query}` mapping; asking for a dialect the active provider does not speak is a loud
error, never a silent downgrade.
"""
import hashlib
import json
import os

from fleet import Job, provider, semantics_of
from rola_results import Store, key

RUNTIME_ROOT = os.environ.get("ROLA_BENCH_RUNTIME",
                              os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "runtime"))
DEFAULT_IMAGE = "blakeresearch/rola-bench:v40"   # keep in sync with the specs' fleet.image pins; bumps are manual
# Default offer query PER DIALECT. The vast filter caps bandwidth $/GB (we pull a big image +
# checkpoints) and screens on reliability — both are marketplace concepts with no analogue on a
# provider-owned catalogue, where you pick a SKU and the only real question is whether a node is free.
DEFAULT_OFFER = {
    "vast_filter": ("gpu_name=RTX_4090 num_gpus=1 rentable=true reliability>0.97 inet_down>500 "
                    "inet_down_cost<0.01 inet_up_cost<0.01"),
    "cloudrift_kv": "gpu_name=RTX_4090 num_gpus=1 public_ip=true",
}


def offer_query_for(spec_query, dialect):
    """Pick the offer query written in the ACTIVE provider's dialect.

    `spec_query` is None (use the default), a {dialect: query} mapping, or a plain string — the last
    being every existing spec, all of which are vast filter strings. Handing a vast filter to a
    catalogue provider is the quiet failure this guards: unknown keys are ignored, so you get *a*
    box, just not the GPU you asked for.
    """
    if isinstance(spec_query, dict):
        if dialect not in spec_query:
            raise ValueError(
                f"this spec's offer_query is declared for dialect(s) {sorted(spec_query)}, but the "
                f"active provider speaks {dialect!r}. Add a {dialect!r} entry to the spec's "
                f"fleet.offer_query mapping.")
        return spec_query[dialect]
    if spec_query is not None:
        if dialect != "vast_filter":
            raise ValueError(
                f"this spec's offer_query is a plain string, which means the vast filter dialect, "
                f"but the active provider speaks {dialect!r}. Replace it with a mapping, e.g. "
                f"offer_query: {{vast_filter: '...', {dialect}: '...'}} — a vast filter handed to a "
                f"catalogue provider is silently ignored, not rejected.")
        return spec_query
    try:
        return DEFAULT_OFFER[dialect]
    except KeyError:
        raise ValueError(f"no default offer query for dialect {dialect!r}; "
                         f"set one in rola_bench.fleet.jobs.DEFAULT_OFFER or in the spec") from None


def fleet_store(bench, config, image, job):
    """The job's result sink and completion reader over `rola_results` at `<bench>/<config>`."""
    store = Store(f"{bench}/{config}")

    def semantics(cell):
        return {"bench": bench, "config": config, "cell": cell, "image": image}

    def store_result(row):
        sem = semantics(row["run_id"])
        digest = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
        record = store.get(key(sem))
        if record and any(s.get("row_sha256") == digest for s in record["samples"]):
            return
        if row.get("ok"):
            store.put(sem, output=row, provenance={"job": job}, row_sha256=digest)
        else:
            store.put(sem, error=str(row.get("error") or "the box reported ok=false"), provenance={"job": job},
                      row=row, row_sha256=digest)

    def done_ids():
        return {r["semantics"]["cell"] for r in store.records() if r["semantics"]["image"] == image and Store.complete(r)}

    return store_result, done_ids


def checkpoints(variable="ROLA_BENCH_CKPTS"):
    """The directory this machine keeps a job's pulled checkpoints in, named by `variable` in the environment. There
    is no default: checkpoints are large and live on whatever disk the machine gives them, never in the repository."""
    root = os.environ.get(variable)
    if not root:
        raise RuntimeError(f"set {variable} to the directory this machine keeps fleet checkpoints in")
    return root


def make_job(bench, config, work_list, run_cmd, fleet=None, **job_kwargs):
    fleet = dict(fleet or {})
    dialect = semantics_of(provider()).offer_query_dialect
    runtime = os.path.join(RUNTIME_ROOT, f"{bench}-{config}")
    name = fleet.pop("name", f"{bench}-{config}")
    image = os.environ.get("ROLA_BENCH_IMAGE", fleet.pop("image", DEFAULT_IMAGE))
    store_result, done_ids = fleet_store(bench, config, image, name)
    spec = {
        "name": name,
        "image": image,
        "offer_query": os.environ.get("OFFER_QUERY") or offer_query_for(fleet.pop("offer_query", None),
                                                                     dialect),
        "store_result": store_result,
        "ckpt_dir": job_kwargs.pop("ckpt_dir", None) or checkpoints(),
        "work_list": work_list,
        "done_ids": done_ids,
        "env": {"FLEET_RUN_CMD": run_cmd, **fleet.pop("env", {})},
        "bad_hosts": os.path.join(runtime, "bad_hosts.txt"),
        "state_path": os.path.join(runtime, "hosts.json"),
        "idle_timeout": fleet.pop("idle_timeout", 1200),
        "batch": fleet.pop("batch", 1),
        "per_batch_timeout": fleet.pop("per_batch_timeout", 18000),
    }
    spec.update(fleet)         # remaining plain fleet.Job fields (no_kill, pull_in_progress, ...)
    spec.update(job_kwargs)    # bench callables / path overrides (payload_for, ckpt_dir already popped)
    return Job(**spec)
