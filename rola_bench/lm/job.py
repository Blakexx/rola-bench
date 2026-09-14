"""LM benchmark job builder — iso-backbone RoLA-vs-attention, spec-driven on the global layer.

`lm_job(config)` reads rola_bench/lm/experiments/<config>.yaml (arms × seeds + backbone/geometry/
budget + fleet) and builds the fleet.Job via rola_bench.fleet.jobs.make_job. The spec's backbone/geometry/
budget sections are flattened to the LM_* box env the runner (run.py) + arms.py read, so the spec
file is the single source of truth. Results: `rola_results` at lm/<config> (cell = <arm>_s<seed>).
LM keeps preemption-safe checkpoint resume (payload_for + pull_in_progress + no_kill).

Launch: python -m rola_bench.fleet lm default
"""
import os

from rola_bench.fleet.jobs import checkpoints, make_job
from rola_bench.fleet.spec import env_from, load


def validate_matched_states(spec, box_env) -> None:
    """Every bounded arm of a spec carries the same recurrent content state at the spec's geometry -- that, and only
    that, makes an LM row a matched-state comparison. Read through `rola_bench.lm.arms` at the environment the boxes get
    (RoLA's from a built layer); attention has no bounded state and is skipped."""
    from rola_bench.lm import arms

    g = arms.geometry(box_env)
    states = {arm: arms.recurrent_state_floats(arm, g)[0] for arm in spec["arms"]}
    bounded = {arm: state for arm, state in states.items() if state is not None}
    if len(set(bounded.values())) > 1:
        detail = "; ".join(f"{a}: {s}" for a, s in sorted(bounded.items()))
        raise ValueError(f"the bounded arms of this spec do not carry the same recurrent state: {detail}")


def lm_job(config="default"):
    spec = load("lm", config)
    arms, seeds = spec["arms"], spec["seeds"]
    box_env = env_from("LM", spec.get("backbone"), spec.get("geometry"), spec.get("budget"))
    validate_matched_states(spec, box_env)
    fleet = dict(spec.get("fleet", {}))
    fleet["env"] = {**box_env, **fleet.get("env", {})}
    # PER-CONFIG ckpt dir: cell run_ids are <arm>_s<seed> with no corpus, so two corpora share names; isolating by
    # config keeps one corpus's resume checkpoint from overwriting another's.
    ckpt_dir = os.path.join(checkpoints("ROLA_LM_CKPTS"), config)

    def _payload_for(run_id):  # resume payload = this cell's last in-progress ckpt tar; None on first run
        p = os.path.join(ckpt_dir, run_id + ".pt")
        if not os.path.exists(p):
            return None
        with open(p, "rb") as fh:
            return fh.read()

    return make_job("lm", config, lambda: [f"{a}_s{s}" for a in arms for s in seeds],
                    "python run.py", fleet, ckpt_dir=ckpt_dir, payload_for=_payload_for)
