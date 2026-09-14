"""Box-side LM runner (FLEET_RUN_CMD -> this). For each run_id in RUN_IDS ("<arm>_s<seed>"),
train that arm via train_lm.py and let it append its result row to $FLEET_RESULTS; on a crash we
append an {ok: False} row so the cell is recorded (not silently lost). Mirrors run.py's
RUN_IDS contract for the MQAR fleet.

Corpus / budget come from env (LM_DATASET / LM_MAX_STEPS / ...), defaulting to the canonical
125M-on-FineWeb-Edu comparison config. The checkpoint (save_pretrained dir) lands under
$FLEET_CKPT_DIR/<run_id> for the post-hoc recall-downstream eval pass.
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.environ.get("FLEET_RESULTS", "/workspace/lm_res.jsonl")
CKPT_DIR = os.environ.get("FLEET_CKPT_DIR", "/workspace/lm_ckpts")

# Canonical 125M-on-FineWeb-Edu budget (env-overridable). block_size*batch*grad_accum*max_steps tokens.
DEFAULTS = {"dataset": os.environ.get("LM_DATASET", "fineweb-edu"),
                "dataset_config": os.environ.get("LM_DATASET_CONFIG", "sample-10BT"),
                "max_steps": os.environ.get("LM_MAX_STEPS", "20000"),
                "block_size": os.environ.get("LM_BLOCK_SIZE", "2048"),   # train+eval context 2048
                "batch_size": os.environ.get("LM_BATCH_SIZE", "8"),      # halved vs 1024 to fit memory
                "grad_accum": os.environ.get("LM_GRAD_ACCUM", "4"),      # tokens/step = 2048*8*4 = 65536
                "lr": os.environ.get("LM_LR", "3e-4"),
                "save_steps": os.environ.get("LM_SAVE_STEPS", "500"),    # periodic ckpt cadence (preemption-resume)
                "grad_ckpt": os.environ.get("LM_GRAD_CKPT", "0"),         # 1=grad-checkpointing (frees mem for bigger batch)
                "max_train_samples": os.environ.get("LM_MAX_TRAIN_SAMPLES", "")}  # cap materialized sets (TinyStories)


def main():
    run_ids = [r for r in os.environ.get("RUN_IDS", "").split(",") if r]
    if not run_ids:                                      # fail loud — never silently no-op
        print("run_lm: RUN_IDS empty", flush=True)
        sys.exit(1)
    for rid in run_ids:
        arm, sep, seed = rid.rpartition("_s")
        if not sep or not seed.isdigit():
            with open(RESULTS, "a") as fh:
                fh.write(json.dumps({"run_id": rid, "ok": False, "error": "bad run_id (want <arm>_s<seed>)"}) + "\n")
            print(f"run_lm: bad run_id {rid!r}", flush=True)
            continue
        cmd = [sys.executable, os.path.join(HERE, "train_lm.py"),
               "--arm", arm, "--seed", seed, "--results", RESULTS, "--ckpt_dir", CKPT_DIR,
               "--dataset", DEFAULTS["dataset"], "--dataset_config", DEFAULTS["dataset_config"],
               "--max_steps", DEFAULTS["max_steps"], "--block_size", DEFAULTS["block_size"],
               "--batch_size", DEFAULTS["batch_size"], "--grad_accum", DEFAULTS["grad_accum"],
               "--lr", DEFAULTS["lr"], "--save_steps", DEFAULTS["save_steps"]]
        if DEFAULTS["max_train_samples"]:
            cmd += ["--max_train_samples", DEFAULTS["max_train_samples"]]
        print(f"run_lm: {rid} -> {' '.join(cmd)}", flush=True)
        try:
            subprocess.run(cmd, check=True)              # train_lm writes the success row itself
        except subprocess.CalledProcessError as e:
            with open(RESULTS, "a") as fh:
                fh.write(json.dumps({"run_id": rid, "ok": False, "error": f"train_lm rc={e.returncode}"}) + "\n")
            print(f"run_lm: {rid} FAILED rc={e.returncode}", flush=True)


if __name__ == "__main__":
    main()
