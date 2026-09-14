# rola-bench LM image — FROM the project-agnostic fleet base.
#
# Same pattern as mqar.Dockerfile but for the language-modeling benchmark: fla (the fork) + the HF stack. zoology (the MQAR
# baselines) is not installed.
#
# The box-side contract: run.py maps RUN_IDS ("<arm>_s<seed>") to train_lm.py, which appends {run_id, ok, params_m,
# state_floats, ppl, ...} to $FLEET_RESULTS and saves the checkpoint directory.
ARG FLEET_BASE_TAG=v1
FROM blakeresearch/fleet-base:${FLEET_BASE_TAG}

ENV DEBIAN_FRONTEND=noninteractive PIP_ROOT_USER_ACTION=ignore PYTHONUNBUFFERED=1 \
    WANDB_MODE=offline WANDB_CONSOLE=off

# LM stack: HF + canonical FLA (the rola_hf model imports fla.layers) + datasets/eval.
RUN pip install --no-cache-dir \
        transformers accelerate datasets evaluate einops pydantic numpy tqdm wandb rich pyyaml scipy lm-eval
# flash-attn: ONLY the softmax-attention CEILING arm (fla.layers.Attention) needs it; the 4
# bounded arms (RoLA x3, GDN) do not. Try a prebuilt wheel first; fall back to a source build
# (needs the CUDA toolkit in the base). If it can't install, the `attn` arm fails loudly while the
# bounded arms still run — so the image build does NOT hard-fail on flash-attn.
RUN pip install --no-cache-dir flash-attn --no-build-isolation || \
    echo "WARN: flash-attn not installed — the attn ceiling arm will be unavailable on this image"

COPY fla        /opt/fla
COPY rola-bench /opt/rola-bench
# --no-build-isolation: build against the image's torch (some setup.py's import torch at build).
RUN pip install --no-cache-dir --no-deps --no-build-isolation -e /opt/fla \
 && pip install --no-cache-dir --no-deps --no-build-isolation -e /opt/rola-bench

# Verify the box-side path only (run.py -> train_lm.py -> arms.py and their dependencies, flash-attn for the attn arm),
# from the directory the box runs in. Not rola_bench.lm.job: that is dispatcher-side and builds a fleet.Job.
RUN cd /opt/rola-bench/rola_bench/lm && python -c "import torch, fla.layers, fleet, flash_attn, arms, run; \
print('rola-bench lm image OK; torch', torch.__version__, 'flash_attn', flash_attn.__version__, 'arms', len(arms.ARMS))"

ENV FLEET_WORKDIR=/opt/rola-bench/rola_bench/lm \
    FLEET_VERIFY_CMD="python -m rola_bench.fleet.verify" \
    FLEET_RUN_CMD="python run.py"

WORKDIR /opt/rola-bench/rola_bench/lm
# CMD inherited from fleet-base: python -u -m fleet.box_server
