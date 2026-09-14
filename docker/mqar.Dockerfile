# rola-bench MQAR image — FROM the project-agnostic fleet base.
#
# The base provides torch + the `fleet` package + box_server (the container entrypoint). Here we add
# the research stack (rola, fla from the fork, zoology) + the suite package (rola_bench), and tell box_server
# how to run/verify a batch via FLEET_RUN_CMD / FLEET_VERIFY_CMD. Sources are staged clean by
# build.sh — only runtime code, no .git/CLAUDE.md/data/results (publication-safe).
ARG FLEET_BASE_TAG=v2
FROM blakeresearch/fleet-base:${FLEET_BASE_TAG}

ENV DEBIAN_FRONTEND=noninteractive PIP_ROOT_USER_ACTION=ignore PYTHONUNBUFFERED=1 \
    WANDB_MODE=offline WANDB_CONSOLE=off

# Research stack: canonical FLA (fla.ops + fla.layers) + the libs the MQAR sweep imports.
RUN pip install --no-cache-dir \
        transformers opt_einsum einops pydantic pandas numpy tqdm wandb rich pyyaml scipy datasets

# Clean package sources (staged by build.sh) — install no-deps.
COPY fla        /opt/fla
COPY zoology    /opt/zoology
COPY rola-bench /opt/rola-bench
# --no-build-isolation: use the image's torch (zoology's setup.py imports torch at build time;
# build isolation would give it a torch-less env -> ModuleNotFoundError).
RUN pip install --no-cache-dir --no-deps --no-build-isolation -e /opt/fla \
 && pip install --no-cache-dir --no-deps --no-build-isolation -e /opt/zoology \
 && pip install --no-cache-dir --no-deps --no-build-isolation -e /opt/rola-bench

# Build-time gate: the whole dependency graph imports cleanly (no path hacks).
RUN python -c "import torch, fla.ops.simple_gla, fla.layers, zoology, fleet; \
import rola_bench.mqar.job; from fla.layers import LinearAttention, GatedDeltaNet; \
print('rola-bench mqar image OK; torch', torch.__version__)"

# Tell the (inherited) box_server how to verify the GPU and run a batch of run_ids.
# NB: \$FLEET_RESULTS stays LITERAL in the stored env (Docker would expand it to empty at build);
# box_server sets FLEET_RESULTS at runtime and bash -lc expands it then.
# FLEET_VERIFY_CMD is a FAST GPU smoke (one RoLA layer forward and backward): a fail-fast host gate, not correctness CI.
ENV FLEET_WORKDIR=/opt/rola-bench/rola_bench/mqar \
    FLEET_VERIFY_CMD="python -m rola_bench.fleet.verify" \
    FLEET_RUN_CMD="python run.py --config canonical --results \$FLEET_RESULTS"

WORKDIR /opt/rola-bench/rola_bench/mqar
# CMD inherited from fleet-base: python -u -m fleet.box_server
