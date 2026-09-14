#!/usr/bin/env bash
# One command to build and push the fleet base and the rola-bench image (no GPU required: a pure build).
#
#   FLEET_SRC=<fleet checkout> ROLA_FLA_SRC=<fla fork checkout> ROLA_ZOO_SRC=<zoology checkout> bash docker/build_all.sh
#
# Builds, in order:
#   1. blakeresearch/fleet-base:$IMG_VERSION  -- the fleet base (context: the fleet checkout).
#   2. blakeresearch/rola-bench:$IMG_VERSION  -- FROM that base; the rola_bench package (build.sh stages clean sources).
# then pushes both (Docker Hub credentials from Docker Desktop). After it finishes, from this repository:
#   python -m rola_bench.fleet mqar canonical --boxes 6
set -euo pipefail

V="${IMG_VERSION:-v2}"
FLEET="${FLEET_SRC:?set FLEET_SRC to the fleet checkout}"
RB="$(cd "$(dirname "$0")/.." && pwd)"

# The sub-scripts (build_base.sh/build.sh) invoke bare `docker`, which in WSL is Docker Desktop's
# integration shim. Require it to reach a running daemon; otherwise tell the user exactly what to fix.
docker version >/dev/null 2>&1 || {
  echo "ERROR: 'docker' can't reach a daemon in WSL." >&2
  echo "  1) Start Docker Desktop on Windows, and" >&2
  echo "  2) ensure WSL integration is ON for this distro (Docker Desktop > Settings > Resources > WSL integration)." >&2
  echo "Then re-run: bash rola-bench/docker/build_all.sh" >&2
  exit 1; }

echo "==> [1/2] fleet base  blakeresearch/fleet-base:$V"
bash "$FLEET/docker/build_base.sh" "blakeresearch/fleet-base:$V"
docker push "blakeresearch/fleet-base:$V"

echo "==> [2/2] bench image blakeresearch/rola-bench:$V"
IMG_VERSION="$V" bash "$RB/docker/build.sh" mqar --push

echo "==> done. pushed fleet-base:$V + rola-bench:$V"
echo "    launch:  python -m rola_bench.fleet mqar canonical --boxes 6"
