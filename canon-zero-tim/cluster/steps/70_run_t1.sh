#!/usr/bin/env bash
# Topology admission probes.  Seconds, no model, no checkpoint, no optimizer.
#
# Run these BEFORE anything expensive on a new cluster.  They answer whether the canonical
# switch set transfers to this topology at all: reduction width, device order, slice
# structure, bucket arithmetic, tree cost.
set -euo pipefail
source "$CANON_STATE/env.sh"
export JAX_PLATFORMS="proxy,cpu"
export JAX_BACKEND_TARGET="grpc://localhost:29000"
export PATHWAYS_HEAD="localhost"
export CANON_IN_CONTAINER=1
echo "[t1] activating Pathways backend: JAX_PLATFORMS=$JAX_PLATFORMS target=$JAX_BACKEND_TARGET"
bash "$CANON_PKG/tests/t1_tpu/run.sh"
