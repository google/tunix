#!/usr/bin/env bash
# Topology admission probes.  Seconds, no model, no checkpoint, no optimizer.
#
# Run these BEFORE anything expensive on a new cluster.  They answer whether the canonical
# switch set transfers to this topology at all: reduction width, device order, slice
# structure, bucket arithmetic, tree cost.
set -euo pipefail
source "$CANON_STATE/env.sh"
export CANON_IN_CONTAINER=1
bash "$CANON_PKG/tests/t1_tpu/run.sh"
