#!/bin/bash
# Baseline for experimental/train_v5p_1host_pack.sh: one-host v5p, NO sequence
# packing + optax gradient accumulation (main's path). Identical mesh / batch /
# steps / Perfetto config as the pack+stream run, so the two are DIRECTLY
# comparable -- only the packing (off) and accumulation (optax) dimensions
# differ. Perfetto full-run tracing is inherited (ON by default), so this
# baseline yields convergence (wandb) + full-run performance to compare against
# the pack+stream run at ui.perfetto.dev.
#
# RUN_TAG re-tags the log / xprof trace / PERF_TRACE_DIR, so the baseline and the
# pack+stream run never collide. This is a thin wrapper (same pattern as
# profile_v5p_{optax,stream}.sh): it only sets the three baseline knobs and
# delegates to the single source of truth, so the two runs can never drift.
#
# Usage on the TPU VM (every env override of the pack script passes through):
#   bash experimental/train_v5p_1host_unpack_optax.sh
#   ROLLOUT_HBM=0.4 bash experimental/train_v5p_1host_unpack_optax.sh   # if OOM
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Baseline defaults: packing OFF (MAX_SEQ_TOKEN=0), optax accumulation, own tag.
# Each stays overridable from the environment.
MAX_SEQ_TOKEN="${MAX_SEQ_TOKEN:-0}" \
GRAD_ACCUM="${GRAD_ACCUM:-optax}" \
RUN_TAG="${RUN_TAG:-v5p_1host_unpack_optax}" \
  exec bash "$SCRIPT_DIR/train_v5p_1host_pack.sh"
