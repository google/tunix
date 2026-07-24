#!/bin/bash
# Environment launcher: run the single-host v5p FrozenLake REAL-TRAINING
# convergence run (train_frozenlake_v5p_1host.sh) inside the tunix_base_image
# container on a TPU VM. The FrozenLake analogue of train_v5p_1host_docker.sh
# (which launches the gsm8k train_v5p_1host_pack.sh).
#
# One run yields convergence (wandb loss/reward) + a full-run Perfetto trace +
# a short xprof kernel window, for Gemma4-E2B via the grpo_main CLI.
#
# Usage on the TPU VM (docker preinstalled):
#   # packed (segment-aware CL3 + weighted stream)
#   RUN_TAG=cl3_frozenlake_pack \
#     bash experimental/train_frozenlake_v5p_1host_docker.sh
#   # unpack parity (same stream+weighted accumulation, packing OFF)
#   MAX_TOKEN_PER_TPU=0 RUN_TAG=cl3_frozenlake_unpack \
#     bash experimental/train_frozenlake_v5p_1host_docker.sh
#
# One-time on a fresh VM (artifact-registry pull auth):
#   gcloud auth configure-docker europe-west4-docker.pkg.dev
#
# Every knob of the inner train script passes straight through, e.g.:
#   ROLLOUT_HBM=0.3 MAX_STEPS=50 bash experimental/train_frozenlake_v5p_1host_docker.sh
set -uo pipefail

IMAGE="${IMAGE:-europe-west4-docker.pkg.dev/cloud-tpu-multipod-dev/yuxzhang-repo/tunix_base_image:latest}"
BRANCH="${BRANCH:-yuxzhang/refactor_loss_accum_ablation}"

# Pass the inner train script's knobs through only when the caller set them.
PASS_ENV=()
for var in MAX_TOKEN_PER_TPU MAX_SEGMENTS_PER_ROW ROLLOUT_ENGINE ROLLOUT_HBM \
           MESH_FSDP MESH_TP VLLM_DP VLLM_TP \
           BATCH MINI MICRO NUM_GEN MAX_STEPS NUM_BATCHES RUN_TAG \
           ENABLE_PERF_V1 ENABLE_PERF_V2 PERF_TRACE_DIR \
           TRACE_DEST PROFILER_SKIP PROFILER_STEPS LOG_DIR \
           HF_TOKEN HF_HOME WANDB_MODE WANDB_API_KEY; do
  if [ -n "${!var:-}" ]; then PASS_ENV+=(-e "$var=${!var}"); fi
done

LOG_DIR_HOST="${LOG_DIR:-/tmp/train_frozenlake_logs}"
mkdir -p "$LOG_DIR_HOST"

# Mount the persistent disk when present so the HF model cache survives across
# runs (point HF_HOME at a dir under it to use it).
WS_MOUNT=()
[ -d /mnt/workspace ] && WS_MOUNT=(-v /mnt/workspace:/mnt/workspace)

# --privileged + --net=host: TPU chip access + metadata-server ADC (so the VM's
# service account signs the gs:// trace / Perfetto writes, same as the GKE jobs).
sudo docker run --rm --privileged --net=host \
  -v "$LOG_DIR_HOST":"$LOG_DIR_HOST" \
  "${WS_MOUNT[@]}" \
  "${PASS_ENV[@]}" \
  "$IMAGE" \
  bash -c "
    set -e
    git config --global --add safe.directory \$(pwd)
    git init
    git remote set-url origin https://github.com/google/tunix.git 2>/dev/null \
      || git remote add origin https://github.com/google/tunix.git
    git fetch origin '$BRANCH'
    git reset --hard FETCH_HEAD
    bash experimental/train_frozenlake_v5p_1host.sh
  "
