#!/bin/bash
# Environment launcher: run the single-host v5p FrozenLake REAL-TRAINING
# convergence run (train_frozenlake_v5p_1host.sh) inside the tunix_base_image
# container on a TPU VM. The FrozenLake analogue of train_v5p_1host_docker.sh
# (which launches the gsm8k train_v5p_1host_pack.sh).
#
# Drives examples/frozenlake/train_frozenlake.py (Gemma4-E2B, separate
# rollout/trainer meshes), pinning jax[tpu] and vllm-tpu to the versions the
# recipe is known to converge on and generating the dataset locally.
#
# Usage on the TPU VM (docker preinstalled):
#   # unpacked baseline (the script packs only when told to)
#   RUN_TAG=fl_unpack bash experimental/train_frozenlake_v5p_1host_docker.sh
#   # packed
#   MAX_TOKEN_PER_TPU=4096 RUN_TAG=fl_pack \
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
           BATCH MINI NUM_GEN NUM_BATCHES RUN_TAG \
           JAX_VERSION VLLM_VERSION SKIP_PIP DATA_DIR TB_LOG_DIR LOG_DIR \
           HF_TOKEN HF_HOME WANDB_MODE WANDB_API_KEY WANDB_RUN_NAME; do
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
