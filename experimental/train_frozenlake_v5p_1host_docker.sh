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
# One-time on a fresh VM: build the image (~6 min, no registry involved):
#   bash experimental/build_frozenlake_image.sh
#
# LOCAL_REPO=1 runs the working tree instead of the pushed branch -- for
# iterating on a change that is not committed yet:
#   LOCAL_REPO=1 NUM_BATCHES=2 bash experimental/train_frozenlake_v5p_1host_docker.sh
#
# Every knob of the inner train script passes straight through, e.g.:
#   ROLLOUT_HBM=0.3 MAX_STEPS=50 bash experimental/train_frozenlake_v5p_1host_docker.sh
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TUNIX_DIR="${TUNIX_DIR:-$(dirname "$SCRIPT_DIR")}"

IMAGE="${IMAGE:-tunix_frozenlake_image:vllm-tpu0.25.0}"
BRANCH="${BRANCH:-yuxzhang/refactor_loss_accum_ablation}"

# Pass the inner train script's knobs through only when the caller set them.
PASS_ENV=()
for var in MAX_TOKEN_PER_TPU MAX_SEGMENTS_PER_ROW ROLLOUT_ENGINE ROLLOUT_HBM \
           BATCH MINI NUM_GEN NUM_BATCHES RUN_TAG DRY_RUN \
           VLLM_TPU_VERSION EXPECT_DEVICES MAX_PROMPT MAX_RESPONSE ENV_MAX_STEPS \
           DATA_DIR TB_LOG_DIR LOG_DIR \
           HF_TOKEN HF_HOME WANDB_MODE WANDB_API_KEY WANDB_RUN_NAME; do
  if [ -n "${!var:-}" ]; then PASS_ENV+=(-e "$var=${!var}"); fi
done

LOG_DIR_HOST="${LOG_DIR:-/tmp/train_frozenlake_logs}"
mkdir -p "$LOG_DIR_HOST"

# Mount whatever holds the HF cache so a ~5GB model download survives across
# runs. Following HF_HOME rather than one hardcoded path keeps this working on a
# machine whose persistent disk is mounted somewhere else.
HF_HOME_HOST="${HF_HOME:-$HOME/.cache/huggingface}"
CACHE_MOUNT=()
if mkdir -p "$HF_HOME_HOST" 2>/dev/null; then
  CACHE_MOUNT=(-v "$HF_HOME_HOST":"$HF_HOME_HOST" -e "HF_HOME=$HF_HOME_HOST")
else
  echo "WARNING: cannot create HF_HOME=$HF_HOME_HOST; the model will be" \
       "re-downloaded inside the container on every run."
fi

# LOCAL_REPO=1 runs the working tree instead of the pushed branch. The default
# (fetch) is what a fresh machine wants -- it reproduces a known commit. This
# mode is for iterating here, where the change under test is not committed yet.
REPO_MOUNT=()
if [ -n "${LOCAL_REPO:-}" ]; then
  REPO_MOUNT=(-v "$TUNIX_DIR":/app)
  FETCH_CMD="echo '--- LOCAL_REPO: running the working tree at $TUNIX_DIR ---'"
else
  FETCH_CMD="git config --global --add safe.directory \$(pwd)
    git init
    git remote set-url origin https://github.com/google/tunix.git 2>/dev/null \
      || git remote add origin https://github.com/google/tunix.git
    git fetch origin '$BRANCH'
    git reset --hard FETCH_HEAD"
fi

# --privileged + --net=host: TPU chip access + metadata-server ADC (so the VM's
# service account signs the gs:// trace / Perfetto writes, same as the GKE jobs).
sudo docker run --rm --privileged --net=host \
  -v "$LOG_DIR_HOST":"$LOG_DIR_HOST" \
  "${CACHE_MOUNT[@]}" \
  "${REPO_MOUNT[@]}" \
  "${PASS_ENV[@]}" \
  "$IMAGE" \
  bash -c "
    set -e
    $FETCH_CMD
    bash experimental/train_frozenlake_v5p_1host.sh
  "
