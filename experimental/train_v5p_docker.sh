#!/bin/bash
# Environment launcher: run the grad-accum profiling inside the SAME
# tunix_base_image container the GKE ablation uses, on a single-host TPU VM.
# This is the missing "env setup" step - the profiling scripts themselves
# assume deps (jax/vllm/tpu_inference/grain/tfds) are already present, and
# this image provides all of them.
#
# Usage on the TPU VM (docker is preinstalled on TPU VMs):
#   bash profile_v5p_docker.sh            # both variants (optax then stream)
#   bash profile_v5p_docker.sh optax      # one variant
#   bash profile_v5p_docker.sh stream
#
# One-time on a fresh VM (pull auth for the artifact registry):
#   gcloud auth configure-docker europe-west4-docker.pkg.dev
#
# Knobs of the inner script pass straight through, e.g.:
#   BATCH=8 MINI=8 MICRO=2 LOGPS=2 bash profile_v5p_docker.sh stream
set -uo pipefail

IMAGE="${IMAGE:-europe-west4-docker.pkg.dev/cloud-tpu-multipod-dev/yuxzhang-repo/tunix_base_image:latest}"
BRANCH="${BRANCH:-yuxzhang/refactor_loss_accum_ablation}"
VARIANT="${1:-all}"

case "$VARIANT" in
  optax|stream) INNER_VARIANTS="$VARIANT" ;;
  all) INNER_VARIANTS="optax stream" ;;
  *) echo "usage: $0 [optax|stream|all]"; exit 1 ;;
esac

# Pass the inner script's knobs through only when the caller set them.
PASS_ENV=()
for var in TRACE_DEST RUN_TAG LOG_DIR ROLLOUT_ENGINE MESH_FSDP MESH_TP \
           BATCH MINI MICRO LOGPS MAX_STEPS PROFILER_SKIP PROFILER_STEPS \
           HF_TOKEN WANDB_MODE WANDB_API_KEY; do
  if [ -n "${!var:-}" ]; then PASS_ENV+=(-e "$var=${!var}"); fi
done

LOG_DIR_HOST="${LOG_DIR:-/tmp/grad_accum_logs}"
mkdir -p "$LOG_DIR_HOST"

# --privileged + --net=host: TPU chip access + metadata-server ADC (so the
# VM's service account signs the gs:// trace writes, same as the GKE jobs).
sudo docker run --rm --privileged --net=host \
  -v "$LOG_DIR_HOST":"$LOG_DIR_HOST" \
  -e VARIANTS="$INNER_VARIANTS" \
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
    bash experimental/profile_v5p_grad_accum.sh
  "
