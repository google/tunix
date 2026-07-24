#!/bin/bash
# Environment launcher: run the SFT compile-time repro inside the SAME
# tunix_base_image container the GKE ablation uses, on a single-host TPU VM.
# Same shell as train_v5p_docker.sh except:
#   1. the inner script is experimental/compile_repro_v5p.sh (pure SFT,
#      no vLLM), and
#   2. the persistent disk /mnt/workspace is mounted into the container
#      (the Gemma4-E2B safetensors live there).
#
# Usage on the TPU VM (docker is preinstalled on TPU VMs):
#   bash compile_repro_v5p_docker.sh            # both variants (optax then stream)
#   bash compile_repro_v5p_docker.sh optax      # one variant
#   bash compile_repro_v5p_docker.sh stream
#
# One-time on a fresh VM (pull auth for the artifact registry):
#   gcloud auth configure-docker europe-west4-docker.pkg.dev
#
# Knobs of the inner script pass straight through, e.g. the GQA fallback mesh:
#   MESH_FSDP=4 MESH_TP=1 bash compile_repro_v5p_docker.sh
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
for var in MODEL_PATH MESH_FSDP MESH_TP MAX_STEPS GRAD_ACCUM_STEPS LOG_DIR \
           PROFILE_XPROF HF_TOKEN WANDB_MODE; do
  if [ -n "${!var:-}" ]; then PASS_ENV+=(-e "$var=${!var}"); fi
done

LOG_DIR_HOST="${LOG_DIR:-/tmp/compile_repro_logs}"
mkdir -p "$LOG_DIR_HOST"

# --privileged + --net=host: TPU chip access + metadata-server ADC.
# -v /mnt/workspace: persistent disk with the E2B safetensors.
sudo docker run --rm --privileged --net=host \
  -v "$LOG_DIR_HOST":"$LOG_DIR_HOST" \
  -v /mnt/workspace:/mnt/workspace \
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
    bash experimental/compile_repro_v5p.sh
  "
