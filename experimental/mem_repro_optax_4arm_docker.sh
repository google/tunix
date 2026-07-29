#!/bin/bash
# Environment launcher for optax.MultiSteps parity verification on yuxzhang/fix_accum_fp32.
# Runs inside tunix_base_image on a single-host TPU VM, mounts /mnt/workspace,
# and executes experimental/mem_repro_optax_4arm.sh.
set -uo pipefail

IMAGE="${IMAGE:-europe-west4-docker.pkg.dev/cloud-tpu-multipod-dev/yuxzhang-repo/tunix_base_image:latest}"
BRANCH="${BRANCH:-yuxzhang/fix_accum_fp32}"

PASS_ENV=()
for var in MODEL_PATH MESH_FSDP MESH_TP OUT_DIR HF_TOKEN WANDB_MODE; do
  if [ -n "${!var:-}" ]; then PASS_ENV+=(-e "$var=${!var}"); fi
done

OUT_DIR_HOST="${OUT_DIR:-/mnt/workspace/mem_repro_xprof/optax_4arm}"
mkdir -p "$OUT_DIR_HOST"

sudo docker run --rm --privileged --net=host \
  -v /mnt/workspace:/mnt/workspace \
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
    bash experimental/mem_repro_optax_4arm.sh
  "
