#!/bin/bash
# Environment launcher for the HBM / per-step-op regression repro. Same shell as
# compile_repro_v5p_docker.sh: runs inside tunix_base_image on a single-host TPU
# VM, mounts the persistent disk (/mnt/workspace) for the E2B safetensors and
# the xprof output, and fetches the branch under test.
#
# Usage on the TPU VM:
#   bash mem_repro_v5p_docker.sh
#
# xprof traces land in /mnt/workspace/mem_repro_xprof/<arm>/ (persist on the
# disk); copy them off and open in TensorBoard/xprof.
set -uo pipefail

IMAGE="${IMAGE:-europe-west4-docker.pkg.dev/cloud-tpu-multipod-dev/yuxzhang-repo/tunix_base_image:latest}"
BRANCH="${BRANCH:-yuxzhang/perf-regression-wip}"

# Pass the inner script's knobs through only when the caller set them.
PASS_ENV=()
for var in MODEL_PATH MESH_FSDP MESH_TP MAX_STEPS XPROF_DIR HF_TOKEN WANDB_MODE; do
  if [ -n "${!var:-}" ]; then PASS_ENV+=(-e "$var=${!var}"); fi
done

XPROF_DIR_HOST="${XPROF_DIR:-/mnt/workspace/mem_repro_xprof}"
mkdir -p "$XPROF_DIR_HOST"

# --privileged + --net=host: TPU chip access + metadata-server ADC.
# -v /mnt/workspace: E2B safetensors + persistent xprof output.
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
    bash experimental/mem_repro_v5p.sh
  "
