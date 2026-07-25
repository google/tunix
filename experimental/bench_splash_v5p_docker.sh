#!/bin/bash
# Environment launcher: run the sequence-packing cost-model microbench
# (experimental/bench_splash_v5p.sh) inside the tunix_base_image container on a
# single-host TPU VM. Same shell as compile_repro_v5p_docker.sh, but the inner
# script needs NO model files and no vLLM -- random weights, a few seconds per
# case -- so nothing has to be provisioned first.
#
# Usage on the TPU VM (docker is preinstalled on TPU VMs):
#   bash experimental/bench_splash_v5p_docker.sh
#
# One-time on a fresh VM (pull auth for the artifact registry):
#   gcloud auth configure-docker europe-west4-docker.pkg.dev
#
# Knobs of the inner script pass straight through, e.g.:
#   BUDGETS=8192,4096,2048 bash experimental/bench_splash_v5p_docker.sh
#   MODEL_CONFIG=qwen3_8b bash experimental/bench_splash_v5p_docker.sh
#   WITH_MODULE=1 WITH_LAYER=1 bash experimental/bench_splash_v5p_docker.sh
set -uo pipefail

IMAGE="${IMAGE:-europe-west4-docker.pkg.dev/cloud-tpu-multipod-dev/yuxzhang-repo/tunix_base_image:latest}"
BRANCH="${BRANCH:-yuxzhang/refactor_loss_accum_ablation}"

# Pass the inner script's knobs through only when the caller set them.
PASS_ENV=()
for var in NUM_SEQS SEQ_LEN MIN_TOKENS MAX_TOKENS BUDGETS SEGMENT_SWEEP SWEEP_BUDGET MODEL_CONFIG \
           ITERS WARMUP TRACE_DEST TRACE_ITERS WITH_MODULE WITH_LAYER LOG_DIR RUN_TAG; do
  if [ -n "${!var:-}" ]; then PASS_ENV+=(-e "$var=${!var}"); fi
done

LOG_DIR_HOST="${LOG_DIR:-/tmp/bench_splash_logs}"
mkdir -p "$LOG_DIR_HOST"

# --privileged + --net=host: TPU chip access + metadata-server ADC (so the VM's
# service account signs the gs:// xprof writes).
sudo docker run --rm --privileged --net=host \
  -v "$LOG_DIR_HOST":"$LOG_DIR_HOST" \
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
    bash experimental/bench_splash_v5p.sh
  "
