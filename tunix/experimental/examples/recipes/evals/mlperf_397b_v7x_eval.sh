#!/bin/bash
set -e

# ==============================================================================
# MLPerf DeepSWE evaluation recipe: Qwen3.5-397B-A17B on TPU v7x
# ==============================================================================
# - Inherits cluster, 397B model, rollout, and sandbox settings from mlperf_397b_256_v7x.sh
# - Rollout on 256 chips (16 replicas x 16 chips 2x2x4, DP=2, EP=16, TP=1; no Trainer)
# ==============================================================================

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Fill these before you run.
# k8s has a 63 char limit on total label name, so keep job_prefix unique to your job and short
export JOB_PREFIX="${JOB_PREFIX:-${USER}}"
export EVAL_JOBSET_NAME="${EVAL_JOBSET_NAME:-${JOB_PREFIX}-eval}"
export TUNIX_IMAGE="${TUNIX_IMAGE:-gcr.io/cloud-tpu-multipod-dev/sanbao/trellis:latest}"
export BUCKET="${BUCKET:-gs://atwigg-trellis-us-east1-fast-dev}"
export EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${BUCKET}/eval_results/${JOB_PREFIX}}"

# Eval checkpoint & rollout overrides (post-training eval: no Trainer or Raiden)
export WEIGHT_SYNC_MODE="${WEIGHT_SYNC_MODE:-none}"
export SCAN_LAYERS="${SCAN_LAYERS:-true}"
export CHECKPOINT_STORAGE_USE_OCDBT="${CHECKPOINT_STORAGE_USE_OCDBT:-false}"
export CHECKPOINT_STORAGE_USE_ZARR3="${CHECKPOINT_STORAGE_USE_ZARR3:-false}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.84}"

# Evaluation & Sampling Parameters
export DATASET_SPLIT="${DATASET_SPLIT:-validation}"
export NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
export BATCH_SIZE="${BATCH_SIZE:-64}"
export TASKS_LIMIT="${TASKS_LIMIT:-0}"
export TEMPERATURE="${TEMPERATURE:-0.1}"
export TOP_P="${TOP_P:-0.95}"
export MAX_CONCURRENCY="${MAX_CONCURRENCY:-256}"
export ENABLE_THINKING="${ENABLE_THINKING:-false}"

source "${DIR}/mlperf_397b_256_v7x.sh" "${1:-eval}" "${@:2}"
