#!/bin/bash
set -e

# ==============================================================================
# mlperf_397b_256_v7x.sh with the rollout scaled to 1024 devices (512 chips).
# ==============================================================================
# - Rollout on 512 chips (32 replicas x 16 chips 2x2x4, DP=2, EP=16, TP=1)
# - Trainer and everything else unchanged from the base recipe.
# ==============================================================================

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export ROLLOUT_REPLICAS="${ROLLOUT_REPLICAS:-32}"
export ROLLOUT_TPU_SLICE="${ROLLOUT_TPU_SLICE:-tpu7x:2x2x4}"
export VLLM_DATA_PARALLEL_SIZE="${VLLM_DATA_PARALLEL_SIZE:-2}"
export BATCH_SIZE="${BATCH_SIZE:-32}"
export MAX_STEPS="${MAX_STEPS:-20}"
export CHECKPOINT_SAVE_INTERVAL_STEPS="${CHECKPOINT_SAVE_INTERVAL_STEPS:-1}"
export CHECKPOINT_MAX_TO_KEEP="${CHECKPOINT_MAX_TO_KEEP:-30}"
export LEARNING_RATE="${LEARNING_RATE:-1.4142135624e-6}"
export MAX_GRAD_NORM="${MAX_GRAD_NORM:-0.08838834765}"
export MAX_WARMPOOL_REPLICAS="${MAX_WARMPOOL_REPLICAS:-16}"
export MAX_CONCURRENCY="${MAX_CONCURRENCY:-2048}"
export USER_CONTAINER_MEMORY="${USER_CONTAINER_MEMORY:-96G}"

source "${DIR}/mlperf_397b_256_v7x.sh" "$@"
