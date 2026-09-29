#!/bin/bash
set -e

# ==============================================================================
# mlperf_397b_256_v7x.sh with the rollout scaled to 2048 devices (1024 chips).
# ==============================================================================
# - Rollout on 1024 chips (64 replicas x 16 chips 2x2x4, DP=2, EP=16, TP=1)
# - Trainer and everything else unchanged from the base recipe.
# ==============================================================================

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export ROLLOUT_REPLICAS="${ROLLOUT_REPLICAS:-64}"
export ROLLOUT_TPU_SLICE="${ROLLOUT_TPU_SLICE:-tpu7x:2x2x4}"
export VLLM_DATA_PARALLEL_SIZE="${VLLM_DATA_PARALLEL_SIZE:-2}"
export BATCH_SIZE="${BATCH_SIZE:-64}"
export LEARNING_RATE="${LEARNING_RATE:-2.0e-6}"
export MAX_GRAD_NORM="${MAX_GRAD_NORM:-0.0625}"
export MAX_WARMPOOL_REPLICAS="${MAX_WARMPOOL_REPLICAS:-16}"
export MAX_CONCURRENCY="${MAX_CONCURRENCY:-4096}"
export USER_CONTAINER_MEMORY="${USER_CONTAINER_MEMORY:-160G}"

source "${DIR}/mlperf_397b_256_v7x.sh" "$@"
