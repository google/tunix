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

source "${DIR}/mlperf_397b_256_v7x.sh" "$@"
