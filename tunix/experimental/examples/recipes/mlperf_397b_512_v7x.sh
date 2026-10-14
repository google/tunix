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
export BATCH_SIZE="${BATCH_SIZE:-32}"
export MAX_STEPS="${MAX_STEPS:-20}"
export LEARNING_RATE="${LEARNING_RATE:-1.4142135624e-6}"
export MAX_GRAD_NORM="${MAX_GRAD_NORM:-0.08838834765}"
export MAX_CONCURRENCY="${MAX_CONCURRENCY:-2048}"
export USER_CONTAINER_MEMORY="${USER_CONTAINER_MEMORY:-96G}"

source "${DIR}/mlperf_397b_256_v7x.sh" "$@"
