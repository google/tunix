#!/bin/bash
set -e

# =====================================================
# MLPerf DeepSWE recipe: Qwen3.5-397B-A17B on TPU v7x
# =====================================================

export RCP_LOGGING="${RCP_LOGGING:-true}"
export WEIGHT_SYNC_DISABLE_TIMEOUTS="${WEIGHT_SYNC_DISABLE_TIMEOUTS:-1}"
export RAIDEN_BROADCAST_HOST_RATIO="${RAIDEN_BROADCAST_HOST_RATIO:-1.0}"
export RAIDEN_BROADCAST_PIPELINE_STAGES="${RAIDEN_BROADCAST_PIPELINE_STAGES:-4}"
export FAIL_FAST="${FAIL_FAST:-true}"
export WANDB_RESUME="${WANDB_RESUME:-allow}"
export PATHWAYS_CHECKPOINTING_IMPL="${PATHWAYS_CHECKPOINTING_IMPL:-persistence}"
export CKPT_USE_OCDBT="${CKPT_USE_OCDBT:-0}"
export CHECKPOINT_SAVE_INTERVAL_STEPS="${CHECKPOINT_SAVE_INTERVAL_STEPS:-1}"
export CHECKPOINT_ABANDON_FAILED_SAVES="${CHECKPOINT_ABANDON_FAILED_SAVES:-true}"
export CHECKPOINT_OPTIMIZER_INTERVAL_STEPS="${CHECKPOINT_OPTIMIZER_INTERVAL_STEPS:-0}"
export TUNIX_DEBUG_INFERENCE_LOGS="${TUNIX_DEBUG_INFERENCE_LOGS:-true}"
export MAX_STEPS="${MAX_STEPS:-10}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export ASYNC_WEIGHT_SYNC="${ASYNC_WEIGHT_SYNC:-true}"
export PIPELINE_TRAIN_MICROBATCHES="${PIPELINE_TRAIN_MICROBATCHES:-true}"
export IN_FLIGHT_WEIGHT_UPDATES="${IN_FLIGHT_WEIGHT_UPDATES:-true}"
export MAX_STALENESS="${MAX_STALENESS:-2}"
export OPENHANDS_MULTI_TOOL_CALLS="${OPENHANDS_MULTI_TOOL_CALLS:-true}"
export WEIGHT_SYNC_PARALLEL_H2H="${WEIGHT_SYNC_PARALLEL_H2H:-1}"
export VERIFY_WEIGHTS="${VERIFY_WEIGHTS:-false}"
export BATCH_SIZE="${BATCH_SIZE:-64}"

# Learning rate and gradient clip follow the global batch size when not set:
# GBS = BATCH_SIZE * NUM_GENERATIONS (16), LEARNING_RATE = 1e-6 * sqrt(GBS / 256),
# MAX_GRAD_NORM = 0.125 * sqrt(256 / GBS). GBS 1,024 gives 2e-06 / 0.0625 and
# GBS 1,152 (BATCH_SIZE=72) gives 2.1213203435596424e-06 / 0.05892556509887896.
if [[ ! "${BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: BATCH_SIZE must be a positive integer, got '${BATCH_SIZE}'" >&2
  exit 1
fi
_gbs=$(( BATCH_SIZE * ${NUM_GENERATIONS:-16} ))
export LEARNING_RATE="${LEARNING_RATE:-$(python3 -c "import math; print(f'{1e-6 * math.sqrt(${_gbs} / 256):.16g}')")}"
export MAX_GRAD_NORM="${MAX_GRAD_NORM:-$(python3 -c "import math; print(f'{0.125 * math.sqrt(256 / ${_gbs}):.16g}')")}"
unset _gbs

# Start-up: hold batch 1 175 s after batch 0 and each later start-up batch 75 s
# after the previous one (batch 2 at 250 s), and compile the trainer on a
# synthetic micro-batch before run_start. The longer first gap keeps batch 1 off
# the engines for most of batch 0, which otherwise decodes ~30% slower alongside
# the next batches.
# When STAGGERED_ROLLOUT_POOLS is enabled, independent pools eliminate
# cross-batch contention, so initial dispatch stagger is not needed.
# Each pool only ever holds one batch, so there is nothing to stagger or
# prioritise against either; both default off here (and may still be set
# explicitly).
if [[ "${STAGGERED_ROLLOUT_POOLS:-false}" == "true" || "${STAGGERED_ROLLOUT_POOLS:-false}" == "True" || "${STAGGERED_ROLLOUT_POOLS:-false}" == "1" ]]; then
  export INITIAL_DISPATCH_FIRST_STAGGER_S="${INITIAL_DISPATCH_FIRST_STAGGER_S:-0}"
  export INITIAL_DISPATCH_STAGGER_S="${INITIAL_DISPATCH_STAGGER_S:-0}"
  export ROLLOUT_PRIORITY_SCHEDULING="${ROLLOUT_PRIORITY_SCHEDULING:-false}"
else
  export INITIAL_DISPATCH_FIRST_STAGGER_S="${INITIAL_DISPATCH_FIRST_STAGGER_S:-175}"
  export INITIAL_DISPATCH_STAGGER_S="${INITIAL_DISPATCH_STAGGER_S:-75}"
  export ROLLOUT_PRIORITY_SCHEDULING="${ROLLOUT_PRIORITY_SCHEDULING:-true}"
fi
export TRAINER_WARMUP_COMPILE="${TRAINER_WARMUP_COMPILE:-true}"
export SANDBOX_WAIT_INITIAL="${SANDBOX_WAIT_INITIAL:-false}"

# Rollout attention without data parallelism: 64 sequences per engine, and the
# trainer replicates K/V heads to match (mlperf_base.sh appends kv_tp_size).
export ROLLOUT_ATTENTION_DP="${ROLLOUT_ATTENTION_DP:-false}"

# Dropless MoE trainer. `true` (default) keeps every routed token
# (ragged_buffer_factor=-1); its buffers fit with context offload, four token
# chunks and an XLA HBM budget of 84992 MiB. `false` is the capacity-factor
# trainer of the base recipe (buffer factor 2.0, context remat, two chunks, no
# HBM cap). Each value can still be overridden on its own.
# A finite buffer factor stays dropless only with the replay knobs of
# mlperf_397b_256_v7x.sh (RETRY_WHEN_TOKENS_DROPPED, RETRY_NUM_MOE_TOKEN_CHUNKS,
# RETRY_DROPLESS_FIRST_STEPS, FIRST_PHASE_RAGGED_BUFFER_FACTOR; maxtext #5570),
# which rerun an overflowing micro-batch with a dropless program. For example,
#   RAGGED_BUFFER_FACTOR=2.75 RETRY_WHEN_TOKENS_DROPPED=true
# keeps this block's context offload, four chunks and HBM budget.
export DROPLESS="${DROPLESS:-true}"
case "${DROPLESS}" in
  true)
    export RAGGED_BUFFER_FACTOR="${RAGGED_BUFFER_FACTOR:--1}"
    export REMAT_CONTEXT="${REMAT_CONTEXT:-offload}"
    export NUM_MOE_TOKEN_CHUNKS="${NUM_MOE_TOKEN_CHUNKS:-4}"
    export TRAINER_MAX_HBM_SIZE_MIB="${TRAINER_MAX_HBM_SIZE_MIB:-84992}"
    if [[ "${RAGGED_BUFFER_FACTOR}" != "-1" && "${RETRY_WHEN_TOKENS_DROPPED:-false}" != "true" ]]; then
      echo "ERROR: DROPLESS=true with RAGGED_BUFFER_FACTOR=${RAGGED_BUFFER_FACTOR} drops tokens;" \
           "set RETRY_WHEN_TOKENS_DROPPED=true or DROPLESS=false" >&2
      exit 1
    fi
    ;;
  false)
    ;;
  *)
    echo "ERROR: DROPLESS must be true or false, got '${DROPLESS}'" >&2
    exit 1
    ;;
esac

export TRAINER_FP8="${TRAINER_FP8:-true}"
export ROLLOUT_FP8="${ROLLOUT_FP8:-true}"
export ROLLOUT_REPLICAS="${ROLLOUT_REPLICAS:-128}"
export VLLM_DATA_PARALLEL_SIZE="${VLLM_DATA_PARALLEL_SIZE:-1}"
export ROLLOUT_TPU_SLICE="${ROLLOUT_TPU_SLICE:-tpu7x:2x2x1}"
export ROLLOUT_JOBSET_YAML="${ROLLOUT_JOBSET_YAML:-jobset.tpu.yaml}"
export UBENCH_REPORTING="${UBENCH_REPORTING:-true}"

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${DIR}/mlperf_397b_1024_v7x.sh" "$@"
