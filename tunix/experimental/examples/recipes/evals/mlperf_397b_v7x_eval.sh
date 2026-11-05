#!/bin/bash
set -e

# ==============================================================================
# MLPerf DeepSWE evaluation recipe: Qwen3.5-397B-A17B on TPU v7x
# ==============================================================================
# - Directly inherits cluster, model, rollout worker, and sandbox settings from
#   mlperf_gbs1024_submission.sh (128 replicas x tpu7x:2x2x1 on jobset.tpu.yaml,
#   DP=1, TP=4, EP=8, FP8 rollout; no Pathways, no cpu-np nodes, no Trainer).
# ==============================================================================

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

_eval_no_launch="${MLPERF_NO_LAUNCH:-0}"
_user_batch_size="${BATCH_SIZE:-}"
_user_num_generations="${NUM_GENERATIONS:-}"
_user_max_concurrency="${MAX_CONCURRENCY:-}"
_user_max_warmpool_replicas="${MAX_WARMPOOL_REPLICAS:-}"
_user_prepull_sandbox_images="${PREPULL_SANDBOX_IMAGES:-}"
_user_use_default_sql_store="${USE_DEFAULT_SQL_STORE:-}"
_user_ubench_reporting="${UBENCH_REPORTING:-}"
_eval_pw_worker_extra_env="${PATHWAYS_WORKER_EXTRA_ENV:-}"
_eval_pw_proxy_extra_args="${PATHWAYS_PROXY_EXTRA_ARGS:-}"

export USE_DEFAULT_SQL_STORE="${_user_use_default_sql_store:-false}"
export UBENCH_REPORTING="${_user_ubench_reporting:-false}"
MLPERF_NO_LAUNCH=1 source "${DIR}/mlperf_gbs1024_submission.sh"

export EVAL_JOBSET_NAME="${EVAL_JOBSET_NAME:-${JOB_PREFIX}-eval}"
export EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${BUCKET}/eval_results/${JOB_PREFIX}}"

# Eval checkpoint & rollout overrides (post-training eval: no Trainer or Raiden)
export WEIGHT_SYNC_MODE="${WEIGHT_SYNC_MODE:-none}"
export SCAN_LAYERS="${SCAN_LAYERS:-true}"
export CHECKPOINT_STORAGE_USE_OCDBT="${CHECKPOINT_STORAGE_USE_OCDBT:-false}"
export CHECKPOINT_STORAGE_USE_ZARR3="${CHECKPOINT_STORAGE_USE_ZARR3:-false}"

# Evaluation & Sampling Parameters
export DATASET_SPLIT="${DATASET_SPLIT:-validation}"
export NUM_GENERATIONS="${_user_num_generations:-4}"
export BATCH_SIZE="${_user_batch_size:-128}"
export TASKS_LIMIT="${TASKS_LIMIT:-0}"
# Fixed for MLPerf eval; don't inherit these from a training shell.
export TEMPERATURE="0.1"
export TOP_P="0.95"
export STEP_TIMEOUT_SECS=60
export REWARD_TIMEOUT_SECS=60
export MAX_WARMPOOL_REPLICAS="${_user_max_warmpool_replicas:-32}"
export MAX_CONCURRENCY="${_user_max_concurrency:-$(( 4 * ROLLOUT_REPLICAS ))}"
export PREPULL_SANDBOX_IMAGES="${_user_prepull_sandbox_images:-false}"
export ENABLE_THINKING="${ENABLE_THINKING:-false}"

if [[ "${ROLLOUT_JOBSET_YAML:-}" == *pathways* ]]; then
  export PATHWAYS_WORKER_EXTRA_ENV="${_eval_pw_worker_extra_env:-LIBTPU_INIT_ARGS=${LIBTPU_INIT_ARGS} --megascale_port=-1 --xprof_compress_jftrace=true
SKIP_MEGASCALE_PJRT_CLIENT=true}"
  _rollout_xla_flags=""
  for _f in ${LIBTPU_INIT_ARGS}; do [[ "${_f}" == --xla_* ]] && _rollout_xla_flags+="${_f} "; done
  export PATHWAYS_PROXY_EXTRA_ARGS="${_eval_pw_proxy_extra_args:-${_rollout_xla_flags% }}"
fi

export MLPERF_NO_LAUNCH="${_eval_no_launch}"
source "${DIR}/mlperf_base.sh" "${1:-eval}" "${@:2}"
