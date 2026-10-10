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
export ROLLOUT_JOBSET_YAML="${ROLLOUT_JOBSET_YAML:-jobset.pathways.yaml}"
export SCAN_LAYERS="${SCAN_LAYERS:-true}"
export CHECKPOINT_STORAGE_USE_OCDBT="${CHECKPOINT_STORAGE_USE_OCDBT:-false}"
export CHECKPOINT_STORAGE_USE_ZARR3="${CHECKPOINT_STORAGE_USE_ZARR3:-false}"
export ROLLOUT_REPLICAS="${ROLLOUT_REPLICAS:-16}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.84}"

# Evaluation & Sampling Parameters
export DATASET_SPLIT="${DATASET_SPLIT:-validation}"
export NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
export BATCH_SIZE="${BATCH_SIZE:-256}"
export TASKS_LIMIT="${TASKS_LIMIT:-0}"
# Fixed for MLPerf eval; don't inherit these from a training shell.
export TEMPERATURE="0.1"
export TOP_P="0.95"
export STEP_TIMEOUT_SECS=60
export REWARD_TIMEOUT_SECS=60
export MAX_WARMPOOL_REPLICAS="${MAX_WARMPOOL_REPLICAS:-64}"
export MAX_CONCURRENCY="${MAX_CONCURRENCY:-$(( 32 * ROLLOUT_REPLICAS ))}"
export IMAGE_REWRITE_PREFIX="${IMAGE_REWRITE_PREFIX:-${REGION:-us-east1}-docker.pkg.dev/cloud-tpu-multipod-dev/tunix/}"
export OPENHANDS_SERVER_IMAGE="${OPENHANDS_SERVER_IMAGE:-${REGION:-us-east1}-docker.pkg.dev/cloud-tpu-multipod-dev/tunix/openhands-agent-server:0.62-chardet}"
export ENABLE_THINKING="${ENABLE_THINKING:-false}"

_eval_no_launch="${MLPERF_NO_LAUNCH:-0}"
_eval_pw_worker_extra_env="${PATHWAYS_WORKER_EXTRA_ENV:-}"
_eval_pw_proxy_extra_args="${PATHWAYS_PROXY_EXTRA_ARGS:-}"

MLPERF_NO_LAUNCH=1 source "${DIR}/mlperf_397b_256_v7x.sh"

# Eval runs rollout on Pathways (no Trainer): use rollout LIBTPU_INIT_ARGS for Pathways worker/proxy
export PATHWAYS_WORKER_EXTRA_ENV="${_eval_pw_worker_extra_env:-LIBTPU_INIT_ARGS=${LIBTPU_INIT_ARGS} --megascale_port=-1 --xprof_compress_jftrace=true
SKIP_MEGASCALE_PJRT_CLIENT=true}"
_rollout_xla_flags=""
for _f in ${LIBTPU_INIT_ARGS}; do [[ "${_f}" == --xla_* ]] && _rollout_xla_flags+="${_f} "; done
export PATHWAYS_PROXY_EXTRA_ARGS="${_eval_pw_proxy_extra_args:-${_rollout_xla_flags% }}"

export MLPERF_NO_LAUNCH="${_eval_no_launch}"
source "${DIR}/mlperf_base.sh" "${1:-eval}" "${@:2}"
