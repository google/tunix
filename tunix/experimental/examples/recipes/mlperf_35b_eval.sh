#!/bin/bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Offline pass@4 evaluation of Qwen3.5-35B-A3B on the MLPerf validation split
# (251 instances x 4 rollouts) with 16x 4-chip v5p rollout slices (no trainer).
#
# MLPerf RCP: point CHECKPOINT_MANIFEST_FILE at a training run's
# ${METRIC_LOGGER_DIR}/eval_checkpoints.jsonl (written by mlperf_35b_128_v5p.sh).
# Checkpoints are evaluated in step order and eval_* events are appended to the
# training MLLOG until pass@4 > TARGET_ACCURACY; run_stop is backdated to that
# checkpoint's weight-update timestamp (status=aborted if none converges).
#
# Without a manifest, MAXTEXT_CKPT is evaluated once. Set RCP_LOGGING=true to
# test RCP logging before post-training; CHECKPOINT_STEP, SAMPLES_COUNT and
# CHECKPOINT_TIMESTAMP_MS then default to mock values and events go to
# ${EVAL_OUTPUT_DIR}/mllog (not the training MLLOG) unless METRIC_LOGGER_DIR is set.

# k8s has a 63 char limit on total label name, so keep job_prefix unique to your job and short
export JOB_PREFIX="${JOB_PREFIX:-${USER}}"
export MAXTEXT_OUTPUT_DIR="${MAXTEXT_OUTPUT_DIR:-gs://atwigg-trellis-europe-west4-dev/maxtext/${JOB_PREFIX}}"
export EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-gs://atwigg-trellis-europe-west4-dev/eval_results/${JOB_PREFIX}}"
export TUNIX_IMAGE="${TUNIX_IMAGE:-gcr.io/cloud-tpu-multipod-dev/sanbao/tunix_stack:eval}"

export REGION="europe-west4"
export CLUSTER="bodaborg-v5p-nap"
export K8S_NAMESPACE="trellis"

export PATHWAYS_SERVER_IMAGE="${PATHWAYS_SERVER_IMAGE:-us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/datenglin/unsanitized_server:raiden_20260920_v2}"
export PATHWAYS_PROXY_IMAGE="${PATHWAYS_PROXY_IMAGE:-us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/datenglin/unsanitized_proxy_server:raiden_20260920_v2}"

# Model configuration
export MODEL_NAME="Qwen3.5-35B-A3B"
export MODEL_ID="Qwen/Qwen3.5-35B-A3B"
export TOKENIZER_PATH="Qwen/Qwen3.5-35B-A3B"
export MAXTEXT_MODEL_NAME="qwen3.5-35b-a3b"
export MAXTEXT_CKPT="${MAXTEXT_CKPT:-gs://maxtext-model-checkpoints/qwen3.5-35b-a3b/scanned/0/items}"
export SCAN_LAYERS="${SCAN_LAYERS:-true}"
export CHECKPOINT_STORAGE_USE_OCDBT="${CHECKPOINT_STORAGE_USE_OCDBT:-false}"
export CHECKPOINT_STORAGE_USE_ZARR3="${CHECKPOINT_STORAGE_USE_ZARR3:-false}"

# Rollout topology (Pathways 4-chip 2x2x1 slices, DP=2, FSDP=2, TP=2)
export WEIGHT_SYNC_MODE="none"
export ROLLOUT_JOBSET_YAML="jobset.pathways.yaml"
export ROLLOUT_TPU_SLICE="tpuv5:2x2x1"
export ROLLOUT_MESH_FSDP=2
export ROLLOUT_MESH_TP=2
export ROLLOUT_REPLICAS="${ROLLOUT_REPLICAS:-16}"
export VLLM_DATA_PARALLEL_SIZE=2
export ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-false}"

# Validation: 4 rollouts per instance, temperature 0.1, top_p 0.95
export DATASET_PATH="${DATASET_PATH:-gs://mlperf_dataset/benchmark-r2e-gym-easy}"
export DATASET_SPLIT="${DATASET_SPLIT:-validation}"
export NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
export TEMPERATURE="0.1"
export TOP_P="0.95"
export STEP_TIMEOUT_SECS=60
export REWARD_TIMEOUT_SECS=60

# Sandbox
export SANDBOX_NODE_SELECTOR_VAL="sandbox-cpu-pool"
export IMAGE_REWRITE_PREFIX="${IMAGE_REWRITE_PREFIX:-europe-west4-docker.pkg.dev/cloud-tpu-multipod-dev/tunix/}"

# MLPerf RCP offline eval (see top of file)
export CHECKPOINT_MANIFEST_FILE="${CHECKPOINT_MANIFEST_FILE:-}"
if [[ -n "${CHECKPOINT_MANIFEST_FILE}" ]]; then
  export RCP_LOGGING="${RCP_LOGGING:-true}"
else
  export METRIC_LOGGER_DIR="${METRIC_LOGGER_DIR:-${EVAL_OUTPUT_DIR%/}/mllog}"
fi
export CHECKPOINT_STEP="${CHECKPOINT_STEP:-18}"
export SAMPLES_COUNT="${SAMPLES_COUNT:-4608}"
export CHECKPOINT_TIMESTAMP_MS="${CHECKPOINT_TIMESTAMP_MS:-$(date +%s)000}"
export IS_LAST_CHECKPOINT="${IS_LAST_CHECKPOINT:-true}"

source "${DIR}/mlperf_base.sh" "${1:-eval}" "${@:2}"
