#!/bin/bash
set -e

# ==============================================================================
# MLPerf DeepSWE evaluation recipe: Qwen3.5-397B-A17B on TPU v5p
# ==============================================================================
# - Cluster: bodaborg-v5p-nap in europe-west4
# - Rollout on 16 chips per replica (tpuv5p:2x2x4, EP=16, TP=1; no Trainer)
# - Sandboxes on sandbox-cpu-pool
# ==============================================================================

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Fill these before you run.
# k8s has a 63 char limit on total label name, so keep job_prefix unique to your job and short
export JOB_PREFIX="${JOB_PREFIX:-${USER}}"
export EVAL_JOBSET_NAME="${EVAL_JOBSET_NAME:-${JOB_PREFIX}-eval}"
export TUNIX_IMAGE="${TUNIX_IMAGE:-gcr.io/cloud-tpu-multipod-dev/sanbao/tunix_stack:eval}"

export BUCKET="${BUCKET:-gs://atwigg-trellis-europe-west4-dev}"
export EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${BUCKET}/eval_results/${JOB_PREFIX}}"
export TRAJECTORY_LOG_DIR="${TRAJECTORY_LOG_DIR:-${BUCKET}/trajectories/${JOB_PREFIX}/logger}"
export TRAJECTORY_STORE_ROOT_DIR="${TRAJECTORY_STORE_ROOT_DIR:-${TRAJECTORY_STORE_ROOT:-${BUCKET}/trajectories/${JOB_PREFIX}/store}}"

export REGION="${REGION:-europe-west4}"
export CLUSTER="${CLUSTER:-bodaborg-v5p-nap}"
export K8S_NAMESPACE="${K8S_NAMESPACE:-trellis}"

# Pathways Images and settings
export PATHWAYS_SERVER_IMAGE="${PATHWAYS_SERVER_IMAGE:-us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/datenglin/unsanitized_server:raiden_20260920_v2}"
export PATHWAYS_PROXY_IMAGE="${PATHWAYS_PROXY_IMAGE:-us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/datenglin/unsanitized_proxy_server:raiden_20260920_v2}"

# Model configuration
export MODEL_NAME="Qwen3.5-397B-A17B"
export MODEL_ID="Qwen/Qwen3.5-397B-A17B"
export TOKENIZER_PATH="${TOKENIZER_PATH:-Qwen/Qwen3.5-397B-A17B}"
export MAXTEXT_MODEL_NAME="qwen3.5-397b-a17b"
export MAXTEXT_CKPT="${MAXTEXT_CKPT:-gs://sanbao-europe/qwen35_397b/scanned_reshard_fsdp32_tp2/0/items}"
export SCAN_LAYERS="${SCAN_LAYERS:-true}"
export CHECKPOINT_STORAGE_USE_OCDBT="${CHECKPOINT_STORAGE_USE_OCDBT:-false}"
export CHECKPOINT_STORAGE_USE_ZARR3="${CHECKPOINT_STORAGE_USE_ZARR3:-false}"

# Backend & Rollout Topology (16 chips = 4 hosts per replica, EP=16, TP=1; no Trainer)
export WEIGHT_SYNC_MODE="none"
export ROLLOUT_JOBSET_YAML="${ROLLOUT_JOBSET_YAML:-jobset.pathways.yaml}"
export ROLLOUT_TPU_SLICE="${ROLLOUT_TPU_SLICE:-tpuv5p:2x2x4}"
_rollout_dims="${ROLLOUT_TPU_SLICE#*:}"
export VLLM_DATA_PARALLEL_SIZE="${VLLM_DATA_PARALLEL_SIZE:-1}"
export ROLLOUT_MESH_EXPERT="${ROLLOUT_MESH_EXPERT:-$(( ${_rollout_dims//x/*} / ${VLLM_DATA_PARALLEL_SIZE:-1} ))}"
export ROLLOUT_REPLICAS="${ROLLOUT_REPLICAS:-16}"

# ==============================================================================
# vLLM Rollout Configuration
# ==============================================================================
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.84}"

# Sharding Configs
export VLLM_ADDITIONAL_CONFIG='{"sharding":{"sharding_strategy":{"expert_parallelism":'"${ROLLOUT_MESH_EXPERT}"',"tensor_parallelism":1,"enable_dp_attention":true}},"custom_mamba_cache_multiplier":16,"maxtext_config":{"scan_layers":false,"attention":"vllm_rpa","allow_split_physical_axes":true,"use_multimodal":false,"prefuse_moe_weights":true,"per_device_batch_size":0.0}}'

# ==============================================================================
# Rollout Worker Environment Flags (Optimizations & Runtime Settings)
# ==============================================================================
export ONEHOT_MOE_PERMUTE_THRESHOLD=131072
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
export VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY="RAIDEN_,TPU_"
export VLLM_RAY_EXTRA_ENV_VARS_TO_COPY="ONEHOT_MOE_PERMUTE_THRESHOLD,LIBTPU_INIT_ARGS,RAY_memory_monitor_refresh_ms,VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS,ENABLE_MULTI_NUMA,TPU_RAIDEN_DATA_NICS,FLOAT32_GATE_LOGITS,FLOAT32_LOGITS,NEW_MODEL_DESIGN,ATTN_BUCKETIZED_NUM_REQS,ATTN_CUSTOM_NUM_REQS_BUCKETS,VLLM_MOE_CHUNK_SIZE,SLICE_ROPE_CACHE,DP_SCHED_BATCH_PREFILL"
export ROLLOUT_EXTRA_ENV="${ROLLOUT_EXTRA_ENV:-ONEHOT_MOE_PERMUTE_THRESHOLD=131072 RAY_memory_monitor_refresh_ms=0 RAIDEN_TRANSPORT_COALESCE_WINDOW_BYTES=67108864 RAIDEN_WEIGHT_SYNC_PIPELINE_GROUP_SIZE=16 RAIDEN_PARALLELISM=16 VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800}"
export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:- --xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false --xla_tpu_check_legacy_constraints_in_reduce_scatter_legalizer=false}"
export PATHWAYS_WORKER_EXTRA_ENV="${PATHWAYS_WORKER_EXTRA_ENV:-LIBTPU_INIT_ARGS=${LIBTPU_INIT_ARGS} --megascale_port=-1 --xprof_compress_jftrace=true
SKIP_MEGASCALE_PJRT_CLIENT=true}"
_rollout_xla_flags=""
for _f in ${LIBTPU_INIT_ARGS}; do [[ "${_f}" == --xla_* ]] && _rollout_xla_flags+="${_f} "; done
export PATHWAYS_PROXY_EXTRA_ARGS="${PATHWAYS_PROXY_EXTRA_ARGS:-${_rollout_xla_flags% }}"

# ==============================================================================
# Evaluation & DeepSWE Pipeline Configuration
# ==============================================================================
export NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
export DATASET_SPLIT="${DATASET_SPLIT:-validation}"
export TASKS_LIMIT="${TASKS_LIMIT:-0}"

# Sampling Parameters
export TEMPERATURE="0.1"
export TOP_P="0.95"

export DEBUG=${DEBUG:-0}

# DeepSWE Environment & Agent Sandbox
export DATASET_PATH="${DATASET_PATH:-gs://mlperf_dataset/benchmark-r2e-gym-easy}"
export SANDBOX_NODE_SELECTOR_VAL="${SANDBOX_NODE_SELECTOR_VAL:-sandbox-cpu-pool}"
export IMAGE_REWRITE_PREFIX="${IMAGE_REWRITE_PREFIX:-europe-west4-docker.pkg.dev/cloud-tpu-multipod-dev/tunix/}"
export ENABLE_THINKING="${ENABLE_THINKING:-false}"
export STEP_TIMEOUT_SECS=60
export REWARD_TIMEOUT_SECS=60
export MAX_CONTEXT_LIMIT="${MAX_CONTEXT_LIMIT:-61440}"

source "${DIR}/mlperf_base.sh" "${1:-eval}" "${@:2}"
