#!/bin/bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Fill these before you run.
# k8s has a 63 char limit on total label name, so keep job_prefix unique to your job and short
export JOB_PREFIX="${JOB_PREFIX:-${USER}}"
export EVAL_JOBSET_NAME="${EVAL_JOBSET_NAME:-${JOB_PREFIX}-eval}"
export BUCKET="${BUCKET:-gs://atwigg-trellis-europe-west4-dev}"
export EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${BUCKET}/eval_results/${JOB_PREFIX}}"
export TRAJECTORY_LOG_DIR="${TRAJECTORY_LOG_DIR:-${BUCKET}/trajectories/${JOB_PREFIX}}"
export TUNIX_IMAGE="${TUNIX_IMAGE:-gcr.io/cloud-tpu-multipod-dev/sanbao/tunix_stack:eval}"

export REGION="europe-west4"
export CLUSTER="bodaborg-v5p-nap"
export K8S_NAMESPACE="trellis"

# Model configuration
export MODEL_NAME="Qwen3.5-35B-A3B"
export MODEL_ID="Qwen/Qwen3.5-35B-A3B"
export TOKENIZER_PATH="Qwen/Qwen3.5-35B-A3B"
export MAXTEXT_MODEL_NAME="qwen3.5-35b-a3b"
export MAXTEXT_CKPT="${MAXTEXT_CKPT:-gs://maxtext-model-checkpoints/qwen3.5-35b-a3b/scanned/0/items}"
export SCAN_LAYERS="${SCAN_LAYERS:-true}"
export CHECKPOINT_STORAGE_USE_OCDBT="${CHECKPOINT_STORAGE_USE_OCDBT:-false}"
export CHECKPOINT_STORAGE_USE_ZARR3="${CHECKPOINT_STORAGE_USE_ZARR3:-false}"
export EOS_TOKENS="${EOS_TOKENS:-151645,151643}"

# Backend & Rollout Topology (Pathways 4-chip 2x2x1 slices, mesh_fsdp=2, mesh_tp=2; no Trainer)
export WEIGHT_SYNC_MODE="none"
export ROLLOUT_JOBSET_YAML="jobset.pathways.yaml"
export ROLLOUT_TPU_SLICE="tpuv5:2x2x1"
export ROLLOUT_MESH_FSDP=2
export ROLLOUT_MESH_TP=2
export ROLLOUT_REPLICAS="${ROLLOUT_REPLICAS:-16}"

# ==============================================================================
# vLLM Rollout Configuration
# ==============================================================================
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.84}"

# Sharding Configs
export VLLM_DATA_PARALLEL_SIZE=2

# Prefix Caching Configs
export ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-false}"
export VLLM_PREFIX_CACHE_RETENTION_INTERVAL="${VLLM_PREFIX_CACHE_RETENTION_INTERVAL:-256}"
export VLLM_MAMBA_CACHE_MODE="${VLLM_MAMBA_CACHE_MODE:-${MAMBA_CACHE_MODE:-none}}"

# ==============================================================================
# Evaluation & DeepSWE Pipeline Configuration
# ==============================================================================
export NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
export BATCH_SIZE="${BATCH_SIZE:-64}"
export DATASET_SPLIT="${DATASET_SPLIT:-validation}"
export TASKS_LIMIT="${TASKS_LIMIT:-0}"

# Sampling Parameters
export TEMPERATURE="0.1"
export TOP_P="0.95"

# DeepSWE Environment & Agent Sandbox
export DATASET_PATH="${DATASET_PATH:-gs://mlperf_dataset/benchmark-r2e-gym-easy}"
export MAX_WARMPOOL_REPLICAS="${MAX_WARMPOOL_REPLICAS:-16}"
export MAX_CONCURRENCY="${MAX_CONCURRENCY:-256}"
export SANDBOX_NODE_SELECTOR_VAL="sandbox-cpu-pool"
export IMAGE_REWRITE_PREFIX="${IMAGE_REWRITE_PREFIX:-europe-west4-docker.pkg.dev/cloud-tpu-multipod-dev/tunix/}"
export ENABLE_THINKING="${ENABLE_THINKING:-false}"
export STEP_TIMEOUT_SECS=60
export REWARD_TIMEOUT_SECS=60
export MAX_CONTEXT_LIMIT=61440

source "${DIR}/mlperf_base.sh" "${1:-eval}" "${@:2}"
