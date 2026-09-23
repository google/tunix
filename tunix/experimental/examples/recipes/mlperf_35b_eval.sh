#!/bin/bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Fill these before you run.
# k8s has a 63 char limit on total label name, so keep job_prefix unique to your job and short
export JOB_PREFIX="${JOB_PREFIX:-${USER}}"
export EVAL_JOBSET_NAME="${EVAL_JOBSET_NAME:-${JOB_PREFIX}-eval}"
export EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-gs://atwigg-trellis-europe-west4-dev/eval_results/${JOB_PREFIX}}"
export TRAJECTORY_LOG_DIR="${TRAJECTORY_LOG_DIR:-gs://atwigg-trellis-europe-west4-dev/trajectories/${JOB_PREFIX}}"
export ROLLOUT_PORT="${ROLLOUT_PORT:-20001}"
export TUNIX_IMAGE="${TUNIX_IMAGE:-gcr.io/cloud-tpu-multipod-dev/sanbao/tunix_stack:eval}"

export PROJECT="cloud-tpu-shared-capacity"
export REGION="europe-west4"
export CLUSTER="bodaborg-v5p-nap"
kubectl config use-context "gke_${PROJECT}_${REGION}_${CLUSTER}" || true
kubectl config set-context --current --namespace=trellis || true

export K8S_NAMESPACE="trellis"
export KUEUE_QUEUE="${KUEUE_QUEUE:-multislice-queue}"
export PRIORITY_CLASS="${PRIORITY_CLASS:-medium}"
export SERVICE_ACCOUNT="xpk-sa"
export CPU_MACHINE="n2d-standard-64"

# Pathways Images and settings
export PATHWAYS_SERVER_IMAGE="${PATHWAYS_SERVER_IMAGE:-us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/datenglin/unsanitized_server:raiden_20260920_v2}"
export PATHWAYS_PROXY_IMAGE="${PATHWAYS_PROXY_IMAGE:-us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/datenglin/unsanitized_proxy_server:raiden_20260920_v2}"
export PATHWAYS_PROXY_MEMORY_LIMIT="160G"
export USER_CONTAINER_MEMORY="260G"
export USER_CONTAINER_MEMORY_LIMIT="${USER_CONTAINER_MEMORY_LIMIT:-260G}"
export PREFUSE_MOE_WEIGHTS="true"
export ROLLOUT_PREFUSE_MOE_WEIGHTS="true"

# Model configuration
export MODEL_NAME="Qwen3.5-35B-A3B"
export MODEL_ID="Qwen/Qwen3.5-35B-A3B"
export TOKENIZER_PATH="Qwen/Qwen3.5-35B-A3B"
export MAXTEXT_MODEL_NAME="qwen3.5-35b-a3b"
export MAXTEXT_CKPT="${MAXTEXT_CKPT:-gs://hengtaoguo-maxtext-logs/checkpoints/qwen3.5-35b-a3b/scanned/2026-06-11-10-27/0/items}"
export EOS_TOKENS="${EOS_TOKENS:-151645,151643}"

# Backend & Rollout Topology (Pathways 4-chip 2x2x1 slice, mesh_fsdp=2, mesh_tp=2; no Trainer)
export SAMPLER="vllm"
export WEIGHT_SYNC_MODE="none"
export ROLLOUT_JOBSET_YAML="jobset.pathways.yaml"
export ROLLOUT_TPU_SLICE="tpuv5:2x2x1"
export ROLLOUT_MESH_FSDP=2
export ROLLOUT_MESH_TP=2

# ==============================================================================
# vLLM Rollout Configuration
# ==============================================================================
export VLLM_LOGGING_LEVEL="INFO"
export VLLM_MAX_MODEL_LEN=65536
export VLLM_MAX_NUM_BATCHED_TOKENS=2048
export VLLM_MAX_NUM_SEQS=16
export VLLM_GPU_MEMORY_UTILIZATION="0.9"

# Sharding Configs
export VLLM_DATA_PARALLEL_SIZE=2
export VLLM_ENABLE_EXPERT_PARALLEL="true"

# Prefix Caching Configs
export ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-false}"
export VLLM_PREFIX_CACHE_RETENTION_INTERVAL="${VLLM_PREFIX_CACHE_RETENTION_INTERVAL:-256}"
export VLLM_MAMBA_CACHE_MODE="${VLLM_MAMBA_CACHE_MODE:-${MAMBA_CACHE_MODE:-none}}"

# KV Cache Configs
export ROLLOUT_FREE_KV_CACHE="false"
export VLLM_KV_CACHE_DTYPE="bfloat16"
export VLLM_BLOCK_SIZE=256

# Engine Configs
export VLLM_ASYNC_SCHEDULING="true"
export VLLM_ENABLE_CHUNKED_PREFILL="true"

# Model Configs
export VLLM_LANGUAGE_MODEL_ONLY="true"
export VLLM_REASONING_PARSER="qwen3"
export VLLM_LIMIT_MM_PER_PROMPT='{"image": 0, "video": 0}'

# ==============================================================================
# Rollout Worker Environment Flags (Optimizations & Runtime Settings)
# ==============================================================================
export NUM_PRECOMPILE_WORKERS=8
export NEW_MODEL_DESIGN=1
export ATTN_BUCKETIZED_NUM_REQS=true
export ATTN_CUSTOM_NUM_REQS_BUCKETS=4
export ONEHOT_MOE_PERMUTE_THRESHOLD=32768
export VLLM_MOE_CHUNK_SIZE=256
export SLICE_ROPE_CACHE=1
export DP_SCHED_BATCH_PREFILL=false
export LIBTPU_INIT_ARGS=' --xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false --xla_tpu_check_legacy_constraints_in_reduce_scatter_legalizer=false'
export VLLM_ENABLE_V1_MULTIPROCESSING=0

# ==============================================================================
# Evaluation & DeepSWE Pipeline Configuration
# ==============================================================================
export NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
export DATASET_SPLIT="${DATASET_SPLIT:-validation}"

# Sampling Parameters
export TEMPERATURE="1.0"
export TOP_P="1.0"
export TOP_K="-1"

export EPISODE_TIMEOUT_SECS=1800
export DEBUG=1

# DeepSWE Environment & Agent Sandbox
export DATASET_PATH="gs://mlperf_dataset/benchmark-r2e-gym-easy"
export USE_AGENT_SANDBOX=1
export SCAFFOLD="openhands"
export SANDBOX_NAMESPACE="${SANDBOX_NAMESPACE:-trellis}"
export POOL_NAME_FORMAT="${POOL_NAME_FORMAT:-}"
export TEMPLATE_NAME_PREFIX="${TEMPLATE_NAME_PREFIX:-}"
export SANDBOX_NODE_SELECTOR_KEY="cloud.google.com/gke-nodepool"
export SANDBOX_NODE_SELECTOR_VAL="sandbox-cpu-pool"
export IMAGE_REWRITE_PREFIX="${IMAGE_REWRITE_PREFIX:-europe-west4-docker.pkg.dev/cloud-tpu-multipod-dev/tunix/}"
export MAX_WARMPOOL_REPLICAS=2
export ROLLOUT_MAX_CONCURRENCY=256
export MAX_CONCURRENCY=256
export STEP_TIMEOUT_SECS=300
export REWARD_TIMEOUT_SECS=180
export FLUSH_EVERY_N_STEPS=1
export MAX_TURNS=30
export MAX_PROMPT_LENGTH=4096
export MAX_RESPONSE_LENGTH=61440

# ==============================================================================
# Execution Dispatch
# ==============================================================================
if [ -f "${DIR}/../deepswe_dist/k8s_launcher.sh" ]; then
  LAUNCHER="${DIR}/../deepswe_dist/k8s_launcher.sh"
elif [ -f "${DIR}/tunix/experimental/examples/deepswe_dist/k8s_launcher.sh" ]; then
  LAUNCHER="${DIR}/tunix/experimental/examples/deepswe_dist/k8s_launcher.sh"
elif [ -f "${DIR}/../../../../third_party/py/tunix/experimental/examples/deepswe_dist/k8s_launcher.sh" ]; then
  LAUNCHER="${DIR}/../../../../third_party/py/tunix/experimental/examples/deepswe_dist/k8s_launcher.sh"
elif [ -f "${HOME}/github/tunix_build/tunix/experimental/examples/deepswe_dist/k8s_launcher.sh" ]; then
  LAUNCHER="${HOME}/github/tunix_build/tunix/experimental/examples/deepswe_dist/k8s_launcher.sh"
else
  echo "Error: k8s_launcher.sh not found relative to ${DIR}"
  exit 1
fi

COMMAND="${1:-eval}"
exec "${LAUNCHER}" --command "${COMMAND}" --image "${TUNIX_IMAGE}"
