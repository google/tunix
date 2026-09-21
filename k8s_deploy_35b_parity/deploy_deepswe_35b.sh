#!/usr/bin/env bash
# ==============================================================================
# MLPerf v6.0 35B-A3B Multi-Turn DeepSWE Distributed GRPO Launcher
# Aligned with mlperf_35b_256.sh (bodaborg-v5p-nap) & MLPerf v6.0 Spec
# ==============================================================================
set -euo pipefail

# ------------------------------------------------------------------------------
# 1. GKE Cluster, Namespace, Kueue, and Machine Configuration (bodaborg-v5p-nap)
# ------------------------------------------------------------------------------
export PROJECT="${PROJECT:-cloud-tpu-shared-capacity}"
export REGION="${REGION:-europe-west4}"
export CLUSTER="${CLUSTER:-bodaborg-v5p-nap}"
kubectl config use-context "gke_${PROJECT}_${REGION}_${CLUSTER}" || true
kubectl config set-context --current --namespace="${K8S_NAMESPACE:-trellis}" || true

export K8S_NAMESPACE="${K8S_NAMESPACE:-trellis}"
export SANDBOX_NAMESPACE="${SANDBOX_NAMESPACE:-trellis}"
export KUEUE_QUEUE="${KUEUE_QUEUE:-default}"
export PRIORITY_CLASS="${PRIORITY_CLASS:-medium}"
export SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-xpk-sa}"
export CPU_MACHINE="${CPU_MACHINE:-n2d-standard-64}"

export RUN_PREFIX="${RUN_PREFIX:-${USER}-35b-parity}"
export ORCHESTRATOR_ID="${ORCHESTRATOR_ID:-${RUN_PREFIX}-orch}"
export TRAINER_ID="${TRAINER_ID:-${RUN_PREFIX}-trainer}"
export ROLLOUT_ID="${ROLLOUT_ID:-${RUN_PREFIX}-rollout}"

# ------------------------------------------------------------------------------
# 2. Container Image & Pathways / Raiden Images
# ------------------------------------------------------------------------------
export TUNIX_IMAGE="${TUNIX_IMAGE:-gcr.io/cloud-tpu-multipod-dev/${USER}/tunix_raiden_35b_parity:0918_v1}"
export PATHWAYS_SERVER_IMAGE="${PATHWAYS_SERVER_IMAGE:-us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/datenglin/unsanitized_server:raiden_20260914_fix}"
export PATHWAYS_PROXY_IMAGE="${PATHWAYS_PROXY_IMAGE:-us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/datenglin/unsanitized_proxy_server:raiden_20260914}"
export PATHWAYS_PROXY_MEMORY_LIMIT="${PATHWAYS_PROXY_MEMORY_LIMIT:-160G}"
export USER_CONTAINER_MEMORY="${USER_CONTAINER_MEMORY:-260G}"

# ------------------------------------------------------------------------------
# 3. GCS Bucket Configuration (Checkpoints, Pathways Scratch, Trajectory Logs)
# ------------------------------------------------------------------------------
export GCS_BUCKET="${GCS_BUCKET:-gs://hengtaoguo-maxtext-logs/${USER}/deepswe_35b_parity}"
export RUN_TAG="${RUN_TAG:-$(date +%m%d_%H%M)}"

# Base model checkpoint & tokenizer
export MODEL_NAME="${MODEL_NAME:-Qwen3.5-35B-A3B}"
export MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-35B-A3B}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-Qwen/Qwen3.5-35B-A3B}"
export MODEL_DIR="${MODEL_DIR:-}"
export MAXTEXT_MODEL_NAME="${MAXTEXT_MODEL_NAME:-qwen3.5-35b-a3b}"
export MAXTEXT_CKPT="${MAXTEXT_CKPT:-gs://hengtaoguo-maxtext-logs/checkpoints/qwen3.5-35b-a3b/scanned/2026-06-11-10-27/0/items}"

# Writable GCS destinations (all derived from GCS_BUCKET if not overridden):
export GCS_SCRATCH_LOCATION="${GCS_SCRATCH_LOCATION:-${GCS_BUCKET}/pathways_scratch}"
export MAXTEXT_OUTPUT_DIR="${MAXTEXT_OUTPUT_DIR:-${GCS_BUCKET}/runs/${RUN_TAG}/maxtext_output}"
export LOG_DIR="${LOG_DIR:-${GCS_BUCKET}/runs/${RUN_TAG}/logs}"
export TRAJECTORY_LOG_DIR="${TRAJECTORY_LOG_DIR:-${LOG_DIR}/trajectories}"

# ------------------------------------------------------------------------------
# 4. Weights & Biases (W&B) Configuration
# ------------------------------------------------------------------------------
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_ENTITY="${WANDB_ENTITY:-google-trellis}"
export WANDB_PROJECT="${WANDB_PROJECT:-trellis-deepswe}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-qwen35-35b-parity-${RUN_TAG}}"

# ------------------------------------------------------------------------------
# 5. TPU Slice Topology & Raiden Weight Sync (from mlperf_35b_256.sh)
# ------------------------------------------------------------------------------
export TRAINER_BACKEND="maxtext"
export SAMPLER="${SAMPLER:-vllm}"
export WEIGHT_SYNC_MODE="raiden"
export USE_WEIGHT_CONVERTER="true"
export PREFUSE_MOE_WEIGHTS="true"
export TRAINER_PREFUSE_MOE_WEIGHTS="true"
export ROLLOUT_PREFUSE_MOE_WEIGHTS="true"
export RAIDEN_DEVICES_PER_HOST="4"
export RAIDEN_USE_FFI="1"
export VERIFY_WEIGHTS="${VERIFY_WEIGHTS:-true}"
export TRAINER_PADDED_MOE_MLP_DIM="${TRAINER_PADDED_MOE_MLP_DIM:-}"

# Trainer: 64 chips (tpuv5:4x4x4) with FSDP=1, TP=2, EP=32
export TRAINER_JOBSET_YAML="jobset.pathways.yaml"
export TRAINER_TPU_SLICE="${TRAINER_TPU_SLICE:-tpuv5:4x4x4}"
export TRAINER_MESH_FSDP="${TRAINER_MESH_FSDP:-1}"
export TRAINER_MESH_TP="${TRAINER_MESH_TP:-2}"
export TRAINER_MESH_EXPERT="${TRAINER_MESH_EXPERT:-32}"

# Rollout: 16x 4-chip slices (tpuv5:2x2x2)
export ROLLOUT_JOBSET_YAML="jobset.tpu.yaml"
export ROLLOUT_TPU_SLICE="${ROLLOUT_TPU_SLICE:-tpuv5:2x2x2}"
export ROLLOUT_MESH_FSDP="${ROLLOUT_MESH_FSDP:-1}"
export ROLLOUT_MESH_TP="${ROLLOUT_MESH_TP:-1}"
export ROLLOUT_WORKERS="${ROLLOUT_WORKERS:-16}"
export ROLLOUT_REPLICAS="${ROLLOUT_REPLICAS:-16}"

# ------------------------------------------------------------------------------
# 6. MLPerf v6.0 Hyperparameters & Numerical Parity Flags
# ------------------------------------------------------------------------------
export BATCH_SIZE="${BATCH_SIZE:-16}"
export MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-${BATCH_SIZE}}"
export NUM_GENERATIONS="${NUM_GENERATIONS:-16}"
export NUM_ITERATIONS="${NUM_ITERATIONS:-1}"
export TRAIN_MICRO_BATCH_SIZE="${TRAIN_MICRO_BATCH_SIZE:-32}"
export MAX_STEPS="${MAX_STEPS:-100}"
export MAX_TURNS="${MAX_TURNS:-30}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-61440}"
export TRAINER_MAX_PROMPT_LENGTH="${TRAINER_MAX_PROMPT_LENGTH:-4096}"
export TRAINER_MAX_RESPONSE_LENGTH="${TRAINER_MAX_RESPONSE_LENGTH:-8192}"
export TUNIX_OUTLIER_DUMP_PATH="${TUNIX_OUTLIER_DUMP_PATH:-/tmp/outlier_dump}"
export COMPUTE_LOGPS_CHUNK_SIZE="${COMPUTE_LOGPS_CHUNK_SIZE:-512}"
export CHECKPOINT_SAVE_INTERVAL_STEPS="${CHECKPOINT_SAVE_INTERVAL_STEPS:-2}"
export CHECKPOINT_MAX_TO_KEEP="${CHECKPOINT_MAX_TO_KEEP:-10}"

# Sampling Parameters (MLPerf: temp=1.0, top_p=0.95, top_k=20)
export TEMPERATURE="${TEMPERATURE:-1.0}"
export TOP_P="${TOP_P:-0.95}"
export TOP_K="${TOP_K:-20}"

# Optimizer & Schedule + Freeze MoE Router Gate Weights
export LEARNING_RATE="${LEARNING_RATE:-1e-6}"
export ADAM_B1="${ADAM_B1:-0.9}"
export ADAM_B2="${ADAM_B2:-0.999}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
export MAX_GRAD_NORM="${MAX_GRAD_NORM:-0.125}"
export WARMUP_STEPS_FRACTION="${WARMUP_STEPS_FRACTION:-0.0}"
export LEARNING_RATE_FINAL_FRACTION="${LEARNING_RATE_FINAL_FRACTION:-1.0}"
export REMAT_POLICY="${REMAT_POLICY:-decoder}"
export TRAINABLE_PARAMETERS_MASK="${TRAINABLE_PARAMETERS_MASK:-[\"^(?!.*routed_experts/gate/kernel).*\"]}"

# GRPO / DAPO Loss & Sequence-Mask-TIS
export BETA="${BETA:-0.0}"
export EPSILON="${EPSILON:-0.2}"
export EPSILON_HIGH="${EPSILON_HIGH:-0.28}"
export ADVANTAGE_ESTIMATOR="${ADVANTAGE_ESTIMATOR:-grpo-loo}"
export LOSS_AGG_MODE="${LOSS_AGG_MODE:-token-mean}"
export FORCE_ON_POLICY_RATIO="${FORCE_ON_POLICY_RATIO:-true}"
export USE_ROLLOUT_LOGPS="${USE_ROLLOUT_LOGPS:-false}"
export OVERLONG_FILTER="${OVERLONG_FILTER:-true}"
export OVERLONG_LOSS_MASKING="${OVERLONG_LOSS_MASKING:-true}"
export SEQ_LOGPROB_ERROR_THRESHOLD="${SEQ_LOGPROB_ERROR_THRESHOLD:-2.0}"
export TRUNCATED_IMPORTANCE_SAMPLING_TYPE="${TRUNCATED_IMPORTANCE_SAMPLING_TYPE:-${TIS_TYPE:-seq-mask-tis}}"
export TRUNCATED_IMPORTANCE_SAMPLING_RATIO_MIN="${TRUNCATED_IMPORTANCE_SAMPLING_RATIO_MIN:-${TIS_RATIO_MIN:-0.999}}"
export TRUNCATED_IMPORTANCE_SAMPLING_RATIO="${TRUNCATED_IMPORTANCE_SAMPLING_RATIO:-${TIS_RATIO_MAX:-1.002}}"

# Numerical Parity Flags (FP32 Router Gate, FP32 Weight Sum, FP32 LM Head Logits, GDN/Norm FP32, Router Replay)
export FLOAT32_GATE_LOGITS="${FLOAT32_GATE_LOGITS:-true}"
export FLOAT32_WEIGHT_SUM="${FLOAT32_WEIGHT_SUM:-true}"
export FLOAT32_QK_PRODUCT="${FLOAT32_QK_PRODUCT:-true}"
export FLOAT32_LOGITS="${FLOAT32_LOGITS:-true}"
export ENABLE_ROUTER_REPLAY="${ENABLE_ROUTER_REPLAY:-1}"
export TRAINER_MAXTEXT_ATTENTION="${TRAINER_MAXTEXT_ATTENTION:-flash}"
export ROLLOUT_MAXTEXT_ATTENTION="${ROLLOUT_MAXTEXT_ATTENTION:-vllm_rpa}"

# vLLM Rollout Engine Configuration
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-65536}"
export VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-2048}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-8}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.9}"
export VLLM_DATA_PARALLEL_SIZE="${VLLM_DATA_PARALLEL_SIZE:-1}"
export VLLM_ENABLE_EXPERT_PARALLEL="${VLLM_ENABLE_EXPERT_PARALLEL:-true}"
export VLLM_ADDITIONAL_CONFIG="${VLLM_ADDITIONAL_CONFIG:-{\"sharding\":{\"sharding_strategy\":{\"expert_parallelism\":4,\"tensor_parallelism\":1,\"enable_dp_attention\":true}},\"custom_mamba_cache_multiplier\":16,\"maxtext_config\":{\"scan_layers\":false,\"attention\":\"vllm_rpa\",\"allow_split_physical_axes\":true,\"use_multimodal\":false,\"prefuse_moe_weights\":true}}}"
export ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-true}"
export VLLM_PREFIX_CACHE_RETENTION_INTERVAL="${VLLM_PREFIX_CACHE_RETENTION_INTERVAL:-256}"
export VLLM_KV_CACHE_DTYPE="${VLLM_KV_CACHE_DTYPE:-bfloat16}"
export VLLM_BLOCK_SIZE="${VLLM_BLOCK_SIZE:-256}"
export VLLM_ASYNC_SCHEDULING="${VLLM_ASYNC_SCHEDULING:-false}"
export VLLM_ENABLE_CHUNKED_PREFILL="${VLLM_ENABLE_CHUNKED_PREFILL:-true}"
export VLLM_LANGUAGE_MODEL_ONLY="${VLLM_LANGUAGE_MODEL_ONLY:-true}"
export VLLM_REASONING_PARSER="${VLLM_REASONING_PARSER:-qwen3}"
export VLLM_LIMIT_MM_PER_PROMPT="${VLLM_LIMIT_MM_PER_PROMPT:-{\"image\": 0, \"video\": 0}}"

# Rollout Worker Environment Flags
export NUM_PRECOMPILE_WORKERS="${NUM_PRECOMPILE_WORKERS:-8}"
export NEW_MODEL_DESIGN="${NEW_MODEL_DESIGN:-1}"
export ATTN_BUCKETIZED_NUM_REQS="${ATTN_BUCKETIZED_NUM_REQS:-true}"
export ATTN_CUSTOM_NUM_REQS_BUCKETS="${ATTN_CUSTOM_NUM_REQS_BUCKETS:-4}"
export ONEHOT_MOE_PERMUTE_THRESHOLD="${ONEHOT_MOE_PERMUTE_THRESHOLD:-32768}"
export VLLM_MOE_CHUNK_SIZE="${VLLM_MOE_CHUNK_SIZE:-256}"
export SLICE_ROPE_CACHE="${SLICE_ROPE_CACHE:-1}"
export DP_SCHED_BATCH_PREFILL="${DP_SCHED_BATCH_PREFILL:-false}"
export ROLLOUT_ENV_FLAGS="${ROLLOUT_ENV_FLAGS:-TPU_CHIPS_PER_HOST_BOUNDS=2,2,1 TPU_HOST_BOUNDS=1,1,1}"
export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:---deepsea_chips_per_host_bounds=2,2,1 --deepsea_host_bounds=1,1,1 --xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false --xla_tpu_check_legacy_constraints_in_reduce_scatter_legalizer=false}"
export VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-0}"

# DeepSWE Environment & Agent Sandbox (on namespace 'trellis')
export USE_AGENT_SANDBOX="${USE_AGENT_SANDBOX:-1}"
export SANDBOX_NODE_SELECTOR_KEY="${SANDBOX_NODE_SELECTOR_KEY:-cloud.google.com/gke-nodepool}"
export SANDBOX_NODE_SELECTOR_VAL="${SANDBOX_NODE_SELECTOR_VAL:-sandbox-cpu-pool}"
export MAX_WARMPOOL_REPLICAS="${MAX_WARMPOOL_REPLICAS:-2}"
export ROLLOUT_MAX_CONCURRENCY="${ROLLOUT_MAX_CONCURRENCY:-256}"
export MAX_CONCURRENCY="${MAX_CONCURRENCY:-256}"
export STEP_TIMEOUT_SECS="${STEP_TIMEOUT_SECS:-300}"
export REWARD_TIMEOUT_SECS="${REWARD_TIMEOUT_SECS:-180}"
export EPISODE_TIMEOUT_SECS="${EPISODE_TIMEOUT_SECS:-1800}"
export FLUSH_EVERY_N_STEPS="${FLUSH_EVERY_N_STEPS:-1}"
export DEBUG="${DEBUG:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/tunix/tunix/experimental/examples/deepswe_dist/k8s_launcher.sh" ]]; then
  LAUNCHER="${SCRIPT_DIR}/tunix/tunix/experimental/examples/deepswe_dist/k8s_launcher.sh"
elif [[ -f "${SCRIPT_DIR}/../tunix/experimental/examples/deepswe_dist/k8s_launcher.sh" ]]; then
  LAUNCHER="${SCRIPT_DIR}/../tunix/experimental/examples/deepswe_dist/k8s_launcher.sh"
else
  LAUNCHER="tunix/experimental/examples/deepswe_dist/k8s_launcher.sh"
fi

COMMAND="${1:-start}"
exec "${LAUNCHER}" --command "${COMMAND}" --image "${TUNIX_IMAGE}"
