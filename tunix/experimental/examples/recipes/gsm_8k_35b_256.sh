#!/bin/bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ==============================================================================
# Recipe: Qwen3.5-35B-A3B Distributed GSM8K (256 Concurrency / 128 TPU Chips)
# ==============================================================================
# Architecture:
# - Trainer: MaxText on Pathways (64 TPU v5p chips, 4x4x4, FSDP=32, TP=2, Expert=1)
#   with Raiden FFI enabled (RAIDEN_USE_FFI=1, 4 devices/host).
# - Rollout: 16 vLLM replicas (4 TPU v5p chips each, 2x2x1, 64 TPU chips total)
#   with Expert Parallelism (EP=4, TP=1, enable_dp_attention=true), TCP transport.
# - Weight Sync: Raiden native TPU sync with weight converter & prefused MoE weights.
# - Concurrency: Batch size 16 * 16 generations = 256 trajectories per step.
#
# GSM8K uses in-process math reward scoring without agent sandboxes or warmpools,
# providing a lightweight, low-friction setup to quickly test trainer and sampler
# configurations at full scale.
# ==============================================================================

set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Job Identification & Cloud Storage ---
export JOB_PREFIX="${JOB_PREFIX:-${USER}-$(date +%Y%m%d-%H%M%S)}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-${JOB_PREFIX}-gsm8k-35b-256}"
export MAXTEXT_OUTPUT_DIR="${MAXTEXT_OUTPUT_DIR:-gs://atwigg-trellis-europe-west4-dev/maxtext/${JOB_PREFIX}}"
export TRAJECTORY_LOG_DIR="${TRAJECTORY_LOG_DIR:-gs://atwigg-trellis-europe-west4-dev/trajectories/${JOB_PREFIX}}"
export ORCHESTRATOR_PORT="${ORCHESTRATOR_PORT:-20000}"
export ROLLOUT_PORT="${ROLLOUT_PORT:-20001}"
export TRAINER_PORT="${TRAINER_PORT:-20002}"
export PROFILER_STEPS="${PROFILER_STEPS:-0}"
export SKIP_FIRST_N_PROFILER_STEPS="${SKIP_FIRST_N_PROFILER_STEPS:--1}"
export TUNIX_IMAGE="${TUNIX_IMAGE:-gcr.io/cloud-tpu-multipod-dev/atwigg/trellis-35b:latest}"

# --- Cluster & Kubernetes Infrastructure ---
export PROJECT="${PROJECT:-cloud-tpu-shared-capacity}"
export REGION="${REGION:-europe-west4}"
export CLUSTER="${CLUSTER:-bodaborg-v5p-nap}"
if [[ "${DRY_RUN:-false}" != "true" ]]; then
  kubectl config use-context "gke_${PROJECT}_${REGION}_${CLUSTER}" || true
  kubectl config set-context --current --namespace=trellis || true
fi

export K8S_NAMESPACE="${K8S_NAMESPACE:-trellis}"
export KUEUE_QUEUE="${KUEUE_QUEUE:-multislice-queue}"
export KUEUE_QUEUE_NAME="${KUEUE_QUEUE_NAME:-${KUEUE_QUEUE}}"
export PRIORITY_CLASS="${PRIORITY_CLASS:-medium}"
export SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-xpk-sa}"
export CPU_MACHINE="${CPU_MACHINE:-n2d-standard-64}"

# --- Pathways & Raiden Settings ---
export PATHWAYS_SERVER_IMAGE="${PATHWAYS_SERVER_IMAGE:-us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/datenglin/unsanitized_server:raiden_20260920_v2}"
export PATHWAYS_PROXY_IMAGE="${PATHWAYS_PROXY_IMAGE:-us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/datenglin/unsanitized_proxy_server:raiden_20260920_v2}"
export PATHWAYS_PROXY_MEMORY_LIMIT="${PATHWAYS_PROXY_MEMORY_LIMIT:-160G}"
export USER_CONTAINER_MEMORY="${USER_CONTAINER_MEMORY:-260G}"
export USER_CONTAINER_MEMORY_LIMIT="${USER_CONTAINER_MEMORY_LIMIT:-260G}"
export RAIDEN_DEVICES_PER_HOST="${RAIDEN_DEVICES_PER_HOST:-4}"
export USE_WEIGHT_CONVERTER="${USE_WEIGHT_CONVERTER:-true}"
export PREFUSE_MOE_WEIGHTS="${PREFUSE_MOE_WEIGHTS:-true}"
export TRAINER_PREFUSE_MOE_WEIGHTS="${TRAINER_PREFUSE_MOE_WEIGHTS:-true}"
export ROLLOUT_PREFUSE_MOE_WEIGHTS="${ROLLOUT_PREFUSE_MOE_WEIGHTS:-true}"
export VERIFY_WEIGHTS="${VERIFY_WEIGHTS:-true}"
export TRAINER_PADDED_MOE_MLP_DIM="${TRAINER_PADDED_MOE_MLP_DIM:-}"

# --- WandB Monitoring ---
export WANDB_ENTITY="${WANDB_ENTITY:-google-trellis}"
export WANDB_PROJECT="${WANDB_PROJECT:-trellis-gsm8k}"

# --- Model Configuration ---
export MODEL_NAME="${MODEL_NAME:-Qwen3.5-35B-A3B}"
export MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-35B-A3B}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-Qwen/Qwen3.5-35B-A3B}"
export MAXTEXT_MODEL_NAME="${MAXTEXT_MODEL_NAME:-qwen3.5-35b-a3b}"

export RETURN_ROUTED_EXPERTS="${RETURN_ROUTED_EXPERTS:-false}"

if [[ "${MAXTEXT_MODEL_NAME}" == "qwen3-0.6b" || "${MODEL_NAME}" == "Qwen3-0.6B" ]]; then
  export MAXTEXT_CKPT="${MAXTEXT_CKPT:-gs://maxtext-model-checkpoints/qwen3-0.6b/2025-10-27/scanned/0/items}"
  export TRAINER_BASE_NUM_KV_HEADS="${TRAINER_BASE_NUM_KV_HEADS:-8}"
  export PREFUSE_MOE_WEIGHTS="${PREFUSE_MOE_WEIGHTS:-false}"
  export TRAINER_PREFUSE_MOE_WEIGHTS="${TRAINER_PREFUSE_MOE_WEIGHTS:-false}"
  export ROLLOUT_PREFUSE_MOE_WEIGHTS="${ROLLOUT_PREFUSE_MOE_WEIGHTS:-false}"
  export RETURN_ROUTED_EXPERTS="${RETURN_ROUTED_EXPERTS:-false}"
  export VLLM_ENABLE_EXPERT_PARALLEL="${VLLM_ENABLE_EXPERT_PARALLEL:-false}"
  export VLLM_PREFIX_CACHE_RETENTION_INTERVAL="${VLLM_PREFIX_CACHE_RETENTION_INTERVAL:-0}"
  if [[ -z "${VLLM_ADDITIONAL_CONFIG:-}" ]]; then
    export VLLM_ADDITIONAL_CONFIG='{"maxtext_config":{"scan_layers":false,"attention":"vllm_rpa","allow_split_physical_axes":true,"use_multimodal":false,"prefuse_moe_weights":false}}'
  fi
  export TRAINABLE_PARAMETERS_MASK="${TRAINABLE_PARAMETERS_MASK:-.*}"
else
  export MAXTEXT_CKPT="${MAXTEXT_CKPT:-gs://hengtaoguo-maxtext-logs/checkpoints/qwen3.5-35b-a3b/scanned/2026-06-11-10-27/0/items}"
  export TRAINER_BASE_NUM_KV_HEADS="${TRAINER_BASE_NUM_KV_HEADS:-2}"
  export TRAINABLE_PARAMETERS_MASK="${TRAINABLE_PARAMETERS_MASK:-^(?!.*routed_experts/gate/kernel).*}"
  export VLLM_ENABLE_EXPERT_PARALLEL="${VLLM_ENABLE_EXPERT_PARALLEL:-true}"
  export VLLM_PREFIX_CACHE_RETENTION_INTERVAL="${VLLM_PREFIX_CACHE_RETENTION_INTERVAL:-0}"
  if [[ -z "${VLLM_ADDITIONAL_CONFIG:-}" ]]; then
    export VLLM_ADDITIONAL_CONFIG='{"sharding":{"sharding_strategy":{"expert_parallelism":4,"tensor_parallelism":1,"enable_dp_attention":true}},"custom_mamba_cache_multiplier":16,"maxtext_config":{"scan_layers":false,"attention":"vllm_rpa","allow_split_physical_axes":true,"use_multimodal":false,"prefuse_moe_weights":true}}'
  fi
fi

# Qwen3.5 <|im_end|>, <|endoftext|> (generation_config.json eos_token_id).
# 151645/151643 are the Qwen2.5/Qwen3 ids and are ordinary tokens in the
# Qwen3.5 vocab.
export EOS_TOKENS="${EOS_TOKENS:-248046,248044}"

# --- Backend Configuration ---
export TRAINER_BACKEND="${TRAINER_BACKEND:-maxtext}"
export SAMPLER="${SAMPLER:-vllm}"
export WEIGHT_SYNC_MODE="${WEIGHT_SYNC_MODE:-raiden}"

# --- Topologies (64-chip Trainer 4x4x4, 16x 4-chip Rollout slices) ---
export TRAINER_JOBSET_YAML="${TRAINER_JOBSET_YAML:-jobset.pathways.yaml}"
export TRAINER_TPU_SLICE="${TRAINER_TPU_SLICE:-tpuv5:4x4x4}"
export TRAINER_MESH_FSDP="${TRAINER_MESH_FSDP:-32}"
export TRAINER_MESH_TP="${TRAINER_MESH_TP:-2}"
export TRAINER_MESH_EXPERT="${TRAINER_MESH_EXPERT:-1}"

export ROLLOUT_JOBSET_YAML="${ROLLOUT_JOBSET_YAML:-jobset.tpu.yaml}"
export ROLLOUT_TPU_SLICE="${ROLLOUT_TPU_SLICE:-tpuv5:2x2x1}"
export ROLLOUT_MESH_FSDP="${ROLLOUT_MESH_FSDP:-1}"
export ROLLOUT_MESH_TP="${ROLLOUT_MESH_TP:-1}"
export ROLLOUT_WORKERS="${ROLLOUT_WORKERS:-16}"
export ROLLOUT_REPLICAS="${ROLLOUT_REPLICAS:-16}"

# ==============================================================================
# vLLM Rollout Configuration
# ==============================================================================
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-65536}"
export VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-2048}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-16}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.9}"

# Sharding Configs
export VLLM_DATA_PARALLEL_SIZE="${VLLM_DATA_PARALLEL_SIZE:-1}"

# Prefix Caching Configs
export ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-false}"
export VLLM_MAMBA_CACHE_MODE="${VLLM_MAMBA_CACHE_MODE:-none}"

# KV Cache Configs
export ROLLOUT_FREE_KV_CACHE="${ROLLOUT_FREE_KV_CACHE:-false}"
export VLLM_KV_CACHE_DTYPE="${VLLM_KV_CACHE_DTYPE:-bfloat16}"
export VLLM_BLOCK_SIZE="${VLLM_BLOCK_SIZE:-256}"

# Engine Configs
export VLLM_ASYNC_SCHEDULING="${VLLM_ASYNC_SCHEDULING:-true}"
export VLLM_ENABLE_CHUNKED_PREFILL="${VLLM_ENABLE_CHUNKED_PREFILL:-true}"

# Model Configs
export VLLM_LANGUAGE_MODEL_ONLY="${VLLM_LANGUAGE_MODEL_ONLY:-true}"
export VLLM_REASONING_PARSER="${VLLM_REASONING_PARSER:-qwen3}"
if [[ -z "${VLLM_LIMIT_MM_PER_PROMPT:-}" ]]; then
  export VLLM_LIMIT_MM_PER_PROMPT='{"image": 0, "video": 0}'
fi

# ==============================================================================
# Rollout Worker Environment Flags (Optimizations & Runtime Settings)
# ==============================================================================
export NUM_PRECOMPILE_WORKERS="${NUM_PRECOMPILE_WORKERS:-8}"
export NEW_MODEL_DESIGN="${NEW_MODEL_DESIGN:-1}"
export ATTN_BUCKETIZED_NUM_REQS="${ATTN_BUCKETIZED_NUM_REQS:-true}"
export ATTN_CUSTOM_NUM_REQS_BUCKETS="${ATTN_CUSTOM_NUM_REQS_BUCKETS:-4}"
export ONEHOT_MOE_PERMUTE_THRESHOLD="${ONEHOT_MOE_PERMUTE_THRESHOLD:-32768}"
export VLLM_MOE_CHUNK_SIZE="${VLLM_MOE_CHUNK_SIZE:-256}"
export SLICE_ROPE_CACHE="${SLICE_ROPE_CACHE:-1}"
export DP_SCHED_BATCH_PREFILL="${DP_SCHED_BATCH_PREFILL:-false}"
export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:- --xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false --xla_tpu_check_legacy_constraints_in_reduce_scatter_legalizer=false}"
export VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-0}"

# ==============================================================================
# Hyperparameters & Loss Configuration
# ==============================================================================
export MAX_STEPS="${MAX_STEPS:-50}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
export MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-${BATCH_SIZE}}"
export NUM_GENERATIONS="${NUM_GENERATIONS:-16}"
_TOTAL_TRAJECTORIES=$(( MINI_BATCH_SIZE * NUM_GENERATIONS ))
if [[ -z "${TRAIN_MICRO_BATCH_SIZE:-}" ]]; then
  if (( _TOTAL_TRAJECTORIES < 32 )); then
    export TRAIN_MICRO_BATCH_SIZE="${_TOTAL_TRAJECTORIES}"
  else
    export TRAIN_MICRO_BATCH_SIZE=32
  fi
else
  export TRAIN_MICRO_BATCH_SIZE="${TRAIN_MICRO_BATCH_SIZE}"
fi
export CHECKPOINT_SAVE_INTERVAL_STEPS="${CHECKPOINT_SAVE_INTERVAL_STEPS:-0}"
export CHECKPOINT_MAX_TO_KEEP="${CHECKPOINT_MAX_TO_KEEP:-10}"
export MAX_STALENESS="${MAX_STALENESS:-0}"

# Sampling Parameters
export TEMPERATURE="${TEMPERATURE:-1.0}"
export TOP_P="${TOP_P:-1.0}"
export TOP_K="${TOP_K:--1}"

# Algorithmic & Loss Hyperparameters
export BETA="${BETA:-0.0}"
export EPSILON="${EPSILON:-0.2}"
export EPSILON_HIGH="${EPSILON_HIGH:-0.28}"
export USE_ROLLOUT_LOGPS="${USE_ROLLOUT_LOGPS:-false}"
export OVERLONG_FILTER="${OVERLONG_FILTER:-true}"
export OVERLONG_LOSS_MASKING="${OVERLONG_LOSS_MASKING:-true}"
export SEQ_LOGPROB_ERROR_THRESHOLD="${SEQ_LOGPROB_ERROR_THRESHOLD:-2.0}"
export TRUNCATED_IMPORTANCE_SAMPLING_TYPE="${TRUNCATED_IMPORTANCE_SAMPLING_TYPE:-seq-mask-tis}"
export TRUNCATED_IMPORTANCE_SAMPLING_RATIO_MIN="${TRUNCATED_IMPORTANCE_SAMPLING_RATIO_MIN:-0.999}"
export TRUNCATED_IMPORTANCE_SAMPLING_RATIO="${TRUNCATED_IMPORTANCE_SAMPLING_RATIO:-1.002}"
export ADVANTAGE_ESTIMATOR="${ADVANTAGE_ESTIMATOR:-grpo-loo}"
export LOSS_AGG_MODE="${LOSS_AGG_MODE:-token-mean}"
export FLOAT32_GATE_LOGITS="${FLOAT32_GATE_LOGITS:-true}"
export FLOAT32_LOGITS="${FLOAT32_LOGITS:-true}"

# Optimizer Hyperparameters
export LEARNING_RATE="${LEARNING_RATE:-1e-6}"
export ADAM_B1="${ADAM_B1:-0.9}"
export ADAM_B2="${ADAM_B2:-0.999}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
export MAX_GRAD_NORM="${MAX_GRAD_NORM:-0.125}"
export WARMUP_STEPS_FRACTION="${WARMUP_STEPS_FRACTION:-0.0}"
export LEARNING_RATE_FINAL_FRACTION="${LEARNING_RATE_FINAL_FRACTION:-1.0}"

# Architecture & Rematerialization
export REMAT_POLICY="${REMAT_POLICY:-full}"
export TRAINER_MAXTEXT_ATTENTION="${TRAINER_MAXTEXT_ATTENTION:-flash}"
export COMPUTE_LOGPS_CHUNK_SIZE="${COMPUTE_LOGPS_CHUNK_SIZE:-512}"

export EPISODE_TIMEOUT_SECS="${EPISODE_TIMEOUT_SECS:-1800}"
export DEBUG="${DEBUG:-1}"

# ==============================================================================
# GSM8K Dataset & Sequence Parameters (No Sandboxing)
# ==============================================================================
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
export ROLLOUT_MAX_CONCURRENCY="${ROLLOUT_MAX_CONCURRENCY:-256}"
export MAX_CONCURRENCY="${MAX_CONCURRENCY:-${ROLLOUT_MAX_CONCURRENCY}}"
export FLUSH_EVERY_N_STEPS="${FLUSH_EVERY_N_STEPS:-1}"
export FLUSH_METRICS_EVERY_N_STEPS="${FLUSH_METRICS_EVERY_N_STEPS:-${FLUSH_EVERY_N_STEPS}}"
export REWARD_MODE="${REWARD_MODE:-env}"
export TFDS_DATA_DIR="${TFDS_DATA_DIR:-/tmp/gsm8k_data}"
export TFDS_SPLIT="${TFDS_SPLIT:-train}"
export SHUFFLE="${SHUFFLE:-true}"
export SEED="${SEED:-42}"

# ==============================================================================
# Worker Passthrough Environments & Arguments
# ==============================================================================
export ORCHESTRATOR_EXTRA_ENV="${ORCHESTRATOR_EXTRA_ENV:-} WANDB_ENTITY=\"${WANDB_ENTITY}\""

export TRAINER_EXTRA_ENV="${TRAINER_EXTRA_ENV:-} RAIDEN_DEVICES_PER_HOST=${RAIDEN_DEVICES_PER_HOST} USE_WEIGHT_CONVERTER=${USE_WEIGHT_CONVERTER} PREFUSE_MOE_WEIGHTS=${TRAINER_PREFUSE_MOE_WEIGHTS} ROLLOUT_PREFUSE_MOE_WEIGHTS=${ROLLOUT_PREFUSE_MOE_WEIGHTS} ROLLOUT_MESH_TP=${ROLLOUT_MESH_TP} ROLLOUT_TENSOR_PARALLEL_SIZE=${ROLLOUT_MESH_TP} FLOAT32_GATE_LOGITS=${FLOAT32_GATE_LOGITS} FLOAT32_LOGITS=${FLOAT32_LOGITS}"
export TRAINER_EXTRA_ARGS="${TRAINER_EXTRA_ARGS:-} --prefuse_moe_weights=${TRAINER_PREFUSE_MOE_WEIGHTS} --trainable_parameters_mask='${TRAINABLE_PARAMETERS_MASK}' --remat_policy=${REMAT_POLICY} --maxtext_attention=${TRAINER_MAXTEXT_ATTENTION} --base_num_kv_heads=${TRAINER_BASE_NUM_KV_HEADS} --maxtext_warmup_steps_fraction=${WARMUP_STEPS_FRACTION} --learning_rate_final_fraction=${LEARNING_RATE_FINAL_FRACTION}"

ROLLOUT_RAIDEN_ENV=""
if [[ "${WEIGHT_SYNC_MODE}" == "raiden" ]]; then
  ROLLOUT_RAIDEN_ENV="USE_RAIDEN_FFI=false RAIDEN_USE_FFI=0"
fi
export ROLLOUT_EXTRA_ENV="${ROLLOUT_EXTRA_ENV:-} PYTHONUNBUFFERED=1 TUNIX_IS_INTERNAL_ENV=false RAIDEN_DEVICES_PER_HOST=${RAIDEN_DEVICES_PER_HOST} PREFUSE_MOE_WEIGHTS=${ROLLOUT_PREFUSE_MOE_WEIGHTS} ROLLOUT_PREFUSE_MOE_WEIGHTS=${ROLLOUT_PREFUSE_MOE_WEIGHTS} ROLLOUT_MESH_TP=${ROLLOUT_MESH_TP} ROLLOUT_TENSOR_PARALLEL_SIZE=${ROLLOUT_MESH_TP} ENABLE_PREFIX_CACHING=${ENABLE_PREFIX_CACHING} VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS} VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION} NUM_PRECOMPILE_WORKERS=${NUM_PRECOMPILE_WORKERS} NEW_MODEL_DESIGN=${NEW_MODEL_DESIGN} ATTN_BUCKETIZED_NUM_REQS=${ATTN_BUCKETIZED_NUM_REQS} ATTN_CUSTOM_NUM_REQS_BUCKETS=${ATTN_CUSTOM_NUM_REQS_BUCKETS} ONEHOT_MOE_PERMUTE_THRESHOLD=${ONEHOT_MOE_PERMUTE_THRESHOLD} VLLM_MOE_CHUNK_SIZE=${VLLM_MOE_CHUNK_SIZE} SLICE_ROPE_CACHE=${SLICE_ROPE_CACHE} DP_SCHED_BATCH_PREFILL=${DP_SCHED_BATCH_PREFILL} VLLM_ENABLE_V1_MULTIPROCESSING=${VLLM_ENABLE_V1_MULTIPROCESSING} VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL} LIBTPU_INIT_ARGS=\"${LIBTPU_INIT_ARGS}\" ${ROLLOUT_RAIDEN_ENV}"
export ROLLOUT_EXTRA_ARGS="${ROLLOUT_EXTRA_ARGS:-} --prefuse_moe_weights=${ROLLOUT_PREFUSE_MOE_WEIGHTS} --tensor_parallel_size=${ROLLOUT_MESH_TP} --return_routed_experts=${RETURN_ROUTED_EXPERTS} --free_kv_cache_during_weight_sync=${ROLLOUT_FREE_KV_CACHE} --max_concurrency=${ROLLOUT_MAX_CONCURRENCY}"

# ==============================================================================
# Execution Dispatch to math_gsm8k_dist Launcher
# ==============================================================================
if [ -f "${DIR}/../math_gsm8k_dist/k8s_launcher.sh" ]; then
  LAUNCHER="${DIR}/../math_gsm8k_dist/k8s_launcher.sh"
elif [ -f "${DIR}/tunix/experimental/examples/math_gsm8k_dist/k8s_launcher.sh" ]; then
  LAUNCHER="${DIR}/tunix/experimental/examples/math_gsm8k_dist/k8s_launcher.sh"
elif [ -f "${DIR}/../../../../third_party/py/tunix/experimental/examples/math_gsm8k_dist/k8s_launcher.sh" ]; then
  LAUNCHER="${DIR}/../../../../third_party/py/tunix/experimental/examples/math_gsm8k_dist/k8s_launcher.sh"
elif [ -f "${HOME}/github/tunix_build/tunix/experimental/examples/math_gsm8k_dist/k8s_launcher.sh" ]; then
  LAUNCHER="${HOME}/github/tunix_build/tunix/experimental/examples/math_gsm8k_dist/k8s_launcher.sh"
else
  echo "Error: math_gsm8k_dist/k8s_launcher.sh not found relative to ${DIR}"
  exit 1
fi

COMMAND="start"
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    start|stop|orchestrator|trainer|rollout)
      COMMAND="$1"
      shift
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

exec "${LAUNCHER}" --command "${COMMAND}" --image "${TUNIX_IMAGE}" "${EXTRA_ARGS[@]}"
