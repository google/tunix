#!/bin/bash
set -e

# ==============================================================================
# MLPerf DeepSWE Shared Base Configuration & Dispatcher
# ==============================================================================
# This script defines shared defaults for MLPerf distributed recipes (35B, 397B)
# on TPU clusters (v5p, v7x) and dispatches execution to deepswe_dist/k8s_launcher.sh.
#
# Recipe scripts specify model, cluster, topology, and hardware-specific flags,
# then source this base script at the end:
#   source "${DIR}/mlperf_base.sh" "$@"
# ==============================================================================

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ==============================================================================
# Container Image, User Identity & WandB
# ==============================================================================
export TUNIX_IMAGE="${TUNIX_IMAGE:-gcr.io/cloud-tpu-multipod-dev/atwigg/trellis:latest}"
export JOB_PREFIX="${JOB_PREFIX:-${USER}}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export ORCHESTRATOR_PORT="${ORCHESTRATOR_PORT:-20000}"
export ROLLOUT_PORT="${ROLLOUT_PORT:-20001}"
export TRAINER_PORT="${TRAINER_PORT:-20002}"
export PROFILER_STEPS=${PROFILER_STEPS:-0}
export SKIP_FIRST_N_PROFILER_STEPS=${SKIP_FIRST_N_PROFILER_STEPS:--1}

# ==============================================================================
# Cluster Context & Kueue / Priority
# ==============================================================================
export PROJECT="${PROJECT:-cloud-tpu-shared-capacity}"
if [[ -n "${REGION:-}" && -n "${CLUSTER:-}" ]]; then
  kubectl config use-context "gke_${PROJECT}_${REGION}_${CLUSTER}" || true
  if [[ -n "${K8S_NAMESPACE:-}" ]]; then
    kubectl config set-context --current --namespace="${K8S_NAMESPACE}" || true
  fi
fi

export KUEUE_QUEUE="${KUEUE_QUEUE:-multislice-queue}"
export PRIORITY_CLASS="${PRIORITY_CLASS:-medium}"
# yaml_generator reads KUEUE_PRIORITY_CLASS (not PRIORITY_CLASS) to render
# ${PRIORITY_CLASS_LINE}; without it the trainer admits at priority 0 and is
# evictable by any prioritised workload.
export KUEUE_PRIORITY_CLASS="${KUEUE_PRIORITY_CLASS:-${PRIORITY_CLASS}}"
export SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-xpk-sa}"
export CPU_MACHINE="${CPU_MACHINE:-n2d-standard-64}"

# ==============================================================================
# Pathways & Raiden Weight Sync Defaults
# ==============================================================================
source "${DIR}/mlperf_pathways_config.sh"

export USE_WEIGHT_CONVERTER="true"
export PREFUSE_MOE_WEIGHTS="true"
export TRAINER_PREFUSE_MOE_WEIGHTS="true"
export ROLLOUT_PREFUSE_MOE_WEIGHTS="true"
export VERIFY_WEIGHTS="true"
export TRAINER_PADDED_MOE_MLP_DIM=""
export WEIGHT_SYNC_MODE="raiden"

# ==============================================================================
# WandB Configuration
# ==============================================================================
export WANDB_ENTITY="${WANDB_ENTITY:-google-trellis}"
export WANDB_PROJECT="${WANDB_PROJECT:-trellis-deepswe}"

# ==============================================================================
# Common Model & Backend Configuration
# ==============================================================================
export TRAINER_BACKEND="maxtext"
export SAMPLER="vllm"
export TRAINABLE_PARAMETERS_MASK='^(?!.*routed_experts/gate/kernel).*'
# Qwen3.5 <|im_end|>, <|endoftext|> (generation_config.json eos_token_id).
# 151645/151643 are the Qwen2.5/Qwen3 ids and are ordinary tokens in the
# Qwen3.5 vocab.
export EOS_TOKENS="${EOS_TOKENS:-248046,248044}"
export TRAINER_BASE_NUM_KV_HEADS=2
export ROLLOUT_MESH_FSDP=1
export ROLLOUT_MESH_TP=1

# ==============================================================================
# MLPerf RCP Logging
# ==============================================================================
export RCP_LOGGING="${RCP_LOGGING:-false}"
if [[ -n "${MAXTEXT_OUTPUT_DIR:-}" ]]; then
  export METRIC_LOGGER_DIR="${METRIC_LOGGER_DIR:-${MAXTEXT_OUTPUT_DIR}/mllog}"
fi
export TARGET_ACCURACY="${TARGET_ACCURACY:-0.69}"

# ==============================================================================
# vLLM Rollout Configuration
# ==============================================================================
export VLLM_LOGGING_LEVEL="INFO"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-65536}"
export VLLM_MAX_NUM_BATCHED_TOKENS=2048
export VLLM_MAX_NUM_SEQS=16
export VLLM_GPU_MEMORY_UTILIZATION="0.9"

# Sharding Configs
export VLLM_DATA_PARALLEL_SIZE=1
export VLLM_ENABLE_EXPERT_PARALLEL="true"

# Prefix Caching Configs
export ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-true}"
export VLLM_PREFIX_CACHE_RETENTION_INTERVAL="${VLLM_PREFIX_CACHE_RETENTION_INTERVAL:-0}"
export MAMBA_CACHE_MODE="${MAMBA_CACHE_MODE:-align}"
export VLLM_MAMBA_CACHE_MODE="${VLLM_MAMBA_CACHE_MODE:-${MAMBA_CACHE_MODE}}"

# Router replay
export RETURN_ROUTED_EXPERTS="${RETURN_ROUTED_EXPERTS:-true}"

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
export ONEHOT_MOE_PERMUTE_THRESHOLD="${ONEHOT_MOE_PERMUTE_THRESHOLD:-32768}"
export VLLM_MOE_CHUNK_SIZE=256
export SLICE_ROPE_CACHE=1
export DP_SCHED_BATCH_PREFILL=false
export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:- --xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false --xla_tpu_check_legacy_constraints_in_reduce_scatter_legalizer=false}"
export VLLM_ENABLE_V1_MULTIPROCESSING=0

# ==============================================================================
# Hyperparameters & DeepSWE Pipeline Configuration
# ==============================================================================
export MAX_STEPS=${MAX_STEPS:-50}
export BATCH_SIZE=${BATCH_SIZE:-16}
export MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-${BATCH_SIZE}}
export NUM_GENERATIONS=16
export TRAIN_MICRO_BATCH_SIZE="${TRAIN_MICRO_BATCH_SIZE:-32}"
export CHECKPOINT_SAVE_INTERVAL_STEPS=${CHECKPOINT_SAVE_INTERVAL_STEPS:-0}
export CHECKPOINT_MAX_TO_KEEP=10
export CHECKPOINT_ASYNC=${CHECKPOINT_ASYNC:-true}
export ENABLE_PATHWAYS_PERSISTENCE=${ENABLE_PATHWAYS_PERSISTENCE:-1}
export MAX_STALENESS=${MAX_STALENESS:-0}
export TRAJECTORY_GROUP_ORDER=${TRAJECTORY_GROUP_ORDER:-arrival}

# Sequence packing
export MAX_SEQ_TOKEN_PER_TPU=${MAX_SEQ_TOKEN_PER_TPU:-65536}
# Stopgap: one trajectory per packed row. With maxtext >= 6c2adb7be (#5351), rows holding
# several trajectories give NaN grad_norm (GDN segment-reset backward); 16 was NaN 8/8, 1 is
# finite. Restore 16 once the maxtext fix (#5364 or later) is verified on packed rows.
export MAX_SEGMENTS_PER_PACKED_ROW=${MAX_SEGMENTS_PER_PACKED_ROW:-1}

# Sampling Parameters (explicitly disable top-k, set top-p 1.0 and temperature 1.0)
export TEMPERATURE="1.0"
export TOP_P="1.0"
export TOP_K="-1"

# Algorithmic & Loss Hyperparameters
export BETA=0.0
export EPSILON=0.2
export EPSILON_HIGH=0.28
export USE_ROLLOUT_LOGPS="false"
export EXACT_TOKEN_CONTINUITY="${EXACT_TOKEN_CONTINUITY:-true}"
export OVERLONG_FILTER="true"
export OVERLONG_LOSS_MASKING="true"
export SEQ_LOGPROB_ERROR_THRESHOLD=2.0
export TRUNCATED_IMPORTANCE_SAMPLING_TYPE="seq-mask-tis"
export TRUNCATED_IMPORTANCE_SAMPLING_RATIO_MIN=0.999
export TRUNCATED_IMPORTANCE_SAMPLING_RATIO=1.002
export ADVANTAGE_ESTIMATOR="grpo-loo"
export LOSS_AGG_MODE="token-mean"
export FLOAT32_GATE_LOGITS="true"
export FLOAT32_LOGITS="true"

# Optimizer Hyperparameters
export LEARNING_RATE="1e-6"
export ADAM_B1=0.9
export ADAM_B2=0.999
export WEIGHT_DECAY=0.0
export MAX_GRAD_NORM="0.125"
export WARMUP_STEPS_FRACTION=0.0
export LEARNING_RATE_FINAL_FRACTION=1.0
export SKIP_STEP_ON_SPIKES="${SKIP_STEP_ON_SPIKES:-false}"
export SKIP_STEP_ON_NAN="${SKIP_STEP_ON_NAN:-true}"

# Architecture & Rematerialization
export REMAT_POLICY="${REMAT_POLICY:-full}"
export TRAINER_MAXTEXT_ATTENTION="flash"
export COMPUTE_LOGPS_CHUNK_SIZE=512

export EPISODE_TIMEOUT_SECS=1800
export DEBUG=${DEBUG:-1}

# ==============================================================================
# DeepSWE Environment & Agent Sandbox
# ==============================================================================
export DATASET_PATH="gs://mlperf_dataset/benchmark-r2e-gym-easy"
export USE_AGENT_SANDBOX=1
export SCAFFOLD="openhands"
export SANDBOX_NAMESPACE="${SANDBOX_NAMESPACE:-${K8S_NAMESPACE:-trellis}}"
export POOL_NAME_FORMAT="${POOL_NAME_FORMAT:-}"
export TEMPLATE_NAME_PREFIX="${TEMPLATE_NAME_PREFIX:-}"
export SANDBOX_NODE_SELECTOR_KEY="cloud.google.com/gke-nodepool"
export SANDBOX_NODE_SELECTOR_VAL="${SANDBOX_NODE_SELECTOR_VAL:-sandbox-np}"
export MAX_WARMPOOL_REPLICAS=2
export ROLLOUT_MAX_CONCURRENCY="${ROLLOUT_MAX_CONCURRENCY:-256}"
export MAX_CONCURRENCY="${MAX_CONCURRENCY:-256}"
export STEP_TIMEOUT_SECS=300
export REWARD_TIMEOUT_SECS=180
export FLUSH_EVERY_N_STEPS=1
export MAX_TURNS=30
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-61440}"

# ==============================================================================
# Execution Dispatch
# ==============================================================================
if [[ -z "${LAUNCHER:-}" ]]; then
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
fi

if [[ "${MLPERF_NO_LAUNCH:-0}" != "1" ]]; then
  COMMAND="${1:-start}"
  shift || true
  exec "${LAUNCHER}" --command "${COMMAND}" --image "${TUNIX_IMAGE}" "$@"
fi
