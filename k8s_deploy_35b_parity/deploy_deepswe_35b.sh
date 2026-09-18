#!/usr/bin/env bash
# ==============================================================================
# MLPerf v6.0 35B-A3B Multi-Turn DeepSWE Distributed GRPO Launcher
# 1-to-1 Aligned with training/llm_post_training/README.md & mlperf_35b_256.sh
# ==============================================================================
set -euo pipefail

# ------------------------------------------------------------------------------
# 1. Container Image & Pathways Images
# ------------------------------------------------------------------------------
export TUNIX_IMAGE="${TUNIX_IMAGE:-gcr.io/cloud-tpu-multipod-dev/${USER}/tunix_raiden_35b_parity:0918_v1}"
export PATHWAYS_SERVER_IMAGE="${PATHWAYS_SERVER_IMAGE:-us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/custom/yixuannwang_google_com-server:fe26c8fb60c2}"
export PATHWAYS_PROXY_IMAGE="${PATHWAYS_PROXY_IMAGE:-us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/custom/yixuannwang_google_com-proxy_server:c3f41a2495b7}"
export PATHWAYS_PROXY_MEMORY_LIMIT="${PATHWAYS_PROXY_MEMORY_LIMIT:-160Gi}"

# ------------------------------------------------------------------------------
# 2. GCS Bucket Configuration (Checkpoints, Pathways Scratch, Trajectory Logs)
# ------------------------------------------------------------------------------
# Set GCS_BUCKET to your own GCS bucket (e.g. gs://my-mlperf-bucket/jfacevedo)
export GCS_BUCKET="${GCS_BUCKET:-gs://cloud-tpu-multipod-dev-shared/${USER}/deepswe_35b_parity}"
export RUN_TAG="${RUN_TAG:-$(date +%m%d_%H%M)}"

# Base model checkpoint & tokenizer (read-only shared bucket by default)
export MAXTEXT_CKPT="${MAXTEXT_CKPT:-gs://tunix-eval/qwen3.5-35b-a3b-orbax/0/items}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-gs://tunix-eval/qwen3.5-35b-a3b-hf}"
export MODEL_DIR="${MODEL_DIR:-${TOKENIZER_PATH}}"

# Writable GCS destinations (all automatically derived from GCS_BUCKET if not overridden):
# - GCS_SCRATCH_LOCATION : Pathways compilation & runtime scratch directory
# - MAXTEXT_OUTPUT_DIR   : Trainer Orbax checkpoints & MaxText TensorBoard outputs
# - LOG_DIR              : Orchestrator CLU/TensorBoard metrics & step timing events
# - TRAJECTORY_LOG_DIR   : Collected SWE-bench multi-turn rollout trajectories (.jsonl)
export GCS_SCRATCH_LOCATION="${GCS_SCRATCH_LOCATION:-${GCS_BUCKET}/pathways_scratch}"
export MAXTEXT_OUTPUT_DIR="${MAXTEXT_OUTPUT_DIR:-${GCS_BUCKET}/runs/${RUN_TAG}/maxtext_output}"
export LOG_DIR="${LOG_DIR:-${GCS_BUCKET}/runs/${RUN_TAG}/logs}"
export TRAJECTORY_LOG_DIR="${TRAJECTORY_LOG_DIR:-${LOG_DIR}/trajectories}"

# ------------------------------------------------------------------------------
# 3. Weights & Biases (W&B) Configuration
# ------------------------------------------------------------------------------
# Export WANDB_API_KEY in your shell before running, or pass it inline:
#   WANDB_API_KEY="your_40_char_key" ./deploy_deepswe_35b.sh start
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_PROJECT="${WANDB_PROJECT:-mlperf-qwen35-35b-deepswe}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-qwen35-35b-parity-${RUN_TAG}}"

# ------------------------------------------------------------------------------
# 4. Cluster & Topology (1x v5p-128 Trainer + 4x v5p-32 Rollout Replicas)
# ------------------------------------------------------------------------------
export K8S_NAMESPACE="${K8S_NAMESPACE:-default}"
export RUN_PREFIX="${RUN_PREFIX:-${USER}-35b-parity}"
export ORCHESTRATOR_ID="${RUN_PREFIX}-orch"
export TRAINER_ID="${RUN_PREFIX}-trainer"
export ROLLOUT_ID="${RUN_PREFIX}-rollout"

export TRAINER_BACKEND="maxtext"
export MAXTEXT_MODEL_NAME="qwen3_5-35b-a3b"
export MODEL_NAME="qwen3_5-35b-a3b"
export MODEL_ID="Qwen/Qwen3.5-35B-A3B"

export TRAINER_TPU_SLICE="${TRAINER_TPU_SLICE:-v5p-128}"
export TRAINER_MESH_FSDP="${TRAINER_MESH_FSDP:-16}"
export TRAINER_MESH_TP="${TRAINER_MESH_TP:-4}"
export TRAINER_MESH_EXPERT="${TRAINER_MESH_EXPERT:-1}"
export TRAINER_PADDED_MOE_MLP_DIM="${TRAINER_PADDED_MOE_MLP_DIM:-512}"

export SAMPLER="inprocess_vllm"
export ROLLOUT_TPU_SLICE="${ROLLOUT_TPU_SLICE:-v5p-32}"
export ROLLOUT_REPLICAS="${ROLLOUT_REPLICAS:-4}"
export ROLLOUT_WORKERS="${ROLLOUT_REPLICAS}"
export ROLLOUT_MESH_TP="${ROLLOUT_MESH_TP:-4}"
export ROLLOUT_MESH_FSDP="${ROLLOUT_MESH_FSDP:-1}"

export WEIGHT_SYNC_MODE="raiden"
export USE_WEIGHT_CONVERTER="true"
export ROLLOUT_PREFUSE_MOE_WEIGHTS="true"
export TRAINER_PREFUSE_MOE_WEIGHTS="false"
export RAIDEN_DEVICES_PER_HOST="4"
export RAIDEN_USE_FFI="1"
export VERIFY_WEIGHTS="${VERIFY_WEIGHTS:-0}"

# ------------------------------------------------------------------------------
# 5. MLPerf v6.0 Hyperparameters (1-to-1 Match with training/llm_post_training/README.md)
# ------------------------------------------------------------------------------
# Batching & Rollout Lengths
export BATCH_SIZE="${BATCH_SIZE:-16}"                       # 16 prompts per step
export NUM_GENERATIONS="${NUM_GENERATIONS:-8}"              # G = 8 rollouts per prompt (128 trajectories/step)
export MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-16}"             # 16 prompt groups per optimizer step (1 update/step)
export NUM_ITERATIONS="${NUM_ITERATIONS:-1}"                # 1 epoch per rollout batch
export TRAIN_MICRO_BATCH_SIZE="${TRAIN_MICRO_BATCH_SIZE:-1}" # 1 sequence per micro-batch (grad_accum = 128)
export MAX_STEPS="${MAX_STEPS:-150}"
export MAX_TURNS="${MAX_TURNS:-50}"                         # MLPerf max_turns: 50
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"       # MLPerf max_prompt_length: 4096
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-28672}"  # 4096 + 28672 = 32768 total sequence length
export COMPUTE_LOGPS_CHUNK_SIZE="${COMPUTE_LOGPS_CHUNK_SIZE:-2048}"

# Sampling Parameters (MLPerf: temp=1.0, top_p=0.95, top_k=20)
export TEMPERATURE="${TEMPERATURE:-1.0}"
export TOP_P="${TOP_P:-0.95}"
export TOP_K="${TOP_K:-20}"

# Optimizer & Schedule (MLPerf: AdamW lr=3e-6, b1=0.9, b2=0.999, eps=1e-8, wd=0.0, clip=0.125, constant schedule)
export LEARNING_RATE="${LEARNING_RATE:-3e-6}"
export ADAM_B1="${ADAM_B1:-0.9}"
export ADAM_B2="${ADAM_B2:-0.999}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
export MAX_GRAD_NORM="${MAX_GRAD_NORM:-0.125}"
export WARMUP_STEPS_FRACTION="${WARMUP_STEPS_FRACTION:-0.0}"
export LEARNING_RATE_FINAL_FRACTION="${LEARNING_RATE_FINAL_FRACTION:-1.0}"
export REMAT_POLICY="${REMAT_POLICY:-custom}"

# GRPO / DAPO Loss & Sequence-Mask-TIS
# NOTE: In Tunix, FORCE_ON_POLICY_RATIO="true" + USE_ROLLOUT_LOGPS="false" sets
# old_per_token_logps=None (giving PPO ratio = 1.0) while STILL populating
# rollout_per_token_logps so seq-mask-tis weights the loss by stop_gradient(p_trainer/q_sampler) * keep_mask.
export BETA="${BETA:-0.0}"
export EPSILON="${EPSILON:-0.2}"
export EPSILON_HIGH="${EPSILON_HIGH:-0.28}"
export ADVANTAGE_ESTIMATOR="${ADVANTAGE_ESTIMATOR:-grpo_no_std}"
export LOSS_AGG_MODE="${LOSS_AGG_MODE:-token-mean}"
export FORCE_ON_POLICY_RATIO="${FORCE_ON_POLICY_RATIO:-true}"
export USE_ROLLOUT_LOGPS="${USE_ROLLOUT_LOGPS:-false}"
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
export TRAINER_MAXTEXT_ATTENTION="${TRAINER_MAXTEXT_ATTENTION:-dot_product}"
export ROLLOUT_MAXTEXT_ATTENTION="${ROLLOUT_MAXTEXT_ATTENTION:-vllm_rpa}"

# vLLM Rollout Engine Configuration
# Note: VLLM_ASYNC_SCHEDULING is set to "false" because enable_return_routed_experts=True
# (required for MoE router replay) is incompatible with vLLM async scheduling.
export VLLM_ADDITIONAL_CONFIG="${VLLM_ADDITIONAL_CONFIG:-{\"sharding\":{\"sharding_strategy\":{\"expert_parallelism\":2,\"tensor_parallelism\":1,\"enable_dp_attention\":true}},\"custom_mamba_cache_multiplier\":16,\"maxtext_config\":{\"scan_layers\":false,\"attention\":\"vllm_rpa\",\"allow_split_physical_axes\":true,\"use_multimodal\":false,\"prefuse_moe_weights\":true}}}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
export VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-1024}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-32}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.65}"
export VLLM_BLOCK_SIZE="${VLLM_BLOCK_SIZE:-256}"
export VLLM_ASYNC_SCHEDULING="${VLLM_ASYNC_SCHEDULING:-false}"
export VLLM_ENABLE_CHUNKED_PREFILL="${VLLM_ENABLE_CHUNKED_PREFILL:-true}"
export VLLM_LANGUAGE_MODEL_ONLY="${VLLM_LANGUAGE_MODEL_ONLY:-true}"
export VLLM_PREFIX_CACHE_RETENTION_INTERVAL="${VLLM_PREFIX_CACHE_RETENTION_INTERVAL:-0}"
export VLLM_KV_CACHE_DTYPE="${VLLM_KV_CACHE_DTYPE:-bfloat16}"

# DeepSWE Dataset & Sandbox Configuration
export DATASET_NAME="${DATASET_NAME:-R2E-Gym/R2E-Gym-Subset}"
export DATASET_SPLIT="${DATASET_SPLIT:-train}"
export SHUFFLE="${SHUFFLE:-true}"
export SEED="${SEED:-42}"
export SCAFFOLD="${SCAFFOLD:-r2egym}"
export ENV_BACKEND="${ENV_BACKEND:-kubernetes}"
export USE_AGENT_SANDBOX="${USE_AGENT_SANDBOX:-1}"
export SANDBOX_NAMESPACE="${SANDBOX_NAMESPACE:-rl-tunix-swebench}"
export SANDBOX_NODE_SELECTOR_KEY="${SANDBOX_NODE_SELECTOR_KEY:-workload}"
export SANDBOX_NODE_SELECTOR_VAL="${SANDBOX_NODE_SELECTOR_VAL:-swe-bench}"
export STEP_TIMEOUT_SECS="${STEP_TIMEOUT_SECS:-90}"
export REWARD_TIMEOUT_SECS="${REWARD_TIMEOUT_SECS:-300}"
export EPISODE_TIMEOUT_SECS="${EPISODE_TIMEOUT_SECS:-2700}"
export ROLLOUT_MAX_CONCURRENCY="${ROLLOUT_MAX_CONCURRENCY:-32}"
export MAX_CONCURRENCY="${MAX_CONCURRENCY:-128}"
export MAX_WARMPOOL_REPLICAS="${MAX_WARMPOOL_REPLICAS:-4}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/tunix/tunix/experimental/examples/deepswe_dist/k8s_launcher.sh" "$@"
