#!/bin/bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Fill these before you run.
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export MAXTEXT_OUTPUT_DIR="${MAXTEXT_OUTPUT_DIR:-gs://atwigg-trellis-europe-west4-dev/maxtext}"
export PROFILER_STEPS=0
export SKIP_FIRST_N_PROFILER_STEPS=-1
export TUNIX_IMAGE="${TUNIX_IMAGE:-gcr.io/cloud-tpu-multipod-dev/${USER:-niting}/trellis-35b:latest}"

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

# Pathways & Raiden Images and settings (from Google doc)
export PATHWAYS_SERVER_IMAGE="${PATHWAYS_SERVER_IMAGE:-us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/datenglin/unsanitized_server:raiden_20260920_v2}"
export PATHWAYS_PROXY_IMAGE="${PATHWAYS_PROXY_IMAGE:-us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/datenglin/unsanitized_proxy_server:raiden_20260920_v2}"
export PATHWAYS_PROXY_MEMORY_LIMIT="160G"
export USER_CONTAINER_MEMORY="260G"
export USER_CONTAINER_MEMORY_LIMIT="${USER_CONTAINER_MEMORY_LIMIT:-260G}"
export RAIDEN_DEVICES_PER_HOST=4
export USE_WEIGHT_CONVERTER="true"
export PREFUSE_MOE_WEIGHTS="true"
export TRAINER_PREFUSE_MOE_WEIGHTS="true"
export ROLLOUT_PREFUSE_MOE_WEIGHTS="true"
export VERIFY_WEIGHTS="true"
export TRAINER_PADDED_MOE_MLP_DIM=""

# WandB configuration
export WANDB_ENTITY="google-trellis"
export WANDB_PROJECT="trellis-deepswe"

# Model configuration
export MODEL_NAME="Qwen3.5-35B-A3B"
export MODEL_ID="Qwen/Qwen3.5-35B-A3B"
export TOKENIZER_PATH="Qwen/Qwen3.5-35B-A3B"
export MAXTEXT_MODEL_NAME="qwen3.5-35b-a3b"
export MAXTEXT_CKPT="gs://hengtaoguo-maxtext-logs/checkpoints/qwen3.5-35b-a3b/scanned/2026-06-11-10-27/0/items"
export TRAINABLE_PARAMETERS_MASK='^(?!.*routed_experts/gate/kernel).*'
export EOS_TOKENS="${EOS_TOKENS:-151645,151643}"
export TRAJECTORY_LOG_DIR="${TRAJECTORY_LOG_DIR:-gs://atwigg-trellis-europe-west4-dev/trajectories}"

# Backend configuration
export TRAINER_BACKEND="maxtext"
export SAMPLER="vllm"
export WEIGHT_SYNC_MODE="raiden"

# Topologies (64 chips Trainer 4x4x4, 16x 4-chip Rollout slices)
export TRAINER_JOBSET_YAML="jobset.pathways.yaml"
export TRAINER_TPU_SLICE="tpuv5:4x4x4"
export TRAINER_MESH_FSDP=32
export TRAINER_MESH_TP=2
export TRAINER_MESH_EXPERT=1
export TRAINER_BASE_NUM_KV_HEADS=2

export ROLLOUT_JOBSET_YAML="jobset.tpu.yaml"
export ROLLOUT_TPU_SLICE="tpuv5:2x2x1"
export ROLLOUT_MESH_FSDP=1
export ROLLOUT_MESH_TP=1
export ROLLOUT_WORKERS=16
export ROLLOUT_REPLICAS=16

# ==============================================================================
# vLLM Rollout Configuration (from paste.googleplex.com/5903655694368768)
# ==============================================================================
export VLLM_LOGGING_LEVEL="INFO"
export VLLM_MAX_MODEL_LEN=65536
export VLLM_MAX_NUM_BATCHED_TOKENS=2048
export VLLM_MAX_NUM_SEQS=16
export VLLM_GPU_MEMORY_UTILIZATION="0.9"

# Sharding Configs
export VLLM_DATA_PARALLEL_SIZE=1
export VLLM_ENABLE_EXPERT_PARALLEL="true"
# Note: enable_nnx and pure_nnx_decoder are internal to MaxTextForCausalLM and are
# not accepted by MaxText pyconfig HyperParameters.
export VLLM_ADDITIONAL_CONFIG='{"sharding":{"sharding_strategy":{"expert_parallelism":4,"tensor_parallelism":1,"enable_dp_attention":true}},"custom_mamba_cache_multiplier":16,"maxtext_config":{"scan_layers":false,"attention":"vllm_rpa","allow_split_physical_axes":true,"use_multimodal":false,"prefuse_moe_weights":true}}'

# Prefix Caching Configs
export ENABLE_PREFIX_CACHING="false"
export VLLM_PREFIX_CACHE_RETENTION_INTERVAL=256
export VLLM_MAMBA_CACHE_MODE="none"

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
# Note: enable_auto_tool_choice, tool_call_parser, and default_chat_template_kwargs
# are vLLM API server flags; DeepSWE handles tool calling and chat formatting in
# SWEAgent/SWEEnv directly, and vLLM AsyncEngineArgs does not accept them.
# export VLLM_ENABLE_AUTO_TOOL_CHOICE="true"
# export VLLM_TOOL_CALL_PARSER="qwen3_coder"
# export VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS='{"enable_thinking": false}'

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
# Hyperparameters & DeepSWE Pipeline Configuration
# ==============================================================================
export MAX_STEPS=${MAX_STEPS:-50}
export BATCH_SIZE=16
export MINI_BATCH_SIZE=${BATCH_SIZE}
export NUM_GENERATIONS=16
export TRAIN_MICRO_BATCH_SIZE="${TRAIN_MICRO_BATCH_SIZE:-32}"
export CHECKPOINT_SAVE_INTERVAL_STEPS=0
export CHECKPOINT_MAX_TO_KEEP=10
export MAX_STALENESS=1

# Sampling Parameters (explicitly disable top-k, set top-p 1.0 and temperature 1.0)
export TEMPERATURE="1.0"
export TOP_P="1.0"
export TOP_K="-1"

# Algorithmic & Loss Hyperparameters
export BETA=0.0
export EPSILON=0.2
export EPSILON_HIGH=0.28
export USE_ROLLOUT_LOGPS="false"
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

# Architecture & Rematerialization
export REMAT_POLICY="full"
export TRAINER_MAXTEXT_ATTENTION="flash"
export COMPUTE_LOGPS_CHUNK_SIZE=512

export EPISODE_TIMEOUT_SECS=1800
export DEBUG=1

# DeepSWE Environment & Agent Sandbox
export DATASET_PATH="gs://mlperf_dataset/r2e-gym-easy"
export USE_AGENT_SANDBOX=1
export SANDBOX_NAMESPACE="trellis"
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

COMMAND="${1:-start}"
exec "${LAUNCHER}" --command "${COMMAND}" --image "${TUNIX_IMAGE}"
