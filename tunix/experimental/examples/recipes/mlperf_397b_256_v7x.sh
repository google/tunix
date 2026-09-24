#!/bin/bash
set -e

# ==============================================================================
# MLPerf DeepSWE recipe: Qwen3.5-397B-A17B on TPU v7x (bodaborg-tpu7x-gsc)
# ==============================================================================
# Derived from mlperf_35b_256_v7x.sh and mlperf_397b_v7x.sh, configured with:
# - TPU7x dynamic slicing on bodaborg-tpu7x-gsc (priority-dev namespace)
# - Trainer on 256 chips (4x8x8 = 512 devices, FSDP=64, TP=1, EXPERT=2, CP=4)
# - Rollout on 256 chips (32 replicas x 8 chips 2x2x2, EP=16, TP=1)
# - Sandbox configured for sandbox-np nodepool with workload tolerations
# ==============================================================================

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Fill these before you run.
export JOB_PREFIX="${JOB_PREFIX:-$USER}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-${JOB_PREFIX}-mlperf-397b-v7x}"
# us-central1 buckets, co-located with the cluster.
export MAXTEXT_OUTPUT_DIR="${MAXTEXT_OUTPUT_DIR:-gs://bodaborg-tpu7x-nap-us-central1/maxtext/${JOB_PREFIX}}"
export TRAJECTORY_LOG_DIR="${TRAJECTORY_LOG_DIR:-gs://bodaborg-tpu7x-nap-us-central1/trajectories/${JOB_PREFIX}}"
export ORCHESTRATOR_PORT="${ORCHESTRATOR_PORT:-20000}"
export ROLLOUT_PORT="${ROLLOUT_PORT:-20001}"
export TRAINER_PORT="${TRAINER_PORT:-20002}"
export TUNIX_IMAGE="${TUNIX_IMAGE:-gcr.io/cloud-tpu-multipod-dev/${USER:-atwigg}/trellis-397b:latest}"

# Profiling off.
export PROFILER_STEPS=0
export SKIP_FIRST_N_PROFILER_STEPS=-1

# Regional cluster: us-central1 bodaborg-tpu7x-gsc.
export PROJECT="cloud-tpu-shared-capacity"
export REGION="us-central1"
export CLUSTER="bodaborg-tpu7x-gsc"
kubectl config use-context "gke_${PROJECT}_${REGION}_${CLUSTER}" || true
kubectl config set-context --current --namespace=priority-dev || true

export K8S_NAMESPACE="priority-dev"
export KUEUE_QUEUE="${KUEUE_QUEUE:-multislice-queue}"
export PRIORITY_CLASS="${PRIORITY_CLASS:-medium}"
export KUEUE_PRIORITY_CLASS="${KUEUE_PRIORITY_CLASS:-${PRIORITY_CLASS}}"
export SERVICE_ACCOUNT="xpk-sa"
export CPU_MACHINE="n2d-standard-64"
export USE_DYNAMIC_SLICING="true"
export TPU_RESERVATION="${TPU_RESERVATION:-ghostfish-pogoag4tylwed}"
export ENABLE_PATHWAYS_PERSISTENCE=1

# Pathways shared configuration
source "${DIR}/mlperf_pathways_config.sh"
export ENABLE_MULTI_NUMA="${ENABLE_MULTI_NUMA:-0}"

export RAIDEN_DEVICES_PER_HOST=8
export USE_WEIGHT_CONVERTER="true"
export PREFUSE_MOE_WEIGHTS="true"
export TRAINER_PREFUSE_MOE_WEIGHTS="true"
export ROLLOUT_PREFUSE_MOE_WEIGHTS="true"
export VERIFY_WEIGHTS="true"
export TRAINER_PADDED_MOE_MLP_DIM=""
export TPU_RAIDEN_DATA_NICS="eth0"

# WandB configuration
export WANDB_ENTITY="google-trellis"
export WANDB_PROJECT="trellis-deepswe"

# Model configuration
export MODEL_NAME="Qwen3.5-397B-A17B"
export MODEL_ID="Qwen/Qwen3.5-397B-A17B"
export TOKENIZER_PATH="${TOKENIZER_PATH:-/app/Qwen/Qwen3.5-397B-A17B}"
export MAXTEXT_MODEL_NAME="qwen3.5-397b-a17b"
export MAXTEXT_CKPT="${MAXTEXT_CKPT:-gs://mlperf-6-1-submission/ckpt/qwen35_397b/scanned_reshard_fsdp32_tp2/0/items}"
export TRAINABLE_PARAMETERS_MASK='^(?!.*routed_experts/gate/kernel).*'
export EOS_TOKENS="${EOS_TOKENS:-151645,151643}"

# Backend configuration
export TRAINER_BACKEND="maxtext"
export SAMPLER="vllm"
export WEIGHT_SYNC_MODE="raiden"

# Topologies (128 chips / 256 devices Trainer 4x4x8, 8x 8-chip Rollout slices on TPU7x dynamic slicing)
# Mesh product is DEVICES, and v7x has 2 devices/chip at ~95 GB each.
# 4x4x8 = 128 chips = 256 devices.
# Sharding: TP=1, FSDP=32, CP=4, EP=2, cp-as-ep.
# Mesh product: 32 * 1 * 2 * 4 = 256 devices.
export TRAINER_JOBSET_YAML="jobset.pathways.qwen3.5-397b.yaml"
export TRAINER_TPU_SLICE="tpu7x:4x4x8"           # 128 chips = 256 devices
export TRAINER_MESH_FSDP=32
export TRAINER_MESH_TP=1
export TRAINER_MESH_EXPERT=2
export TRAINER_MESH_CONTEXT=4
export TRAINER_BASE_NUM_KV_HEADS=2

# Rollout: 8 chips = 16 devices = 2 hosts per replica. tp * expert must equal the
# DEVICE count, so expert=16 on 8 chips -- the same ep=16 that v5p ran on 16 chips,
# because v7x exposes 2 devices per chip. 1520 GB vs the 794 GB of bf16 weights
# leaves ~726 GB of KV cache. 4 chips (8 devices, 760 GB) does not even hold the
# weights, so single-host is out. Keep tp=1: tp=2 x expert=8 gave incoherent
# rollouts on v5p.
export ROLLOUT_JOBSET_YAML="jobset.mcjax.ray.yaml"   # 8 chips = 2 hosts -> still Ray multihost
export ROLLOUT_TPU_SLICE="tpu7x:2x2x2"   # 8 chips
export ROLLOUT_MESH_FSDP=1
export ROLLOUT_MESH_TP=1
export ROLLOUT_MESH_EXPERT=16

# 16 replicas x 8 chips = 128 rollout chips. ROLLOUT_WORKERS wins where both are
# read (k8s_launcher.sh:320), so keep them equal.
export ROLLOUT_WORKERS="${ROLLOUT_WORKERS:-16}"
export ROLLOUT_REPLICAS="${ROLLOUT_REPLICAS:-16}"

# ==============================================================================
# vLLM Rollout Configuration
# ==============================================================================
export VLLM_LOGGING_LEVEL="INFO"
export VLLM_MAX_NUM_BATCHED_TOKENS=2048
export VLLM_MAX_NUM_SEQS=16
export VLLM_GPU_MEMORY_UTILIZATION="0.9"
export VLLM_DATA_PARALLEL_SIZE=1
export VLLM_ENABLE_EXPERT_PARALLEL="true"
export VLLM_ADDITIONAL_CONFIG='{"sharding":{"sharding_strategy":{"expert_parallelism":16,"tensor_parallelism":1,"enable_dp_attention":true}},"custom_mamba_cache_multiplier":16,"maxtext_config":{"scan_layers":false,"attention":"vllm_rpa","allow_split_physical_axes":true,"use_multimodal":false,"prefuse_moe_weights":true,"per_device_batch_size":0.0}}'

export ENABLE_PREFIX_CACHING="false"
export VLLM_PREFIX_CACHE_RETENTION_INTERVAL="${VLLM_PREFIX_CACHE_RETENTION_INTERVAL:-0}"
export MAMBA_CACHE_MODE="${MAMBA_CACHE_MODE:-align}"
export VLLM_MAMBA_CACHE_MODE="${VLLM_MAMBA_CACHE_MODE:-${MAMBA_CACHE_MODE}}"

export ROLLOUT_FREE_KV_CACHE="false"
export VLLM_KV_CACHE_DTYPE="bfloat16"
export VLLM_BLOCK_SIZE=256
export VLLM_ASYNC_SCHEDULING="true"
export VLLM_ENABLE_CHUNKED_PREFILL="true"
export VLLM_LANGUAGE_MODEL_ONLY="true"
export VLLM_REASONING_PARSER="qwen3"
export VLLM_LIMIT_MM_PER_PROMPT='{"image": 0, "video": 0}'

# ==============================================================================
# Rollout Worker Environment Flags
# ==============================================================================
export NUM_PRECOMPILE_WORKERS=8
export NEW_MODEL_DESIGN=1
export ATTN_BUCKETIZED_NUM_REQS=true
export ATTN_CUSTOM_NUM_REQS_BUCKETS=4
export ONEHOT_MOE_PERMUTE_THRESHOLD=131072
export VLLM_MOE_CHUNK_SIZE=256
export SLICE_ROPE_CACHE=1
export DP_SCHED_BATCH_PREFILL=false
export LIBTPU_INIT_ARGS=' --xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false --xla_tpu_check_legacy_constraints_in_reduce_scatter_legalizer=false'
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800

# Raiden tuning carried over from the 397B GSM8K recipe.
export VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY="RAIDEN_,TPU_"
export VLLM_RAY_EXTRA_ENV_VARS_TO_COPY="ONEHOT_MOE_PERMUTE_THRESHOLD,LIBTPU_INIT_ARGS,RAY_memory_monitor_refresh_ms,VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS,ENABLE_MULTI_NUMA,TPU_RAIDEN_DATA_NICS"
export ORCHESTRATOR_EXTRA_ENV="${ORCHESTRATOR_EXTRA_ENV:-WEIGHT_SYNC_H2D_TIMEOUT_S=3600 WEIGHT_SYNC_TRANSFER_TIMEOUT_S=7200 TPU_RAIDEN_DATA_NICS=eth0}"
export ROLLOUT_EXTRA_ENV="${ROLLOUT_EXTRA_ENV:-ONEHOT_MOE_PERMUTE_THRESHOLD=131072 RAY_memory_monitor_refresh_ms=0 RAIDEN_TRANSPORT_COALESCE_WINDOW_BYTES=67108864 RAIDEN_WEIGHT_SYNC_PIPELINE_GROUP_SIZE=16 RAIDEN_PARALLELISM=16 VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 TPU_RAIDEN_DATA_NICS=eth0}"
# Trainer XLA flags. Note LIBTPU_INIT_ARGS above is the ROLLOUT's; the trainer
# needs its own, with sparsecore collective offloading and a raised scoped
# vmem limit. The 64k config was AOT-compiled with exactly these set.
export TRAINER_LIBTPU_INIT_ARGS="${TRAINER_LIBTPU_INIT_ARGS:---DANGEROUS_tpu_runtime_abi_verification_disabled=true --xla_tpu_use_tc_device_shape_on_sc=true --xla_sc_disable_megacore_partitioning=true --xla_tpu_enable_offloading_gather_to_sparsecore=true --xla_tpu_enable_sparse_core_collective_offload_all_gather=true --xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true --xla_tpu_enable_sparse_core_reduce_scatter_v2=true --xla_tpu_use_single_sparse_core_for_all_gather_offload=true --xla_tpu_enable_concurrent_sparse_core_offloading=true --xla_tpu_aggressive_opt_barrier_removal=true --xla_tpu_scoped_vmem_limit_kib=65536 --xla_tpu_enable_sublane_major_scaling_bitcast_fusion=false}"
export TRAINER_EXTRA_ENV="${TRAINER_EXTRA_ENV:-ONEHOT_MOE_PERMUTE_THRESHOLD=131072 RAIDEN_TRANSPORT_COALESCE_WINDOW_BYTES=67108864 RAIDEN_WEIGHT_SYNC_PIPELINE_GROUP_SIZE=16 TPU_RAIDEN_DATA_NICS=eth0 LIBTPU_INIT_ARGS='${TRAINER_LIBTPU_INIT_ARGS}'}"

# ==============================================================================
# Hyperparameters & DeepSWE Pipeline Configuration
# ==============================================================================
# GBS = 1024, NUM_GENERATIONS = 16 => BATCH_SIZE = MINI_BATCH_SIZE = 1024 / 16 = 64
# MicroBatchSize = 64 => PerDeviceBatchSize = 64 / 256 devices = 0.25
# GradientAccumulationSteps = (MINI_BATCH_SIZE * NUM_GENERATIONS) / TRAIN_MICRO_BATCH_SIZE = (64 * 16) / 64 = 16
# Divisibility: MicroBatchSize (64) is a multiple of FSDP * EXPERT (32 * 2 = 64)
export MAX_STEPS=${MAX_STEPS:-50}
export BATCH_SIZE=${BATCH_SIZE:-16}
export MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-${BATCH_SIZE}}
export NUM_GENERATIONS=16
export TRAIN_MICRO_BATCH_SIZE="${TRAIN_MICRO_BATCH_SIZE:-32}"
export CHECKPOINT_SAVE_INTERVAL_STEPS=${CHECKPOINT_SAVE_INTERVAL_STEPS:-0}
export CHECKPOINT_MAX_TO_KEEP=10
# When saving is enabled, default to Pathways persistence; the fallback OOMs the proxy at 397B.
if [[ "${CHECKPOINT_SAVE_INTERVAL_STEPS}" -gt 0 ]]; then
  export ENABLE_PATHWAYS_PERSISTENCE=${ENABLE_PATHWAYS_PERSISTENCE:-1}
fi
export MAX_STALENESS=0

# Step 0 is a cold single-threaded Pallas lowering of the MoE and GDN kernels
# with the TPUs idle; at 256 chips that ran past the 1800 s default.
export RPC_TIMEOUT_S="${RPC_TIMEOUT_S:-10800}"

export TEMPERATURE="1.0"
export TOP_P="1.0"
export TOP_K="-1"

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

export LEARNING_RATE="1e-6"
export ADAM_B1=0.9
export ADAM_B2=0.999
export WEIGHT_DECAY=0.0
export MAX_GRAD_NORM="0.125"
export WARMUP_STEPS_FRACTION=0.0
export LEARNING_RATE_FINAL_FRACTION=1.0

export REMAT_POLICY="full"
export TRAINER_MAXTEXT_ATTENTION="flash"

# custom_mesh_and_rule=cp-as-ep shards experts across both context and expert axes (8 experts/device).
export MAXTEXT_EXTRA_FLAGS="${MAXTEXT_EXTRA_FLAGS:-custom_mesh_and_rule=cp-as-ep}"
export COMPUTE_LOGPS_CHUNK_SIZE=512

export EPISODE_TIMEOUT_SECS=1800
export DEBUG=${DEBUG:-0}

# DeepSWE Environment & Agent Sandbox
export DATASET_PATH="gs://mlperf_dataset/r2e-gym-easy"
export USE_AGENT_SANDBOX=1
export SCAFFOLD="openhands"
export SANDBOX_NAMESPACE="${SANDBOX_NAMESPACE:-${K8S_NAMESPACE}}"
export POOL_NAME_FORMAT="${POOL_NAME_FORMAT:-}"
export TEMPLATE_NAME_PREFIX="${TEMPLATE_NAME_PREFIX:-}"
export SANDBOX_NODE_SELECTOR_KEY="cloud.google.com/gke-nodepool"
export SANDBOX_NODE_SELECTOR_VAL="sandbox-np"
export SANDBOX_TOLERATIONS=""
export IMAGE_REWRITE_PREFIX="${IMAGE_REWRITE_PREFIX:-us-central1-docker.pkg.dev/cloud-tpu-multipod-dev/tunix/}"
export MAX_WARMPOOL_REPLICAS=2
export STEP_TIMEOUT_SECS=300
export REWARD_TIMEOUT_SECS=180
export FLUSH_EVERY_N_STEPS=1
export MAX_TURNS=30

# Context length and concurrency. 64k window (4096 prompt + 61440 response = 65536) with
# 256-way concurrency, inherited from the 35B recipe. At 397B the rollout has 8
# chips per replica (16 devices) rather than 4 and the KV cache per sequence is far larger.
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-61440}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-65536}"
export MAX_SEQ_TOKEN_PER_TPU="${MAX_SEQ_TOKEN_PER_TPU:-65536}"
export ROLLOUT_MAX_CONCURRENCY="${ROLLOUT_MAX_CONCURRENCY:-256}"
export MAX_CONCURRENCY="${MAX_CONCURRENCY:-256}"

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
shift || true
exec "${LAUNCHER}" --command "${COMMAND}" --image "${TUNIX_IMAGE}" "$@"
