#!/bin/bash
set -e

# ==============================================================================
# MLPerf DeepSWE recipe: Qwen3.5-397B-A17B on TPU v7x (bodaborg-tpu7x-gsc)
# ==============================================================================
# - TPU7x dynamic slicing on bodaborg-tpu7x-gsc (priority-dev namespace)
# - Trainer on 256 chips (4x8x8 = 512 devices, FSDP=16, TP=1, EXPERT=2, CP=8)
# - Rollout on 256 chips (16 replicas x 8 chips 2x2x2, EP=16, TP=1)
# - Sandbox configured for sandbox-np nodepool with workload tolerations
# ==============================================================================

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Fill these before you run.
export JOB_PREFIX="${JOB_PREFIX:-$USER}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-${JOB_PREFIX}-mlperf-397b-v7x}"
# us-central1 regional bucket, co-located with the cluster.
export MAXTEXT_OUTPUT_DIR="${MAXTEXT_OUTPUT_DIR:-gs://atwigg-trellis-v7x/maxtext/${JOB_PREFIX}}"
export TRAJECTORY_LOG_DIR="${TRAJECTORY_LOG_DIR:-gs://atwigg-trellis-v7x/trajectories/${JOB_PREFIX}/logger}"
export TRAJECTORY_STORE_ROOT_DIR="${TRAJECTORY_STORE_ROOT_DIR:-${TRAJECTORY_STORE_ROOT:-gs://atwigg-trellis-v7x/trajectories/${JOB_PREFIX}/store}}"
export TUNIX_IMAGE="${TUNIX_IMAGE:-gcr.io/cloud-tpu-multipod-dev/${USER:-atwigg}/trellis-397b:latest}"

# Regional cluster: us-central1 bodaborg-tpu7x-gsc.
export PROJECT="cloud-tpu-shared-capacity"
export REGION="us-central1"
export CLUSTER="bodaborg-tpu7x-gsc"
export K8S_NAMESPACE="priority-dev"
export USE_DYNAMIC_SLICING="true"
export TPU_RESERVATION="${TPU_RESERVATION:-ghostfish-pogoag4tylwed}"
export ENABLE_PATHWAYS_PERSISTENCE=1
export ENABLE_MULTI_NUMA="${ENABLE_MULTI_NUMA:-0}"

export RAIDEN_DEVICES_PER_HOST=8
export TPU_RAIDEN_DATA_NICS="eth0"

# Model configuration
export MODEL_NAME="Qwen3.5-397B-A17B"
export MODEL_ID="Qwen/Qwen3.5-397B-A17B"
export TOKENIZER_PATH="${TOKENIZER_PATH:-/app/Qwen/Qwen3.5-397B-A17B}"
export MAXTEXT_MODEL_NAME="qwen3.5-397b-a17b"
export MAXTEXT_CKPT="${MAXTEXT_CKPT:-gs://mlperf-6-1-submission/ckpt/qwen35_397b/scanned_reshard_fsdp32_tp2/0/items}"

# Topologies (128 chips / 256 devices Trainer 4x4x8, 16x 8-chip Rollout slices on TPU7x dynamic slicing)
# Mesh product is DEVICES, and v7x has 2 devices/chip at ~95 GB each.
# 4x4x8 = 128 chips = 256 devices.
# Sharding: TP=1, FSDP=16, CP=8, EP=2, cp-as-ep.
# Mesh product: 16 * 1 * 2 * 8 = 256 devices.
export TRAINER_JOBSET_YAML="jobset.pathways.qwen3.5-397b.yaml"
export TRAINER_TPU_SLICE="tpu7x:4x4x8"
export TRAINER_MESH_FSDP=16
export TRAINER_MESH_TP=1
export TRAINER_MESH_EXPERT=2
export TRAINER_MESH_CONTEXT=8

# Rollout: 8 chips = 16 devices = 2 hosts per replica. tp * expert must equal the
# DEVICE count, so expert=16 on 8 chips.
export ROLLOUT_JOBSET_YAML="jobset.mcjax.ray.yaml"
export ROLLOUT_TPU_SLICE="tpu7x:2x2x2"
export ROLLOUT_MESH_EXPERT=16
export ROLLOUT_WORKERS="${ROLLOUT_WORKERS:-16}"
export ROLLOUT_REPLICAS="${ROLLOUT_REPLICAS:-16}"

# vLLM Rollout Configuration
export VLLM_ADDITIONAL_CONFIG='{"sharding":{"sharding_strategy":{"expert_parallelism":16,"tensor_parallelism":1,"enable_dp_attention":true}},"custom_mamba_cache_multiplier":16,"maxtext_config":{"scan_layers":false,"attention":"vllm_rpa","allow_split_physical_axes":true,"use_multimodal":false,"prefuse_moe_weights":true,"per_device_batch_size":0.0}}'
export MAMBA_CACHE_MODE="${MAMBA_CACHE_MODE:-align}"
export VLLM_MAMBA_CACHE_MODE="${VLLM_MAMBA_CACHE_MODE:-${MAMBA_CACHE_MODE}}"

# Rollout Worker Flags & Raiden tuning
export ONEHOT_MOE_PERMUTE_THRESHOLD=131072
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
export VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY="RAIDEN_,TPU_"
export VLLM_RAY_EXTRA_ENV_VARS_TO_COPY="ONEHOT_MOE_PERMUTE_THRESHOLD,LIBTPU_INIT_ARGS,RAY_memory_monitor_refresh_ms,VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS,ENABLE_MULTI_NUMA,TPU_RAIDEN_DATA_NICS"
export ORCHESTRATOR_EXTRA_ENV="${ORCHESTRATOR_EXTRA_ENV:-WEIGHT_SYNC_H2D_TIMEOUT_S=3600 WEIGHT_SYNC_TRANSFER_TIMEOUT_S=7200 TPU_RAIDEN_DATA_NICS=eth0}"
export ROLLOUT_EXTRA_ENV="${ROLLOUT_EXTRA_ENV:-ONEHOT_MOE_PERMUTE_THRESHOLD=131072 RAY_memory_monitor_refresh_ms=0 RAIDEN_TRANSPORT_COALESCE_WINDOW_BYTES=67108864 RAIDEN_WEIGHT_SYNC_PIPELINE_GROUP_SIZE=16 RAIDEN_PARALLELISM=16 VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 TPU_RAIDEN_DATA_NICS=eth0}"
export TRAINER_LIBTPU_INIT_ARGS="${TRAINER_LIBTPU_INIT_ARGS:---DANGEROUS_tpu_runtime_abi_verification_disabled=true --xla_tpu_use_tc_device_shape_on_sc=true --xla_sc_disable_megacore_partitioning=true --xla_tpu_enable_offloading_gather_to_sparsecore=true --xla_tpu_enable_sparse_core_collective_offload_all_gather=true --xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true --xla_tpu_enable_sparse_core_reduce_scatter_v2=true --xla_tpu_use_single_sparse_core_for_all_gather_offload=true --xla_tpu_enable_concurrent_sparse_core_offloading=true --xla_tpu_aggressive_opt_barrier_removal=true --xla_tpu_scoped_vmem_limit_kib=65536 --xla_tpu_enable_sublane_major_scaling_bitcast_fusion=false}"
export TRAINER_EXTRA_ENV="${TRAINER_EXTRA_ENV:-ONEHOT_MOE_PERMUTE_THRESHOLD=131072 RAIDEN_TRANSPORT_COALESCE_WINDOW_BYTES=67108864 RAIDEN_WEIGHT_SYNC_PIPELINE_GROUP_SIZE=16 TPU_RAIDEN_DATA_NICS=eth0 LIBTPU_INIT_ARGS='${TRAINER_LIBTPU_INIT_ARGS}'}"

# Hyperparameters & DeepSWE Pipeline Configuration
export TRAIN_MICRO_BATCH_SIZE="${TRAIN_MICRO_BATCH_SIZE:-16}"
export CHECKPOINT_SAVE_INTERVAL_STEPS=${CHECKPOINT_SAVE_INTERVAL_STEPS:-0}
if [[ "${CHECKPOINT_SAVE_INTERVAL_STEPS}" -gt 0 ]]; then
  export ENABLE_PATHWAYS_PERSISTENCE=${ENABLE_PATHWAYS_PERSISTENCE:-1}
fi

export RPC_TIMEOUT_S="${RPC_TIMEOUT_S:-10800}"
export MAXTEXT_EXTRA_FLAGS="${MAXTEXT_EXTRA_FLAGS:-custom_mesh_and_rule=cp-as-ep}"
export DEBUG=${DEBUG:-0}

# DeepSWE Environment & Agent Sandbox
export SANDBOX_NODE_SELECTOR_VAL="sandbox-np"
export SANDBOX_TOLERATIONS='[{"key":"workload","operator":"Equal","value":"sandbox","effect":"NoSchedule"}]'
export IMAGE_REWRITE_PREFIX="${IMAGE_REWRITE_PREFIX:-us-central1-docker.pkg.dev/cloud-tpu-multipod-dev/tunix/}"

source "${DIR}/mlperf_base.sh" "$@"
