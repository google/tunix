#!/bin/bash
set -e

# ==============================================================================
# MLPerf DeepSWE recipe: Qwen3.5-397B-A17B on TPU v5p
# ==============================================================================
# - Cluster: bodaborg-v5p-nap in europe-west4
# - Trainer on 256 chips (tpuv5p:4x8x8, FSDP=16, TP=1, EXPERT=2, CONTEXT=8)
# - Rollout on 384 chips (4 replicas x 16 chips tpuv5p:2x2x4, EP=16, TP=1)
# - Sandboxes on sandbox-cpu-pool
# ==============================================================================

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Fill these before you run.
export JOB_PREFIX="${JOB_PREFIX:-$USER}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-${JOB_PREFIX}-mlperf-397b}"
export MAXTEXT_OUTPUT_DIR="${MAXTEXT_OUTPUT_DIR:-gs://atwigg-trellis-europe-west4-dev/maxtext/${JOB_PREFIX}}"
export TRAJECTORY_LOG_DIR="${TRAJECTORY_LOG_DIR:-gs://atwigg-trellis-europe-west4-dev/trajectories/${JOB_PREFIX}/logger}"
export TRAJECTORY_STORE_ROOT_DIR="${TRAJECTORY_STORE_ROOT_DIR:-${TRAJECTORY_STORE_ROOT:-gs://atwigg-trellis-europe-west4-dev/trajectories/${JOB_PREFIX}/store}}"

export PROJECT="cloud-tpu-shared-capacity"
export REGION="europe-west4"
export CLUSTER="bodaborg-v5p-nap"
export K8S_NAMESPACE="trellis"

export RAIDEN_DEVICES_PER_HOST=4

# Model configuration
export MODEL_NAME="Qwen3.5-397B-A17B"
export MODEL_ID="Qwen/Qwen3.5-397B-A17B"
export TOKENIZER_PATH="${TOKENIZER_PATH:-Qwen/Qwen3.5-397B-A17B}"
export MAXTEXT_MODEL_NAME="qwen3.5-397b-a17b"
export MAXTEXT_CKPT="${MAXTEXT_CKPT:-gs://sanbao-europe/qwen35_397b/scanned_reshard_fsdp32_tp2/0/items}"

# Trainer: 256 chips. TRAINER_MESH_EXPERT=2 is required, not a tuning choice:
# at expert=1 the GMM_v2 kernel overflows smem by ~8.6K.
export TRAINER_JOBSET_YAML="jobset.pathways.qwen3.5-397b.yaml"
export TRAINER_TPU_SLICE="tpuv5p:4x8x8"          # 256 chips
export TRAINER_MESH_FSDP=16
export TRAINER_MESH_TP=1
export TRAINER_MESH_EXPERT=2
export TRAINER_MESH_CONTEXT=8                    # 16*1*2*8 = 256

# Rollout: 16 chips = 4 hosts per replica. Use tp=1 x expert=16.
export ROLLOUT_JOBSET_YAML="jobset.mcjax.ray.yaml"   # 16 chips = 4 hosts -> Ray multihost
export ROLLOUT_TPU_SLICE="tpuv5p:2x2x4"
export ROLLOUT_MESH_EXPERT=16
export ROLLOUT_WORKERS="${ROLLOUT_WORKERS:-4}"
export ROLLOUT_REPLICAS="${ROLLOUT_REPLICAS:-4}"

# vLLM Rollout Configuration
export VLLM_ADDITIONAL_CONFIG='{"sharding":{"sharding_strategy":{"expert_parallelism":16,"tensor_parallelism":1,"enable_dp_attention":true}},"custom_mamba_cache_multiplier":16,"maxtext_config":{"scan_layers":false,"attention":"vllm_rpa","allow_split_physical_axes":true,"use_multimodal":false,"prefuse_moe_weights":true}}'
export ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-false}"
export VLLM_PREFIX_CACHE_RETENTION_INTERVAL="${VLLM_PREFIX_CACHE_RETENTION_INTERVAL:-256}"
export MAMBA_CACHE_MODE="${MAMBA_CACHE_MODE:-none}"
export VLLM_MAMBA_CACHE_MODE="${VLLM_MAMBA_CACHE_MODE:-${MAMBA_CACHE_MODE}}"

# Rollout Worker Flags & Raiden tuning
export ONEHOT_MOE_PERMUTE_THRESHOLD=131072
export ORCHESTRATOR_EXTRA_ENV="${ORCHESTRATOR_EXTRA_ENV:-WEIGHT_SYNC_TIMEOUT_H2D=1800 RAIDEN_PARALLELISM=16}"
export ROLLOUT_EXTRA_ENV="${ROLLOUT_EXTRA_ENV:-RAY_memory_monitor_refresh_ms=0 RAIDEN_TRANSPORT_COALESCE_WINDOW_BYTES=67108864 RAIDEN_WEIGHT_SYNC_PIPELINE_GROUP_SIZE=16 RAIDEN_PARALLELISM=16}"
export TRAINER_LIBTPU_INIT_ARGS="${TRAINER_LIBTPU_INIT_ARGS:---DANGEROUS_tpu_runtime_abi_verification_disabled=true --xla_tpu_use_tc_device_shape_on_sc=true --xla_sc_disable_megacore_partitioning=true --xla_tpu_enable_offloading_gather_to_sparsecore=true --xla_tpu_enable_sparse_core_collective_offload_all_gather=true --xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true --xla_tpu_enable_sparse_core_reduce_scatter_v2=true --xla_tpu_use_single_sparse_core_for_all_gather_offload=true --xla_tpu_enable_concurrent_sparse_core_offloading=true --xla_tpu_aggressive_opt_barrier_removal=true --xla_tpu_scoped_vmem_limit_kib=65536 --xla_tpu_enable_sublane_major_scaling_bitcast_fusion=false}"
export TRAINER_EXTRA_ENV="${TRAINER_EXTRA_ENV:-RAIDEN_TRANSPORT_COALESCE_WINDOW_BYTES=67108864 RAIDEN_WEIGHT_SYNC_PIPELINE_GROUP_SIZE=16 LIBTPU_INIT_ARGS='${TRAINER_LIBTPU_INIT_ARGS}'}"

# Hyperparameters
export CHECKPOINT_SAVE_INTERVAL_STEPS=${CHECKPOINT_SAVE_INTERVAL_STEPS:-0}
export RPC_TIMEOUT_S="${RPC_TIMEOUT_S:-10800}"
export DEBUG=${DEBUG:-0}

# Sandbox
export SANDBOX_NODE_SELECTOR_VAL="sandbox-cpu-pool"
export IMAGE_REWRITE_PREFIX="${IMAGE_REWRITE_PREFIX:-europe-west4-docker.pkg.dev/cloud-tpu-multipod-dev/tunix/}"

source "${DIR}/mlperf_base.sh" "$@"
