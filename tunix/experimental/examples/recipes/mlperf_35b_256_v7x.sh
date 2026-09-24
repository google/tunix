#!/bin/bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Fill these before you run.
# k8s has a 63 char limit on total label name, so keep job_prefix unique to your job and short
export JOB_PREFIX="${JOB_PREFIX:-${USER}}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-${JOB_PREFIX}-mlperf-35b-v7x}"
export MAXTEXT_OUTPUT_DIR="${MAXTEXT_OUTPUT_DIR:-gs://niting-maxtext-storage/maxtext/${JOB_PREFIX}}"
export TRAJECTORY_LOG_DIR="${TRAJECTORY_LOG_DIR:-gs://niting-maxtext-storage/trajectories/${JOB_PREFIX}}"
export TUNIX_IMAGE="${TUNIX_IMAGE:-gcr.io/cloud-tpu-multipod-dev/niting/trellis-35b:latest}"

export PROJECT="cloud-tpu-shared-capacity"
export REGION="us-central1"
export CLUSTER="bodaborg-tpu7x-gsc"
export K8S_NAMESPACE="priority-dev"
export USE_DYNAMIC_SLICING="true"
export TPU_RESERVATION="${TPU_RESERVATION:-ghostfish-pogoag4tylwed}"
export ENABLE_PATHWAYS_PERSISTENCE=1

# Container memory overrides for v7x
export PATHWAYS_PROXY_MEMORY_LIMIT="100G"
export USER_CONTAINER_MEMORY="48G"
export USER_CONTAINER_MEMORY_LIMIT="${USER_CONTAINER_MEMORY_LIMIT:-70G}"
export TPU_RAIDEN_DATA_NICS="eth0"

# Model configuration
export MODEL_NAME="Qwen3.5-35B-A3B"
export MODEL_ID="Qwen/Qwen3.5-35B-A3B"
export TOKENIZER_PATH="Qwen/Qwen3.5-35B-A3B"
export MAXTEXT_MODEL_NAME="qwen3.5-35b-a3b"
export MAXTEXT_CKPT="${MAXTEXT_CKPT:-gs://hengtaoguo-maxtext-logs/checkpoints/qwen3.5-35b-a3b/scanned/2026-06-11-10-27/0/items}"

# Topologies (64 chips Trainer 4x4x4, 16x 4-chip Rollout slices on TPU7x dynamic slicing)
export TRAINER_JOBSET_YAML="jobset.pathways.yaml"
export TRAINER_TPU_SLICE="tpu7x:4x4x4"
export TRAINER_MESH_FSDP=32
export TRAINER_MESH_TP=2
export TRAINER_MESH_EXPERT=1
export TRAINER_MESH_CONTEXT=2

export ROLLOUT_JOBSET_YAML="jobset.tpu.yaml"
export ROLLOUT_TPU_SLICE="tpu7x:2x2x1"
export ROLLOUT_MESH_EXPERT="${ROLLOUT_MESH_EXPERT:-4}"
export ROLLOUT_REPLICAS="${ROLLOUT_REPLICAS:-16}"

# vLLM Rollout Configuration (from paste.googleplex.com/5903655694368768)
export VLLM_ADDITIONAL_CONFIG='{"sharding":{"sharding_strategy":{"expert_parallelism":4,"tensor_parallelism":1,"enable_dp_attention":true}},"custom_mamba_cache_multiplier":16,"maxtext_config":{"scan_layers":false,"attention":"vllm_rpa","allow_split_physical_axes":true,"use_multimodal":false,"prefuse_moe_weights":true}}'

# Rollout Worker Environment Flags
export LIBTPU_INIT_ARGS=' --xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false --xla_tpu_check_legacy_constraints_in_reduce_scatter_legalizer=false --xla_tpu_dvfs_p_state=7'

# Sandbox
export SANDBOX_NODE_SELECTOR_VAL="sandbox-np"
export IMAGE_REWRITE_PREFIX="${IMAGE_REWRITE_PREFIX:-us-central1-docker.pkg.dev/cloud-tpu-multipod-dev/tunix/}"

source "${DIR}/mlperf_base.sh" "$@"
