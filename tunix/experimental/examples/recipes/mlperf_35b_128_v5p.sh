#!/bin/bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Fill these before you run.
# k8s has a 63 char limit on total label name, so keep job_prefix unique to your job and short
export JOB_PREFIX="${JOB_PREFIX:-${USER}}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-${JOB_PREFIX}-mlperf-35b}"


# Model configuration
export MODEL_NAME="Qwen3.5-35B-A3B"
export MODEL_ID="Qwen/Qwen3.5-35B-A3B"
export TOKENIZER_PATH="Qwen/Qwen3.5-35B-A3B"
export MAXTEXT_MODEL_NAME="qwen3.5-35b-a3b"
if [[ -z "${MAXTEXT_CKPT:-}" ]]; then
  echo "Error: MAXTEXT_CKPT must be set (e.g. export MAXTEXT_CKPT=\"gs://<your-bucket>/checkpoints/...\")" >&2
  exit 1
fi

# Topologies (64 chips Trainer 4x4x4, 16x 4-chip Rollout slices)
export TRAINER_JOBSET_YAML="jobset.pathways.yaml"
export TRAINER_TPU_SLICE="tpuv5:4x4x4"
export TRAINER_MESH_FSDP=32
export TRAINER_MESH_TP=2
export TRAINER_MESH_EXPERT=1

export ROLLOUT_JOBSET_YAML="jobset.tpu.yaml"
export ROLLOUT_TPU_SLICE="tpuv5:2x2x1"
export ROLLOUT_MESH_EXPERT="${ROLLOUT_MESH_EXPERT:-4}"
export ROLLOUT_REPLICAS="${ROLLOUT_REPLICAS:-16}"

# vLLM Rollout Configuration
export VLLM_ADDITIONAL_CONFIG='{"sharding":{"sharding_strategy":{"expert_parallelism":4,"tensor_parallelism":1,"enable_dp_attention":true}},"custom_mamba_cache_multiplier":16,"maxtext_config":{"scan_layers":false,"attention":"vllm_rpa","allow_split_physical_axes":true,"use_multimodal":false,"prefuse_moe_weights":true}}'

# Sandbox
export SANDBOX_NODE_SELECTOR_VAL="sandbox-cpu-pool"

source "${DIR}/mlperf_base.sh" "$@"
