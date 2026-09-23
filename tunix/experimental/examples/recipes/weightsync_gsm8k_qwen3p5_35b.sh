#!/usr/bin/env bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ==============================================================================
# Recipe: Qwen3.5-35B-A3B Distributed GSM8K Weight Sync Benchmark (Raiden & GCS)
# ==============================================================================
# Launches Qwen3.5-35B-A3B GRPO training on GSM8K across a multi-host Pathways
# Trainer (tpuv5:2x2x2, 8 chips) and 2 vLLM Rollout replicas (tpuv5:2x2x1, 4 chips
# each) on GKE, supporting both WEIGHT_SYNC_MODE=raiden and WEIGHT_SYNC_MODE=gcs.
#
# Usage:
#   USER=noghabi WEIGHT_SYNC_MODE=raiden bash tunix/experimental/examples/recipes/weightsync_gsm8k_qwen3p5_35b.sh start
#   USER=noghabi WEIGHT_SYNC_MODE=gcs    bash tunix/experimental/examples/recipes/weightsync_gsm8k_qwen3p5_35b.sh start
#   USER=noghabi bash tunix/experimental/examples/recipes/weightsync_gsm8k_qwen3p5_35b.sh stop
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="${SCRIPT_DIR}/../math_gsm8k_dist/k8s_launcher.sh"

# --- Cluster & Container Infrastructure ---
export CLUSTER="${CLUSTER:-bodaborg-v5p-nap}"
export ZONE="${ZONE:-europe-west4}"
export PROJECT="${PROJECT:-cloud-tpu-shared-capacity}"
export K8S_NAMESPACE="${K8S_NAMESPACE:-trellis}"
export KUEUE_QUEUE="${KUEUE_QUEUE:-multislice-queue}"
export KUEUE_QUEUE_NAME="${KUEUE_QUEUE_NAME:-${KUEUE_QUEUE}}"
export PRIORITY_CLASS="${PRIORITY_CLASS:-medium}"
export SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-xpk-sa}"
export CPU_MACHINE="${CPU_MACHINE:-n2d-standard-64}"

export TUNIX_IMAGE="${TUNIX_IMAGE:-gcr.io/cloud-tpu-multipod-dev/atwigg/trellis-35b:latest}"
export PATHWAYS_SERVER_IMAGE="${PATHWAYS_SERVER_IMAGE:-us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/datenglin/unsanitized_server:raiden_20260923}"
export PATHWAYS_PROXY_IMAGE="${PATHWAYS_PROXY_IMAGE:-us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/datenglin/unsanitized_proxy_server:raiden_20260923}"

export WANDB_PROJECT="${WANDB_PROJECT:-trellis-gsm8k-qwen35-35b}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-${USER:-user}-qwen35-35b-$(date +%m%d-%H%M)}"
export LOG_DIR="/tmp/trellis_gsm8k"
export TRAJECTORY_LOG_DIR="/tmp/trellis_gsm8k/trajectories"

export ENABLE_PROFILER="${ENABLE_PROFILER:-false}"
export PROFILER_PATH="${PROFILER_PATH:-gs://cloud-tpu-tunix-dev/profiles/${WANDB_RUN_NAME}}"
export FIRST_PROFILE_STEP="${FIRST_PROFILE_STEP:-1}"
export NUM_PROFILE_STEPS="${NUM_PROFILE_STEPS:-3}"

# --- Model & Checkpoint Configuration ---
export MODEL_ID="Qwen/Qwen3.5-35B-A3B"
export TOKENIZER_PATH="Qwen/Qwen3.5-35B-A3B"
export MODEL_NAME="qwen3.5-35b-a3b"
export MAXTEXT_MODEL_NAME="qwen3.5-35b-a3b"
export MAXTEXT_CKPT="${MAXTEXT_CKPT:-gs://hengtaoguo-maxtext-logs/checkpoints/qwen3.5-35b-a3b/scanned/2026-06-11-10-27/0/items}"
export MAXTEXT_OUTPUT_DIR="${MAXTEXT_OUTPUT_DIR:-gs://cloud-tpu-tunix-dev/maxtext-outputs}"

export TRAINER_BACKEND="maxtext"
export SAMPLER="vllm"

# --- Weight Synchronization & MoE Configuration ---
export WEIGHT_SYNC_MODE="${WEIGHT_SYNC_MODE:-raiden}"
export WEIGHT_SYNC_GCS_DIR="${WEIGHT_SYNC_GCS_DIR:-gs://noghabi-trellis-ane1/weight_sync_staging_k8s_35b}"
export WEIGHT_SYNC_GCS_MAX_TO_KEEP="${WEIGHT_SYNC_GCS_MAX_TO_KEEP:-1}"
export USE_WEIGHT_CONVERTER="true"
export PREFUSE_MOE_WEIGHTS="true"
export TRAINER_PREFUSE_MOE_WEIGHTS="false"
export ROLLOUT_PREFUSE_MOE_WEIGHTS="true"
export VERIFY_WEIGHTS="${VERIFY_WEIGHTS:-true}"
export RAIDEN_DEVICES_PER_HOST="4"
export USE_ROLLOUT_LOGPS="false"

# --- Trainer Topology & Memory (2 Hosts, 8 TPU v5p Chips) ---
export TRAINER_JOBSET_YAML="jobset.pathways.yaml"
export TRAINER_TPU_SLICE="${TRAINER_TPU_SLICE:-tpuv5:2x2x2}"
export TRAINER_MESH_FSDP="${TRAINER_MESH_FSDP:-4}"
export TRAINER_MESH_TP="${TRAINER_MESH_TP:-2}"
export TRAINER_MESH_EXPERT="${TRAINER_MESH_EXPERT:-1}"
export TRAINER_BASE_NUM_KV_HEADS="2"
export TRAINABLE_PARAMETERS_MASK='^(?!.*routed_experts/gate/kernel).*'
export REMAT_POLICY="full"
export TRAINER_MAXTEXT_ATTENTION="flash"
export CKPT_D2H_CONCURRENT_GB="96"

export PATHWAYS_PROXY_MEMORY_LIMIT="300G"
export PATHWAYS_PROXY_MEMORY="240G"
export PATHWAYS_RM_MEMORY="32G"
export USER_CONTAINER_MEMORY="260G"
export USER_CONTAINER_MEMORY_LIMIT="300G"
export PATHWAYS_WORKER_MEMORY="100G"

# --- Rollout Topology & vLLM Configuration (2x 4-Chip TPU v5p Replicas) ---
export ROLLOUT_JOBSET_YAML="jobset.tpu.yaml"
export ROLLOUT_TPU_SLICE="${ROLLOUT_TPU_SLICE:-tpuv5:2x2x1}"
export ROLLOUT_REPLICAS="${ROLLOUT_REPLICAS:-2}"
export ROLLOUT_MESH_FSDP="4"
export ROLLOUT_MESH_TP="1"

export VLLM_MAX_MODEL_LEN="65536"
export VLLM_MAX_NUM_BATCHED_TOKENS="2048"
export VLLM_MAX_NUM_SEQS="8"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.5}"
export VLLM_DATA_PARALLEL_SIZE="1"
export VLLM_ENABLE_EXPERT_PARALLEL="true"
export VLLM_KV_CACHE_DTYPE="fp8"
export VLLM_ASYNC_SCHEDULING="true"
export VLLM_ADDITIONAL_CONFIG='{"sharding":{"sharding_strategy":{"expert_parallelism":4,"tensor_parallelism":1}}}'
export VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS='{"enable_thinking":true}'
export VLLM_LIMIT_MM_PER_PROMPT='{"image":0,"video":0,"audio":0}'
export VLLM_LANGUAGE_MODEL_ONLY="true"
export VLLM_ENABLE_AUTO_TOOL_CHOICE="true"
export VLLM_TOOL_CALL_PARSER="qwen3_coder"
export VLLM_REASONING_PARSER="qwen3"

# --- GRPO Hyperparameters ---
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
export BATCH_SIZE="${BATCH_SIZE:-2}"
export MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-${BATCH_SIZE}}"
export NUM_GENERATIONS="${NUM_GENERATIONS:-2}"
export MAX_STEPS="${MAX_STEPS:-1}"
export TRAIN_MICRO_BATCH_SIZE="${TRAIN_MICRO_BATCH_SIZE:-4}"
export COMPUTE_LOGPS_CHUNK_SIZE="4"
export LEARNING_RATE="3e-6"
export SCHEDULE_TYPE="constant"
export WARMUP_STEPS="0"
export OPT_CHAIN_TYPE="clip_by_global_norm"
export MAX_GRAD_NORM="1.0"
export CHECKPOINT_SAVE_INTERVAL_STEPS="10"
export CHECKPOINT_MAX_TO_KEEP="3"
export CHECKPOINT_ROOT_DIRECTORY="gs://cloud-tpu-tunix-dev/checkpoints/${WANDB_RUN_NAME}"
export DATA_DIR="/tmp/gsm8k_data"
export REWARD_MODE="env"

# --- Build vLLM Config JSON & Recipe-Specific Passthrough Flags ---
VLLM_JSON=$("${PYTHON:-python3}" -c '
import json, os
cfg = {
    "max_model_len": int(os.environ["VLLM_MAX_MODEL_LEN"]),
    "max_num_batched_tokens": int(os.environ["VLLM_MAX_NUM_BATCHED_TOKENS"]),
    "max_num_seqs": int(os.environ["VLLM_MAX_NUM_SEQS"]),
    "gpu_memory_utilization": float(os.environ["VLLM_GPU_MEMORY_UTILIZATION"]),
    "data_parallel_size": int(os.environ["VLLM_DATA_PARALLEL_SIZE"]),
    "enable_expert_parallel": os.environ["VLLM_ENABLE_EXPERT_PARALLEL"].lower() in ("true", "1"),
    "kv_cache_dtype": os.environ["VLLM_KV_CACHE_DTYPE"],
    "async_scheduling": os.environ["VLLM_ASYNC_SCHEDULING"].lower() in ("true", "1"),
    "language_model_only": os.environ["VLLM_LANGUAGE_MODEL_ONLY"].lower() in ("true", "1"),
    "enable_auto_tool_choice": os.environ["VLLM_ENABLE_AUTO_TOOL_CHOICE"].lower() in ("true", "1"),
    "tool_call_parser": os.environ["VLLM_TOOL_CALL_PARSER"],
    "reasoning_parser": os.environ["VLLM_REASONING_PARSER"],
    "additional_config": json.loads(os.environ["VLLM_ADDITIONAL_CONFIG"]),
    "default_chat_template_kwargs": json.loads(os.environ["VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS"]),
    "limit_mm_per_prompt": json.loads(os.environ["VLLM_LIMIT_MM_PER_PROMPT"]),
}
print(json.dumps(cfg, separators=(",", ":")))
')

export ORCHESTRATOR_EXTRA_ENV="WEIGHT_SYNC_MODE=\"${WEIGHT_SYNC_MODE}\" WEIGHT_SYNC_GCS_DIR=\"${WEIGHT_SYNC_GCS_DIR}\" WEIGHT_SYNC_GCS_MAX_TO_KEEP=${WEIGHT_SYNC_GCS_MAX_TO_KEEP}"

export TRAINER_EXTRA_ENV="WEIGHT_SYNC_MODE=\"${WEIGHT_SYNC_MODE}\" WEIGHT_SYNC_GCS_DIR=\"${WEIGHT_SYNC_GCS_DIR}\" WEIGHT_SYNC_GCS_MAX_TO_KEEP=${WEIGHT_SYNC_GCS_MAX_TO_KEEP} RAIDEN_DEVICES_PER_HOST=${RAIDEN_DEVICES_PER_HOST} USE_WEIGHT_CONVERTER=${USE_WEIGHT_CONVERTER} PREFUSE_MOE_WEIGHTS=${TRAINER_PREFUSE_MOE_WEIGHTS} ROLLOUT_PREFUSE_MOE_WEIGHTS=${ROLLOUT_PREFUSE_MOE_WEIGHTS} ROLLOUT_MESH_TP=${ROLLOUT_MESH_TP} ROLLOUT_TENSOR_PARALLEL_SIZE=${ROLLOUT_MESH_TP} FLOAT32_GATE_LOGITS=true FLOAT32_LOGITS=true"
export TRAINER_EXTRA_ARGS="--prefuse_moe_weights=${TRAINER_PREFUSE_MOE_WEIGHTS} --trainable_parameters_mask='${TRAINABLE_PARAMETERS_MASK}' --remat_policy=${REMAT_POLICY} --maxtext_attention=${TRAINER_MAXTEXT_ATTENTION} --base_num_kv_heads=${TRAINER_BASE_NUM_KV_HEADS}"

ROLLOUT_RAIDEN_ENV=""
if [[ "${WEIGHT_SYNC_MODE}" == "raiden" ]]; then
  ROLLOUT_RAIDEN_ENV="USE_RAIDEN_FFI=false RAIDEN_USE_FFI=0"
fi
export ROLLOUT_EXTRA_ENV="PYTHONUNBUFFERED=1 TUNIX_IS_INTERNAL_ENV=false WEIGHT_SYNC_MODE=\"${WEIGHT_SYNC_MODE}\" WEIGHT_SYNC_GCS_DIR=\"${WEIGHT_SYNC_GCS_DIR}\" WEIGHT_SYNC_GCS_MAX_TO_KEEP=${WEIGHT_SYNC_GCS_MAX_TO_KEEP} RAIDEN_DEVICES_PER_HOST=${RAIDEN_DEVICES_PER_HOST} PREFUSE_MOE_WEIGHTS=${ROLLOUT_PREFUSE_MOE_WEIGHTS} ROLLOUT_PREFUSE_MOE_WEIGHTS=${ROLLOUT_PREFUSE_MOE_WEIGHTS} ROLLOUT_MESH_TP=${ROLLOUT_MESH_TP} ROLLOUT_TENSOR_PARALLEL_SIZE=${ROLLOUT_MESH_TP} ENABLE_PREFIX_CACHING=false VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS} VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION} NUM_PRECOMPILE_WORKERS=4 NEW_MODEL_DESIGN=1 ATTN_BUCKETIZED_NUM_REQS=0 ATTN_CUSTOM_NUM_REQS_BUCKETS=1,4,8 ONEHOT_MOE_PERMUTE_THRESHOLD=32768 VLLM_MOE_CHUNK_SIZE=256 SLICE_ROPE_CACHE=1 DP_SCHED_BATCH_PREFILL=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_LOGGING_LEVEL=INFO LIBTPU_INIT_ARGS=\"--xla_tpu_NextDone_wait_WarnThresholdSeconds=180 --xla_tpu_dvfs_p_state=7\" ${ROLLOUT_RAIDEN_ENV}"
export ROLLOUT_EXTRA_ARGS="--prefuse_moe_weights=${ROLLOUT_PREFUSE_MOE_WEIGHTS} --tensor_parallel_size=${ROLLOUT_MESH_TP} --vllm_config_json='${VLLM_JSON}'"

exec bash "${LAUNCHER}" "$@"
