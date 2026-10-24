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

set -euo pipefail

# ==============================================================================
# MLPerf DeepSWE recipe: Qwen3.5-35B-A3B on TPU v5p (128 chips)
# ==============================================================================
# - Trainer on 64 chips (tpuv5:4x4x4 = 64 devices, FSDP=32, TP=1, EXPERT=1, CP=2)
# - Rollout on 64 chips (16 replicas x 4 chips tpuv5:2x2x1, EP=4, TP=1)
# - Sandboxes on Kubernetes Agent Sandbox warmpool
# ==============================================================================

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f "${DIR}/google/mlperf_35b_internal_env.sh" ]]; then
  # shellcheck source=/dev/null
  source "${DIR}/google/mlperf_35b_internal_env.sh"
fi

# Allow --image=... or --image ... on the CLI to satisfy TUNIX_IMAGE check.
_cli_prev=""
for _arg in "$@"; do
  if [[ "${_cli_prev}" == "--image" ]]; then
    export TUNIX_IMAGE="${_arg}"
    _cli_prev=""
    continue
  fi
  case "${_arg}" in
    --image=*) export TUNIX_IMAGE="${_arg#*=}" ;;
    --image) _cli_prev="--image" ;;
  esac
done

for _req_var in TUNIX_IMAGE PATHWAYS_SERVER_IMAGE PATHWAYS_PROXY_IMAGE BUCKET MAXTEXT_CKPT DATASET_PATH; do
  if [[ -z "${!_req_var:-}" ]]; then
    echo "Error: ${_req_var} must be set in the environment before launching mlperf_35b_128_v5p.sh." >&2
    exit 1
  fi
done

export JOB_PREFIX="${JOB_PREFIX:-${USER}}"
export ORCHESTRATOR_ID="${ORCHESTRATOR_ID:-${JOB_PREFIX}-orch}"
export RUN_ID="${RUN_ID:-${JOB_PREFIX}}"
export WANDB_ENTITY="${WANDB_ENTITY:-google-trellis}"
export WANDB_PROJECT="${WANDB_PROJECT:-trellis-deepswe}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-${JOB_PREFIX}-mlperf-35b}"

export NAMESPACE="${NAMESPACE:-${K8S_NAMESPACE:-default}}"
export K8S_NAMESPACE="${K8S_NAMESPACE:-${NAMESPACE}}"
export SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-xpk-sa}"
export PRIORITY_CLASS="${PRIORITY_CLASS:-medium}"
export CPU_MACHINE="${CPU_MACHINE:-n2-standard-64}"
export CPU_NODEPOOL="${CPU_NODEPOOL:-}"
export CPU_MEMORY="${CPU_MEMORY:-240G}"

export MAXTEXT_OUTPUT_DIR="${MAXTEXT_OUTPUT_DIR:-${BUCKET}/maxtext/${JOB_PREFIX}}"
export LOG_DIR="${LOG_DIR:-${BUCKET}/logs/${JOB_PREFIX}}"
export TRAJECTORY_LOG_DIR="${TRAJECTORY_LOG_DIR:-${BUCKET}/trajectories/${JOB_PREFIX}/logger}"
export TRAJECTORY_LOGGING="${TRAJECTORY_LOGGING:-false}"
export TRAJECTORY_STORE_ENABLED="${TRAJECTORY_STORE_ENABLED:-true}"
export TRAJECTORY_STORE_BACKEND="${TRAJECTORY_STORE_BACKEND:-file}"
export TRAJECTORY_STORE_DIR="${TRAJECTORY_STORE_DIR:-${TRAJECTORY_STORE_ROOT_DIR:-${BUCKET}/trajectories/${JOB_PREFIX}/store}}"
export CHECKPOINT_ROOT_DIRECTORY="${CHECKPOINT_ROOT_DIRECTORY:-${BUCKET}/checkpoints/${JOB_PREFIX}}"
export GCS_SCRATCH_LOCATION="${GCS_SCRATCH_LOCATION:-${BUCKET}/pathways-staging/${JOB_PREFIX}}"
export DATASET_CACHE_DIR="${DATASET_CACHE_DIR:-/tmp/hf_datasets_cache}"
export LOCAL_JAX_CACHE_DIR="${LOCAL_JAX_CACHE_DIR:-/tmp/jax_cache}"
export ROLLOUT_JAX_CACHE_GCS_DIR="${ROLLOUT_JAX_CACHE_GCS_DIR:-${BUCKET}/jax_cache/vllm_rpa_ep4_v1}"
export SAVE_JAX_CACHE="${SAVE_JAX_CACHE:-true}"

# Pathways + Raiden image and memory configuration
export PATHWAYS_PROXY_MEMORY="${PATHWAYS_PROXY_MEMORY:-${PATHWAYS_PROXY_MEMORY_REQUEST:-16G}}"
export PATHWAYS_PROXY_MEMORY_LIMIT="${PATHWAYS_PROXY_MEMORY_LIMIT:-160G}"
export USER_CONTAINER_MEMORY="${USER_CONTAINER_MEMORY:-${PATHWAYS_USER_MEMORY_REQUEST:-260G}}"
export USER_CONTAINER_MEMORY_LIMIT="${USER_CONTAINER_MEMORY_LIMIT:-${PATHWAYS_USER_MEMORY_LIMIT:-260G}}"
export PATHWAYS_WORKER_MEMORY="${PATHWAYS_WORKER_MEMORY:-100G}"
export PATHWAYS_RM_MEMORY="${PATHWAYS_RM_MEMORY:-4G}"

# v5p has 1 device/chip and 4 chips (4 devices) per host.
export RAIDEN_DEVICES_PER_HOST="${RAIDEN_DEVICES_PER_HOST:-4}"
export RAIDEN_FFI_USE_DIRECT_DEVICE_BUFFER="${RAIDEN_FFI_USE_DIRECT_DEVICE_BUFFER:-0}"
export RAIDEN_BROADCAST_HOST_RATIO="${RAIDEN_BROADCAST_HOST_RATIO:-1.0}"
export RAIDEN_BROADCAST_PIPELINE_STAGES="${RAIDEN_BROADCAST_PIPELINE_STAGES:-4}"
export ENABLE_MULTI_NUMA="${ENABLE_MULTI_NUMA:-0}"
export TPU_RAIDEN_DATA_NICS="${TPU_RAIDEN_DATA_NICS:-eth0}"

# Model configuration
export MODEL_NAME="${MODEL_NAME:-Qwen3.5-35B-A3B}"
export MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-35B-A3B}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_ID}}"
export MAXTEXT_MODEL_NAME="${MAXTEXT_MODEL_NAME:-qwen3.5-35b-a3b}"

# Topologies (64 chips / 64 devices Trainer 4x4x4, 16x 4-chip Rollout slices)
export TRAINER_BACKEND="${TRAINER_BACKEND:-maxtext}"
export TRAINER_PRECOMPILE="${TRAINER_PRECOMPILE:-true}"
export TRAINER_JOBSET_YAML="${TRAINER_JOBSET_YAML:-jobset.pathways.yaml}"
export TRAINER_IFRT_PROXY_LARGE_TRANSFER_THRESHOLD_BYTES="${TRAINER_IFRT_PROXY_LARGE_TRANSFER_THRESHOLD_BYTES:-16777216}"
export TRAINER_TPU_SLICE="${TRAINER_TPU_SLICE:-tpuv5:4x4x4}"
export TRAINER_MESH_FSDP="${TRAINER_MESH_FSDP:-32}"
export TRAINER_MESH_TP="${TRAINER_MESH_TP:-1}"
export TRAINER_MESH_EXPERT="${TRAINER_MESH_EXPERT:-1}"
export TRAINER_MESH_CONTEXT="${TRAINER_MESH_CONTEXT:-2}"
export TRAINER_MAXTEXT_ATTENTION="${TRAINER_MAXTEXT_ATTENTION:-flash}"
export TRAINER_BASE_NUM_KV_HEADS="${TRAINER_BASE_NUM_KV_HEADS:-2}"
export TRAINER_PADDED_MOE_MLP_DIM="${TRAINER_PADDED_MOE_MLP_DIM:-}"
export COMPUTE_LOGPS_CHUNK_SIZE="${COMPUTE_LOGPS_CHUNK_SIZE:-512}"
export PREFUSE_MOE_WEIGHTS="${PREFUSE_MOE_WEIGHTS:-true}"
export REMAT_POLICY="${REMAT_POLICY:-custom}"
export FLOAT32_GATE_LOGITS="${FLOAT32_GATE_LOGITS:-true}"
export FLOAT32_LOGITS="${FLOAT32_LOGITS:-true}"

# Rollout: 16 replicas x 4 chips (tpuv5:2x2x1), EP=4, TP=1.
export ROLLOUT_JOBSET_YAML="${ROLLOUT_JOBSET_YAML:-jobset.tpu.yaml}"
export ROLLOUT_TPU_SLICE="${ROLLOUT_TPU_SLICE:-tpuv5:2x2x1}"
export ROLLOUT_MESH_TP="${ROLLOUT_MESH_TP:-1}"
export ROLLOUT_MESH_FSDP="${ROLLOUT_MESH_FSDP:-1}"
export VLLM_DATA_PARALLEL_SIZE="${VLLM_DATA_PARALLEL_SIZE:-1}"
_rollout_dims="${ROLLOUT_TPU_SLICE#*:}"
export ROLLOUT_MESH_EXPERT="${ROLLOUT_MESH_EXPERT:-$(( ${_rollout_dims//x/*} / ${VLLM_DATA_PARALLEL_SIZE:-1} ))}"
export ROLLOUT_REPLICAS="${ROLLOUT_REPLICAS:-16}"
export MIN_ROLLOUT_WORKERS="${MIN_ROLLOUT_WORKERS:-${ROLLOUT_REPLICAS}}"
export ROLLOUT_SKIP_JAX_PRECOMPILE="${ROLLOUT_SKIP_JAX_PRECOMPILE:-0}"
export EOS_TOKENS="${EOS_TOKENS:-248046,248044}"

# RL & Optimizer Hyperparameters (aligned with v13 / consensus §5.1; OVERLONG_FILTER=false per user request)
export MAX_SEQ_TOKEN_PER_TPU="${MAX_SEQ_TOKEN_PER_TPU:-65536}"
export MAX_SEGMENTS_PER_PACKED_ROW="${MAX_SEGMENTS_PER_PACKED_ROW:-16}"
export SEGMENT_ALIGNMENT_BOUNDARY="${SEGMENT_ALIGNMENT_BOUNDARY:-64}"
export MAX_STEPS="${MAX_STEPS:-50}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
export MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-${BATCH_SIZE}}"
export NUM_GENERATIONS="${NUM_GENERATIONS:-16}"
export TRAIN_MICRO_BATCH_SIZE="${TRAIN_MICRO_BATCH_SIZE:-64}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8192}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-57344}"
export MAX_TURNS="${MAX_TURNS:-30}"
export MAX_STALENESS="${MAX_STALENESS:-1}"
export TRAJECTORY_GROUP_ORDER="${TRAJECTORY_GROUP_ORDER:-prompt_arrival}"

export TEMPERATURE="${TEMPERATURE:-1.0}"
export TOP_P="${TOP_P:-1.0}"
export TOP_K="${TOP_K:--1}"
export LEARNING_RATE="${LEARNING_RATE:-1e-6}"
export WARMUP_STEPS_FRACTION="${WARMUP_STEPS_FRACTION:-0.0}"
export ADAM_B1="${ADAM_B1:-0.9}"
export ADAM_B2="${ADAM_B2:-0.999}"
export ADAM_EPS="${ADAM_EPS:-1e-8}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
export OPT_CHAIN_TYPE="${OPT_CHAIN_TYPE:-clip_by_global_norm}"
export MAX_GRAD_NORM="${MAX_GRAD_NORM:-0.125}"
export BETA="${BETA:-0.0}"
export EPSILON="${EPSILON:-0.2}"
export EPSILON_HIGH="${EPSILON_HIGH:-0.28}"
export ADVANTAGE_ESTIMATOR="${ADVANTAGE_ESTIMATOR:-grpo-loo}"
export LOSS_AGG_MODE="${LOSS_AGG_MODE:-token-mean}"
export SAMPLER_IS="${SAMPLER_IS:-token}"
export SAMPLER_IS_THRESHOLD="${SAMPLER_IS_THRESHOLD:-none}"
export SAMPLER_RS="${SAMPLER_RS:-geometric}"
export SAMPLER_RS_MIN="${SAMPLER_RS_MIN:-0.999}"
export SAMPLER_RS_MAX="${SAMPLER_RS_MAX:-1.002}"
export SEQ_LOGPROB_ERROR_THRESHOLD="${SEQ_LOGPROB_ERROR_THRESHOLD:-2.0}"
export USE_ROLLOUT_LOGPS="${USE_ROLLOUT_LOGPS:-true}"
export OVERLONG_FILTER="${OVERLONG_FILTER:-false}"
export RETURN_ROUTED_EXPERTS="${RETURN_ROUTED_EXPERTS:-true}"
export ROUTED_EXPERTS_ENCODE_MODE="${ROUTED_EXPERTS_ENCODE_MODE:-auto}"
export ROUTED_EXPERTS_NUM_EXPERTS="${ROUTED_EXPERTS_NUM_EXPERTS:-256}"
export FLUSH_EVERY_N_STEPS="${FLUSH_EVERY_N_STEPS:-1}"

# Weight Sync & Checkpointing
export SAMPLER="${SAMPLER:-vllm}"
export WEIGHT_SYNC_MODE="${WEIGHT_SYNC_MODE:-raiden}"
export USE_WEIGHT_CONVERTER="${USE_WEIGHT_CONVERTER:-true}"
export IN_FLIGHT_WEIGHT_UPDATES="${IN_FLIGHT_WEIGHT_UPDATES:-false}"
export VERIFY_WEIGHTS="${VERIFY_WEIGHTS:-true}"
export CHECKPOINT_SAVE_INTERVAL_STEPS="${CHECKPOINT_SAVE_INTERVAL_STEPS:-0}"
export CHECKPOINT_MAX_TO_KEEP="${CHECKPOINT_MAX_TO_KEEP:-35}"
export CKPT_D2H_CONCURRENT_GB="${CKPT_D2H_CONCURRENT_GB:-96}"
export ENABLE_PATHWAYS_PERSISTENCE="${ENABLE_PATHWAYS_PERSISTENCE:-1}"

# Dataset & Sandbox Environment
export DATASET_NAME="${DATASET_NAME:-R2E-Gym/R2E-Gym-Subset}"
export DATASET_SPLIT="${DATASET_SPLIT:-train}"
export SHUFFLE="${SHUFFLE:-false}"
export SEED="${SEED:-42}"
export SCAFFOLD="${SCAFFOLD:-openhands}"
export OPENHANDS_MULTI_TOOL_CALLS="${OPENHANDS_MULTI_TOOL_CALLS:-false}"
export ENV_BACKEND="${ENV_BACKEND:-kubernetes-sandbox}"
export USE_AGENT_SANDBOX="${USE_AGENT_SANDBOX:-1}"
export SANDBOX_NAMESPACE="${SANDBOX_NAMESPACE:-${K8S_NAMESPACE}}"
export NODE_SELECTOR_KEY="${NODE_SELECTOR_KEY:-${SANDBOX_NODE_SELECTOR_KEY:-cloud.google.com/gke-nodepool}}"
export NODE_SELECTOR_VAL="${NODE_SELECTOR_VAL:-${SANDBOX_NODE_SELECTOR_VAL:-}}"
export SANDBOX_NODE_SELECTOR_KEY="${SANDBOX_NODE_SELECTOR_KEY:-${NODE_SELECTOR_KEY}}"
export SANDBOX_NODE_SELECTOR_VAL="${SANDBOX_NODE_SELECTOR_VAL:-${NODE_SELECTOR_VAL}}"
export IMAGE_REWRITE_PREFIX="${IMAGE_REWRITE_PREFIX:-}"
export OPENHANDS_SERVER_IMAGE="${OPENHANDS_SERVER_IMAGE:-}"
export MAX_WARMPOOL_REPLICAS="${MAX_WARMPOOL_REPLICAS:-16}"
export MAX_CONCURRENCY="${MAX_CONCURRENCY:-1024}"
export ROLLOUT_MAX_CONCURRENCY="${ROLLOUT_MAX_CONCURRENCY:-256}"
export STEP_TIMEOUT_SECS="${STEP_TIMEOUT_SECS:-60}"
export REWARD_TIMEOUT_SECS="${REWARD_TIMEOUT_SECS:-60}"
export EPISODE_TIMEOUT_SECS="${EPISODE_TIMEOUT_SECS:-1800}"
export RPC_TIMEOUT_S="${RPC_TIMEOUT_S:-10800}"
export INIT_TIMEOUT_S="${INIT_TIMEOUT_S:-7200}"

# vLLM Rollout Configuration
export USE_BATCHED_RPA_LONG_CTX_KERNEL="${USE_BATCHED_RPA_LONG_CTX_KERNEL:-false}"
if [[ "${USE_BATCHED_RPA_LONG_CTX_KERNEL}" == "1" || "${USE_BATCHED_RPA_LONG_CTX_KERNEL}" == "true" ]]; then
  export ROLLOUT_MAXTEXT_ATTENTION="${ROLLOUT_MAXTEXT_ATTENTION:-vllm_batched_rpa}"
else
  export ROLLOUT_MAXTEXT_ATTENTION="${ROLLOUT_MAXTEXT_ATTENTION:-vllm_rpa}"
fi
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-65536}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-16}"
export VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-2048}"
export VLLM_ASYNC_SCHEDULING="${VLLM_ASYNC_SCHEDULING:-true}"
export VLLM_HBM_UTILIZATION="${VLLM_HBM_UTILIZATION:-0.9}"
export ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-true}"
if [[ -z "${VLLM_CONFIG_JSON:-}" ]]; then
  export VLLM_CONFIG_JSON="{\"enable_expert_parallel\":true,\"prefix_cache_retention_interval\":0,\"mamba_cache_mode\":\"align\",\"kv_cache_dtype\":\"bfloat16\",\"block_size\":256,\"enable_chunked_prefill\":true,\"language_model_only\":true,\"reasoning_parser\":\"qwen3\",\"limit_mm_per_prompt\":{\"image\":0,\"video\":0},\"additional_config\":{\"sharding\":{\"sharding_strategy\":{\"expert_parallelism\":${ROLLOUT_MESH_EXPERT},\"tensor_parallelism\":1,\"enable_dp_attention\":true}},\"custom_mamba_cache_multiplier\":16,\"maxtext_config\":{\"use_multimodal\":false,\"per_device_batch_size\":0.0}}}"
fi
python3 -c 'import json, sys; json.loads(sys.argv[1])' "${VLLM_CONFIG_JSON}"

# Rollout Worker Flags & Raiden tuning
export ONEHOT_MOE_PERMUTE_THRESHOLD="${ONEHOT_MOE_PERMUTE_THRESHOLD:-32768}"
export RAGGED_GATHER_REDUCE_VERSION="${RAGGED_GATHER_REDUCE_VERSION:-v2}"
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1800}"
export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:- --xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false --xla_tpu_check_legacy_constraints_in_reduce_scatter_legalizer=false}"
export VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY="${VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY:-RAIDEN_,TPU_,WEIGHT_SYNC_}"
export VLLM_RAY_EXTRA_ENV_VARS_TO_COPY="${VLLM_RAY_EXTRA_ENV_VARS_TO_COPY:-USE_BATCHED_RPA_LONG_CTX_KERNEL,ONEHOT_MOE_PERMUTE_THRESHOLD,RAGGED_GATHER_REDUCE_VERSION,LIBTPU_INIT_ARGS,RAY_memory_monitor_refresh_ms,VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS,ENABLE_MULTI_NUMA,TPU_RAIDEN_DATA_NICS,FLOAT32_GATE_LOGITS,FLOAT32_LOGITS,NEW_MODEL_DESIGN,ATTN_BUCKETIZED_NUM_REQS,ATTN_CUSTOM_NUM_REQS_BUCKETS,VLLM_MOE_CHUNK_SIZE,SLICE_ROPE_CACHE,DP_SCHED_BATCH_PREFILL,WEIGHT_SYNC_PARALLEL_H2H,TPU_SMALL_ALL_REDUCE,TPU_MOE_SINGLE_SORT,TPU_MOE_ROUTER_TOPK,TPU_MOE_LOWTOK}"
export ORCHESTRATOR_EXTRA_ENV="${ORCHESTRATOR_EXTRA_ENV:-WEIGHT_SYNC_BIND_TIMEOUT_S=900 WEIGHT_SYNC_METADATA_TIMEOUT_S=900 WEIGHT_SYNC_H2D_TIMEOUT_S=3600 WEIGHT_SYNC_TRANSFER_TIMEOUT_S=7200 RAIDEN_PARALLELISM=16 TPU_RAIDEN_DATA_NICS=eth0 RAIDEN_BROADCAST_HOST_RATIO=${RAIDEN_BROADCAST_HOST_RATIO} RAIDEN_BROADCAST_PIPELINE_STAGES=${RAIDEN_BROADCAST_PIPELINE_STAGES}}"
export ROLLOUT_EXTRA_ENV="${ROLLOUT_EXTRA_ENV:-RAIDEN_DEVICES_PER_HOST=${RAIDEN_DEVICES_PER_HOST} NUM_PRECOMPILE_WORKERS=8 NEW_MODEL_DESIGN=1 ATTN_BUCKETIZED_NUM_REQS=true ATTN_CUSTOM_NUM_REQS_BUCKETS=4 ONEHOT_MOE_PERMUTE_THRESHOLD=${ONEHOT_MOE_PERMUTE_THRESHOLD} RAGGED_GATHER_REDUCE_VERSION=${RAGGED_GATHER_REDUCE_VERSION} VLLM_MOE_CHUNK_SIZE=256 SLICE_ROPE_CACHE=1 DP_SCHED_BATCH_PREFILL=false VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_LOGGING_LEVEL=INFO TPU_SMALL_ALL_REDUCE=0 TPU_MOE_SINGLE_SORT=0 TPU_MOE_ROUTER_TOPK=sort TPU_MOE_LOWTOK=0 USE_BATCHED_RPA_LONG_CTX_KERNEL=${USE_BATCHED_RPA_LONG_CTX_KERNEL} RAY_memory_monitor_refresh_ms=0 RAIDEN_TRANSPORT_COALESCE_WINDOW_BYTES=67108864 RAIDEN_WEIGHT_SYNC_PIPELINE_GROUP_SIZE=16 RAIDEN_PARALLELISM=16 VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 ENABLE_MULTI_NUMA=${ENABLE_MULTI_NUMA} TPU_RAIDEN_DATA_NICS=eth0 RAIDEN_BROADCAST_HOST_RATIO=${RAIDEN_BROADCAST_HOST_RATIO} RAIDEN_BROADCAST_PIPELINE_STAGES=${RAIDEN_BROADCAST_PIPELINE_STAGES} VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY='${VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY}' VLLM_RAY_EXTRA_ENV_VARS_TO_COPY='${VLLM_RAY_EXTRA_ENV_VARS_TO_COPY}' LIBTPU_INIT_ARGS='${LIBTPU_INIT_ARGS}'}"

# Trainer XLA & MaxText Flags
export TRAINER_LIBTPU_INIT_ARGS="${TRAINER_LIBTPU_INIT_ARGS:---DANGEROUS_tpu_runtime_abi_verification_disabled=true --xla_tpu_use_tc_device_shape_on_sc=true --xla_sc_disable_megacore_partitioning=true --xla_tpu_enable_offloading_gather_to_sparsecore=true --xla_tpu_enable_sparse_core_collective_offload_all_gather=true --xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true --xla_tpu_enable_sparse_core_reduce_scatter_v2=true --xla_tpu_use_single_sparse_core_for_all_gather_offload=true --xla_tpu_enable_concurrent_sparse_core_offloading=true --xla_tpu_aggressive_opt_barrier_removal=true --xla_tpu_scoped_vmem_limit_kib=65536 --xla_tpu_enable_sublane_major_scaling_bitcast_fusion=false}"
export TRAINER_EXTRA_ENV="${TRAINER_EXTRA_ENV:-ONEHOT_MOE_PERMUTE_THRESHOLD=${ONEHOT_MOE_PERMUTE_THRESHOLD} RAIDEN_TRANSPORT_COALESCE_WINDOW_BYTES=67108864 RAIDEN_WEIGHT_SYNC_PIPELINE_GROUP_SIZE=16 ENABLE_MULTI_NUMA=${ENABLE_MULTI_NUMA} TPU_RAIDEN_DATA_NICS=eth0 RAIDEN_BROADCAST_HOST_RATIO=${RAIDEN_BROADCAST_HOST_RATIO} RAIDEN_BROADCAST_PIPELINE_STAGES=${RAIDEN_BROADCAST_PIPELINE_STAGES} LIBTPU_INIT_ARGS='${TRAINER_LIBTPU_INIT_ARGS}'}"
export PATHWAYS_WORKER_EXTRA_ENV="${PATHWAYS_WORKER_EXTRA_ENV:-LIBTPU_INIT_ARGS=${TRAINER_LIBTPU_INIT_ARGS} --megascale_port=-1 --xprof_compress_jftrace=true
SKIP_MEGASCALE_PJRT_CLIENT=true
TPU_RAIDEN_DATA_NICS=eth0
RAIDEN_BROADCAST_HOST_RATIO=${RAIDEN_BROADCAST_HOST_RATIO}
RAIDEN_BROADCAST_PIPELINE_STAGES=${RAIDEN_BROADCAST_PIPELINE_STAGES}}"

_trainer_xla_flags=""
for _f in ${TRAINER_LIBTPU_INIT_ARGS}; do [[ "${_f}" == --xla_* ]] && _trainer_xla_flags+="${_f} "; done
export PATHWAYS_PROXY_EXTRA_ARGS="${PATHWAYS_PROXY_EXTRA_ARGS:-${_trainer_xla_flags% }}"

export MAXTEXT_EXTRA_FLAGS="${MAXTEXT_EXTRA_FLAGS:-remat_policy=custom gdn_chunk_size=64 \
decoder_layer_input=offload context=remat \
megablox=true sparse_matmul=true use_tokamax_gmm=true use_gmm_v2=true \
use_gmm_v2_heuristic_tiling=true merge_gating_gmm=false \
use_ragged_sort=false use_custom_sort_vjp=false \
use_tokamax_splash=true use_splash_scheduler=true \
sa_block_q=1024 sa_block_kv=4096 sa_block_kv_compute=512 \
sa_block_q_dkv=2048 sa_block_kv_dkv=2048 sa_block_kv_dkv_compute=512 \
sa_fuse_reciprocal=false sa_use_base2_exp=true dq_reduction_steps=3 \
context_parallel_strategy=ring context_parallel_load_balance=false allow_split_physical_axes=true \
num_vocab_tiling=16 use_iota_embed=false mu_dtype=float32 grad_dtype=float32 \
checkpoint_storage_concurrent_gb=96 \
checkpoint_storage_use_ocdbt=false checkpoint_storage_use_zarr3=false \
packing=True optimizer_memory_host_offload=true}"
export DEBUG="${DEBUG:-0}"

if [[ $# -eq 0 ]]; then
  exec bash "${DIR}/../deepswe_dist/k8s_launcher.sh" --command start --image "${TUNIX_IMAGE}"
elif [[ "$1" == "start" || "$1" == "stop" || "$1" == "orchestrator" || "$1" == "trainer" || "$1" == "rollout" ]]; then
  _cmd="$1"
  shift
  exec bash "${DIR}/../deepswe_dist/k8s_launcher.sh" --command "${_cmd}" --image "${TUNIX_IMAGE}" "$@"
else
  exec bash "${DIR}/../deepswe_dist/k8s_launcher.sh" --image "${TUNIX_IMAGE}" "$@"
fi
