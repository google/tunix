#!/bin/bash
set -e

# ==============================================================================
# Self-Contained MLPerf DeepSWE Recipe: Qwen3.5-397B-A17B on TPU v7x (GBS=1152)
# ==============================================================================
# Flattens and replaces the stacked script chain:
#   mlperf_gbs1024_submission.sh
#     -> mlperf_397b_1024_v7x.sh
#     -> mlperf_397b_256_v7x.sh (and rollout_attention.sh)
#     -> mlperf_base.sh (and mlperf_pathways_config.sh, fp8_moe.sh,
#                        jax_cache_config.sh, ubench_reporting.sh)
# with the exact configuration used for run "at-1015c-s4".
#
# Commonly modified arguments can be overridden via environment variables:
#   SEED=... JOB_PREFIX=... K8S_NAMESPACE=... WANDB_API_KEY=... \
#     bash tunix/experimental/examples/recipes/mlperf_gbs1152_standalone.sh start
# ==============================================================================

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ==============================================================================
# 1. Commonly Modified Arguments
# ==============================================================================
export SEED="${SEED:-42}"
export JOB_PREFIX="${JOB_PREFIX:-${USER}}"
export K8S_NAMESPACE="${K8S_NAMESPACE:-}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export TUNIX_IMAGE="${TUNIX_IMAGE:-gcr.io/cloud-tpu-multipod-dev/atwigg/trellis-experimental:1015c}"

# ==============================================================================
# 2. Submission-Level Overrides (from mlperf_gbs1024_submission.sh)
# ==============================================================================
export TUNIX_DEBUG_INFERENCE_LOGS="${TUNIX_DEBUG_INFERENCE_LOGS:-false}"
export DISABLE_TRAJECTORY_STORE="${DISABLE_TRAJECTORY_STORE:-true}"
export RAGGED_BUFFER_FACTOR="${RAGGED_BUFFER_FACTOR:-2.75}"
export CPU_NODEPOOL="${CPU_NODEPOOL:-cpu-np}"
export CPU_MEMORY="${CPU_MEMORY:-220Gi}"
export ROLLOUT_REPLICAS="${ROLLOUT_REPLICAS:-128}"
export BATCH_SIZE="${BATCH_SIZE:-72}"
export INITIAL_DISPATCH_FIRST_STAGGER_S="${INITIAL_DISPATCH_FIRST_STAGGER_S:-75}"
export PREPULL_SANDBOX_IMAGES="${PREPULL_SANDBOX_IMAGES:-false}"

export RCP_LOGGING="${RCP_LOGGING:-true}"
export WEIGHT_SYNC_DISABLE_TIMEOUTS="${WEIGHT_SYNC_DISABLE_TIMEOUTS:-1}"
export RAIDEN_BROADCAST_HOST_RATIO="${RAIDEN_BROADCAST_HOST_RATIO:-1.0}"
export RAIDEN_BROADCAST_PIPELINE_STAGES="${RAIDEN_BROADCAST_PIPELINE_STAGES:-4}"
export FAIL_FAST="${FAIL_FAST:-true}"
export WANDB_RESUME="${WANDB_RESUME:-allow}"
export PATHWAYS_CHECKPOINTING_IMPL="${PATHWAYS_CHECKPOINTING_IMPL:-persistence}"
export CKPT_USE_OCDBT="${CKPT_USE_OCDBT:-0}"
export CHECKPOINT_SAVE_INTERVAL_STEPS="${CHECKPOINT_SAVE_INTERVAL_STEPS:-1}"
export CHECKPOINT_ABANDON_FAILED_SAVES="${CHECKPOINT_ABANDON_FAILED_SAVES:-true}"
export CHECKPOINT_OPTIMIZER_INTERVAL_STEPS="${CHECKPOINT_OPTIMIZER_INTERVAL_STEPS:-0}"
export MAX_STEPS="${MAX_STEPS:-10}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export ASYNC_WEIGHT_SYNC="${ASYNC_WEIGHT_SYNC:-true}"
export PIPELINE_TRAIN_MICROBATCHES="${PIPELINE_TRAIN_MICROBATCHES:-true}"
export IN_FLIGHT_WEIGHT_UPDATES="${IN_FLIGHT_WEIGHT_UPDATES:-true}"
export MAX_STALENESS="${MAX_STALENESS:-2}"
export ROLLOUT_PRIORITY_SCHEDULING="${ROLLOUT_PRIORITY_SCHEDULING:-true}"
export OPENHANDS_MULTI_TOOL_CALLS="${OPENHANDS_MULTI_TOOL_CALLS:-true}"
export WEIGHT_SYNC_PARALLEL_H2H="${WEIGHT_SYNC_PARALLEL_H2H:-1}"
export VERIFY_WEIGHTS="${VERIFY_WEIGHTS:-false}"

# Learning rate and gradient clip follow the global batch size when not set:
# GBS = BATCH_SIZE * NUM_GENERATIONS (16), LEARNING_RATE = 1e-6 * sqrt(GBS / 256),
# MAX_GRAD_NORM = 0.125 * sqrt(256 / GBS).
if [[ ! "${BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: BATCH_SIZE must be a positive integer, got '${BATCH_SIZE}'" >&2
  exit 1
fi
_gbs=$(( BATCH_SIZE * ${NUM_GENERATIONS:-16} ))
export LEARNING_RATE="${LEARNING_RATE:-$(python3 -c "import math; print(f'{1e-6 * math.sqrt(${_gbs} / 256):.16g}')")}"
export MAX_GRAD_NORM="${MAX_GRAD_NORM:-$(python3 -c "import math; print(f'{0.125 * math.sqrt(256 / ${_gbs}):.16g}')")}"
unset _gbs

export INITIAL_DISPATCH_STAGGER_S="${INITIAL_DISPATCH_STAGGER_S:-75}"
export TRAINER_WARMUP_COMPILE="${TRAINER_WARMUP_COMPILE:-true}"
export SANDBOX_WAIT_INITIAL="${SANDBOX_WAIT_INITIAL:-false}"
export ROLLOUT_ATTENTION_DP="${ROLLOUT_ATTENTION_DP:-false}"

export TRAINER_TUNED_FLAGS="${TRAINER_TUNED_FLAGS:-true}"
if [[ "${TRAINER_TUNED_FLAGS}" == "true" ]]; then
  export COMPUTE_LOGPS_CHUNK_SIZE="${COMPUTE_LOGPS_CHUNK_SIZE:-1024}"
fi
export DROPLESS="${DROPLESS:-true}"
case "${DROPLESS}" in
  true)
    if [[ "${TRAINER_TUNED_FLAGS}" == "true" ]]; then
      export RAGGED_BUFFER_FACTOR="${RAGGED_BUFFER_FACTOR:-2.75}"
    fi
    export RAGGED_BUFFER_FACTOR="${RAGGED_BUFFER_FACTOR:--1}"
    if [[ "${TRAINER_TUNED_FLAGS}" == "true" && "${RAGGED_BUFFER_FACTOR}" != "-1" ]]; then
      export RETRY_WHEN_TOKENS_DROPPED="${RETRY_WHEN_TOKENS_DROPPED:-true}"
      export MOE_EXCLUDE_PADDING="${MOE_EXCLUDE_PADDING:-true}"
      export LOG_MOE_DROPS="${LOG_MOE_DROPS:-true}"
      if [[ "${RETRY_DROPLESS_FINAL_MICRO_BATCH:-false}" != "true" \
            && -z "${FIRST_PHASE_RAGGED_BUFFER_FACTOR:-}" ]]; then
        export REMAT_CONTEXT="${REMAT_CONTEXT:-device}"
        export NUM_MOE_TOKEN_CHUNKS="${NUM_MOE_TOKEN_CHUNKS:-2}"
      fi
    fi
    export REMAT_CONTEXT="${REMAT_CONTEXT:-offload}"
    export NUM_MOE_TOKEN_CHUNKS="${NUM_MOE_TOKEN_CHUNKS:-4}"
    export TRAINER_MAX_HBM_SIZE_MIB="${TRAINER_MAX_HBM_SIZE_MIB:-84992}"
    if [[ "${RAGGED_BUFFER_FACTOR}" != "-1" && "${RETRY_WHEN_TOKENS_DROPPED:-false}" != "true" ]]; then
      echo "ERROR: DROPLESS=true with RAGGED_BUFFER_FACTOR=${RAGGED_BUFFER_FACTOR} drops tokens;" \
           "set RETRY_WHEN_TOKENS_DROPPED=true or DROPLESS=false" >&2
      exit 1
    fi
    ;;
  false)
    ;;
  *)
    echo "ERROR: DROPLESS must be true or false, got '${DROPLESS}'" >&2
    exit 1
    ;;
esac

export TUNIX_LOGPS_SHARDED_HEAD="${TUNIX_LOGPS_SHARDED_HEAD:-local}"
export PIPELINE_STEP_BOUNDARY="${PIPELINE_STEP_BOUNDARY:-1}"
export STAGE_NEXT_MICROBATCH="${STAGE_NEXT_MICROBATCH:-${PIPELINE_STEP_BOUNDARY}}"

export TRAINER_FP8="${TRAINER_FP8:-true}"
export ROLLOUT_FP8="${ROLLOUT_FP8:-true}"
export VLLM_DATA_PARALLEL_SIZE="${VLLM_DATA_PARALLEL_SIZE:-1}"
export ROLLOUT_TPU_SLICE="${ROLLOUT_TPU_SLICE:-tpu7x:2x2x1}"
export ROLLOUT_JOBSET_YAML="${ROLLOUT_JOBSET_YAML:-jobset.tpu.yaml}"
export UBENCH_REPORTING="${UBENCH_REPORTING:-true}"

# ==============================================================================
# 3. Scale Overrides (from mlperf_397b_1024_v7x.sh)
# ==============================================================================
export CHECKPOINT_MAX_TO_KEEP="${CHECKPOINT_MAX_TO_KEEP:-30}"
export MAX_CONCURRENCY="${MAX_CONCURRENCY:-4096}"
export USER_CONTAINER_MEMORY="${USER_CONTAINER_MEMORY:-160G}"

# ==============================================================================
# 4. 397B v7x Model & Topology Configuration (from mlperf_397b_256_v7x.sh)
# ==============================================================================
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-${JOB_PREFIX}-mlperf-397b-v7x}"
export POD="${POD:-pod1}"

export REGION="${REGION:-us-central1}"
export CLUSTER="${CLUSTER:-bodaborg-tpu7x-gsc}"
export BUCKET="${BUCKET:-gs://atwigg-trellis-us-central1-fast-dev}"
export TPU_RESERVATION="${TPU_RESERVATION:-ghostfish-tb2qe8ysltros}"
export MAXTEXT_CKPT="${MAXTEXT_CKPT:-gs://mlperf-6-1-submission/ckpt/qwen35_397b/scanned_reshard_fsdp32_tp2/0/items}"

export MAXTEXT_OUTPUT_DIR="${MAXTEXT_OUTPUT_DIR:-${BUCKET}/maxtext/${JOB_PREFIX}}"
export TRAJECTORY_LOG_DIR="${TRAJECTORY_LOG_DIR:-${BUCKET}/trajectories/${JOB_PREFIX}/logger}"
if [[ "${DISABLE_TRAJECTORY_STORE}" =~ ^(1|true|True|TRUE)$ ]] || [[ "${ENABLE_TRAJECTORY_STORE:-true}" =~ ^(0|false|False|FALSE)$ ]]; then
  export TRAJECTORY_STORE_ROOT_DIR=""
else
  export TRAJECTORY_STORE_ROOT_DIR="${TRAJECTORY_STORE_ROOT_DIR:-${TRAJECTORY_STORE_ROOT:-${BUCKET}/trajectories/${JOB_PREFIX}/store}}"
fi

export USE_DYNAMIC_SLICING="true"
export ENABLE_MULTI_NUMA="${ENABLE_MULTI_NUMA:-0}"
export TPU_RAIDEN_DATA_NICS="${TPU_RAIDEN_DATA_NICS:-eth0}"
export RAIDEN_DEVICES_PER_HOST=8

# Model configuration
export MODEL_NAME="Qwen3.5-397B-A17B"
export MODEL_ID="Qwen/Qwen3.5-397B-A17B"
export TOKENIZER_PATH="${TOKENIZER_PATH:-Qwen/Qwen3.5-397B-A17B}"
export MAXTEXT_MODEL_NAME="qwen3.5-397b-a17b"

# Trainer topology (128 chips / 256 devices: 4x4x8, FSDP=32, TP=1, EXPERT=2, CONTEXT=4)
export TRAINER_JOBSET_YAML="${TRAINER_JOBSET_YAML:-jobset.pathways.yaml}"
export TRAINER_TPU_SLICE="${TRAINER_TPU_SLICE:-tpu7x:4x4x8}"
export TRAINER_MESH_FSDP="${TRAINER_MESH_FSDP:-32}"
export TRAINER_MESH_TP="${TRAINER_MESH_TP:-1}"
export TRAINER_MESH_EXPERT="${TRAINER_MESH_EXPERT:-2}"
export TRAINER_MESH_CONTEXT="${TRAINER_MESH_CONTEXT:-4}"

# Rollout topology (128 replicas x 2x2x1 = 4 chips = 8 devices per replica, DP=1, EP=8, TP=1)
_rollout_dims="${ROLLOUT_TPU_SLICE#*:}"
export ROLLOUT_MESH_EXPERT="${ROLLOUT_MESH_EXPERT:-$(( 2 * ${_rollout_dims//x/*} / ${VLLM_DATA_PARALLEL_SIZE:-1} ))}"

# Rollout attention kernel (inlined from rollout_attention.sh)
export USE_BATCHED_RPA_LONG_CTX_KERNEL="${USE_BATCHED_RPA_LONG_CTX_KERNEL:-true}"
case "${USE_BATCHED_RPA_LONG_CTX_KERNEL}" in
  true)
    export ROLLOUT_MAXTEXT_ATTENTION="${ROLLOUT_MAXTEXT_ATTENTION:-vllm_batched_rpa_long_ctx}"
    ;;
  false)
    export ROLLOUT_MAXTEXT_ATTENTION="${ROLLOUT_MAXTEXT_ATTENTION:-vllm_rpa}"
    ;;
  *)
    echo "ERROR: USE_BATCHED_RPA_LONG_CTX_KERNEL must be true or false, got '${USE_BATCHED_RPA_LONG_CTX_KERNEL}'" >&2
    exit 1
    ;;
esac

export VLLM_ADDITIONAL_CONFIG='{"sharding":{"sharding_strategy":{"expert_parallelism":'"${ROLLOUT_MESH_EXPERT}"',"tensor_parallelism":1,"enable_dp_attention":true}},"custom_mamba_cache_multiplier":16,"maxtext_config":{"scan_layers":false,"attention":"'"${ROLLOUT_MAXTEXT_ATTENTION}"'","allow_split_physical_axes":true,"use_multimodal":false,"prefuse_moe_weights":true,"per_device_batch_size":0.0}}'

# Rollout Worker Flags & Raiden tuning
export ONEHOT_MOE_PERMUTE_THRESHOLD="${ONEHOT_MOE_PERMUTE_THRESHOLD:-2048}"
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
export VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY="RAIDEN_,TPU_,WEIGHT_SYNC_"
export TPU_RAIDEN_TCP_CONNECT_TIMEOUT_MS="${TPU_RAIDEN_TCP_CONNECT_TIMEOUT_MS:-120000}"
export TPU_RAIDEN_TCP_CONNECT_MAX_ATTEMPTS="${TPU_RAIDEN_TCP_CONNECT_MAX_ATTEMPTS:-5}"
export VLLM_RAY_EXTRA_ENV_VARS_TO_COPY="USE_BATCHED_RPA_LONG_CTX_KERNEL,ONEHOT_MOE_PERMUTE_THRESHOLD,LIBTPU_INIT_ARGS,RAY_memory_monitor_refresh_ms,VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS,ENABLE_MULTI_NUMA,TPU_RAIDEN_DATA_NICS,FLOAT32_GATE_LOGITS,FLOAT32_LOGITS,NEW_MODEL_DESIGN,ATTN_BUCKETIZED_NUM_REQS,ATTN_CUSTOM_NUM_REQS_BUCKETS,VLLM_MOE_CHUNK_SIZE,SLICE_ROPE_CACHE,DP_SCHED_BATCH_PREFILL,TPU_RAIDEN_TCP_CONNECT_TIMEOUT_MS,TPU_RAIDEN_TCP_CONNECT_MAX_ATTEMPTS,WEIGHT_SYNC_PARALLEL_H2H"
export ORCHESTRATOR_EXTRA_ENV="${ORCHESTRATOR_EXTRA_ENV:-WEIGHT_SYNC_BIND_TIMEOUT_S=900 WEIGHT_SYNC_METADATA_TIMEOUT_S=900 WEIGHT_SYNC_H2D_TIMEOUT_S=3600 WEIGHT_SYNC_TRANSFER_TIMEOUT_S=7200 RAIDEN_PARALLELISM=16 TPU_RAIDEN_DATA_NICS=${TPU_RAIDEN_DATA_NICS} RAIDEN_BROADCAST_HOST_RATIO=${RAIDEN_BROADCAST_HOST_RATIO} RAIDEN_BROADCAST_PIPELINE_STAGES=${RAIDEN_BROADCAST_PIPELINE_STAGES}${TPU_RAIDEN_TCP_CONNECT_TIMEOUT_MS:+ TPU_RAIDEN_TCP_CONNECT_TIMEOUT_MS=${TPU_RAIDEN_TCP_CONNECT_TIMEOUT_MS}}${TPU_RAIDEN_TCP_CONNECT_MAX_ATTEMPTS:+ TPU_RAIDEN_TCP_CONNECT_MAX_ATTEMPTS=${TPU_RAIDEN_TCP_CONNECT_MAX_ATTEMPTS}}}"
export ROLLOUT_EXTRA_ENV="${ROLLOUT_EXTRA_ENV:-USE_BATCHED_RPA_LONG_CTX_KERNEL=${USE_BATCHED_RPA_LONG_CTX_KERNEL} ONEHOT_MOE_PERMUTE_THRESHOLD=${ONEHOT_MOE_PERMUTE_THRESHOLD} RAY_memory_monitor_refresh_ms=0 VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 ENABLE_MULTI_NUMA=${ENABLE_MULTI_NUMA} TPU_RAIDEN_DATA_NICS=${TPU_RAIDEN_DATA_NICS} RAIDEN_BROADCAST_HOST_RATIO=${RAIDEN_BROADCAST_HOST_RATIO} RAIDEN_BROADCAST_PIPELINE_STAGES=${RAIDEN_BROADCAST_PIPELINE_STAGES}${TPU_RAIDEN_TCP_CONNECT_TIMEOUT_MS:+ TPU_RAIDEN_TCP_CONNECT_TIMEOUT_MS=${TPU_RAIDEN_TCP_CONNECT_TIMEOUT_MS}}${TPU_RAIDEN_TCP_CONNECT_MAX_ATTEMPTS:+ TPU_RAIDEN_TCP_CONNECT_MAX_ATTEMPTS=${TPU_RAIDEN_TCP_CONNECT_MAX_ATTEMPTS}}}"

# Rollout (vLLM) & Trainer libtpu flags
export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:- --xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false --xla_tpu_check_legacy_constraints_in_reduce_scatter_legalizer=false --xla_tpu_dvfs_p_state=7}"
export TRAINER_LIBTPU_INIT_ARGS="${TRAINER_LIBTPU_INIT_ARGS:---DANGEROUS_tpu_runtime_abi_verification_disabled=true --xla_tpu_use_tc_device_shape_on_sc=true --xla_sc_disable_megacore_partitioning=true --xla_tpu_enable_offloading_gather_to_sparsecore=true --xla_tpu_enable_sparse_core_collective_offload_all_gather=true --xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true --xla_tpu_enable_sparse_core_reduce_scatter_v2=true --xla_tpu_use_single_sparse_core_for_all_gather_offload=false --xla_tpu_enable_concurrent_sparse_core_offloading=true --xla_tpu_aggressive_opt_barrier_removal=false --xla_tpu_scoped_vmem_limit_kib=65472 --xla_tpu_enable_sublane_major_scaling_bitcast_fusion=false --xla_tpu_use_single_sparse_core_for_reduce_scatter_offload=false --xla_tpu_dvfs_p_state=7}"
if [[ -n "${TRAINER_MAX_HBM_SIZE_MIB:-}" ]]; then
  export TRAINER_LIBTPU_INIT_ARGS="${TRAINER_LIBTPU_INIT_ARGS} --xla_tpu_max_hbm_size_mib=${TRAINER_MAX_HBM_SIZE_MIB}"
fi
case "${TRAINER_TUNED_FLAGS:-false}" in
  true)
    if [[ " ${TRAINER_LIBTPU_INIT_ARGS} " != *" --xla_tpu_enable_offloading_copy_to_sparsecore="* ]]; then
      export TRAINER_LIBTPU_INIT_ARGS="${TRAINER_LIBTPU_INIT_ARGS} --xla_tpu_enable_offloading_copy_to_sparsecore=false"
    fi
    ;;
  false)
    ;;
  *)
    echo "ERROR: TRAINER_TUNED_FLAGS must be true or false, got '${TRAINER_TUNED_FLAGS}'" >&2
    exit 1
    ;;
esac

export TRAINER_EXTRA_ENV="${TRAINER_EXTRA_ENV:-ONEHOT_MOE_PERMUTE_THRESHOLD=${ONEHOT_MOE_PERMUTE_THRESHOLD} ENABLE_MULTI_NUMA=${ENABLE_MULTI_NUMA} TPU_RAIDEN_DATA_NICS=${TPU_RAIDEN_DATA_NICS} RAIDEN_BROADCAST_HOST_RATIO=${RAIDEN_BROADCAST_HOST_RATIO} RAIDEN_BROADCAST_PIPELINE_STAGES=${RAIDEN_BROADCAST_PIPELINE_STAGES} LIBTPU_INIT_ARGS='${TRAINER_LIBTPU_INIT_ARGS}'${TPU_RAIDEN_TCP_CONNECT_TIMEOUT_MS:+ TPU_RAIDEN_TCP_CONNECT_TIMEOUT_MS=${TPU_RAIDEN_TCP_CONNECT_TIMEOUT_MS}}${TPU_RAIDEN_TCP_CONNECT_MAX_ATTEMPTS:+ TPU_RAIDEN_TCP_CONNECT_MAX_ATTEMPTS=${TPU_RAIDEN_TCP_CONNECT_MAX_ATTEMPTS}}}"
export PATHWAYS_WORKER_EXTRA_ENV="${PATHWAYS_WORKER_EXTRA_ENV:-LIBTPU_INIT_ARGS=${TRAINER_LIBTPU_INIT_ARGS} --megascale_port=-1 --xprof_compress_jftrace=true
SKIP_MEGASCALE_PJRT_CLIENT=true
ENABLE_MULTI_NUMA=${ENABLE_MULTI_NUMA}
TPU_RAIDEN_DATA_NICS=${TPU_RAIDEN_DATA_NICS}
RAIDEN_BROADCAST_HOST_RATIO=${RAIDEN_BROADCAST_HOST_RATIO}
RAIDEN_BROADCAST_PIPELINE_STAGES=${RAIDEN_BROADCAST_PIPELINE_STAGES}${TPU_RAIDEN_TCP_CONNECT_TIMEOUT_MS:+
TPU_RAIDEN_TCP_CONNECT_TIMEOUT_MS=${TPU_RAIDEN_TCP_CONNECT_TIMEOUT_MS}}${TPU_RAIDEN_TCP_CONNECT_MAX_ATTEMPTS:+
TPU_RAIDEN_TCP_CONNECT_MAX_ATTEMPTS=${TPU_RAIDEN_TCP_CONNECT_MAX_ATTEMPTS}}}"

_trainer_xla_flags=""
for _f in ${TRAINER_LIBTPU_INIT_ARGS}; do [[ "${_f}" == --xla_* ]] && _trainer_xla_flags+="${_f} "; done
export PATHWAYS_PROXY_EXTRA_ARGS="${PATHWAYS_PROXY_EXTRA_ARGS:-${_trainer_xla_flags% }}"

export TRAIN_MICRO_BATCH_SIZE="${TRAIN_MICRO_BATCH_SIZE:-64}"
export RPC_TIMEOUT_S="${RPC_TIMEOUT_S:-10800}"
export REMAT_POLICY="${REMAT_POLICY:-custom}"
export REMAT_CONTEXT="${REMAT_CONTEXT:-remat}"
export NUM_MOE_TOKEN_CHUNKS="${NUM_MOE_TOKEN_CHUNKS:-2}"
export MAXTEXT_EXTRA_FLAGS="${MAXTEXT_EXTRA_FLAGS:-custom_mesh_and_rule=cp-as-ep \
use_gdn_kernel=true gdn_cp_mode=head gdn_chunk_size=64 \
decoder_layer_input=offload context=${REMAT_CONTEXT} gdn=remat gdn_conv=remat gdn_states=remat \
megablox=true sparse_matmul=true use_tokamax_gmm=true use_gmm_v2=true \
use_gmm_v2_heuristic_tiling=false merge_gating_gmm=false \
wi_tile_fwd_batch_seq=256 wi_tile_fwd_embed_dim=4096 wi_tile_fwd_mlp_dim=1024 \
wi_tile_dlhs_batch_seq=256 wi_tile_dlhs_embed_dim=4096 wi_tile_dlhs_mlp_dim=1024 \
wi_tile_drhs_batch_seq=256 wi_tile_drhs_embed_dim=4096 wi_tile_drhs_mlp_dim=1024 \
wo_tile_fwd_batch_seq=256 wo_tile_fwd_embed_dim=4096 wo_tile_fwd_mlp_dim=1024 \
wo_tile_dlhs_batch_seq=256 wo_tile_dlhs_embed_dim=4096 wo_tile_dlhs_mlp_dim=1024 \
wo_tile_drhs_batch_seq=256 wo_tile_drhs_embed_dim=4096 wo_tile_drhs_mlp_dim=1024 \
use_ring_of_experts=true num_moe_token_chunks=${NUM_MOE_TOKEN_CHUNKS} moe_chunk_barrier=false \
ring_of_experts_local_routing=true moe_expert_weight_prefetch=forward \
moe_chunk_pipeline=true ring_of_experts_row_major_reduce_scatter=true \
use_ragged_sort=true use_custom_sort_vjp=false ragged_buffer_factor=${RAGGED_BUFFER_FACTOR} \
use_tokamax_splash=true use_splash_scheduler=true \
sa_block_q=1024 sa_block_kv=4096 sa_block_kv_compute=512 \
sa_block_q_dkv=2048 sa_block_kv_dkv=2048 sa_block_kv_dkv_compute=512 \
sa_fuse_reciprocal=false sa_use_base2_exp=true dq_reduction_steps=3 \
context_parallel_strategy=ring context_parallel_load_balance=false allow_split_physical_axes=false \
context_parallel_attention_load_balance=true \
num_vocab_tiling=16 use_iota_embed=false mu_dtype=float32 grad_dtype=float32 \
checkpoint_storage_concurrent_gb=96 \
packing=True optimizer_memory_host_offload=true}"

if [[ "${RETRY_WHEN_TOKENS_DROPPED:-false}" == "true" ]]; then
  export MAXTEXT_EXTRA_FLAGS="${MAXTEXT_EXTRA_FLAGS} retry_when_tokens_dropped=true retry_num_moe_token_chunks=${RETRY_NUM_MOE_TOKEN_CHUNKS:-8}"
  if [[ -n "${RETRY_DROPLESS_FIRST_STEPS:-}" ]]; then
    export MAXTEXT_EXTRA_FLAGS="${MAXTEXT_EXTRA_FLAGS} retry_dropless_first_steps=${RETRY_DROPLESS_FIRST_STEPS}"
  fi
  if [[ -n "${FIRST_PHASE_RAGGED_BUFFER_FACTOR:-}" ]]; then
    export MAXTEXT_EXTRA_FLAGS="${MAXTEXT_EXTRA_FLAGS} first_phase_ragged_buffer_factor=${FIRST_PHASE_RAGGED_BUFFER_FACTOR}"
  fi
  if [[ "${RETRY_DROPLESS_FINAL_MICRO_BATCH:-false}" == "true" ]]; then
    export MAXTEXT_EXTRA_FLAGS="${MAXTEXT_EXTRA_FLAGS} retry_dropless_final_micro_batch=true"
  fi
fi
if [[ "${MOE_EXCLUDE_PADDING:-false}" == "true" ]]; then
  export MAXTEXT_EXTRA_FLAGS="${MAXTEXT_EXTRA_FLAGS} moe_exclude_padding_from_routing=true"
fi
if [[ "${LOG_MOE_DROPS:-false}" == "true" ]]; then
  export MAXTEXT_EXTRA_FLAGS="${MAXTEXT_EXTRA_FLAGS} log_moe_drops=true"
fi
if [[ "${TRAINER_TUNED_FLAGS:-false}" == "true" ]]; then
  export MAXTEXT_EXTRA_FLAGS="${MAXTEXT_EXTRA_FLAGS} context_parallel_strategy=all_gather sa_fuse_reciprocal=true sa_use_base2_exp=false gdn_cp_mode=seq gdn_cp_state=device gdn_cp_matmul_precision=high cast_grads_after_all_reduce=true"
fi
export DEBUG=${DEBUG:-0}

export SANDBOX_NODE_SELECTOR_KEY="${SANDBOX_NODE_SELECTOR_KEY:-workload}"
export SANDBOX_NODE_SELECTOR_VAL="${SANDBOX_NODE_SELECTOR_VAL:-sandbox}"
export SANDBOX_TOLERATIONS='[{"key":"workload","operator":"Equal","value":"sandbox","effect":"NoSchedule"}]'
export IMAGE_REWRITE_PREFIX="${IMAGE_REWRITE_PREFIX:-us-central1-docker.pkg.dev/cloud-tpu-multipod-dev/tunix/}"

# ==============================================================================
# 5. Base Configuration & Dispatcher (from mlperf_base.sh & helper scripts)
# ==============================================================================
export ORCHESTRATOR_PORT="${ORCHESTRATOR_PORT:-20000}"
export ROLLOUT_PORT="${ROLLOUT_PORT:-20001}"
export TRAINER_PORT="${TRAINER_PORT:-20002}"
export PROFILER_STEPS=${PROFILER_STEPS:-0}
export SKIP_FIRST_N_PROFILER_STEPS=${SKIP_FIRST_N_PROFILER_STEPS:--1}

export TPU_PROFILE_CHIPS_PER_TASK="${TPU_PROFILE_CHIPS_PER_TASK:-1}"
export TPU_PROFILE_SPARSE_CORES="${TPU_PROFILE_SPARSE_CORES:-1}"
export TPU_PROFILE_SPARSE_CORE_TILES="${TPU_PROFILE_SPARSE_CORE_TILES:-1}"
if [[ "${PROFILER_STEPS}" =~ ^[0-9]+$ && "${PROFILER_STEPS}" -gt 0 ]]; then
  tpu_profiling_flags=(
    "enable_tpu_profiling_options=true"
    "tpu_num_chips_to_profile_per_task=${TPU_PROFILE_CHIPS_PER_TASK}"
    "tpu_num_sparse_cores_to_trace=${TPU_PROFILE_SPARSE_CORES}"
    "tpu_num_sparse_core_tiles_to_trace=${TPU_PROFILE_SPARSE_CORE_TILES}"
    "upload_all_profiler_results=false"
  )
  export MAXTEXT_EXTRA_FLAGS="${MAXTEXT_EXTRA_FLAGS:+$MAXTEXT_EXTRA_FLAGS }${tpu_profiling_flags[*]}"
  unset tpu_profiling_flags
fi

# Cluster Context & Kueue / Priority
export PROJECT="${PROJECT:-cloud-tpu-shared-capacity}"
if [[ -n "${REGION:-}" && -n "${CLUSTER:-}" ]]; then
  kubectl config use-context "gke_${PROJECT}_${REGION}_${CLUSTER}" || true
  if [[ -n "${K8S_NAMESPACE:-}" ]]; then
    kubectl config set-context --current --namespace="${K8S_NAMESPACE}" || true
  fi
fi

export KUEUE_QUEUE="${KUEUE_QUEUE:-multislice-queue}"
export PREEMPTIBLE="${PREEMPTIBLE:-${preemptible:-false}}"
export GANG_ID="${GANG_ID:-${JOB_PREFIX}}"
export PRIORITY_CLASS="${PRIORITY_CLASS:-medium}"
export KUEUE_PRIORITY_CLASS="${KUEUE_PRIORITY_CLASS:-${PRIORITY_CLASS}}"
export SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-xpk-sa}"
export CPU_MACHINE="${CPU_MACHINE:-n2d-standard-64}"

export FT_STARTUP_RETRIES="${FT_STARTUP_RETRIES:-3}"
export FT_SANDBOX_READY_TIMEOUT_S="${FT_SANDBOX_READY_TIMEOUT_S:-600}"
export FT_SANDBOX_ACQUIRE_RETRIES="${FT_SANDBOX_ACQUIRE_RETRIES:-2}"

# Pathways & Raiden Weight Sync Defaults (inlined from mlperf_pathways_config.sh)
export PATHWAYS_SERVER_IMAGE="${PATHWAYS_SERVER_IMAGE:-us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/datenglin/sanitized_server:raiden_20261013212200}"
export PATHWAYS_PROXY_IMAGE="${PATHWAYS_PROXY_IMAGE:-us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/datenglin/sanitized_proxy_server:raiden_20261013212200}"
export PATHWAYS_PROXY_MEMORY_LIMIT="${PATHWAYS_PROXY_MEMORY_LIMIT:-160G}"
export USER_CONTAINER_MEMORY="${USER_CONTAINER_MEMORY:-260G}"
export USER_CONTAINER_MEMORY_LIMIT="${USER_CONTAINER_MEMORY_LIMIT:-260G}"

export USE_WEIGHT_CONVERTER="true"
export PREFUSE_MOE_WEIGHTS="true"
export TRAINER_PREFUSE_MOE_WEIGHTS="true"
export ROLLOUT_PREFUSE_MOE_WEIGHTS="true"
export TRAINER_PADDED_MOE_MLP_DIM=""
export WEIGHT_SYNC_MODE="${WEIGHT_SYNC_MODE:-raiden}"
export RAIDEN_FFI_USE_DIRECT_DEVICE_BUFFER="${RAIDEN_FFI_USE_DIRECT_DEVICE_BUFFER:-0}"

export RAIDEN_PARALLELISM="${RAIDEN_PARALLELISM:-16}"
export TRAINER_RAIDEN_COALESCE_WINDOW_BYTES="${TRAINER_RAIDEN_COALESCE_WINDOW_BYTES:-67108864}"
export TRAINER_RAIDEN_PIPELINE_GROUP_SIZE="${TRAINER_RAIDEN_PIPELINE_GROUP_SIZE:-16}"
export TRAINER_EXTRA_ENV="${TRAINER_EXTRA_ENV} RAIDEN_PARALLELISM=${RAIDEN_PARALLELISM} RAIDEN_TRANSPORT_COALESCE_WINDOW_BYTES=${TRAINER_RAIDEN_COALESCE_WINDOW_BYTES} RAIDEN_WEIGHT_SYNC_PIPELINE_GROUP_SIZE=${TRAINER_RAIDEN_PIPELINE_GROUP_SIZE}"
export ROLLOUT_EXTRA_ENV="${ROLLOUT_EXTRA_ENV} RAIDEN_PARALLELISM=${RAIDEN_PARALLELISM} RAIDEN_TRANSPORT_COALESCE_WINDOW_BYTES=67108864 RAIDEN_WEIGHT_SYNC_PIPELINE_GROUP_SIZE=16"
export PATHWAYS_WORKER_EXTRA_ENV="${PATHWAYS_WORKER_EXTRA_ENV}
RAIDEN_TRANSPORT_COALESCE_WINDOW_BYTES=${TRAINER_RAIDEN_COALESCE_WINDOW_BYTES}
RAIDEN_WEIGHT_SYNC_PIPELINE_GROUP_SIZE=${TRAINER_RAIDEN_PIPELINE_GROUP_SIZE}"

# WandB Configuration
export WANDB_ENTITY="${WANDB_ENTITY:-google-trellis}"
export WANDB_PROJECT="${WANDB_PROJECT:-trellis-deepswe}"

# Common Model & Backend Configuration
export TRAINER_BACKEND="maxtext"
export SAMPLER="vllm"
export TRAINABLE_PARAMETERS_MASK='^(?!.*routed_experts/gate/kernel).*'
export EOS_TOKENS="${EOS_TOKENS:-248046,248044}"
export TRAINER_BASE_NUM_KV_HEADS=2
export ROLLOUT_MESH_FSDP="${ROLLOUT_MESH_FSDP:-1}"
export ROLLOUT_MESH_TP="${ROLLOUT_MESH_TP:-1}"

# MLPerf RCP Logging
if [[ -n "${MAXTEXT_OUTPUT_DIR:-}" ]]; then
  export METRIC_LOGGER_DIR="${METRIC_LOGGER_DIR:-${MAXTEXT_OUTPUT_DIR}/mllog}"
fi
export TARGET_ACCURACY="${TARGET_ACCURACY:-0.69}"

# vLLM Rollout Configuration
export VLLM_LOGGING_LEVEL="INFO"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-65536}"
export VLLM_MAX_NUM_BATCHED_TOKENS=2048

case "${ROLLOUT_ATTENTION_DP}" in
  true)
    export VLLM_MAX_NUM_SEQS=16
    ;;
  false)
    export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-64}"
    VLLM_ADDITIONAL_CONFIG="$(python3 -c '
import json, os
cfg = json.loads(os.environ.get("VLLM_ADDITIONAL_CONFIG") or "{}")
strategy = cfg.setdefault("sharding", {}).setdefault("sharding_strategy", {})
strategy["enable_dp_attention"] = False
strategy["tensor_parallelism"] = 1
print(json.dumps(cfg, separators=(",", ":")))
')"
    export VLLM_ADDITIONAL_CONFIG
    _rollout_kv_tp="$(python3 -c '
import json, os
s = json.loads(os.environ["VLLM_ADDITIONAL_CONFIG"])["sharding"]["sharding_strategy"]
print(int(s.get("expert_parallelism", 1)) * int(s["tensor_parallelism"]))
')"
    export MAXTEXT_EXTRA_FLAGS="${MAXTEXT_EXTRA_FLAGS:+$MAXTEXT_EXTRA_FLAGS }kv_tp_size=${_rollout_kv_tp}"
    unset _rollout_kv_tp
    ;;
  *)
    echo "ERROR: ROLLOUT_ATTENTION_DP must be true or false, got '${ROLLOUT_ATTENTION_DP}'" >&2
    exit 1
    ;;
esac

export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.9}"
export VLLM_ENABLE_EXPERT_PARALLEL="true"

# Prefix Caching Configs
export ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-true}"
export VLLM_PREFIX_CACHE_RETENTION_INTERVAL="${VLLM_PREFIX_CACHE_RETENTION_INTERVAL:-0}"
if [[ "${ENABLE_PREFIX_CACHING}" == "true" ]]; then
  export MAMBA_CACHE_MODE="${MAMBA_CACHE_MODE:-align}"
else
  export MAMBA_CACHE_MODE="${MAMBA_CACHE_MODE:-none}"
fi
export VLLM_MAMBA_CACHE_MODE="${VLLM_MAMBA_CACHE_MODE:-${MAMBA_CACHE_MODE}}"

# FP8 MoE rollout/trainer (inlined from fp8_moe.sh)
for _var in ROLLOUT_FP8 TRAINER_FP8; do
  if [[ "${!_var}" != "true" && "${!_var}" != "false" ]]; then
    echo "ERROR: ${_var} must be true or false, got '${!_var}'" >&2
    exit 1
  fi
done
if [[ "${TRAINER_FP8}" == "true" && "${ROLLOUT_FP8}" != "true" ]]; then
  echo "ERROR: TRAINER_FP8=true matches the trainer to an FP8 rollout; set ROLLOUT_FP8=true too" >&2
  exit 1
fi
if [[ "${ROLLOUT_FP8}" == "true" ]]; then
  VLLM_ADDITIONAL_CONFIG="$(python3 -c '
import json, os
cfg = json.loads(os.environ.get("VLLM_ADDITIONAL_CONFIG") or "{}")
cfg.setdefault("maxtext_config", {})["fp8_moe"] = True
print(json.dumps(cfg, separators=(",", ":")))
')"
  export VLLM_ADDITIONAL_CONFIG
  export MAXTEXT_EXTRA_FLAGS="${MAXTEXT_EXTRA_FLAGS:-} rollout_fp8_moe=true"
fi
if [[ "${TRAINER_FP8}" == "true" ]]; then
  export MAXTEXT_EXTRA_FLAGS="${MAXTEXT_EXTRA_FLAGS:-} fp8_moe_fake_quant=true"
fi

# Router replay & KV Cache Configs
export RETURN_ROUTED_EXPERTS="${RETURN_ROUTED_EXPERTS:-true}"
export ROLLOUT_FREE_KV_CACHE="false"
export VLLM_KV_CACHE_DTYPE="bfloat16"
export VLLM_BLOCK_SIZE=256

# Engine & Model Configs
export VLLM_ASYNC_SCHEDULING="true"
export VLLM_ENABLE_CHUNKED_PREFILL="true"
export VLLM_LANGUAGE_MODEL_ONLY="true"
export VLLM_REASONING_PARSER="qwen3"
export VLLM_LIMIT_MM_PER_PROMPT='{"image": 0, "video": 0}'

# Rollout Worker Environment Flags (Optimizations & Runtime Settings)
export ROLLOUT_SKIP_JAX_PRECOMPILE="${ROLLOUT_SKIP_JAX_PRECOMPILE:-0}"
export NUM_PRECOMPILE_WORKERS=8
export NEW_MODEL_DESIGN=1
export ATTN_BUCKETIZED_NUM_REQS=true
export ATTN_CUSTOM_NUM_REQS_BUCKETS=4
export RAGGED_GATHER_REDUCE_VERSION="${RAGGED_GATHER_REDUCE_VERSION:-v3}"
case ",${VLLM_RAY_EXTRA_ENV_VARS_TO_COPY:-}," in
  *,RAGGED_GATHER_REDUCE_VERSION,*) ;;
  *) export VLLM_RAY_EXTRA_ENV_VARS_TO_COPY="${VLLM_RAY_EXTRA_ENV_VARS_TO_COPY:+${VLLM_RAY_EXTRA_ENV_VARS_TO_COPY},}RAGGED_GATHER_REDUCE_VERSION" ;;
esac
export VLLM_MOE_CHUNK_SIZE=256
export SLICE_ROPE_CACHE=1
export DP_SCHED_BATCH_PREFILL=false
export VLLM_ENABLE_V1_MULTIPROCESSING=0

# MoE and collective optimization switches
export TPU_SMALL_ALL_REDUCE="${TPU_SMALL_ALL_REDUCE:-1}"
export TPU_MOE_SINGLE_SORT="${TPU_MOE_SINGLE_SORT:-1}"
export TPU_MOE_ROUTER_TOPK="${TPU_MOE_ROUTER_TOPK:-rowmax}"
export TPU_MOE_LOWTOK="${TPU_MOE_LOWTOK:-1}"

for _opt_var in TPU_SMALL_ALL_REDUCE TPU_MOE_SINGLE_SORT TPU_MOE_ROUTER_TOPK TPU_MOE_LOWTOK; do
  if [[ ",${VLLM_RAY_EXTRA_ENV_VARS_TO_COPY:-}," != *",${_opt_var},"* ]]; then
    export VLLM_RAY_EXTRA_ENV_VARS_TO_COPY="${VLLM_RAY_EXTRA_ENV_VARS_TO_COPY:+${VLLM_RAY_EXTRA_ENV_VARS_TO_COPY},}${_opt_var}"
  fi
done

# Phased inference profiling
export PROFILE_SAMPLER="${PROFILE_SAMPLER:-false}"
if [[ "${PROFILE_SAMPLER}" == "true" || "${PROFILE_SAMPLER}" == "True" || "${PROFILE_SAMPLER}" == "1" ]]; then
  export PHASED_PROFILING_DIR="${PHASED_PROFILING_DIR:-${MAXTEXT_OUTPUT_DIR:+${MAXTEXT_OUTPUT_DIR}/inference_profiles}}"
  export PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR="${PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR:-3}"
  export PHASED_PROFILER_NUM_DECODE_STEPS_TO_SKIP="${PHASED_PROFILER_NUM_DECODE_STEPS_TO_SKIP:-10}"
else
  export PHASED_PROFILING_DIR="${PHASED_PROFILING_DIR:-}"
  export PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR="${PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR:-}"
  export PHASED_PROFILER_NUM_DECODE_STEPS_TO_SKIP="${PHASED_PROFILER_NUM_DECODE_STEPS_TO_SKIP:-}"
fi

# ==============================================================================
# JAX Compilation Cache (inlined from jax_cache_config.sh)
# ==============================================================================
export LOCAL_JAX_CACHE_DIR=${LOCAL_JAX_CACHE_DIR:-${JAX_CACHE_DIR:-/tmp/jax_cache}}
export JAX_CACHE_GCS_DIR=${JAX_CACHE_GCS_DIR:-}
export JAX_CACHE_BUCKET="${JAX_CACHE_BUCKET:-${BUCKET:-}}"

if [[ -z "${JAX_CACHE_GCS_DIR}" && -z "${ROLLOUT_JAX_CACHE_GCS_DIR:-}" ]]; then
  _cache_bucket="${JAX_CACHE_BUCKET:-${BUCKET:-}}"
  if [[ -z "${_cache_bucket}" && -n "${MAXTEXT_OUTPUT_DIR:-}" ]]; then
    _cache_bucket="$(echo "${MAXTEXT_OUTPUT_DIR}" | grep -o '^gs://[^/]*' || true)"
  fi

  if [[ -n "${_cache_bucket}" ]]; then
    _hw="unknown"
    if [[ "${ROLLOUT_TPU_SLICE:-}" == tpuv5* ]]; then
      _hw="v5p"
    elif [[ "${ROLLOUT_TPU_SLICE:-}" == tpu7x* ]]; then
      _hw="v7x"
    fi

    _model_slug="$(echo "${MODEL_NAME:-${MAXTEXT_MODEL_NAME:-model}}" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9._' '-' | sed 's/-$//')"
    _rollout_topo="${ROLLOUT_TPU_SLICE#*:}"
    _rollout_ep="${ROLLOUT_MESH_EXPERT:-1}"
    if [[ "${_rollout_ep}" == "1" && "${VLLM_ADDITIONAL_CONFIG:-}" =~ \"expert_parallelism\":\ *([0-9]+) ]]; then
      _rollout_ep="${BASH_REMATCH[1]}"
    fi
    _rollout_tp="${ROLLOUT_MESH_TP:-1}"
    _use_batched_rpa="${USE_BATCHED_RPA_KERNEL:-${ROLLOUT_USE_BATCHED_RPA:-0}}"

    _setup_payload=$(cat <<EOF
MODEL_NAME=${MODEL_NAME:-${MAXTEXT_MODEL_NAME:-model}}
ROLLOUT_FP8=${ROLLOUT_FP8:-true}
ROLLOUT_QUANTIZATION=${ROLLOUT_QUANTIZATION:-none}
FLOAT32_LOGITS=${FLOAT32_LOGITS:-true}
ROLLOUT_TPU_SLICE=${ROLLOUT_TPU_SLICE:-}
ROLLOUT_MESH_EXPERT=${_rollout_ep}
ROLLOUT_MESH_TP=${_rollout_tp}
VLLM_DATA_PARALLEL_SIZE=${VLLM_DATA_PARALLEL_SIZE:-1}
LIBTPU_INIT_ARGS=${LIBTPU_INIT_ARGS:-}
VLLM_MOE_CHUNK_SIZE=${VLLM_MOE_CHUNK_SIZE:-}
ONEHOT_MOE_PERMUTE_THRESHOLD=${ONEHOT_MOE_PERMUTE_THRESHOLD:-}
RAGGED_GATHER_REDUCE_VERSION=${RAGGED_GATHER_REDUCE_VERSION:-v3}
ATTN_CUSTOM_NUM_REQS_BUCKETS=${ATTN_CUSTOM_NUM_REQS_BUCKETS:-}
USE_BATCHED_RPA_KERNEL=${_use_batched_rpa}
USE_BATCHED_RPA_LONG_CTX_KERNEL=${USE_BATCHED_RPA_LONG_CTX_KERNEL:-true}
TPU_SMALL_ALL_REDUCE=${TPU_SMALL_ALL_REDUCE:-1}
TPU_MOE_SINGLE_SORT=${TPU_MOE_SINGLE_SORT:-1}
TPU_MOE_ROUTER_TOPK=${TPU_MOE_ROUTER_TOPK:-rowmax}
TPU_MOE_LOWTOK=${TPU_MOE_LOWTOK:-1}
EOF
)
    if command -v sha256sum >/dev/null 2>&1; then
      _setup_hash=$(printf '%s' "${_setup_payload}" | sha256sum | cut -c1-10)
    elif command -v shasum >/dev/null 2>&1; then
      _setup_hash=$(printf '%s' "${_setup_payload}" | shasum -a 256 | cut -c1-10)
    else
      _setup_hash=$(echo -n "${_setup_payload}" | cksum | awk '{print $1}')
    fi

    export ROLLOUT_JAX_CACHE_GCS_DIR="${_cache_bucket}/jax_cache/${_hw}/${_model_slug}/rollout_${_rollout_topo}_ep${_rollout_ep}_tp${_rollout_tp}_${_setup_hash}"
  fi
fi

export ROLLOUT_JAX_CACHE_GCS_DIR="${ROLLOUT_JAX_CACHE_GCS_DIR:-${JAX_CACHE_GCS_DIR:+${JAX_CACHE_GCS_DIR}/rollout}}"
export SAVE_JAX_CACHE="${SAVE_JAX_CACHE:-true}"
export SKIP_JAX_PRECOMPILE="${SKIP_JAX_PRECOMPILE:-1}"

# ==============================================================================
# Hyperparameters & DeepSWE Pipeline Configuration
# ==============================================================================
export MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-${BATCH_SIZE}}
export NUM_GENERATIONS="${NUM_GENERATIONS:-16}"
export NUM_GENERATIONS_TO_DISPATCH="${NUM_GENERATIONS_TO_DISPATCH:-}"
export PREFER_VALID_ROLLOUTS="${PREFER_VALID_ROLLOUTS:-true}"
export CHECKPOINT_ASYNC=${CHECKPOINT_ASYNC:-true}
export ENABLE_PATHWAYS_PERSISTENCE=${ENABLE_PATHWAYS_PERSISTENCE:-1}
export COLOCATED_PYTHON_SIDECAR_IMAGE="${COLOCATED_PYTHON_SIDECAR_IMAGE:-gcr.io/cloud-tpu-multipod-dev/yixuannwang_google_com-runner@sha256:6682fc0f364d2246a8768265c8c06ef50e213d86e1639d367543c78a90a8f0f2}"
export COLOCATED_PYTHON_SIDECAR_MEMORY="${COLOCATED_PYTHON_SIDECAR_MEMORY:-16Gi}"
export COLOCATED_PYTHON_SIDECAR_SHM="${COLOCATED_PYTHON_SIDECAR_SHM:-1}"
export CKPT_D2H_CONCURRENT_GB="${CKPT_D2H_CONCURRENT_GB:-32}"

if [[ "${ASYNC_WEIGHT_SYNC}" == "true" && "${WEIGHT_SYNC_MODE}" == "raiden" \
      && "${RAIDEN_FFI_USE_DIRECT_DEVICE_BUFFER}" != "0" ]]; then
  echo "ASYNC_WEIGHT_SYNC=true requires RAIDEN_FFI_USE_DIRECT_DEVICE_BUFFER=0" >&2
  exit 1
fi

case "${PIPELINE_STEP_BOUNDARY}" in
  true|True|1) _pipeline_step_boundary_on=true ;;
  *) _pipeline_step_boundary_on=false ;;
esac
case "${STAGE_NEXT_MICROBATCH}" in
  true|True|1) _stage_next_microbatch_on=true ;;
  *) _stage_next_microbatch_on=false ;;
esac
if [[ "${_stage_next_microbatch_on}" == "true" && "${_pipeline_step_boundary_on}" != "true" ]]; then
  echo "STAGE_NEXT_MICROBATCH=${STAGE_NEXT_MICROBATCH} requires PIPELINE_STEP_BOUNDARY on (true, True or 1), got '${PIPELINE_STEP_BOUNDARY}'" >&2
  exit 1
fi
unset _pipeline_step_boundary_on _stage_next_microbatch_on
export TRAJECTORY_GROUP_ORDER=${TRAJECTORY_GROUP_ORDER:-prompt_batch}

# Trajectory Store
if [[ "${DISABLE_TRAJECTORY_STORE}" =~ ^(1|true|True|TRUE)$ ]] || [[ "${ENABLE_TRAJECTORY_STORE:-true}" =~ ^(0|false|False|FALSE)$ ]]; then
  export DISABLE_TRAJECTORY_STORE="true"
  export TRAJECTORY_STORE_ROOT_DIR=""
  export TRAJECTORY_STORE_ROOT=""
  export TRAJECTORY_STORE_DB_URL=""
  export USE_DEFAULT_SQL_STORE="false"
else
  export USE_DEFAULT_SQL_STORE="${USE_DEFAULT_SQL_STORE:-true}"
  if [[ "${DRY_RUN:-false}" != "true" ]] && [[ -z "${TRAJECTORY_STORE_DB_URL:-}" ]] && [[ "${USE_DEFAULT_SQL_STORE}" =~ ^(1|true|True|TRUE)$ ]]; then
    if [[ -n "${PROJECT:-}" ]]; then
      TRAJECTORY_STORE_DB_URL="$(timeout 10s gcloud secrets versions access latest --secret=tunix-sql-store-uri --project="${PROJECT}" --quiet 2>/dev/null || true)"
    fi
    if [[ -z "${TRAJECTORY_STORE_DB_URL}" ]]; then
      echo "WARNING: USE_DEFAULT_SQL_STORE is enabled but TRAJECTORY_STORE_DB_URL could not be retrieved from GCP Secret Manager for project '${PROJECT:-}'; SQL trajectory store will be skipped." >&2
    fi
  fi
fi
export TRAJECTORY_STORE_DB_URL="${TRAJECTORY_STORE_DB_URL:-}"

# Sequence packing
export MAX_SEQ_TOKEN_PER_TPU=${MAX_SEQ_TOKEN_PER_TPU:-65536}
export MAX_SEGMENTS_PER_PACKED_ROW=${MAX_SEGMENTS_PER_PACKED_ROW:-16}

# Sampling Parameters
export TEMPERATURE="${TEMPERATURE:-1.0}"
export TOP_P="${TOP_P:-1.0}"
export TOP_K="-1"

# Algorithmic & Loss Hyperparameters
export BETA=0.0
export EPSILON=0.2
export EPSILON_HIGH=0.28
export USE_ROLLOUT_LOGPS="false"
export EXACT_TOKEN_CONTINUITY="${EXACT_TOKEN_CONTINUITY:-true}"
export OVERLONG_FILTER="${OVERLONG_FILTER:-false}"
export OVERLONG_LOSS_MASKING="${OVERLONG_LOSS_MASKING:-true}"
export SEQ_LOGPROB_ERROR_THRESHOLD=2.0
export TRUNCATED_IMPORTANCE_SAMPLING_TYPE="seq-mask-tis"
export TRUNCATED_IMPORTANCE_SAMPLING_RATIO_MIN=0.999
export TRUNCATED_IMPORTANCE_SAMPLING_RATIO=1.002
export ADVANTAGE_ESTIMATOR="grpo-loo"
export LOSS_AGG_MODE="token-mean"
export FLOAT32_GATE_LOGITS="true"
export FLOAT32_LOGITS="true"

# Optimizer Hyperparameters
export ADAM_B1=0.9
export ADAM_B2=0.999
export WEIGHT_DECAY=0.0
export OPT_CHAIN_TYPE="clip_by_global_norm"
export WARMUP_STEPS_FRACTION=0.0
export LEARNING_RATE_FINAL_FRACTION=1.0
export SKIP_STEP_ON_SPIKES="${SKIP_STEP_ON_SPIKES:-false}"
export SKIP_STEP_ON_NAN="${SKIP_STEP_ON_NAN:-true}"

# Architecture & Rematerialization
export TRAINER_MAXTEXT_ATTENTION="flash"
export COMPUTE_LOGPS_CHUNK_SIZE="${COMPUTE_LOGPS_CHUNK_SIZE:-512}"
export EPISODE_TIMEOUT_SECS=1800

# DeepSWE Environment & Agent Sandbox
export DATASET_PATH="${DATASET_PATH:-gs://mlperf_dataset/benchmark-r2e-gym-easy}"
export SHUFFLE="${SHUFFLE:-false}"
export USE_AGENT_SANDBOX=1
export SCAFFOLD="openhands"
export SANDBOX_NAMESPACE="${SANDBOX_NAMESPACE:-${K8S_NAMESPACE:-trellis}}"
export POOL_NAME_FORMAT="${POOL_NAME_FORMAT:-}"
export TEMPLATE_NAME_PREFIX="${TEMPLATE_NAME_PREFIX:-}"
export MAX_WARMPOOL_REPLICAS="${MAX_WARMPOOL_REPLICAS:-${NUM_GENERATIONS_TO_DISPATCH:-${NUM_GENERATIONS:-16}}}"
export SANDBOX_IMAGE_PULL_POLICY="${SANDBOX_IMAGE_PULL_POLICY:-IfNotPresent}"
export PREPULL_IMAGES_FILE="${PREPULL_IMAGES_FILE:-${DATASET_PATH}/sandbox_images.txt}"
export PREWARM_IMAGES_FILE="${PREWARM_IMAGES_FILE:-${PREPULL_IMAGES_FILE}}"
export PREWARM_MAX_WORKERS="${PREWARM_MAX_WORKERS:-32}"
export ROLLOUT_MAX_CONCURRENCY="${ROLLOUT_MAX_CONCURRENCY:-256}"
export STEP_TIMEOUT_SECS="${STEP_TIMEOUT_SECS:-60}"
export REWARD_TIMEOUT_SECS="${REWARD_TIMEOUT_SECS:-60}"
export FLUSH_EVERY_N_STEPS=1
export MAX_TURNS=30
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8192}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-57344}"

# ==============================================================================
# uBench Reporting (inlined from ubench_reporting.sh)
# ==============================================================================
_ubench_log() {
  echo "UBENCH_REPORTING: $*" >&2
}

_ubench_num_chips() {
  local chips=1 dim
  local -a dims
  IFS=x read -ra dims <<< "$1"
  for dim in "${dims[@]}"; do
    chips=$((chips * dim))
  done
  echo "${chips}"
}

_ubench_upload() {
  gcloud storage cp - "$1" --no-user-output-enabled
}

_ubench_main() {
  if [[ "${1:-start}" != "start" || "${MLPERF_NO_LAUNCH:-0}" == "1" ]]; then
    return 0
  fi
  local dry_run="${DRY_RUN:-false}" arg
  for arg in "$@"; do
    case "${arg}" in
      --dry-run|--dry_run|--render) dry_run="true" ;;
    esac
  done

  if [[ "${TRAINER_TPU_SLICE:-}" != tpu7x:* ||
        "${ROLLOUT_TPU_SLICE:-}" != tpu7x:* ]]; then
    _ubench_log "Not recording this run for uBench, because only tpu7x" \
      "slices are supported."
    return 0
  fi
  if [[ "${JOB_PREFIX:-}" == "${USER:-}" ]]; then
    _ubench_log "Not recording this run for uBench, because JOB_PREFIX is" \
      "the default (\$USER). To record runs, use a new JOB_PREFIX for each" \
      "run."
    return 0
  fi
  if [[ ! "${JOB_PREFIX:-}" =~ ^[a-z][a-z0-9-]*$ ]]; then
    _ubench_log "JOB_PREFIX '${JOB_PREFIX:-}' must start with a lowercase" \
      "letter and use only lowercase letters, digits, and '-'."
    return 1
  fi
  local slice_re='^tpu7x:[1-9][0-9]*(x[1-9][0-9]*)*$'
  if [[ ! "${TRAINER_TPU_SLICE:-}" =~ ${slice_re} ||
        ! "${ROLLOUT_TPU_SLICE:-}" =~ ${slice_re} ]]; then
    _ubench_log "Only tpu7x slices with a topology such as tpu7x:4x4x4 are" \
      "supported. Got TRAINER_TPU_SLICE=${TRAINER_TPU_SLICE:-}," \
      "ROLLOUT_TPU_SLICE=${ROLLOUT_TPU_SLICE:-}."
    return 1
  fi
  local rollout_replicas="${ROLLOUT_REPLICAS:-1}"
  if [[ ! "${rollout_replicas}" =~ ^[1-9][0-9]*$ ]]; then
    _ubench_log "ROLLOUT_REPLICAS must be a positive integer. Got" \
      "'${rollout_replicas}'."
    return 1
  fi
  local run_dir="${MAXTEXT_OUTPUT_DIR:-}"
  run_dir="${run_dir%/}"
  if [[ "${run_dir}" != gs://*/"${JOB_PREFIX}" ]]; then
    _ubench_log "MAXTEXT_OUTPUT_DIR must be a gs:// path that ends with" \
      "/${JOB_PREFIX}, because uBench uses it as the run directory. Got" \
      "'${MAXTEXT_OUTPUT_DIR:-}'."
    return 1
  fi
  local root="${run_dir%/*}"
  case "${RCP_LOGGING:-false}" in
    1|true|True)
      if [[ "${METRIC_LOGGER_DIR:-}" != "${run_dir}/mllog"* ]]; then
        _ubench_log "Warning: uBench reads the MLPerf log from" \
          "${run_dir}/mllog, but METRIC_LOGGER_DIR is" \
          "'${METRIC_LOGGER_DIR:-}'. uBench won't report MLPerf metrics."
      fi
      ;;
    *)
      _ubench_log "Warning: RCP_LOGGING is off, so the run writes no MLPerf" \
        "log, and uBench won't report MLPerf metrics."
      ;;
  esac

  if [[ -z "${PROJECT:-}" || -z "${REGION:-}" || -z "${CLUSTER:-}" ]]; then
    _ubench_log "Warning: PROJECT, REGION, or CLUSTER is not set. uBench might not be able to find the cluster or report metrics."
  fi
  local trainer_topology="${TRAINER_TPU_SLICE#tpu7x:}"
  local rollout_topology="${ROLLOUT_TPU_SLICE#tpu7x:}"
  local trainer_chips rollout_chips
  trainer_chips="$(_ubench_num_chips "${trainer_topology}")"
  rollout_chips="$(( $(_ubench_num_chips "${rollout_topology}") * rollout_replicas ))"

  if [[ "${dry_run}" == "true" ]]; then
    _ubench_log "Dry run. Not writing files to ${run_dir}."
    return 0
  fi
  if ! command -v gcloud &> /dev/null; then
    _ubench_log "gcloud CLI is not installed or not in PATH."
    return 1
  fi
  if gcloud storage ls "${run_dir}/logs/resolved_config.yaml" > /dev/null 2>&1; then
    _ubench_log "${run_dir} already has uBench files from an earlier run." \
      "Use a new JOB_PREFIX."
    return 1
  fi

  local rl_topology
  rl_topology="$(cat <<EOF
trainer:
  accelerator: tpu7x
  topology: "${trainer_topology}"
  replicas: 1
  chips: ${trainer_chips}
rollout:
  accelerator: tpu7x
  topology: "${rollout_topology}"
  replicas: ${rollout_replicas}
  chips: ${rollout_chips}
total_chips: $((trainer_chips + rollout_chips))
EOF
)"
  local resolved_config
  resolved_config="$(cat <<EOF
# Written by tunix/experimental/examples/recipes/ubench_reporting.sh.
benchmark_type: DISTRIBUTED
run_name: "${JOB_PREFIX}"
cluster:
  gke:
    project: "${PROJECT:-}"
    zone: "${REGION:-}"
    cluster_name: "${CLUSTER:-}"
distributed:
  recipe:
    generator_type: HELM
    artifacts_gcs_root_path: "${root}"
  workload:
    type: TUNIX_RL
    model: "${MODEL_ID:-}"
    hardware: V7X_${trainer_topology//x/X}
    num_steps: ${MAX_STEPS:-}
    image: "${TUNIX_IMAGE:-}"
    recipe_type: MLPERF
    # Same as the uBench default: skip the first (warmup) step.
    metrics_config:
      start_step: 1
EOF
)"
  if ! printf '%s\n' "${rl_topology}" |
         _ubench_upload "${run_dir}/logs/rl_topology.yaml" ||
     ! echo "job_start_time: '$(date -u +%Y-%m-%dT%H:%M:%SZ)'" |
         _ubench_upload "${run_dir}/workload_timestamps.yaml" ||
     ! printf '%s\n' "${resolved_config}" |
         _ubench_upload "${run_dir}/logs/resolved_config.yaml"; then
    _ubench_log "Could not write the uBench files to ${run_dir}."
    return 1
  fi

  _ubench_log "After the run finishes, report it with:"
  echo "  ubench benchmark report --run-name ${JOB_PREFIX} --gcs-root-path ${root}" >&2
}

if [[ "${UBENCH_REPORTING:-false}" =~ ^(1|true|True|TRUE)$ ]]; then
  ( _ubench_main "$@" ) ||
    echo "UBENCH_REPORTING: Not recording this run for uBench. The launch continues." >&2
fi

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
  if [[ "${COMMAND}" == "eval" && -n "${CHECKPOINT_MANIFEST_FILE:-}" ]]; then
    echo "Running sequential offline evaluation from manifest: ${CHECKPOINT_MANIFEST_FILE}"
    MANIFEST_ROWS_TSV="$(python3 -c '
import json, subprocess, sys
path = sys.argv[1]
if path.startswith("gs://"):
    text = subprocess.check_output(["gsutil", "cat", path], text=True)
else:
    with open(path, encoding="utf-8") as f:
        text = f.read()
records = sorted(
    (json.loads(line) for line in text.splitlines() if line.strip()),
    key=lambda r: int(r["step"]),
)
if not records:
    sys.exit(f"Checkpoint manifest is empty: {path}")
steps = [int(r["step"]) for r in records]
first = int(records[0].get("val_start_at", steps[0]))
if steps != list(range(first, first + len(steps))):
    sys.exit(f"Manifest steps must be contiguous from val_start_at={first}: {steps}")
for i, r in enumerate(records):
    print("\t".join([
        str(int(r["step"])),
        str(int(r["samples_count"])),
        str(int(r["timestamp_ms"])),
        str(r["checkpoint_path"]),
        "true" if i == len(records) - 1 else "false",
        str(r.get("mllog_file") or ""),
    ]))
' "${CHECKPOINT_MANIFEST_FILE}")"
    mapfile -t MANIFEST_ROWS <<< "${MANIFEST_ROWS_TSV}"
    BASE_EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${MAXTEXT_OUTPUT_DIR}/eval_results}"
    EVAL_JOBSET_NAME="${EVAL_JOBSET_NAME:-${JOB_PREFIX}-eval}"
    for row in "${MANIFEST_ROWS[@]}"; do
      IFS=$'\t' read -r STEP SAMPLES TS_MS CKPT_PATH IS_LAST MLLOG_FILE <<< "${row}"

      echo "=== Evaluating checkpoint step=${STEP} samples=${SAMPLES} is_last=${IS_LAST} path=${CKPT_PATH} ==="
      export MAXTEXT_CKPT="${CKPT_PATH}"
      export CHECKPOINT_STEP="${STEP}"
      export SAMPLES_COUNT="${SAMPLES}"
      export CHECKPOINT_TIMESTAMP_MS="${TS_MS}"
      export IS_LAST_CHECKPOINT="${IS_LAST}"
      export METRIC_LOGGER_DIR="${MLLOG_FILE:-${METRIC_LOGGER_DIR:-}}"
      export EVAL_OUTPUT_DIR="${BASE_EVAL_OUTPUT_DIR%/}/step_${STEP}"
      "${LAUNCHER}" --command eval --image "${TUNIX_IMAGE}" "$@"

      if [[ "${DRY_RUN:-false}" != "true" ]]; then
        HEAD_JOBSET="${EVAL_JOBSET_NAME}"
        if [[ "${ROLLOUT_REPLICAS:-1}" -gt 1 ]]; then
          HEAD_JOBSET="${EVAL_JOBSET_NAME}-0"
        fi
        echo "Waiting for evaluation JobSet ${HEAD_JOBSET} (main container) in namespace ${K8S_NAMESPACE}..."
        while true; do
          if ! kubectl get jobset "${HEAD_JOBSET}" -n "${K8S_NAMESPACE}" &>/dev/null; then
            echo "JobSet ${HEAD_JOBSET} no longer exists."
            break
          fi
          MAIN_EXIT="$(kubectl get pods -n "${K8S_NAMESPACE}" -l "jobset.sigs.k8s.io/jobset-name=${HEAD_JOBSET},jobset.sigs.k8s.io/replicatedjob-name=proc" -o jsonpath='{.items[0].status.containerStatuses[?(@.name=="main")].state.terminated.exitCode}' 2>/dev/null || true)"
          if [[ -n "${MAIN_EXIT}" ]]; then
            echo "Main evaluation container finished with exit code ${MAIN_EXIT}."
            break
          fi
          sleep 10
        done
        "${LAUNCHER}" --command stop_eval --image "${TUNIX_IMAGE}" || true

        TARGET_REACHED="$(
          python3 -c '
import glob, json, os, subprocess, sys
out_dir = os.environ["EVAL_OUTPUT_DIR"].rstrip("/")
if out_dir.startswith("gs://"):
    res = subprocess.run(["gsutil", "ls", f"{out_dir}/*/summary.json"], capture_output=True, text=True, check=False)
    if res.returncode == 0 and res.stdout.strip():
        matches = sorted(line.strip() for line in res.stdout.splitlines() if line.strip())
        if matches:
            res_cat = subprocess.run(["gsutil", "cat", matches[-1]], capture_output=True, text=True, check=False)
            if res_cat.returncode == 0 and res_cat.stdout.strip():
                data = json.loads(res_cat.stdout)
                print("true" if data.get("target_reached") else "false")
                sys.exit(0)
else:
    matches = sorted(glob.glob(f"{out_dir}/*/summary.json"))
    if matches:
        with open(matches[-1], "r", encoding="utf-8") as f:
            data = json.load(f)
        print("true" if data.get("target_reached") else "false")
        sys.exit(0)
print("false")
'
        )"
        if [[ "${TARGET_REACHED}" == "true" ]]; then
          echo "Target accuracy ${TARGET_ACCURACY} reached at step ${STEP}. Stopping offline evaluation loop."
          break
        fi
      fi
    done
    exit 0
  fi
  exec "${LAUNCHER}" --command "${COMMAND}" --image "${TUNIX_IMAGE}" "$@"
fi
