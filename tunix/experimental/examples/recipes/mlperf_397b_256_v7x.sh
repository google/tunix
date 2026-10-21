#!/bin/bash
set -e

# ==============================================================================
# MLPerf DeepSWE recipe: Qwen3.5-397B-A17B on TPU v7x
# ==============================================================================
# - TPU7x dynamic slicing on pod1 (bodaborg-tpu7x-gsc) or pod2 (bodaborg-tpu7x-gsc-elm)
# - Trainer on 128 chips (4x4x8 = 256 devices, FSDP=32, TP=1, EXPERT=2, CP=4), or on 256
#   chips with TRAINER_TPU_SLICE=tpu7x:4x8x8 and TRAINER_MESH_FSDP/EXPERT/CONTEXT 32/2/8 or
#   64/1/8 (TRAINER_TUNED_FLAGS then adds those meshes' keys, see below)
# - Rollout on 256 chips (32 replicas x 8 chips 2x2x2, EP=16, TP=1)
# - Sandbox configured for sandbox-np nodepool with workload tolerations
# ==============================================================================

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Fill these before you run.
export JOB_PREFIX="${JOB_PREFIX:-$USER}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-${JOB_PREFIX}-mlperf-397b-v7x}"

# Select pod: pod1 (bodaborg-tpu7x-gsc, us-central1) or pod2 (bodaborg-tpu7x-gsc-elm, us-east1).
export POD="${POD:-pod1}"

export REGION="${REGION:-us-central1}"
export CLUSTER="${CLUSTER:-bodaborg-tpu7x-gsc}"
export BUCKET="${BUCKET:-gs://atwigg-trellis-us-central1-fast-dev}"
export TPU_RESERVATION="${TPU_RESERVATION:-ghostfish-tb2qe8ysltros}"
export MAXTEXT_CKPT="${MAXTEXT_CKPT:-gs://mlperf-6-1-submission/ckpt/qwen35_397b/scanned_reshard_fsdp32_tp2/0/items}"

export MAXTEXT_OUTPUT_DIR="${MAXTEXT_OUTPUT_DIR:-${BUCKET}/maxtext/${JOB_PREFIX}}"
export TRAJECTORY_LOG_DIR="${TRAJECTORY_LOG_DIR:-${BUCKET}/trajectories/${JOB_PREFIX}/logger}"
export DISABLE_TRAJECTORY_STORE="${DISABLE_TRAJECTORY_STORE:-false}"
if [[ "${DISABLE_TRAJECTORY_STORE}" =~ ^(1|true|True|TRUE)$ ]] || [[ "${ENABLE_TRAJECTORY_STORE:-true}" =~ ^(0|false|False|FALSE)$ ]]; then
  export TRAJECTORY_STORE_ROOT_DIR=""
else
  export TRAJECTORY_STORE_ROOT_DIR="${TRAJECTORY_STORE_ROOT_DIR:-${TRAJECTORY_STORE_ROOT:-${BUCKET}/trajectories/${JOB_PREFIX}/store}}"
fi

export K8S_NAMESPACE="${K8S_NAMESPACE:-priority-dev}"
export USE_DYNAMIC_SLICING="true"
export ENABLE_MULTI_NUMA="${ENABLE_MULTI_NUMA:-0}"
export TPU_RAIDEN_DATA_NICS="${TPU_RAIDEN_DATA_NICS:-eth0}"

# Head pod lands on cpu-np (~257G allocatable). mlperf_pathways_config.sh's
# 260G user-container request plus proxy/rm requests (~280G) never schedules
# there. Only the request matters for placement; the limits stay as configured.
export USER_CONTAINER_MEMORY="${USER_CONTAINER_MEMORY:-48G}"

export RAIDEN_DEVICES_PER_HOST=8
export RAIDEN_BROADCAST_HOST_RATIO="${RAIDEN_BROADCAST_HOST_RATIO:-1.0}"
export RAIDEN_BROADCAST_PIPELINE_STAGES="${RAIDEN_BROADCAST_PIPELINE_STAGES:-4}"

# Model configuration
export MODEL_NAME="Qwen3.5-397B-A17B"
export MODEL_ID="Qwen/Qwen3.5-397B-A17B"
export TOKENIZER_PATH="${TOKENIZER_PATH:-Qwen/Qwen3.5-397B-A17B}"
export MAXTEXT_MODEL_NAME="qwen3.5-397b-a17b"

# Topologies:
# - Default 128-chip trainer (tpu7x:4x4x8 = 256 devices): FSDP=32, TP=1, EP=2, CP=4.
# - 256-chip trainer (TRAINER_TPU_SLICE=tpu7x:4x8x8 = 512 devices):
#     With TRAINER_TUNED_FLAGS=true: FSDP=64, TP=1, EP=1, CP=8 (num_moe_token_chunks=1).
#     With TRAINER_TUNED_FLAGS=false: FSDP=32, TP=1, EP=2, CP=8.
# Mesh rule: cp-as-ep (cp-as-ep-dense-no-cp with TRAINER_TUNED_FLAGS=true, see below).
export TRAINER_JOBSET_YAML="${TRAINER_JOBSET_YAML:-jobset.pathways.yaml}"
export TRAINER_TPU_SLICE="${TRAINER_TPU_SLICE:-tpu7x:4x4x8}"
if [[ "${TRAINER_TPU_SLICE}" == "tpu7x:4x8x8" ]]; then
  if [[ "${TRAINER_TUNED_FLAGS:-false}" == "true" ]]; then
    export TRAINER_MESH_FSDP="${TRAINER_MESH_FSDP:-64}"
    export TRAINER_MESH_TP="${TRAINER_MESH_TP:-1}"
    export TRAINER_MESH_EXPERT="${TRAINER_MESH_EXPERT:-1}"
    export TRAINER_MESH_CONTEXT="${TRAINER_MESH_CONTEXT:-8}"
  else
    export TRAINER_MESH_FSDP="${TRAINER_MESH_FSDP:-32}"
    export TRAINER_MESH_TP="${TRAINER_MESH_TP:-1}"
    export TRAINER_MESH_EXPERT="${TRAINER_MESH_EXPERT:-2}"
    export TRAINER_MESH_CONTEXT="${TRAINER_MESH_CONTEXT:-8}"
  fi
else
  export TRAINER_MESH_FSDP="${TRAINER_MESH_FSDP:-32}"
  export TRAINER_MESH_TP="${TRAINER_MESH_TP:-1}"
  export TRAINER_MESH_EXPERT="${TRAINER_MESH_EXPERT:-2}"
  export TRAINER_MESH_CONTEXT="${TRAINER_MESH_CONTEXT:-4}"
fi

# Rollout: 8 chips = 16 devices = 2 hosts per replica. dp * tp * expert must equal
# the DEVICE count, so expert=16 on 8 chips. Scaled rollout recipes
# (mlperf_397b_1024_v7x.sh) override ROLLOUT_REPLICAS, ROLLOUT_TPU_SLICE and
# VLLM_DATA_PARALLEL_SIZE; expert then fills the slice's devices (2 per chip) / dp.
export ROLLOUT_JOBSET_YAML="${ROLLOUT_JOBSET_YAML:-jobset.mcjax.ray.yaml}"
export ROLLOUT_TPU_SLICE="${ROLLOUT_TPU_SLICE:-tpu7x:2x2x4}"
export VLLM_DATA_PARALLEL_SIZE="${VLLM_DATA_PARALLEL_SIZE:-2}"
_rollout_dims="${ROLLOUT_TPU_SLICE#*:}"
export ROLLOUT_MESH_EXPERT="${ROLLOUT_MESH_EXPERT:-$(( 2 * ${_rollout_dims//x/*} / ${VLLM_DATA_PARALLEL_SIZE:-1} ))}"
export ROLLOUT_REPLICAS="${ROLLOUT_REPLICAS:-16}"

# Sandbox Concurrency
export MAX_CONCURRENCY="${MAX_CONCURRENCY:-1024}"

# vLLM Rollout Configuration
# Rollout attention kernel (USE_BATCHED_RPA_LONG_CTX_KERNEL -> ROLLOUT_MAXTEXT_ATTENTION); see rollout_attention.sh.
source "${DIR}/rollout_attention.sh"
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
# Rollout (vLLM) libtpu flags; mlperf_base.sh only applies its default when unset.
export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:- --xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false --xla_tpu_check_legacy_constraints_in_reduce_scatter_legalizer=false --xla_tpu_dvfs_p_state=7}"
# TRAINER_TUNED_FLAGS=true at the 128-chip trainer mesh 32x2x4 sets
# --xla_tpu_use_single_sparse_core_for_reduce_scatter_offload=true in the default trainer libtpu
# flags below (false elsewhere, as before): XLA issues every offloaded reduce-scatter on one
# SparseCore. On its own it made the step slower on the bench (about +0.9 s/step); with
# gdn_cp_dh_scan, which the block also sets only there (part 2), the two together are faster
# (numbers in part 2). A TRAINER_LIBTPU_INIT_ARGS set by the caller is kept as it is.
if [[ "${TRAINER_TUNED_FLAGS:-false}" == "true" \
      && "${TRAINER_MESH_FSDP}x${TRAINER_MESH_EXPERT}x${TRAINER_MESH_CONTEXT}" == "32x2x4" ]]; then
  _single_sc_reduce_scatter=true
else
  _single_sc_reduce_scatter=false
fi
# v7x caps scoped VMEM at 67043328 bytes (65472 KiB); 65536 is rejected per compile
# (INVALID_ARGUMENT in pathways-rm) and the compiler falls back to its default.
export TRAINER_LIBTPU_INIT_ARGS="${TRAINER_LIBTPU_INIT_ARGS:---DANGEROUS_tpu_runtime_abi_verification_disabled=true --xla_tpu_use_tc_device_shape_on_sc=true --xla_sc_disable_megacore_partitioning=true --xla_tpu_enable_offloading_gather_to_sparsecore=true --xla_tpu_enable_sparse_core_collective_offload_all_gather=true --xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true --xla_tpu_enable_sparse_core_reduce_scatter_v2=true --xla_tpu_use_single_sparse_core_for_all_gather_offload=false --xla_tpu_enable_concurrent_sparse_core_offloading=true --xla_tpu_aggressive_opt_barrier_removal=false --xla_tpu_scoped_vmem_limit_kib=65472 --xla_tpu_enable_sublane_major_scaling_bitcast_fusion=false --xla_tpu_use_single_sparse_core_for_reduce_scatter_offload=${_single_sc_reduce_scatter} --xla_tpu_dvfs_p_state=7}"
# Optional XLA HBM budget for the trainer, in MiB (unset = no cap). Appended here,
# before the trainer env, Pathways worker env and proxy args below copy the flags.
if [[ -n "${TRAINER_MAX_HBM_SIZE_MIB:-}" ]]; then
  export TRAINER_LIBTPU_INIT_ARGS="${TRAINER_LIBTPU_INIT_ARGS} --xla_tpu_max_hbm_size_mib=${TRAINER_MAX_HBM_SIZE_MIB}"
fi
# TRAINER_TUNED_FLAGS (true or false): trainer settings measured on the RL trainer
# bench (AI-Hypercomputer/maxtext#5701) at this recipe's 128-chip trainer cell. Most
# are settings of the MaxText pretraining recipe for this model
# (AI-Hypercomputer/maxtext#5409) that this recipe does not use yet. Off here and in
# the recipes that source this one; on in mlperf_gbs1024_submission.sh. Part 1, the
# trainer libtpu flag, is appended here so the trainer env, the Pathways workers and
# the proxy (which compiles the trainer program) all copy it. A value already in
# TRAINER_LIBTPU_INIT_ARGS is kept.
#   --xla_tpu_enable_offloading_copy_to_sparsecore=false: XLA does not offload copies
#     to the SparseCores. Bench, alone: 130.8 -> 126.4 s/step.
# Part 2, the MaxText keys, follows MAXTEXT_EXTRA_FLAGS below.
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
# Under Pathways the trainer's TPU program runs in the pathways-worker container,
# so the trainer libtpu flags must be set there (yaml_generator.py renders one
# KEY=VALUE per line into the worker env).
export PATHWAYS_WORKER_EXTRA_ENV="${PATHWAYS_WORKER_EXTRA_ENV:-LIBTPU_INIT_ARGS=${TRAINER_LIBTPU_INIT_ARGS} --megascale_port=-1 --xprof_compress_jftrace=true
SKIP_MEGASCALE_PJRT_CLIENT=true
ENABLE_MULTI_NUMA=${ENABLE_MULTI_NUMA}
TPU_RAIDEN_DATA_NICS=${TPU_RAIDEN_DATA_NICS}
RAIDEN_BROADCAST_HOST_RATIO=${RAIDEN_BROADCAST_HOST_RATIO}
RAIDEN_BROADCAST_PIPELINE_STAGES=${RAIDEN_BROADCAST_PIPELINE_STAGES}${TPU_RAIDEN_TCP_CONNECT_TIMEOUT_MS:+
TPU_RAIDEN_TCP_CONNECT_TIMEOUT_MS=${TPU_RAIDEN_TCP_CONNECT_TIMEOUT_MS}}${TPU_RAIDEN_TCP_CONNECT_MAX_ATTEMPTS:+
TPU_RAIDEN_TCP_CONNECT_MAX_ATTEMPTS=${TPU_RAIDEN_TCP_CONNECT_MAX_ATTEMPTS}}}"

# XLA compiler flags must also reach the pathways-proxy: it is what compiles the
# trainer program, and without them it uses the 32 MiB default scoped VMEM limit
# (splash-attention dkv needs ~37 MiB at these block sizes -> CompileTimeScopedVmemOom).
_trainer_xla_flags=""
for _f in ${TRAINER_LIBTPU_INIT_ARGS}; do [[ "${_f}" == --xla_* ]] && _trainer_xla_flags+="${_f} "; done
export PATHWAYS_PROXY_EXTRA_ARGS="${PATHWAYS_PROXY_EXTRA_ARGS:-${_trainer_xla_flags% }}"

# Hyperparameters & DeepSWE Pipeline Configuration
# Must be a multiple of TRAINER_MESH_FSDP. Defaults to TRAINER_MESH_FSDP * TRAINER_MESH_EXPERT
# (64 for 32x2x4 and 64x1x8; 128 for 64x2x4) to match SequencePackedBatchAssembler's pack_size.
export TRAIN_MICRO_BATCH_SIZE="${TRAIN_MICRO_BATCH_SIZE:-$(( TRAINER_MESH_FSDP * TRAINER_MESH_EXPERT ))}"
export MAX_STEPS="${MAX_STEPS:-30}"
export CHECKPOINT_SAVE_INTERVAL_STEPS="${CHECKPOINT_SAVE_INTERVAL_STEPS:-1}"
export CHECKPOINT_MAX_TO_KEEP="${CHECKPOINT_MAX_TO_KEEP:-30}"

export RPC_TIMEOUT_S="${RPC_TIMEOUT_S:-10800}"
# Architecture & Rematerialization: custom remat with per-tensor policies below
# (mlperf_base.sh defaults REMAT_POLICY to full).
export REMAT_POLICY="${REMAT_POLICY:-custom}"
export RAGGED_BUFFER_FACTOR="${RAGGED_BUFFER_FACTOR:-2.0}"
export REMAT_CONTEXT="${REMAT_CONTEXT:-remat}"
# TRAINER_TUNED_FLAGS=true at the 256-chip trainer mesh 64x1x8 runs one MoE token chunk (see
# the end of the TRAINER_TUNED_FLAGS block); an explicit NUM_MOE_TOKEN_CHUNKS wins.
if [[ "${TRAINER_TUNED_FLAGS:-false}" == "true" \
      && "${TRAINER_MESH_FSDP}x${TRAINER_MESH_EXPERT}x${TRAINER_MESH_CONTEXT}" == "64x1x8" ]]; then
  export NUM_MOE_TOKEN_CHUNKS="${NUM_MOE_TOKEN_CHUNKS:-1}"
fi
export NUM_MOE_TOKEN_CHUNKS="${NUM_MOE_TOKEN_CHUNKS:-2}"
# MaxText refuses moe_chunk_pipeline=true with one MoE token chunk, so one chunk turns the
# pipeline off.
if [[ "${NUM_MOE_TOKEN_CHUNKS}" == "1" ]]; then
  _moe_chunk_pipeline=false
else
  _moe_chunk_pipeline=true
fi
# TRAINER_TUNED_FLAGS=true at the 128-chip trainer mesh 32x2x4 gives the fused wi forward
# GMM N tiles of 2048 instead of 1024 (wi_tile_fwd_mlp_dim below, one value), next to the
# 2-deep weight pipeline that tile needs to fit VMEM (wi_tile_fwd_rhs_buffer_count=2, at
# the end of the TRAINER_TUNED_FLAGS block).
if [[ "${TRAINER_TUNED_FLAGS:-false}" == "true" \
      && "${TRAINER_MESH_FSDP}x${TRAINER_MESH_EXPERT}x${TRAINER_MESH_CONTEXT}" == "32x2x4" ]]; then
  _wi_tile_fwd_mlp_dim=2048
else
  _wi_tile_fwd_mlp_dim=1024
fi
export MAXTEXT_EXTRA_FLAGS="${MAXTEXT_EXTRA_FLAGS:-custom_mesh_and_rule=cp-as-ep \
use_gdn_kernel=true gdn_cp_mode=head gdn_chunk_size=64 \
decoder_layer_input=offload context=${REMAT_CONTEXT} gdn=remat gdn_conv=remat gdn_states=remat \
megablox=true sparse_matmul=true use_tokamax_gmm=true use_gmm_v2=true \
use_gmm_v2_heuristic_tiling=false merge_gating_gmm=false \
wi_tile_fwd_batch_seq=256 wi_tile_fwd_embed_dim=4096 wi_tile_fwd_mlp_dim=${_wi_tile_fwd_mlp_dim} \
wi_tile_dlhs_batch_seq=256 wi_tile_dlhs_embed_dim=4096 wi_tile_dlhs_mlp_dim=1024 \
wi_tile_drhs_batch_seq=256 wi_tile_drhs_embed_dim=4096 wi_tile_drhs_mlp_dim=1024 \
wo_tile_fwd_batch_seq=256 wo_tile_fwd_embed_dim=4096 wo_tile_fwd_mlp_dim=1024 \
wo_tile_dlhs_batch_seq=256 wo_tile_dlhs_embed_dim=4096 wo_tile_dlhs_mlp_dim=1024 \
wo_tile_drhs_batch_seq=256 wo_tile_drhs_embed_dim=4096 wo_tile_drhs_mlp_dim=1024 \
use_ring_of_experts=true num_moe_token_chunks=${NUM_MOE_TOKEN_CHUNKS} moe_chunk_barrier=false \
ring_of_experts_local_routing=true moe_expert_weight_prefetch=forward \
moe_chunk_pipeline=${_moe_chunk_pipeline} ring_of_experts_row_major_reduce_scatter=true \
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
# Opt-in: RETRY_WHEN_TOKENS_DROPPED=true reruns a micro-batch whose MoE ragged buffer overflows
# with a dropless program (RETRY_NUM_MOE_TOKEN_CHUNKS scanned token chunks, ~63 GiB/device)
# instead of dropping its tokens. Needs RAGGED_BUFFER_FACTOR > 0 and a MaxText with the
# training-engine dropless replay (AI-Hypercomputer/maxtext#5570).
# Optionally set RETRY_DROPLESS_FIRST_STEPS and FIRST_PHASE_RAGGED_BUFFER_FACTOR to run initial
# steps directly dropless or with an expanded buffer factor.
if [[ "${RETRY_WHEN_TOKENS_DROPPED:-false}" == "true" ]]; then
  export MAXTEXT_EXTRA_FLAGS="${MAXTEXT_EXTRA_FLAGS} retry_when_tokens_dropped=true retry_num_moe_token_chunks=${RETRY_NUM_MOE_TOKEN_CHUNKS:-8}"
  if [[ -n "${RETRY_DROPLESS_FIRST_STEPS:-}" ]]; then
    export MAXTEXT_EXTRA_FLAGS="${MAXTEXT_EXTRA_FLAGS} retry_dropless_first_steps=${RETRY_DROPLESS_FIRST_STEPS}"
  fi
  if [[ -n "${FIRST_PHASE_RAGGED_BUFFER_FACTOR:-}" ]]; then
    export MAXTEXT_EXTRA_FLAGS="${MAXTEXT_EXTRA_FLAGS} first_phase_ragged_buffer_factor=${FIRST_PHASE_RAGGED_BUFFER_FACTOR}"
  fi
  # Opt-in: the final micro-batch of each step runs the dropless program directly
  # rather than overflowing and replaying.
  if [[ "${RETRY_DROPLESS_FINAL_MICRO_BATCH:-false}" == "true" ]]; then
    export MAXTEXT_EXTRA_FLAGS="${MAXTEXT_EXTRA_FLAGS} retry_dropless_final_micro_batch=true"
  fi
fi
# Opt-in: MOE_EXCLUDE_PADDING=true keeps padding tokens out of MoE routing
# (moe_exclude_padding_from_routing), and LOG_MOE_DROPS=true logs a per-micro-batch
# MoE drop audit (log_moe_drops, MOE_DROP_AUDIT lines). Both need a MaxText with
# those flags.
if [[ "${MOE_EXCLUDE_PADDING:-false}" == "true" ]]; then
  export MAXTEXT_EXTRA_FLAGS="${MAXTEXT_EXTRA_FLAGS} moe_exclude_padding_from_routing=true"
fi
if [[ "${LOG_MOE_DROPS:-false}" == "true" ]]; then
  export MAXTEXT_EXTRA_FLAGS="${MAXTEXT_EXTRA_FLAGS} log_moe_drops=true"
fi
# TRAINER_TUNED_FLAGS=true, part 2 (part 1 is with the trainer libtpu flags above): the
# MaxText keys. Appended last: MaxText keeps the last value of a repeated key, so they
# replace the values above.
#   context_parallel_strategy=all_gather: attention all-gathers K/V over the context
#     axis instead of passing it around a ring.
#   sa_fuse_reciprocal=true: a Splash attention kernel option (SplashConfig
#     fuse_reciprocal). With context_parallel_strategy=all_gather and
#     sa_use_base2_exp=false, without the libtpu flag, on the bench: 130.8 -> 127.5
#     s/step.
#   sa_use_base2_exp stays at true (set above). The block set it to false until the
#     bench measured the base-2 exponentials 0.025 s faster per micro-batch
#     (o275x-gx-b2e); q * log2(e) is rounded in bf16, and its numerics run (nxgx-b2e)
#     passes all six gates.
#   gdn_cp_mode=seq: GDN context parallelism shards the sequence instead of the heads,
#     which drops head mode's all-to-alls (MaxText's `auto` picks seq at context 4).
#   gdn_cp_state=device: keeps seq mode's carries (conv halo, local transition, incoming
#     state), so the backward does not replay the halo exchange and the first pass.
#   moe_route stays at its default (remat). The bench image's MaxText tags nothing with
#     it (it tags the routing arrays moe_routing_maps, below), and neither does
#     AI-Hypercomputer/maxtext#5758, so the moe_route=device of the measured runs changed
#     nothing there. MaxText at this recipe's pin tags the ragged sort's integer routing
#     results with it, where device would keep them on the chip at a memory cost no run
#     has measured.
#   num_moe_token_chunks=2 comes from NUM_MOE_TOKEN_CHUNKS: the default here, and set by
#     mlperf_gbs1024_submission.sh for its replay layout.
# The keys above with two token chunks, on top of the libtpu flag (bench, same pod, with
# the inert moe_route=device): 126.4 -> 121.7 s/step.
#   gdn_cp_matmul_precision=high: seq mode composes GDN state in 3 bf16 passes, not 6.
#   cast_grads_after_all_reduce=true: all-reduce-only bf16 grads cast to f32 after it.
# The two together (bench, same pod, the sharded log-prob head on in both arms):
# 107.59 -> 106.24 s/step; grad_norm moved 1.5e-5 relative, less than a rerun (1.9e-5).
#   wi_tile_dlhs_mlp_dim=2048: the K tile of wi's input-gradient GMM. The trainer fuses
#     wi's gate and up weights (prefuse_moe_weights, on in mlperf_base.sh), so that
#     contraction is 2048 wide, and a 1024 tile takes two K steps that each fetch the
#     weight block again.
#   decoder_layer_input=device: each decoder layer's input stays in HBM instead of being
#     offloaded to host memory.
#   moe_routing_maps=device: the backward reuses the forward's MoE routing maps (top-k
#     expert ids, ragged-sort permutations, sorted token indices, group sizes) instead of
#     running top-k and the sorts again. It saves values; it does not change them.
#   moe_gmm_v2_dlhs_transpose_rhs=true: the input-gradient GMMs read the expert weights
#     through the kernel's transpose_rhs instead of from a transposed copy of them.
# These four on the bench, as the time of a regular micro-batch (a step has nine; the
# median of micro-batches 3 to 9 over steps 1 to 4), with no token dropped in any arm:
#   the tile alone, same pod: 10.773 -> 10.605 s (o275x-hd -> o275x-t2048, 106.3 -> 104.7
#     s/step);
#   decoder_layer_input=device alone, other pods: 10.754 -> 10.712 s (x3l -> x3dli, both
#     with the staged step boundary: 98.3 -> 98.0 s/step);
#   those two on top of the GDN change of AI-Hypercomputer/maxtext#5667, other pods:
#     10.582 -> 10.319 s (o275x-t67 -> o275x-stk1a, 104.5 -> 102.1 s/step); against two
#     reference runs the loss moved 4.6e-8 and grad_norm 3.9e-5 relative, a rerun 1.9e-5
#     (nxstk1a-rp);
#   on top of that, same pod: moe_routing_maps=device 10.319 -> 10.189 s (o275x-stk1a-a,
#     102.1 -> 100.9 s/step) and moe_gmm_v2_dlhs_transpose_rhs=true 10.319 -> 10.195 s
#     (o275x-stk1a-b, 102.1 -> 100.9 s/step).
#   moe_routing_index_math=count: the ring-of-experts ragged sort builds its routing
#     index maps with one sort of a packed key and by counting, instead of stable
#     argsorts: the same maps.
#   moe_unsplit_prefused_wi=true: the fused wi = [w0 | w1] stays whole up to the GEMM
#     instead of being split at the layer's entry and joined again before it.
#   moe_roe_packed_routing_gather=true: the layer routes its tokens once and all-gathers
#     their routing weights and expert ids as one array, not twice per token chunk.
#   moe_router_grad_from_wo_input=true: the down projection and the unsort run as one op
#     whose backward takes the routing-weight gradient from the GEMM's input, so the
#     rematerialized backward does not run the down-projection GEMM again.
#   moe_wi_gmm_fused_swiglu=true: the wi GEMM applies the SwiGLU to its f32 accumulator
#     in its epilogue, instead of a separate pass over its [rows, 2 * mlp] output.
#   gdn_conv1d_input_in_qkv_dtype=true: the GDN Conv1D backward keeps its halo-extended
#     input in qkv's dtype and casts each tap where it reads it, instead of writing an
#     f32 copy of it.
#   gdn_conv1d_dense_token_stack=true: the GDN forward kernel assembles its Conv1D output
#     from dense per-token views, so the SiLU after it does not run on rows padded from 1
#     to 8 sublanes.
#   wi_tile_fwd_batch_seq, wi_tile_drhs_batch_seq, wo_tile_fwd_batch_seq,
#   wo_tile_dlhs_batch_seq and wo_tile_drhs_batch_seq=512: row tiles of 512 for the five
#     MoE GMMs with 1024 tiles. wi_tile_dlhs_batch_seq stays 256 next to its 2048 K tile.
#   context_parallel_attention_load_balance_hoist_ids=true: remat_policy=custom saves
#     each attention layer's load-balance-ordered positions and segment ids. They depend
#     only on the batch, so jax.lax.scan keeps one copy outside the layer loop, and the
#     remat replay of every attention layer reads it instead of reordering them again.
#     The same values. Compiled for TPU v7x, XLA already moves the reorder's
#     collective-permutes out of the loop without the key; the key moves out the rest
#     of the reorder. MaxText refuses the key without
#     context_parallel_attention_load_balance=true, an even context parallelism of at
#     least 2, remat_policy=custom and scan_layers=true, which this recipe's trainer has
#     (TRAINER_MESH_CONTEXT=4, or 8 on the 256-chip trainer; REMAT_POLICY=custom).
#   moe_wo_gmm_fused_dswiglu=true: the input-gradient GMM of the MoE down projection runs
#     the SwiGLU backward in its epilogue: it turns its f32 accumulator into d[gate | up]
#     (with moe_router_grad_from_wo_input, also into the routing weight's gradient)
#     instead of writing the activation's gradient in bf16 for separate passes to read
#     back. The derivative reads the f32 accumulator, so the gradients of wi and of
#     everything before it change by bf16 rounding. MaxText refuses the key unless that
#     GMM has one n tile (wo_tile_dlhs_mlp_dim = the MoE MLP dim, 1024 here) and, with
#     moe_wi_gmm_fused_swiglu, wi_tile_fwd_mlp_dim halves to that dim or to a multiple of
#     128 dividing it (1024 here: two n tiles of 512 per half); it also needs explicit
#     tiles (use_gmm_v2_heuristic_tiling=false), prefuse_moe_weights, use_tokamax_gmm,
#     use_gmm_v2 and a remat policy that does not save the routed experts' gate or up
#     half, which this recipe's trainer has.
#   query_proj_kernel_gather_once=true: the attention query projection keeps the kernel
#     it gathered in the remat replay for its input gradient, so the backward gathers the
#     query kernel once per attention layer instead of twice. The same values. MaxText
#     refuses the key unless the backward replays the query projection
#     (remat_policy=custom with query_proj=remat, its default) and the kernel's embed
#     dimension is sharded. So the block passes it at every trainer mesh but the
#     256-chip 32x2x8 and 64x1x8, which keep query_proj on device (below): there the
#     backward does not replay the projection, so it reads the query kernel only for
#     the input gradient.
#   fp8_fake_quant_hoist=gather8: with fp8_moe_fake_quant=true (TRAINER_FP8=true) every
#     routed-expert weight is rounded to its per-channel fp8 grid in each layer's forward
#     and remat replay. The key computes the absmax that fixes each channel's scale once
#     for the whole layer stack, before the layer scan (one FSDP max all-reduce per
#     weight instead of one per layer and pass), and the expert weights' FSDP
#     all-gathers move the float8_e4m3fn codes (1 byte per element instead of 2), which
#     are dequantized after the gather. The same values: on CPU the loss, the logits and
#     every gradient are bit-identical to the key off. MaxText raises when the trainer
#     traces it without fp8_moe_fake_quant=true, so the block passes it only with
#     TRAINER_FP8=true, and only at the 128-chip 32x2x4 mesh, the one it was timed at.
#   wi_tile_fwd_mlp_dim=2048 with wi_tile_fwd_rhs_buffer_count=2, at 32x2x4: the fused
#     SwiGLU wi forward GMM (moe_wi_gmm_fused_swiglu) covers each half of its [gate | up]
#     output (the MoE MLP dim, 1024) with one N tile instead of two, so it makes one pass
#     over its input rows instead of two. With the kernel's default three weight buffers
#     that tile needs 66.95 MiB of VMEM, over the 57.60 MiB kernel limit;
#     wi_tile_fwd_rhs_buffer_count=2 keeps two (50.95 MiB; kernel-only compiles for TPU
#     v7x). Buffering only: the same values. The gate before the recipe's MaxText flags
#     sets the recipe's own tile to 2048, so the flags carry one value of it. MaxText
#     refuses the buffer count unless moe_wi_gmm_fused_swiglu=true and
#     use_gmm_v2_heuristic_tiling=false, as here. Both only at the 128-chip 32x2x4 mesh,
#     the one they were timed at.
#   gdn_cp_m_scan=true, at 32x2x4: with gdn_cp_mode=seq, the GDN context-parallel forward
#     folds each rank's chunk transitions into its local transition product with one
#     Pallas scan in VMEM instead of an XLA reduction tree of matmuls. MaxText refuses the
#     key unless use_gdn_kernel=true and gdn_chunk_size=64, as here. Its backward twin,
#     gdn_cp_dh_scan, is set below, with the final stack's keys.
#   The final stack's keys, at 32x2x4 (moe_fp8_bwd_dispatch and moe_fp8_token_all_gather only
#   with TRAINER_FP8=true):
#   gdn_cp_dh_scan=true: the GDN context-parallel backward folds the incoming state gradients
#     with one Pallas scan in VMEM instead of an XLA reduction tree. It pays only with the
#     single-SparseCore reduce-scatter libtpu flag, which the block sets at this mesh (part 1).
#   gdn_cp_pass1_states_and_bf16_qkv=true and gdn_out_proj_token_minor=true: GDN context-parallel
#     pass 1 runs a states-only forward kernel (no output, no chunk states to HBM), the
#     prefill-only forward kernels DMA bfloat16 qkv chunks and upcast them in VMEM instead of
#     materializing a float32 copy, and the gated-norm output is laid out token-minor before
#     out_proj. Bit-identical for bfloat16 activations; MaxText refuses the first unless
#     use_gdn_kernel=true, gdn_chunk_size=64 and dtype=bfloat16, as here.
#   moe_fp8_bwd_dispatch=true: in the ring of experts, the backward all-gather of the MoE output
#     cotangent moves float8_e4m3fn values with one float32 scale per row instead of bfloat16.
#     Lossy, so only with the trainer's fp8 fake-quant on (TRAINER_FP8=true), the cell whose
#     numerics run it passed.
#   ragged_unsort_spread_masked_indices=true: with moe_routing_index_math=count and the packed
#     ring-of-experts buffer, the masked lanes of the unsort backward's slot-order gather (other
#     shards' and dropped slots, about 7 of 8 at expert parallelism 8) read their own slot index
#     instead of all reading table entry 0. The outer where discards them: bit-identical.
#   moe_fallback_topk_capacity=128: under router replay the router's own top-k is computed only
#     for the tokens that use it (at most 128 rows per call, a lax.cond computes every row
#     above that) instead of for every token. Bit-identical.
#   gdn_cp_fast_state_passing=true: the GDN context-parallel cross-rank state and state-gradient
#     scans on 4 ranks take 2 rounds instead of 3 plus a psum. Bit-identical on CPU.
#   fp32_master_storage=residual16: Tunix's float32-master optimizer stores a scanned layer's
#     master as the 16 bits its bfloat16 weight does not hold plus one packed bit per element
#     and rebuilds it bit for bit at each update: the same masters, weights and updates, with
#     2.13 instead of 4 bytes per element stored and copied to and from host memory.
#   moe_fp8_token_all_gather=true: in the ring of experts, the forward token all-gather (the
#     dispatch) moves float8_e4m3fn values with one float32 scale per token row, dequantized
#     to bfloat16 right after the gather; the backward is the plain bfloat16 reduce-scatter.
#     Lossy, so only with the trainer's fp8 fake-quant on (TRAINER_FP8=true), the cell whose
#     numerics run it passed.
#   gdn_bwd_natural_layout=states: the Pallas GDN backward kernel reads the float32 chunk
#     states and t_inv (and dht / dh0) in their natural layouts through its BlockSpec index
#     maps instead of XLA relayouts to and from the head-group-major layout. Bit-identical.
#   gdn_gated_norm_layout=token_minor: the bfloat16 GDN output is laid out token-minor before
#     the gated RMSNorm, so its relayout is one bfloat16 transpose ahead of the float32
#     convert instead of a re-tile, a convert and a float32 transpose per call. Layout only.
#     On CPU the loss and the input gradient stay bit-identical. At the shapes of MaxText's
#     unit test only the float32 sum that forms the norm's scale gradient is reordered
#     (relative 1e-7); at other shapes a few elements of other weight gradients also move,
#     by up to one bfloat16 ulp, and the TPU backward program changes, so bit-identity on
#     TPU is not established. MaxText refuses it unless gdn_out_proj_token_minor=true,
#     as here.
#   context_parallel_attention_load_balance_order=dual_chunk_swap_local_self: the attention
#     layers' re-order into and out of the load-balanced token order leaves out of its two
#     collective-permutes the pairs in which a rank sends its own half to itself (a local
#     copy inside the permute); that rank takes its half with a select. The same token order
#     and values: forward and gradients bit-identical on CPU. MaxText refuses it unless
#     context_parallel_attention_load_balance=true with an even context parallelism, as here.
#   TUNIX_LOGPS_HEAD_GRAD_SCATTER=chunk, an environment variable of the trainer process
#     (tunix/rl/common.py), not a MaxText key, so the block appends it to TRAINER_EXTRA_ENV:
#     the log-prob head's backward reduce-scatters each chunk's vocabulary-table gradient
#     inside its chunk loop and keeps only this device's float32 shard, instead of adding
#     every chunk into a full-size float32 accumulator (3.8 GiB per device for the
#     [4096, 248320] table) and reduce-scattering it once after the loop. The reduce-scatter
#     runs in bfloat16, the dtype the head's vjp returns, so the gradient changes in its low
#     bits: precision class, not bit-identical (its numerics run passed, below). tunix
#     refuses it with the sharded head off, so the block sets it only with
#     TUNIX_LOGPS_SHARDED_HEAD=fused, the head it was timed with.
#   gdn_conv1d_natural_layout=true with gdn_fwd_kernel_head_group_size=4: the prefill-only
#     GDN forward kernels (context-parallel pass 1 and pass 2) run their causal Conv1D + SiLU
#     on the bfloat16 [chunk, dim] qkv buffer they DMA, with tokens on sublanes (each shifted
#     tap a sublane rotation, added in the same order with the same segment masks), instead
#     of storing each token into a compact float32 scratch and loading q, k and v back row by
#     row, and they process 4 kq heads (with their value heads) per group instead of all 16
#     at once. The same values: on CPU the conv stage, and the whole forward and states-only
#     kernels in TPU interpret mode, are bit-identical with the keys on and off. MaxText
#     refuses the first unless gdn_cp_pass1_states_and_bf16_qkv=true (its bf16 DMA buffer),
#     as here, and a group size that does not divide the kq heads (16 here).
#   gdn_in_proj_kernel_order=true: the GDN in_proj_qkvz is two matmuls whose outputs are
#     already in the Pallas kernel's order, qkv = [q | k | v] (all heads of each) and z, from
#     the column-permuted kernel gathered once, instead of one matmul in the per-key-head
#     [q | k | v | z] order followed by a per-head split, transposes and a concatenation into
#     the kernel's qkv (and their assembly back in the backward). On CPU the forward values
#     and the parameter gradients are bit-identical; the input gradient is one float32 sum of
#     the two partial products, so it matches to float32 summation order (its numerics run
#     passed, below). MaxText refuses it unless use_gdn_kernel=true with sequence-sharded GDN
#     context parallelism, as here (gdn_cp_mode=seq).
#   dense_grad_reduce=all_reduce: cp-as-ep-dense-no-cp replicates the dense weights over
#     context, and by default each of their gradients is reduce-scattered over all devices and
#     then all-gathered over context (per micro-batch at this cell, 240 small 256-way
#     reduce-scatters, 240 collective-permutes and 240 all-gathers). all_reduce drops the
#     embed_grad rule, so each of these gradients is reduced by one all-reduce instead. The
#     forward is unchanged and the gradients differ in summation order only (its numerics runs
#     passed, below). MaxText refuses it under a rule set that does not map embed_grad.
#   optimizer_device_resident_gib=6: with optimizer_memory_host_offload=true (as here) every
#     optimizer leaf crosses the host link twice per step; this keeps the largest leaves, up
#     to 6 GiB per device, in device memory between updates, where the update reads and writes
#     them in place. Placement only: the same values. The 6 GiB stay in device memory for the
#     whole step. MaxText refuses it without optimizer_memory_host_offload=true.
#   ragged_buffer_factor=2.0 (the submission launcher's default is 2.75): smaller MoE ragged
#     buffers. Only with RETRY_WHEN_TOKENS_DROPPED=true and a finite RAGGED_BUFFER_FACTOR: a
#     micro-batch whose routing overflows the buffer is then rerun with the dropless program,
#     so no token is dropped. RAGGED_BUFFER_FACTOR=-1 (fully dropless) and launches without
#     the retry keep their own value.
# These on the bench, regular micro-batch or s/step as above, 0 tokens dropped:
#   moe_routing_index_math=count 10.022 -> 9.875 s (o275x-mf-cnt-prof),
#     moe_unsplit_prefused_wi with two exact rewrites this block leaves off 10.022 ->
#     9.852 s (o275x-mf-gb-prof) and moe_router_grad_from_wo_input 10.022 -> 9.765 s
#     (o275x-mf-a14-prof), each against o275x-mf-ctl in the same pod;
#     moe_roe_packed_routing_gather 9.938 s (o275x-mf-pra-prof) against 10.033 s in
#     another pod (o275x-gdn-ctl);
#   the five MoE keys together, same pod: 10.0445 -> 9.510 s (o275x-s5c-l3-prof ->
#     o275x-s5moe5-l3-prof, 91.87 -> 87.04 s/step);
#   gdn_conv1d_input_in_qkv_dtype alone: -0.050 s per micro-batch;
#   then against o275x-s6c-l3 (86.970 s/step) in the same pod: moe_wi_gmm_fused_swiglu
#     86.253 s/step (o275x-s6silu-l3-prof) and the five 512 tiles 86.364 s/step
#     (o275x-s6t512-l3); gdn_conv1d_dense_token_stack 86.926 -> 85.731 s/step, same pod,
#     against an arm whose one change is a no-op (o275x-s6n2-l3-prof ->
#     o275x-s6gr-l3-prof).
#   context_parallel_attention_load_balance_hoist_ids alone, against controls in other
#     pods: 99.504 -> 99.267 s/step (o275x-gdn-ctl -> o275x-s3-x17h), 99.248 -> 99.267
#     (o275x-mf-ctl -> o275x-s3-x17h) and 91.7513 -> 91.8665 (o275x-stk3-l3-prof ->
#     o275x-s5c-l3-prof): -0.237, +0.019 and +0.115 s/step, within the 0.255 s/step
#     spread of one unchanged configuration over four pods (o275x-hd), so no effect is
#     resolved.
#   moe_wo_gmm_fused_dswiglu, switched with an environment variable on an overlay build
#     with this block's keys of AI-Hypercomputer/maxtext#5758, the hoist and padding
#     first, against controls in another pod: 83.316 -> 82.148 s/step, 9.0845 -> 8.9545 s
#     (o275x-c9pfxctl-l3 -> o275x-c9ypfctl-l3); with the fused log-prob head and two
#     changes that no PR has, 82.787 -> 81.604 s/step (o275x-c9pfxall-l3-prof ->
#     o275x-c9ypfall-l3). On one TPU v7x chip its kernels at this cell's shapes match an
#     f32 SwiGLU backward to 1.658e-3 relative L2 (parambole-bck-2d4c2), and its numerics
#     run passes all six gates (nxc9ypfckhv).
#   query_proj_kernel_gather_once: not timed on TPU alone (only in the whole cell below).
#     In an AOT compile of this cell's trainer program (with cp-as-ep) the backward
#     gathers the query kernel once per attention layer instead of twice, at the same
#     device peak; the MaxText change's test, executed on TPU v5p, gives bit-identical
#     values and gradients with it.
#   fp8_fake_quant_hoist=gather8, on the pinned MaxText with this block's other keys at
#     32x2x4, against the same code without it in the same pod: 80.385 -> 79.397 s/step
#     (o275x-v19a-ctl -> o275x-v19a-fq8, 3,669 -> 3,714 tokens/s/chip), 0 tokens
#     dropped. In an AOT compile of that cell fwd_bwd takes 80.26 GiB of HBM (79.96
#     without it).
#   wi_tile_fwd_mlp_dim=2048 with wi_tile_fwd_rhs_buffer_count=2, on the pinned MaxText with
#     this block's other keys at 32x2x4, against the same code with the two keys off in the
#     same pod: 79.342 -> 79.024 s/step (o275x-ec-ctl -> o275x-ec-e6k, 3,717 -> 3,732
#     tokens/s/chip) and 78.985 in a replicate (o275x-ec-e6kr), 0 tokens dropped; 79.140 and
#     79.045 s/step in two other pods (o275x-e6k1, o275x-p3-e6k). Its numerics run passes all
#     six gates, with the loss of the code without the keys (nx-e6k1). In an AOT compile of
#     that cell fwd_bwd takes the same 80.26 GiB of HBM as without the keys, at the same
#     scheduler memory.
#   gdn_cp_m_scan=true, on MaxText with this block's other keys at 32x2x4 (without the 2048
#     tile), against the same code with the key off in the same pod: 79.480 -> 78.822 s/step
#     (o275x-g2-ctl -> o275x-g2-m, 3,711 -> 3,742 tokens/s/chip), 0 tokens dropped; 78.795
#     in a replicate in another pod (o275x-gr-m). Its numerics run passes all six gates with
#     the loss of the code without the key (nx-gm-m). gdn_cp_dh_scan made the step slower:
#     79.403 -> 80.610 s/step alone (o275x-g3-ctl -> o275x-g3-dh) and 80.152 with both
#     scans (o275x-g2-both, against 79.480).
#   The final stack's keys, each against the same code without it in the same pod (later
#   MaxText stacks than the previous pin; steps 1-4, 0 tokens dropped):
#   gdn_cp_pass1_states_and_bf16_qkv with gdn_out_proj_token_minor: 79.491 -> 78.181 s/step
#     (o275x-c3-ctl -> o275x-c3-both); with the 2048 wi tile and gdn_cp_m_scan as well, 77.223
#     (o275x-c3-k1em), numerics 6 of 6 (nx-c3-k1em).
#   gdn_cp_dh_scan with the single-SparseCore reduce-scatter flag, on that stack, both arms with
#     the flag: 78.134 -> 76.221 s/step (o275x-s2-emc -> o275x-s2-emdc, 3,774 -> 3,869
#     tokens/s/chip), numerics 6 of 6 (nx-s2-emdc). The flag alone costs about 0.9 s/step (78.134
#     against 77.223 without it, another pod), so the two together are about -1.0 s/step.
#   moe_fp8_bwd_dispatch on that stack: 76.419 -> 76.183 s/step (o275x-fs-best -> o275x-fs-f8d),
#     76.131 in a replicate in a new pod (o275x-bf-best), numerics 6 of 6 (nx-fs-f8d).
#   ragged_unsort_spread_masked_indices on that stack: 76.405 -> 74.041 s/step (o275x-sgs-ctl ->
#     o275x-sgs-spread, 3,860 -> 3,983 tokens/s/chip), numerics 6 of 6 (nx-sgs-spread).
#   moe_fallback_topk_capacity=128 and fp32_master_storage=residual16 on that stack, one pod:
#     76.413 -> 76.116 and 76.057 s/step (o275x-c1s-ctl -> o275x-c1s-c1, o275x-c1s-s1), numerics
#     6 of 6 with both (nx-c1s1).
#   gdn_cp_fast_state_passing was not timed alone on TPU.
#   On that whole stack (all of the keys above), each against the same code without it in
#   the same pod (MaxText stacks with these three keys; steps 1-4, 0 tokens dropped):
#   moe_fp8_token_all_gather: 73.202 -> 72.692 s/step (o275x-m3b-ctl -> o275x-m3-cb1, 4,029
#     -> 4,057 tokens/s/chip), and 73.139 -> 72.721 in a replicate in a new pod
#     (o275x-m3b-ctlr -> o275x-m3-cb1r), numerics 6 of 6 (nx-m3-cb1).
#   gdn_bwd_natural_layout=states, with a MoE dead-tail-row key this recipe does not set
#     (-0.025 s/step on its own): 73.348 -> 73.140 s/step (o275x-mg3-ctl -> o275x-mg3-all),
#     numerics 6 of 6 (nx-mg3-all); without that key, on top of moe_fp8_token_all_gather:
#     72.554 -> 72.268 s/step (o275x-mg4-ctl -> o275x-mg4-nat).
#   gdn_gated_norm_layout=token_minor on top of both: 72.442 -> 71.700 s/step
#     (o275x-mg4-natc -> o275x-mg4-c0, 4,071 -> 4,113 tokens/s/chip), 71.779 in a replicate
#     in a new pod (o275x-mg4-c0r); the three together: numerics 6 of 6 (nx-mg4-c0).
#   context_parallel_attention_load_balance_order=dual_chunk_swap_local_self, on that whole
#     stack without the three keys just above: 73.633 -> 73.375 s/step (o275x-hs-ctl ->
#     o275x-hs-p1a).
#   TUNIX_LOGPS_HEAD_GRAD_SCATTER=chunk, on that stack without the three keys too: 73.146 ->
#     72.865 s/step (o275x-hr-ctl -> o275x-hr-chunk).
#   The two together on top of all of the keys above: 71.771 -> 71.229 s/step (o275x-mg5-ctl
#     -> o275x-mg5-all, 4,109 -> 4,140 tokens/s/chip), 71.025 in a replicate in a new pod
#     (o275x-mg5-allr), numerics 6 of 6 (nx-mg5-all).
#   On top of all of the keys above, each against the same code without it in one pod
#   (o275x-w28s-ctl: 71.180 s/step, 4,143 tokens/s/chip):
#   gdn_conv1d_natural_layout with gdn_fwd_kernel_head_group_size=4: 69.707 s/step
#     (o275x-gc-on, -1.474 s/step, 4,231 tokens/s/chip), numerics 6 of 6 (nx-gc-on).
#   gdn_in_proj_kernel_order: 69.979 s/step (o275x-qkv-on, -1.201 s/step, 4,214
#     tokens/s/chip), numerics 6 of 6 (nx-qkv-on).
#   dense_grad_reduce=all_reduce: 69.446 s/step (o275x-w27d-ar, -1.734 s/step, 4,247
#     tokens/s/chip).
#   optimizer_device_resident_gib=6: 70.576 s/step (o275x-or-k6, -0.604 s/step, 4,179
#     tokens/s/chip).
#   The four together, on the bench's production routing table, against the same code with
#   the four off in the same pod: 69.339 -> 64.271 s/step (o275x-m6q-ctl -> o275x-m6p4,
#   -5.068 s/step, 4,253 -> 4,589 tokens/s/chip). Numerics 6 of 6 with the four (nx-m6p4)
#   and with all but optimizer_device_resident_gib (nx-m6p3). The two GDN keys together
#   without the other two have not run on TPU.
#   custom_mesh_and_rule=cp-as-ep-dense-no-cp, on the code of
#     AI-Hypercomputer/maxtext#5758 at this cell without
#     context_parallel_attention_load_balance_hoist_ids, moe_wo_gmm_fused_dswiglu and
#     query_proj_kernel_gather_once: 82.111 to 82.269 s/step with cp-as-ep (four runs in
#     three pods, mean 82.185; 8.942 to 8.959 s) -> 81.261 and 81.245 s/step in two pods
#     (o275x-l2-l3-prof; 8.838 s), 3,629 and 3,630 tokens/s/chip. Its numerics runs in
#     both pods pass all six gates (nxl2), with cp-as-ep's loss to the ten digits logged
#     and the gradient norm within 1.4e-4 relative. In an AOT compile of that cell
#     fwd_bwd takes 81.05 GiB of HBM (78.28 with cp-as-ep), under the runs' 84992 MiB
#     budget.
# The TRAINER_TUNED_FLAGS cell of mlperf_gbs1024_submission.sh (these keys, its log-prob
# head and chunk size, padding first in packed rows, the staged step boundary) before
# moe_wo_gmm_fused_dswiglu, query_proj_kernel_gather_once and cp-as-ep-dense-no-cp:
# 82.671 and 82.670 s/step in two pods (o275x-c9pfxhv-l3), against 106.30 s/step before
# these keys, the fused head and padding first (o275x-hd, which also ran without the
# staged step boundary; per regular micro-batch, which the boundary leaves alone, about
# 10.77 -> 9.01 s). That build switched the MaxText changes with environment variables
# (context_parallel_attention_load_balance_hoist_ids as PARAMBOLE_ATTN_HOIST_PERMUTE=1).
# The keys themselves, on the code of AI-Hypercomputer/maxtext#5758 without
# gdn_conv1d_dense_token_stack and context_parallel_attention_load_balance_hoist_ids,
# the fused head and padding first: 85.295 and 85.251 s/step (o275x-prc-l3-prof,
# o275x-prc-l3-r).
# Numerics (loss gates off, random advantages, three steps, against two reference runs
# whose gradient norms differ by 1.9e-5 relative): all six gates pass (nxc9pfxhv).
# The bench cell of this block at 32x2x4 with all four of
# context_parallel_attention_load_balance_hoist_ids, moe_wo_gmm_fused_dswiglu,
# query_proj_kernel_gather_once and cp-as-ep-dense-no-cp, on the MaxText of
# AI-Hypercomputer/maxtext#5758 with the change on top of it (the pinned MaxText without
# #5721's three commits of Rohan Bierneni): 80.398 and 80.342 s/step in two pods
# (o275x-f2-l3-prof, o275x-f2-l3-r), 3,668 and 3,671 tokens/s/chip, regular micro-batch
# 8.748 s, against 82.373 s/step and 8.966 s for the code of #5758 with the keys before
# these four (o275x-prfc-f1, a third pod of the same booking; the two pods' own control
# arms did not finish, so the comparison is across pods). Numerics in both pods (nxf2):
# all six gates pass, the loss is 0.0304453575 in all three steps, as with that code
# under the same protocol (nxprf-l3), and the gradient norm is within 2.0e-4 relative
# of it. Not run with #5721's three commits, at the 256-chip meshes, or end to end.
# moe_routing_maps and moe_gmm_v2_dlhs_transpose_rhs come from
# AI-Hypercomputer/maxtext#5698, moe_routing_index_math, moe_unsplit_prefused_wi,
# moe_roe_packed_routing_gather, moe_router_grad_from_wo_input, moe_wi_gmm_fused_swiglu,
# gdn_conv1d_input_in_qkv_dtype and gdn_conv1d_dense_token_stack from the MaxText change
# of AI-Hypercomputer/maxtext#5758 (on top of #5698), and
# context_parallel_attention_load_balance_hoist_ids, moe_wo_gmm_fused_dswiglu and
# query_proj_kernel_gather_once from the MaxText change on top of
# AI-Hypercomputer/maxtext#5721. All twelve keys, and the cp-as-ep-dense-no-cp rule
# below, are in the MaxText pinned in requirements/maxtext_requirements.txt (branch
# parambole/rl-4600-final-maxtext, on atwigg/mlperf at bbd0f6f10, the merge of
# AI-Hypercomputer/maxtext#5721; bbd0f6f10 itself lacks those three). So is
# fp8_fake_quant_hoist, from the commit after them on that branch (94f11c09d lacks it), and
# wi_tile_fwd_rhs_buffer_count and gdn_cp_m_scan (with gdn_cp_dh_scan), from later ones
# (e1cd8bf1d lacks them, 15285fd74 the scans), and gdn_cp_pass1_states_and_bf16_qkv,
# gdn_out_proj_token_minor, moe_fp8_bwd_dispatch, ragged_unsort_spread_masked_indices,
# moe_fallback_topk_capacity, fp32_master_storage and gdn_cp_fast_state_passing, from the
# commits after 1bf840cff (which lacks all seven), and moe_fp8_token_all_gather,
# gdn_bwd_natural_layout and gdn_gated_norm_layout, from the commits after 6679552770a0 (which
# lacks all three), and context_parallel_attention_load_balance_order, from the commits after
# ae5257c1f (which lacks it), and gdn_conv1d_natural_layout, gdn_fwd_kernel_head_group_size
# and gdn_in_proj_kernel_order, from the commits after 8c56959ae (which lacks all three), and
# dense_grad_reduce and optimizer_device_resident_gib, from the commits after d7c75f205 (which
# lacks both). MaxText refuses a key it does not define when the trainer starts, so an older
# MaxText needs TRAINER_TUNED_FLAGS=false.
# The 256-chip trainer (TRAINER_TPU_SLICE=tpu7x:4x8x8, 512 devices), from Rohan Bierneni's
# 256-chip trainer runs (2026-10-15, RL trainer bench, 9 micro-batches of 64 rows, steps 1-4):
# 32x2x8 60.62 s/step and 64x1x8 55.97 s/step, against 82.17 s/step for 32x2x4 on 128 chips.
#   at 32x2x8 (TRAINER_MESH_FSDP=32, TRAINER_MESH_EXPERT=2, TRAINER_MESH_CONTEXT=8) and
#   64x1x8 (64, 1, 8), query_proj, key_proj, value_proj, out_proj and qkv_proj=device keep
#   those projections' outputs in HBM for the backward instead of recomputing them. MaxText
#   also names the GDN layers' fused qkvz projection qkv_proj, so qkv_proj covers them too.
#   At 64x1x8, NUM_MOE_TOKEN_CHUNKS defaults to 1 (above), so moe_chunk_pipeline is false,
#   and so is moe_roe_packed_routing_gather: MaxText refuses it without the pipeline.
# custom_mesh_and_rule=cp-as-ep-dense-no-cp (AI-Hypercomputer/maxtext#5783, in #5721) comes after
# the default custom_mesh_and_rule=cp-as-ep, and MaxText keeps the last value of a repeated key:
# the dense weights (attention and shared-expert projections, the MoE router and the shared-expert
# gate) are sharded over fsdp x expert and replicated over context, their gradients reduce-scattered
# over all devices and then all-gathered over context (at 32x2x4 all-reduced instead, by
# dense_grad_reduce=all_reduce below); the GDN projections (embed_gdn), routed
# experts, embeddings and activations stay as under cp-as-ep. 128 chips: 82.185 -> 81.253 s/step.
if [[ "${TRAINER_TUNED_FLAGS:-false}" == "true" ]]; then
  export MAXTEXT_EXTRA_FLAGS="${MAXTEXT_EXTRA_FLAGS} context_parallel_strategy=all_gather sa_fuse_reciprocal=true gdn_cp_mode=seq gdn_cp_state=device gdn_cp_matmul_precision=high cast_grads_after_all_reduce=true wi_tile_dlhs_mlp_dim=2048 decoder_layer_input=device moe_routing_maps=device moe_gmm_v2_dlhs_transpose_rhs=true moe_routing_index_math=count moe_unsplit_prefused_wi=true moe_roe_packed_routing_gather=${_moe_chunk_pipeline} moe_router_grad_from_wo_input=true moe_wi_gmm_fused_swiglu=true gdn_conv1d_input_in_qkv_dtype=true gdn_conv1d_dense_token_stack=true wi_tile_fwd_batch_seq=512 wi_tile_drhs_batch_seq=512 wo_tile_fwd_batch_seq=512 wo_tile_dlhs_batch_seq=512 wo_tile_drhs_batch_seq=512 context_parallel_attention_load_balance_hoist_ids=true moe_wo_gmm_fused_dswiglu=true custom_mesh_and_rule=cp-as-ep-dense-no-cp"
  case "${TRAINER_MESH_FSDP}x${TRAINER_MESH_EXPERT}x${TRAINER_MESH_CONTEXT}" in
    32x2x8|64x1x8)
      export MAXTEXT_EXTRA_FLAGS="${MAXTEXT_EXTRA_FLAGS} query_proj=device key_proj=device value_proj=device out_proj=device qkv_proj=device"
      ;;
    *)
      # query_proj stays remat, so the backward replays the query projection, which
      # query_proj_kernel_gather_once needs (MaxText refuses it with query_proj=device).
      export MAXTEXT_EXTRA_FLAGS="${MAXTEXT_EXTRA_FLAGS} query_proj_kernel_gather_once=true"
      ;;
  esac
  # fp8_fake_quant_hoist only at the mesh it was timed at, and only with the trainer's fp8
  # fake-quant on (fp8_moe.sh: TRAINER_FP8 defaults to true), without which MaxText raises.
  if [[ "${TRAINER_MESH_FSDP}x${TRAINER_MESH_EXPERT}x${TRAINER_MESH_CONTEXT}" == "32x2x4" \
        && "${TRAINER_FP8:-true}" == "true" ]]; then
    export MAXTEXT_EXTRA_FLAGS="${MAXTEXT_EXTRA_FLAGS} fp8_fake_quant_hoist=gather8"
  fi
  # wi_tile_fwd_rhs_buffer_count=2 only at the mesh it was timed at, where the recipe's
  # wi_tile_fwd_mlp_dim is 2048 (above): that tile needs the 2-deep weight pipeline to
  # fit VMEM.
  if [[ "${TRAINER_MESH_FSDP}x${TRAINER_MESH_EXPERT}x${TRAINER_MESH_CONTEXT}" == "32x2x4" ]]; then
    export MAXTEXT_EXTRA_FLAGS="${MAXTEXT_EXTRA_FLAGS} wi_tile_fwd_rhs_buffer_count=2"
  fi
  # gdn_cp_m_scan only at the mesh it was timed at.
  if [[ "${TRAINER_MESH_FSDP}x${TRAINER_MESH_EXPERT}x${TRAINER_MESH_CONTEXT}" == "32x2x4" ]]; then
    export MAXTEXT_EXTRA_FLAGS="${MAXTEXT_EXTRA_FLAGS} gdn_cp_m_scan=true"
  fi
  # The final stack's keys only at the mesh they were timed at, where part 1 also sets the
  # single-SparseCore reduce-scatter flag that gdn_cp_dh_scan needs; moe_fp8_bwd_dispatch only
  # with the trainer's fp8 fake-quant on, the cell its numerics were measured on.
  if [[ "${TRAINER_MESH_FSDP}x${TRAINER_MESH_EXPERT}x${TRAINER_MESH_CONTEXT}" == "32x2x4" ]]; then
    export MAXTEXT_EXTRA_FLAGS="${MAXTEXT_EXTRA_FLAGS} gdn_cp_dh_scan=true gdn_cp_pass1_states_and_bf16_qkv=true gdn_out_proj_token_minor=true"
    if [[ "${TRAINER_FP8:-true}" == "true" ]]; then
      export MAXTEXT_EXTRA_FLAGS="${MAXTEXT_EXTRA_FLAGS} moe_fp8_bwd_dispatch=true"
    fi
    export MAXTEXT_EXTRA_FLAGS="${MAXTEXT_EXTRA_FLAGS} ragged_unsort_spread_masked_indices=true moe_fallback_topk_capacity=128 gdn_cp_fast_state_passing=true fp32_master_storage=residual16"
  fi
  # Three keys timed on top of the final stack, only at that mesh: moe_fp8_token_all_gather only
  # with the trainer's fp8 fake-quant on (lossy; its numerics run used that cell), then
  # gdn_bwd_natural_layout=states and gdn_gated_norm_layout=token_minor (which MaxText refuses
  # without gdn_out_proj_token_minor=true, set just above).
  if [[ "${TRAINER_MESH_FSDP}x${TRAINER_MESH_EXPERT}x${TRAINER_MESH_CONTEXT}" == "32x2x4" ]]; then
    if [[ "${TRAINER_FP8:-true}" == "true" ]]; then
      export MAXTEXT_EXTRA_FLAGS="${MAXTEXT_EXTRA_FLAGS} moe_fp8_token_all_gather=true"
    fi
    export MAXTEXT_EXTRA_FLAGS="${MAXTEXT_EXTRA_FLAGS} gdn_bwd_natural_layout=states gdn_gated_norm_layout=token_minor"
  fi
  # Two more timed on top of those, only at that mesh: the attention re-order without its
  # self-copy (MaxText refuses it without context_parallel_attention_load_balance=true, which
  # the recipe sets), and the per-chunk reduce-scatter of the log-prob head's table gradient,
  # which the trainer process reads from its environment (TRAINER_EXTRA_ENV), only with the
  # fused head (tunix refuses it with the sharded head off). A value the caller set, in
  # TRAINER_EXTRA_ENV or as TUNIX_LOGPS_HEAD_GRAD_SCATTER, is kept.
  if [[ "${TRAINER_MESH_FSDP}x${TRAINER_MESH_EXPERT}x${TRAINER_MESH_CONTEXT}" == "32x2x4" ]]; then
    export MAXTEXT_EXTRA_FLAGS="${MAXTEXT_EXTRA_FLAGS} context_parallel_attention_load_balance_order=dual_chunk_swap_local_self"
    _logps_head_mode="$(printf '%s' "${TUNIX_LOGPS_SHARDED_HEAD:-}" | tr -d '[:space:]' | tr '[:upper:]' '[:lower:]')"
    if [[ "${_logps_head_mode}" == "fused" && " ${TRAINER_EXTRA_ENV} " != *" TUNIX_LOGPS_HEAD_GRAD_SCATTER="* ]]; then
      export TRAINER_EXTRA_ENV="${TRAINER_EXTRA_ENV} TUNIX_LOGPS_HEAD_GRAD_SCATTER=${TUNIX_LOGPS_HEAD_GRAD_SCATTER:-chunk}"
    fi
    unset _logps_head_mode
  fi
  # Three more timed on top of those, only at that mesh, right after the re-order key (the order
  # the TPU runs used): the GDN forward kernels' natural-layout Conv1D with 4 kq heads per group
  # (MaxText refuses the first without gdn_cp_pass1_states_and_bf16_qkv=true, set above), and
  # the GDN in-projection computed in the kernel's q|k|v order (MaxText refuses it with
  # head-sharded GDN context parallelism; the block sets gdn_cp_mode=seq).
  if [[ "${TRAINER_MESH_FSDP}x${TRAINER_MESH_EXPERT}x${TRAINER_MESH_CONTEXT}" == "32x2x4" ]]; then
    export MAXTEXT_EXTRA_FLAGS="${MAXTEXT_EXTRA_FLAGS} gdn_conv1d_natural_layout=true gdn_fwd_kernel_head_group_size=4 gdn_in_proj_kernel_order=true"
  fi
  # Three more timed on top of those, only at that mesh, after the GDN keys and in the order the
  # TPU runs appended them: the dense-weight gradients reduced by all-reduce (MaxText refuses it
  # under a rule set without embed_grad; the block sets cp-as-ep-dense-no-cp), the largest
  # optimizer leaves kept in device memory between updates, up to the GiB per device on the
  # second line (MaxText refuses it without optimizer_memory_host_offload=true, which the
  # recipe sets), and the MoE ragged buffer factor 2.0, only with the dropless retry and a
  # finite buffer factor (an overflowing micro-batch is then rerun dropless, so no token is
  # dropped; RAGGED_BUFFER_FACTOR=-1 and launches without the retry keep their own value).
  if [[ "${TRAINER_MESH_FSDP}x${TRAINER_MESH_EXPERT}x${TRAINER_MESH_CONTEXT}" == "32x2x4" ]]; then
    export MAXTEXT_EXTRA_FLAGS="${MAXTEXT_EXTRA_FLAGS} dense_grad_reduce=all_reduce"
    export MAXTEXT_EXTRA_FLAGS="${MAXTEXT_EXTRA_FLAGS} optimizer_device_resident_gib=6"
    if [[ "${RAGGED_BUFFER_FACTOR}" != "-1" && "${RETRY_WHEN_TOKENS_DROPPED:-false}" == "true" ]]; then
      export MAXTEXT_EXTRA_FLAGS="${MAXTEXT_EXTRA_FLAGS} ragged_buffer_factor=2.0"
    fi
  fi
fi
export DEBUG=${DEBUG:-0}

# DeepSWE Environment & Agent Sandbox
export SANDBOX_NODE_SELECTOR_KEY="${SANDBOX_NODE_SELECTOR_KEY:-workload}"
export SANDBOX_NODE_SELECTOR_VAL="${SANDBOX_NODE_SELECTOR_VAL:-sandbox}"
export SANDBOX_TOLERATIONS='[{"key":"workload","operator":"Equal","value":"sandbox","effect":"NoSchedule"}]'
export IMAGE_REWRITE_PREFIX="${IMAGE_REWRITE_PREFIX:-us-central1-docker.pkg.dev/cloud-tpu-multipod-dev/tunix/}"

source "${DIR}/mlperf_base.sh" "$@"
