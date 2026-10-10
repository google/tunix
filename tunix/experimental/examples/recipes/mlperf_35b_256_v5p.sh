#!/bin/bash
set -e

# ==============================================================================
# MLPerf DeepSWE recipe: Qwen3.5-35B-A3B on TPU v5p (Scaled Rollout / Batch 32)
# ==============================================================================
# - Cluster: bodaborg-v5p-nap in europe-west4
# - Trainer on 64 chips (tpuv5:4x4x4 = 64 devices, FSDP=32, TP=1, EXPERT=1, CP=2)
# - Rollout on 128 chips (32 replicas x 4 chips tpuv5:2x2x1, EP=4, TP=1)
# - Batch size 32 (512 sequences/step) with sqrt-scaled LR and grad norm clip
# - Sandboxes on sandbox-cpu-pool
# ==============================================================================

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Fill these before you run.
# k8s has a 63 char limit on total label name, so keep job_prefix unique to your job and short
export JOB_PREFIX="${JOB_PREFIX:-${USER}}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-${JOB_PREFIX}-mlperf-35b-256}"

export REGION="${REGION:-europe-west4}"
export CLUSTER="${CLUSTER:-bodaborg-v5p-nap}"
export BUCKET="${BUCKET:-gs://atwigg-trellis-europe-west4-dev}"
export MAXTEXT_CKPT="${MAXTEXT_CKPT:-gs://hengtaoguo-maxtext-logs/checkpoints/qwen3.5-35b-a3b/scanned/2026-06-11-10-27/0/items}"

export MAXTEXT_OUTPUT_DIR="${MAXTEXT_OUTPUT_DIR:-${BUCKET}/maxtext/${JOB_PREFIX}}"
export TRAJECTORY_LOG_DIR="${TRAJECTORY_LOG_DIR:-${BUCKET}/trajectories/${JOB_PREFIX}/logger}"
export TRAJECTORY_STORE_ROOT_DIR="${TRAJECTORY_STORE_ROOT_DIR:-${TRAJECTORY_STORE_ROOT:-${BUCKET}/trajectories/${JOB_PREFIX}/store}}"

export K8S_NAMESPACE="${K8S_NAMESPACE:-trellis}"

# v5p has 1 device/chip and 4 chips (4 devices) per host.
export RAIDEN_DEVICES_PER_HOST="${RAIDEN_DEVICES_PER_HOST:-4}"
export RAIDEN_BROADCAST_HOST_RATIO="${RAIDEN_BROADCAST_HOST_RATIO:-1.0}"
export RAIDEN_BROADCAST_PIPELINE_STAGES="${RAIDEN_BROADCAST_PIPELINE_STAGES:-4}"

# Model configuration
export MODEL_NAME="Qwen3.5-35B-A3B"
export MODEL_ID="Qwen/Qwen3.5-35B-A3B"
export TOKENIZER_PATH="${TOKENIZER_PATH:-Qwen/Qwen3.5-35B-A3B}"
export MAXTEXT_MODEL_NAME="qwen3.5-35b-a3b"

# Topologies (64 chips / 64 devices Trainer 4x4x4, 32x 4-chip Rollout slices)
# use_gdn_kernel=true requires TRAINER_MESH_TP=1 (the GDN kernel shard_map does not shard over 'tensor').
# Mesh product: 32 * 1 * 1 * 2 = 64 devices.
export TRAINER_JOBSET_YAML="${TRAINER_JOBSET_YAML:-jobset.pathways.yaml}"
export TRAINER_TPU_SLICE="${TRAINER_TPU_SLICE:-tpuv5:4x4x4}"
export TRAINER_MESH_FSDP="${TRAINER_MESH_FSDP:-32}"
export TRAINER_MESH_TP="${TRAINER_MESH_TP:-1}"
export TRAINER_MESH_EXPERT="${TRAINER_MESH_EXPERT:-1}"
export TRAINER_MESH_CONTEXT="${TRAINER_MESH_CONTEXT:-2}"

# Rollout: 32 replicas x 4 chips (tpuv5:2x2x1 = 1 host per replica).
# dp * tp * expert must equal the DEVICE count (1 per chip on v5p).
export ROLLOUT_JOBSET_YAML="${ROLLOUT_JOBSET_YAML:-jobset.tpu.yaml}"
export ROLLOUT_TPU_SLICE="${ROLLOUT_TPU_SLICE:-tpuv5:2x2x1}"
export VLLM_DATA_PARALLEL_SIZE="${VLLM_DATA_PARALLEL_SIZE:-1}"
_rollout_dims="${ROLLOUT_TPU_SLICE#*:}"
export ROLLOUT_MESH_EXPERT="${ROLLOUT_MESH_EXPERT:-$(( ${_rollout_dims//x/*} / ${VLLM_DATA_PARALLEL_SIZE:-1} ))}"
export ROLLOUT_REPLICAS="${ROLLOUT_REPLICAS:-32}"

# Sandbox Concurrency
export MAX_WARMPOOL_REPLICAS="${MAX_WARMPOOL_REPLICAS:-16}"
export MAX_CONCURRENCY="${MAX_CONCURRENCY:-2048}"

# MLPerf RCP logging with deferred offline eval
export RCP_LOGGING="${RCP_LOGGING:-true}"
export BATCH_SIZE="${BATCH_SIZE:-32}"
export MAX_STEPS="${MAX_STEPS:-30}"
export CHECKPOINT_SAVE_INTERVAL_STEPS="${CHECKPOINT_SAVE_INTERVAL_STEPS:-1}"
export CHECKPOINT_MAX_TO_KEEP="${CHECKPOINT_MAX_TO_KEEP:-30}"
export LEARNING_RATE="${LEARNING_RATE:-1.4142135624e-6}"
export MAX_GRAD_NORM="${MAX_GRAD_NORM:-0.08838834765}"

# vLLM Rollout Configuration (from paste.googleplex.com/5903655694368768)
# Rollout attention kernel (USE_BATCHED_RPA_LONG_CTX_KERNEL -> ROLLOUT_MAXTEXT_ATTENTION); see rollout_attention.sh.
source "${DIR}/rollout_attention.sh"
export VLLM_ADDITIONAL_CONFIG='{"sharding":{"sharding_strategy":{"expert_parallelism":'"${ROLLOUT_MESH_EXPERT}"',"tensor_parallelism":1,"enable_dp_attention":true}},"custom_mamba_cache_multiplier":16,"maxtext_config":{"scan_layers":false,"attention":"'"${ROLLOUT_MAXTEXT_ATTENTION}"'","allow_split_physical_axes":true,"use_multimodal":false,"prefuse_moe_weights":true,"per_device_batch_size":0.0}}'

# Rollout Worker Flags & Raiden tuning
export ONEHOT_MOE_PERMUTE_THRESHOLD="${ONEHOT_MOE_PERMUTE_THRESHOLD:-32768}"
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
export VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY="RAIDEN_,TPU_,WEIGHT_SYNC_"
export VLLM_RAY_EXTRA_ENV_VARS_TO_COPY="USE_BATCHED_RPA_LONG_CTX_KERNEL,ONEHOT_MOE_PERMUTE_THRESHOLD,LIBTPU_INIT_ARGS,RAY_memory_monitor_refresh_ms,VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS,ENABLE_MULTI_NUMA,TPU_RAIDEN_DATA_NICS,FLOAT32_GATE_LOGITS,FLOAT32_LOGITS,NEW_MODEL_DESIGN,ATTN_BUCKETIZED_NUM_REQS,ATTN_CUSTOM_NUM_REQS_BUCKETS,VLLM_MOE_CHUNK_SIZE,SLICE_ROPE_CACHE,DP_SCHED_BATCH_PREFILL,WEIGHT_SYNC_PARALLEL_H2H"
export ORCHESTRATOR_EXTRA_ENV="${ORCHESTRATOR_EXTRA_ENV:-WEIGHT_SYNC_BIND_TIMEOUT_S=900 WEIGHT_SYNC_METADATA_TIMEOUT_S=900 WEIGHT_SYNC_H2D_TIMEOUT_S=3600 WEIGHT_SYNC_TRANSFER_TIMEOUT_S=7200 RAIDEN_PARALLELISM=16 TPU_RAIDEN_DATA_NICS=eth0 RAIDEN_BROADCAST_HOST_RATIO=${RAIDEN_BROADCAST_HOST_RATIO} RAIDEN_BROADCAST_PIPELINE_STAGES=${RAIDEN_BROADCAST_PIPELINE_STAGES}}"
export ROLLOUT_EXTRA_ENV="${ROLLOUT_EXTRA_ENV:-USE_BATCHED_RPA_LONG_CTX_KERNEL=${USE_BATCHED_RPA_LONG_CTX_KERNEL} ONEHOT_MOE_PERMUTE_THRESHOLD=${ONEHOT_MOE_PERMUTE_THRESHOLD} RAY_memory_monitor_refresh_ms=0 RAIDEN_TRANSPORT_COALESCE_WINDOW_BYTES=67108864 RAIDEN_WEIGHT_SYNC_PIPELINE_GROUP_SIZE=16 RAIDEN_PARALLELISM=16 VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 ENABLE_MULTI_NUMA=${ENABLE_MULTI_NUMA} TPU_RAIDEN_DATA_NICS=eth0 RAIDEN_BROADCAST_HOST_RATIO=${RAIDEN_BROADCAST_HOST_RATIO} RAIDEN_BROADCAST_PIPELINE_STAGES=${RAIDEN_BROADCAST_PIPELINE_STAGES}}"
export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:- --xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false --xla_tpu_check_legacy_constraints_in_reduce_scatter_legalizer=false}"

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

# Hyperparameters & DeepSWE Pipeline Configuration
export TRAIN_MICRO_BATCH_SIZE="${TRAIN_MICRO_BATCH_SIZE:-64}"
export RPC_TIMEOUT_S="${RPC_TIMEOUT_S:-10800}"
export REMAT_POLICY="${REMAT_POLICY:-custom}"
export MAXTEXT_EXTRA_FLAGS="${MAXTEXT_EXTRA_FLAGS:-use_gdn_kernel=true gdn_cp_mode=head gdn_chunk_size=64 \
decoder_layer_input=offload context=remat gdn=remat gdn_conv=remat gdn_states=remat \
megablox=true sparse_matmul=true use_tokamax_gmm=true use_gmm_v2=true \
use_gmm_v2_heuristic_tiling=true merge_gating_gmm=false \
use_ragged_sort=true use_custom_sort_vjp=false ragged_buffer_factor=2.0 \
use_tokamax_splash=true use_splash_scheduler=true \
sa_block_q=1024 sa_block_kv=4096 sa_block_kv_compute=512 \
sa_block_q_dkv=2048 sa_block_kv_dkv=2048 sa_block_kv_dkv_compute=512 \
sa_fuse_reciprocal=false sa_use_base2_exp=true dq_reduction_steps=3 \
context_parallel_strategy=ring context_parallel_load_balance=false allow_split_physical_axes=false \
context_parallel_attention_load_balance=true \
num_vocab_tiling=16 use_iota_embed=false mu_dtype=float32 grad_dtype=float32 \
checkpoint_storage_concurrent_gb=96 \
checkpoint_storage_use_ocdbt=false checkpoint_storage_use_zarr3=false \
packing=True optimizer_memory_host_offload=true}"
export DEBUG=${DEBUG:-0}

# Sandbox
export SANDBOX_NODE_SELECTOR_VAL="${SANDBOX_NODE_SELECTOR_VAL:-sandbox-cpu-pool}"
export IMAGE_REWRITE_PREFIX="${IMAGE_REWRITE_PREFIX:-europe-west4-docker.pkg.dev/cloud-tpu-multipod-dev/tunix/}"

source "${DIR}/mlperf_base.sh" "$@"
