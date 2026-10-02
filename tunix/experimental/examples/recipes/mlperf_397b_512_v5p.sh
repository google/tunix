#!/bin/bash
set -e

# ==============================================================================
# MLPerf DeepSWE recipe: Qwen3.5-397B-A17B on TPU v5p
# ==============================================================================
# - Cluster: bodaborg-v5p-nap in europe-west4
# - Trainer on 256 chips (tpuv5p:4x8x8, FSDP=16, TP=1, EXPERT=2, CONTEXT=8)
# - Rollout on 256 chips (16 replicas x 16 chips tpuv5p:2x2x4, EP=16, TP=1)
# - Sandboxes on sandbox-cpu-pool
# ==============================================================================

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Fill these before you run.
export JOB_PREFIX="${JOB_PREFIX:-$USER}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-${JOB_PREFIX}-mlperf-397b}"
export MAXTEXT_OUTPUT_DIR="${MAXTEXT_OUTPUT_DIR:-gs://atwigg-trellis-europe-west4-dev/maxtext/${JOB_PREFIX}}"
export TRAJECTORY_LOG_DIR="${TRAJECTORY_LOG_DIR:-gs://atwigg-trellis-europe-west4-dev/trajectories/${JOB_PREFIX}/logger}"
export TRAJECTORY_STORE_ROOT_DIR="${TRAJECTORY_STORE_ROOT_DIR:-${TRAJECTORY_STORE_ROOT:-gs://atwigg-trellis-europe-west4-dev/trajectories/${JOB_PREFIX}/store}}"

export REGION="europe-west4"
export CLUSTER="bodaborg-v5p-nap"
export K8S_NAMESPACE="trellis"

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
export ROLLOUT_WORKERS="${ROLLOUT_WORKERS:-16}"
export ROLLOUT_REPLICAS="${ROLLOUT_REPLICAS:-16}"

# vLLM Rollout Configuration
export VLLM_ADDITIONAL_CONFIG='{"sharding":{"sharding_strategy":{"expert_parallelism":16,"tensor_parallelism":1,"enable_dp_attention":true}},"custom_mamba_cache_multiplier":16,"maxtext_config":{"scan_layers":false,"attention":"vllm_rpa","allow_split_physical_axes":true,"use_multimodal":false,"prefuse_moe_weights":true,"per_device_batch_size":0.0}}'

# Rollout Worker Flags & Raiden tuning
export ONEHOT_MOE_PERMUTE_THRESHOLD=131072
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
export VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY="RAIDEN_,TPU_"
export VLLM_RAY_EXTRA_ENV_VARS_TO_COPY="ONEHOT_MOE_PERMUTE_THRESHOLD,LIBTPU_INIT_ARGS,RAY_memory_monitor_refresh_ms,VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS,ENABLE_MULTI_NUMA,TPU_RAIDEN_DATA_NICS,FLOAT32_GATE_LOGITS,FLOAT32_LOGITS,NEW_MODEL_DESIGN,ATTN_BUCKETIZED_NUM_REQS,ATTN_CUSTOM_NUM_REQS_BUCKETS,VLLM_MOE_CHUNK_SIZE,SLICE_ROPE_CACHE,DP_SCHED_BATCH_PREFILL"
export ORCHESTRATOR_EXTRA_ENV="${ORCHESTRATOR_EXTRA_ENV:-WEIGHT_SYNC_BIND_TIMEOUT_S=900 WEIGHT_SYNC_METADATA_TIMEOUT_S=900 WEIGHT_SYNC_H2D_TIMEOUT_S=3600 WEIGHT_SYNC_TRANSFER_TIMEOUT_S=7200 RAIDEN_PARALLELISM=16}"
export ROLLOUT_EXTRA_ENV="${ROLLOUT_EXTRA_ENV:-ONEHOT_MOE_PERMUTE_THRESHOLD=131072 RAY_memory_monitor_refresh_ms=0 RAIDEN_TRANSPORT_COALESCE_WINDOW_BYTES=67108864 RAIDEN_WEIGHT_SYNC_PIPELINE_GROUP_SIZE=16 RAIDEN_PARALLELISM=16 VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800}"
export TRAINER_LIBTPU_INIT_ARGS="${TRAINER_LIBTPU_INIT_ARGS:---DANGEROUS_tpu_runtime_abi_verification_disabled=true --xla_tpu_use_tc_device_shape_on_sc=true --xla_sc_disable_megacore_partitioning=true --xla_tpu_enable_offloading_gather_to_sparsecore=true --xla_tpu_enable_sparse_core_collective_offload_all_gather=true --xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true --xla_tpu_enable_sparse_core_reduce_scatter_v2=true --xla_tpu_use_single_sparse_core_for_all_gather_offload=true --xla_tpu_enable_concurrent_sparse_core_offloading=true --xla_tpu_aggressive_opt_barrier_removal=true --xla_tpu_scoped_vmem_limit_kib=65536 --xla_tpu_enable_sublane_major_scaling_bitcast_fusion=false}"
export TRAINER_EXTRA_ENV="${TRAINER_EXTRA_ENV:-ONEHOT_MOE_PERMUTE_THRESHOLD=131072 RAIDEN_TRANSPORT_COALESCE_WINDOW_BYTES=67108864 RAIDEN_WEIGHT_SYNC_PIPELINE_GROUP_SIZE=16 LIBTPU_INIT_ARGS='${TRAINER_LIBTPU_INIT_ARGS}'}"
export PATHWAYS_WORKER_EXTRA_ENV="${PATHWAYS_WORKER_EXTRA_ENV:-LIBTPU_INIT_ARGS=${TRAINER_LIBTPU_INIT_ARGS} --megascale_port=-1 --xprof_compress_jftrace=true
SKIP_MEGASCALE_PJRT_CLIENT=true}"

_trainer_xla_flags=""
for _f in ${TRAINER_LIBTPU_INIT_ARGS}; do [[ "${_f}" == --xla_* ]] && _trainer_xla_flags+="${_f} "; done
export PATHWAYS_PROXY_EXTRA_ARGS="${PATHWAYS_PROXY_EXTRA_ARGS:-${_trainer_xla_flags% }}"

# Hyperparameters & DeepSWE Pipeline Configuration
export TRAIN_MICRO_BATCH_SIZE="${TRAIN_MICRO_BATCH_SIZE:-64}"
export RPC_TIMEOUT_S="${RPC_TIMEOUT_S:-10800}"
export REMAT_POLICY="${REMAT_POLICY:-custom}"
export MAXTEXT_EXTRA_FLAGS="${MAXTEXT_EXTRA_FLAGS:-custom_mesh_and_rule=cp-as-ep \
use_gdn_kernel=true gdn_cp_mode=head gdn_chunk_size=64 \
decoder_layer_input=offload context=remat gdn=remat gdn_conv=remat gdn_states=remat \
megablox=true sparse_matmul=true use_tokamax_gmm=true use_gmm_v2=true \
use_gmm_v2_heuristic_tiling=true merge_gating_gmm=false \
use_ring_of_experts=true num_moe_token_chunks=4 moe_chunk_barrier=false \
use_ragged_sort=true use_custom_sort_vjp=false ragged_buffer_factor=2.0 \
use_tokamax_splash=true use_splash_scheduler=true \
sa_block_q=1024 sa_block_kv=4096 sa_block_kv_compute=512 \
sa_block_q_dkv=2048 sa_block_kv_dkv=2048 sa_block_kv_dkv_compute=512 \
sa_fuse_reciprocal=false sa_use_base2_exp=true dq_reduction_steps=3 \
context_parallel_strategy=ring context_parallel_load_balance=false allow_split_physical_axes=true \
num_vocab_tiling=16 use_iota_embed=false mu_dtype=float32 grad_dtype=float32 \
checkpoint_storage_concurrent_gb=96 \
packing=True optimizer_memory_host_offload=true}"
export DEBUG=${DEBUG:-0}

# Sandbox
export SANDBOX_NODE_SELECTOR_VAL="sandbox-cpu-pool"
export IMAGE_REWRITE_PREFIX="${IMAGE_REWRITE_PREFIX:-europe-west4-docker.pkg.dev/cloud-tpu-multipod-dev/tunix/}"

source "${DIR}/mlperf_base.sh" "$@"
