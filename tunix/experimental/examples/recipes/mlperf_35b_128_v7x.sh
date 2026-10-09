#!/bin/bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Fill these before you run.
# k8s has a 63 char limit on total label name, so keep job_prefix unique to your job and short
export JOB_PREFIX="${JOB_PREFIX:-${USER}}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-${JOB_PREFIX}-mlperf-35b-v7x}"
export MAXTEXT_OUTPUT_DIR="${MAXTEXT_OUTPUT_DIR:-gs://atwigg-trellis-us-central1/maxtext/${JOB_PREFIX}}"
export TRAJECTORY_LOG_DIR="${TRAJECTORY_LOG_DIR:-gs://atwigg-trellis-us-central1/trajectories/${JOB_PREFIX}/logger}"
export TRAJECTORY_STORE_ROOT_DIR="${TRAJECTORY_STORE_ROOT_DIR:-${TRAJECTORY_STORE_ROOT:-gs://atwigg-trellis-us-central1/trajectories/${JOB_PREFIX}/store}}"

export REGION="us-central1"
export CLUSTER="bodaborg-tpu7x-gsc"
export K8S_NAMESPACE="priority-dev"
export USE_DYNAMIC_SLICING="true"
export TPU_RESERVATION="${TPU_RESERVATION:-ghostfish-pogoag4tylwed}"

# Container memory overrides for v7x
export PATHWAYS_PROXY_MEMORY_LIMIT="100G"
export USER_CONTAINER_MEMORY="48G"
export USER_CONTAINER_MEMORY_LIMIT="${USER_CONTAINER_MEMORY_LIMIT:-70G}"
export ENABLE_MULTI_NUMA="${ENABLE_MULTI_NUMA:-0}"
export TPU_RAIDEN_DATA_NICS="${TPU_RAIDEN_DATA_NICS:-eth0}"

# Model configuration
export MODEL_NAME="Qwen3.5-35B-A3B"
export MODEL_ID="Qwen/Qwen3.5-35B-A3B"
export TOKENIZER_PATH="Qwen/Qwen3.5-35B-A3B"
export MAXTEXT_MODEL_NAME="qwen3.5-35b-a3b"
export MAXTEXT_CKPT="${MAXTEXT_CKPT:-gs://hengtaoguo-maxtext-logs/checkpoints/qwen3.5-35b-a3b/scanned/2026-06-11-10-27/0/items}"

# Topologies (64 chips / 128 devices Trainer 4x4x4, 16x 4-chip Rollout slices on TPU7x dynamic slicing)
# use_gdn_kernel=true requires TRAINER_MESH_TP=1 (the GDN kernel shard_map does not shard over 'tensor').
export TRAINER_JOBSET_YAML="jobset.pathways.yaml"
export TRAINER_TPU_SLICE="tpu7x:4x4x4"
export TRAINER_MESH_FSDP=64
export TRAINER_MESH_TP=1
export TRAINER_MESH_EXPERT=1
export TRAINER_MESH_CONTEXT=2

export ROLLOUT_JOBSET_YAML="jobset.tpu.yaml"
export ROLLOUT_TPU_SLICE="tpu7x:2x2x1"
export ROLLOUT_MESH_EXPERT="${ROLLOUT_MESH_EXPERT:-8}"
export ROLLOUT_REPLICAS="${ROLLOUT_REPLICAS:-16}"
export RAIDEN_DEVICES_PER_HOST=8

# vLLM Rollout Configuration (from paste.googleplex.com/5903655694368768)
export VLLM_ADDITIONAL_CONFIG='{"sharding":{"sharding_strategy":{"expert_parallelism":8,"tensor_parallelism":1,"enable_dp_attention":true}},"custom_mamba_cache_multiplier":16,"maxtext_config":{"scan_layers":false,  "attention":"vllm_rpa","allow_split_physical_axes":true,"use_multimodal":false,"prefuse_moe_weights":true}}'

# Rollout Worker Environment Flags
export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:- --xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false --xla_tpu_check_legacy_constraints_in_reduce_scatter_legalizer=false --xla_tpu_dvfs_p_state=7}"

# Trainer TPU v7x / Pathways XLA & libtpu flags (scoped VMEM capped at 65472 KiB on v7x)
export TRAINER_LIBTPU_INIT_ARGS="${TRAINER_LIBTPU_INIT_ARGS:---DANGEROUS_tpu_runtime_abi_verification_disabled=true --xla_tpu_use_tc_device_shape_on_sc=true --xla_sc_disable_megacore_partitioning=true --xla_tpu_enable_offloading_gather_to_sparsecore=true --xla_tpu_enable_sparse_core_collective_offload_all_gather=true --xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true --xla_tpu_enable_sparse_core_reduce_scatter_v2=true --xla_tpu_use_single_sparse_core_for_all_gather_offload=false --xla_tpu_enable_concurrent_sparse_core_offloading=true --xla_tpu_aggressive_opt_barrier_removal=true --xla_tpu_scoped_vmem_limit_kib=65472 --xla_tpu_enable_sublane_major_scaling_bitcast_fusion=false --xla_tpu_dvfs_p_state=7}"
export TRAINER_EXTRA_ENV="${TRAINER_EXTRA_ENV:-ENABLE_MULTI_NUMA=${ENABLE_MULTI_NUMA} TPU_RAIDEN_DATA_NICS=${TPU_RAIDEN_DATA_NICS} LIBTPU_INIT_ARGS='${TRAINER_LIBTPU_INIT_ARGS}'}"
export PATHWAYS_WORKER_EXTRA_ENV="${PATHWAYS_WORKER_EXTRA_ENV:-LIBTPU_INIT_ARGS=${TRAINER_LIBTPU_INIT_ARGS} --megascale_port=-1 --xprof_compress_jftrace=true
SKIP_MEGASCALE_PJRT_CLIENT=true
ENABLE_MULTI_NUMA=${ENABLE_MULTI_NUMA}
TPU_RAIDEN_DATA_NICS=${TPU_RAIDEN_DATA_NICS}}"

_trainer_xla_flags=""
for _f in ${TRAINER_LIBTPU_INIT_ARGS}; do [[ "${_f}" == --xla_* ]] && _trainer_xla_flags+="${_f} "; done
export PATHWAYS_PROXY_EXTRA_ARGS="${PATHWAYS_PROXY_EXTRA_ARGS:-${_trainer_xla_flags% }}"

export TRAIN_MICRO_BATCH_SIZE="${TRAIN_MICRO_BATCH_SIZE:-64}"

# Qwen3.5 GDN & Tokamax Splash Attention numerical stability flags for TPU v7x
export MAXTEXT_EXTRA_FLAGS="${MAXTEXT_EXTRA_FLAGS:-use_gdn_kernel=true gdn_cp_mode=head gdn_chunk_size=64 \
megablox=true sparse_matmul=true use_tokamax_gmm=true use_gmm_v2=true \
use_gmm_v2_heuristic_tiling=true merge_gating_gmm=false \
use_custom_sort_vjp=false \
use_tokamax_splash=true use_splash_scheduler=true \
sa_block_q=1024 sa_block_kv=4096 sa_block_kv_compute=512 \
sa_block_q_dkv=2048 sa_block_kv_dkv=2048 sa_block_kv_dkv_compute=512 \
sa_fuse_reciprocal=false sa_use_base2_exp=true dq_reduction_steps=3 \
context_parallel_strategy=ring context_parallel_load_balance=false allow_split_physical_axes=false \
num_vocab_tiling=16 use_iota_embed=false mu_dtype=float32 grad_dtype=float32 \
checkpoint_storage_concurrent_gb=96 \
packing=True}"

# Sandbox
export SANDBOX_NODE_SELECTOR_VAL="${SANDBOX_NODE_SELECTOR_VAL:-sandbox-c3d-np}"
export IMAGE_REWRITE_PREFIX="${IMAGE_REWRITE_PREFIX:-us-central1-docker.pkg.dev/cloud-tpu-multipod-dev/tunix/}"

source "${DIR}/mlperf_base.sh" "$@"

