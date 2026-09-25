#!/bin/bash
set -e

# ==============================================================================
# MLPerf DeepSWE recipe: Qwen3.5-397B-A17B on TPU v7x (bodaborg-tpu7x-gsc)
# ==============================================================================
# - TPU7x dynamic slicing on bodaborg-tpu7x-gsc (priority-dev namespace)
# - Trainer on 128 chips (4x4x8 = 256 devices, FSDP=32, TP=1, EXPERT=2, CP=4)
# - Rollout on 128 chips (16 replicas x 8 chips 2x2x2, EP=16, TP=1)
# - Sandbox configured for sandbox-np nodepool with workload tolerations
# ==============================================================================

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export REGION="${REGION:-us-central1}"

# Fill these before you run.
export JOB_PREFIX="${JOB_PREFIX:-$USER}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-${JOB_PREFIX}-mlperf-397b-v7x}"

if [[ "${REGION}" == us-east1* ]]; then
  export BUCKET="${BUCKET:-gs://atwigg-trellis-us-east1}"
  export CLUSTER="${CLUSTER:-bodaborg-tpu7x-gsc-elm}"
  export TPU_RESERVATION="${TPU_RESERVATION:-ghostfish-ev7rs12wndvw5}"
  export MAXTEXT_CKPT="${MAXTEXT_CKPT:-gs://mlperf-6-submission-us-east1/ckpt/qwen35_397b/scanned_reshard_fsdp32_tp2/0/items}"
else
  export BUCKET="${BUCKET:-gs://atwigg-trellis-us-central1}"
  export CLUSTER="${CLUSTER:-bodaborg-tpu7x-gsc}"
  export TPU_RESERVATION="${TPU_RESERVATION:-ghostfish-pogoag4tylwed}"
  export MAXTEXT_CKPT="${MAXTEXT_CKPT:-gs://mlperf-6-1-submission/ckpt/qwen35_397b/scanned_reshard_fsdp32_tp2/0/items}"
fi

export MAXTEXT_OUTPUT_DIR="${MAXTEXT_OUTPUT_DIR:-${BUCKET}/maxtext/${JOB_PREFIX}}"
export TRAJECTORY_LOG_DIR="${TRAJECTORY_LOG_DIR:-${BUCKET}/trajectories/${JOB_PREFIX}/logger}"
export TRAJECTORY_STORE_ROOT_DIR="${TRAJECTORY_STORE_ROOT_DIR:-${TRAJECTORY_STORE_ROOT:-${BUCKET}/trajectories/${JOB_PREFIX}/store}}"

export K8S_NAMESPACE="priority-dev"
export USE_DYNAMIC_SLICING="true"
# Raiden weight sync: working 397B runs use ENABLE_MULTI_NUMA=0 and the default
# RAIDEN_BROADCAST_K (64 -> every slice pushed direct from the trainer). K=3 routes
# slices through the receiver relay tree, which fails with "Incoming push size
# mismatch" on the first sync; MULTI_NUMA=1 doubles the listeners per rollout worker.
export ENABLE_MULTI_NUMA="${ENABLE_MULTI_NUMA:-0}"

# Head pod lands on cpu-np (~257G allocatable). mlperf_pathways_config.sh's
# 260G user-container request plus proxy/rm requests (~280G) never schedules
# there. Only the request matters for placement; the limits stay as configured.
export USER_CONTAINER_MEMORY="${USER_CONTAINER_MEMORY:-48G}"

export RAIDEN_DEVICES_PER_HOST=8
export TPU_RAIDEN_DATA_NICS="eth0"
export RAIDEN_BROADCAST_K=64

# Model configuration
export MODEL_NAME="Qwen3.5-397B-A17B"
export MODEL_ID="Qwen/Qwen3.5-397B-A17B"
export TOKENIZER_PATH="${TOKENIZER_PATH:-/app/Qwen/Qwen3.5-397B-A17B}"
export MAXTEXT_MODEL_NAME="qwen3.5-397b-a17b"

# Topologies (128 chips / 256 devices Trainer 4x4x8, 16x 8-chip Rollout slices on TPU7x dynamic slicing)
# Mesh product is DEVICES, and v7x has 2 devices/chip at ~95 GB each.
# 4x4x8 = 128 chips = 256 devices.
# Sharding: TP=1, FSDP=32, CP=4, EP=2, cp-as-ep.
# Mesh product: 32 * 1 * 2 * 4 = 256 devices.
export TRAINER_JOBSET_YAML="jobset.pathways.yaml"
export TRAINER_TPU_SLICE="tpu7x:4x4x8"
export TRAINER_MESH_FSDP=32
export TRAINER_MESH_TP=1
export TRAINER_MESH_EXPERT=2
export TRAINER_MESH_CONTEXT=4

# Rollout: 8 chips = 16 devices = 2 hosts per replica. tp * expert must equal the
# DEVICE count, so expert=16 on 8 chips.
export ROLLOUT_JOBSET_YAML="jobset.mcjax.ray.yaml"
export ROLLOUT_TPU_SLICE="tpu7x:2x2x2"
export ROLLOUT_MESH_EXPERT=16
export ROLLOUT_REPLICAS="${ROLLOUT_REPLICAS:-16}"

# vLLM Rollout Configuration
export VLLM_ADDITIONAL_CONFIG='{"sharding":{"sharding_strategy":{"expert_parallelism":16,"tensor_parallelism":1,"enable_dp_attention":true}},"custom_mamba_cache_multiplier":16,"maxtext_config":{"scan_layers":false,"attention":"vllm_rpa","allow_split_physical_axes":true,"use_multimodal":false,"prefuse_moe_weights":true,"per_device_batch_size":0.0}}'

# Rollout Worker Flags & Raiden tuning
export ONEHOT_MOE_PERMUTE_THRESHOLD=131072
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
export VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY="RAIDEN_,TPU_"
export VLLM_RAY_EXTRA_ENV_VARS_TO_COPY="ONEHOT_MOE_PERMUTE_THRESHOLD,LIBTPU_INIT_ARGS,RAY_memory_monitor_refresh_ms,VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS,ENABLE_MULTI_NUMA,TPU_RAIDEN_DATA_NICS"
export ORCHESTRATOR_EXTRA_ENV="${ORCHESTRATOR_EXTRA_ENV:-WEIGHT_SYNC_H2D_TIMEOUT_S=3600 WEIGHT_SYNC_TRANSFER_TIMEOUT_S=7200 RAIDEN_PARALLELISM=16 TPU_RAIDEN_DATA_NICS=eth0 RAIDEN_BROADCAST_K=64}"
export ROLLOUT_EXTRA_ENV="${ROLLOUT_EXTRA_ENV:-ONEHOT_MOE_PERMUTE_THRESHOLD=131072 RAY_memory_monitor_refresh_ms=0 RAIDEN_TRANSPORT_COALESCE_WINDOW_BYTES=67108864 RAIDEN_WEIGHT_SYNC_PIPELINE_GROUP_SIZE=16 RAIDEN_PARALLELISM=16 VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 ENABLE_MULTI_NUMA=${ENABLE_MULTI_NUMA} TPU_RAIDEN_DATA_NICS=eth0 RAIDEN_BROADCAST_K=64}"
# Rollout (vLLM) libtpu flags; mlperf_base.sh only applies its default when unset.
export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:- --xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false --xla_tpu_check_legacy_constraints_in_reduce_scatter_legalizer=false --xla_tpu_dvfs_p_state=7}"
# v7x caps scoped VMEM at 67043328 bytes (65472 KiB); 65536 is rejected per compile
# (INVALID_ARGUMENT in pathways-rm) and the compiler falls back to its default.
export TRAINER_LIBTPU_INIT_ARGS="${TRAINER_LIBTPU_INIT_ARGS:---DANGEROUS_tpu_runtime_abi_verification_disabled=true --xla_tpu_use_tc_device_shape_on_sc=true --xla_sc_disable_megacore_partitioning=true --xla_tpu_enable_offloading_gather_to_sparsecore=true --xla_tpu_enable_sparse_core_collective_offload_all_gather=true --xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true --xla_tpu_enable_sparse_core_reduce_scatter_v2=true --xla_tpu_use_single_sparse_core_for_all_gather_offload=false --xla_tpu_enable_concurrent_sparse_core_offloading=true --xla_tpu_aggressive_opt_barrier_removal=true --xla_tpu_scoped_vmem_limit_kib=65472 --xla_tpu_enable_sublane_major_scaling_bitcast_fusion=false}"
export TRAINER_EXTRA_ENV="${TRAINER_EXTRA_ENV:-ONEHOT_MOE_PERMUTE_THRESHOLD=131072 RAIDEN_TRANSPORT_COALESCE_WINDOW_BYTES=67108864 RAIDEN_WEIGHT_SYNC_PIPELINE_GROUP_SIZE=16 ENABLE_MULTI_NUMA=${ENABLE_MULTI_NUMA} TPU_RAIDEN_DATA_NICS=eth0 RAIDEN_BROADCAST_K=64 LIBTPU_INIT_ARGS='${TRAINER_LIBTPU_INIT_ARGS}'}"
# Under Pathways the trainer's TPU program runs in the pathways-worker container,
# so the trainer libtpu flags must be set there (yaml_generator.py renders one
# KEY=VALUE per line into the worker env).
export PATHWAYS_WORKER_EXTRA_ENV="${PATHWAYS_WORKER_EXTRA_ENV:-LIBTPU_INIT_ARGS=${TRAINER_LIBTPU_INIT_ARGS} --megascale_port=-1 --xprof_compress_jftrace=true
SKIP_MEGASCALE_PJRT_CLIENT=true
TPU_RAIDEN_DATA_NICS=eth0
RAIDEN_BROADCAST_K=64}"

# XLA compiler flags must also reach the pathways-proxy: it is what compiles the
# trainer program, and without them it uses the 32 MiB default scoped VMEM limit
# (splash-attention dkv needs ~37 MiB at these block sizes -> CompileTimeScopedVmemOom).
_trainer_xla_flags=""
for _f in ${TRAINER_LIBTPU_INIT_ARGS}; do [[ "${_f}" == --xla_* ]] && _trainer_xla_flags+="${_f} "; done
export PATHWAYS_PROXY_EXTRA_ARGS="${PATHWAYS_PROXY_EXTRA_ARGS:-${_trainer_xla_flags% }}"

# Hyperparameters & DeepSWE Pipeline Configuration
# Must be a multiple of TRAINER_MESH_FSDP=32. Only the padded batch assembler reads it;
# sequence packing (MAX_SEQ_TOKEN_PER_TPU set) sizes micro steps from the trainer mesh.
export TRAIN_MICRO_BATCH_SIZE="${TRAIN_MICRO_BATCH_SIZE:-64}"

export RPC_TIMEOUT_S="${RPC_TIMEOUT_S:-10800}"
# Architecture & Rematerialization: custom remat with per-tensor policies below
# (mlperf_base.sh defaults REMAT_POLICY to full).
export REMAT_POLICY="${REMAT_POLICY:-custom}"
export MAXTEXT_EXTRA_FLAGS="${MAXTEXT_EXTRA_FLAGS:-custom_mesh_and_rule=cp-as-ep \
use_gdn_kernel=true gdn_cp_mode=head \
decoder_layer_input=device context=remat gdn=remat gdn_conv=remat gdn_states=remat \
megablox=true sparse_matmul=true use_tokamax_gmm=true use_gmm_v2=true \
use_gmm_v2_heuristic_tiling=true merge_gating_gmm=false \
use_ring_of_experts=true num_moe_token_chunks=4 \
use_ragged_sort=true use_custom_sort_vjp=false ragged_buffer_factor=2.0 \
use_tokamax_splash=true use_splash_scheduler=true \
sa_block_q=512 sa_block_kv=1024 sa_block_kv_compute=512 \
sa_block_q_dkv=1024 sa_block_kv_dkv=2048 sa_block_kv_dkv_compute=1024 \
sa_fuse_reciprocal=false sa_use_base2_exp=true dq_reduction_steps=3 \
context_parallel_load_balance=false allow_split_physical_axes=false \
num_vocab_tiling=16 mu_dtype=bfloat16 \
checkpoint_storage_concurrent_gb=96 \
checkpoint_storage_use_ocdbt=false checkpoint_storage_use_zarr3=false}"
export DEBUG=${DEBUG:-0}

# DeepSWE Environment & Agent Sandbox
export SANDBOX_TOLERATIONS='[{"key":"workload","operator":"Equal","value":"sandbox","effect":"NoSchedule"}]'
export IMAGE_REWRITE_PREFIX="${IMAGE_REWRITE_PREFIX:-us-central1-docker.pkg.dev/cloud-tpu-multipod-dev/tunix/}"

source "${DIR}/mlperf_base.sh" "$@"
