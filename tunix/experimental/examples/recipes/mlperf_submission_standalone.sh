#!/bin/bash
set -e

# ==============================================================================
# Self-Contained MLPerf DeepSWE Recipe: Qwen3.5-397B-A17B on TPU v7x (GBS=1152)
# ==============================================================================
# Standalone configuration with fixed submission arguments (run "at-1015c-s4").
#
# Only SEED, JOB_PREFIX, K8S_NAMESPACE, WANDB_API_KEY, and TUNIX_IMAGE can be
# overridden via environment variables:
#   SEED=... JOB_PREFIX=... K8S_NAMESPACE=... WANDB_API_KEY=... \
#     bash tunix/experimental/examples/recipes/mlperf_submission_standalone.sh start
# ==============================================================================

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ==============================================================================
# 1. Overridable Arguments
# ==============================================================================
export SEED="${SEED:-42}"
export JOB_PREFIX="${JOB_PREFIX:-${USER}}"
export K8S_NAMESPACE="${K8S_NAMESPACE:-priority-dev}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export TUNIX_IMAGE="${TUNIX_IMAGE:-gcr.io/cloud-tpu-multipod-dev/atwigg/trellis-experimental:1015c}"

# ==============================================================================
# 2. Cluster, Infrastructure & Storage Configuration
# ==============================================================================
export PROJECT="cloud-tpu-shared-capacity"
export REGION="us-central1"
export CLUSTER="bodaborg-tpu7x-gsc"
export BUCKET="gs://atwigg-trellis-us-central1-fast-dev"
export TPU_RESERVATION="ghostfish-tb2qe8ysltros"
export KUEUE_QUEUE="multislice-queue"
export PREEMPTIBLE="false"
export GANG_ID="${JOB_PREFIX}"
export PRIORITY_CLASS="medium"
export KUEUE_PRIORITY_CLASS="medium"
export SERVICE_ACCOUNT="xpk-sa"
export CPU_MACHINE="n2d-standard-64"
export CPU_NODEPOOL="cpu-np"
export CPU_MEMORY="220Gi"
export USER_CONTAINER_MEMORY="160G"
export USER_CONTAINER_MEMORY_LIMIT="260G"

export ORCHESTRATOR_PORT="20000"
export ROLLOUT_PORT="20001"
export TRAINER_PORT="20002"
export RPC_TIMEOUT_S="10800"

export FAIL_FAST="true"
export FT_STARTUP_RETRIES="3"
export FT_SANDBOX_READY_TIMEOUT_S="600"
export FT_SANDBOX_ACQUIRE_RETRIES="2"

kubectl config use-context "gke_${PROJECT}_${REGION}_${CLUSTER}" || true
if [[ -n "${K8S_NAMESPACE}" ]]; then
  kubectl config set-context --current --namespace="${K8S_NAMESPACE}" || true
fi

# ==============================================================================
# 3. Checkpointing, Logging & Telemetry
# ==============================================================================
export MAXTEXT_CKPT="gs://mlperf-6-1-submission/ckpt/qwen35_397b/scanned_reshard_fsdp32_tp2/0/items"
export MAXTEXT_OUTPUT_DIR="${BUCKET}/maxtext/${JOB_PREFIX}"
export TRAJECTORY_LOG_DIR="${BUCKET}/trajectories/${JOB_PREFIX}/logger"
export DISABLE_TRAJECTORY_STORE="true"

export PATHWAYS_CHECKPOINTING_IMPL="persistence"
export ENABLE_PATHWAYS_PERSISTENCE="1"
export CKPT_USE_OCDBT="0"
export CHECKPOINT_ASYNC="true"
export CHECKPOINT_SAVE_INTERVAL_STEPS="1"
export CHECKPOINT_MAX_TO_KEEP="30"
export CHECKPOINT_ABANDON_FAILED_SAVES="true"
export CHECKPOINT_OPTIMIZER_INTERVAL_STEPS="0"
export CKPT_D2H_CONCURRENT_GB="32"

export WANDB_ENTITY="google-trellis"
export WANDB_PROJECT="trellis-deepswe"
export WANDB_RUN_NAME="${JOB_PREFIX}-mlperf-397b-v7x"

export RCP_LOGGING="true"
export METRIC_LOGGER_DIR="${MAXTEXT_OUTPUT_DIR}/mllog"
export TARGET_ACCURACY="0.69"

export DEBUG="0"
export TUNIX_DEBUG_INFERENCE_LOGS="false"

# ==============================================================================
# 4. Model, Topologies & Weight Sync (Pathways + Raiden)
# ==============================================================================
export MODEL_NAME="Qwen3.5-397B-A17B"
export MODEL_ID="Qwen/Qwen3.5-397B-A17B"
export TOKENIZER_PATH="Qwen/Qwen3.5-397B-A17B"
export MAXTEXT_MODEL_NAME="qwen3.5-397b-a17b"
export TRAINER_BACKEND="maxtext"
export SAMPLER="vllm"
export TRAINABLE_PARAMETERS_MASK='^(?!.*routed_experts/gate/kernel).*'
export EOS_TOKENS="248046,248044"
export TRAINER_BASE_NUM_KV_HEADS="2"
export USE_DYNAMIC_SLICING="true"

# Trainer topology (128 chips / 256 devices: 4x4x8, FSDP=32, TP=1, EXPERT=2, CONTEXT=4)
export TRAINER_JOBSET_YAML="jobset.pathways.yaml"
export TRAINER_TPU_SLICE="tpu7x:4x4x8"
export TRAINER_MESH_FSDP="32"
export TRAINER_MESH_TP="1"
export TRAINER_MESH_EXPERT="2"
export TRAINER_MESH_CONTEXT="4"

# Rollout topology (128 replicas x 2x2x1 = 4 chips = 8 devices per replica, DP=1, EP=8, TP=1)
export ROLLOUT_JOBSET_YAML="jobset.tpu.yaml"
export ROLLOUT_TPU_SLICE="tpu7x:2x2x1"
export ROLLOUT_REPLICAS="128"
export VLLM_DATA_PARALLEL_SIZE="1"
export ROLLOUT_MESH_EXPERT="8"
export ROLLOUT_MESH_FSDP="1"
export ROLLOUT_MESH_TP="1"

# Pathways & Raiden Weight Sync
export PATHWAYS_SERVER_IMAGE="us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/datenglin/sanitized_server:raiden_20261013212200"
export PATHWAYS_PROXY_IMAGE="us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/datenglin/sanitized_proxy_server:raiden_20261013212200"
export PATHWAYS_PROXY_MEMORY_LIMIT="160G"

export WEIGHT_SYNC_MODE="raiden"
export ASYNC_WEIGHT_SYNC="true"
export IN_FLIGHT_WEIGHT_UPDATES="true"
export WEIGHT_SYNC_DISABLE_TIMEOUTS="1"
export WEIGHT_SYNC_PARALLEL_H2H="1"
export VERIFY_WEIGHTS="false"
export USE_WEIGHT_CONVERTER="true"
export TRAINER_PREFUSE_MOE_WEIGHTS="true"
export ROLLOUT_PREFUSE_MOE_WEIGHTS="true"

export ENABLE_MULTI_NUMA="0"
export TPU_RAIDEN_DATA_NICS="eth0"
export RAIDEN_DEVICES_PER_HOST="8"
export RAIDEN_FFI_USE_DIRECT_DEVICE_BUFFER="0"

# ==============================================================================
# 5. Trainer Flags & Environment
# ==============================================================================
export TRAINER_FP8="true"
export REMAT_POLICY="custom"
export TRAINER_MAXTEXT_ATTENTION="flash"
export COMPUTE_LOGPS_CHUNK_SIZE="1024"
export TUNIX_LOGPS_SHARDED_HEAD="local"
export TRAINER_WARMUP_COMPILE="true"

_TRAINER_LIBTPU_INIT_ARGS="--DANGEROUS_tpu_runtime_abi_verification_disabled=true --xla_tpu_use_tc_device_shape_on_sc=true --xla_sc_disable_megacore_partitioning=true --xla_tpu_enable_offloading_gather_to_sparsecore=true --xla_tpu_enable_sparse_core_collective_offload_all_gather=true --xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true --xla_tpu_enable_sparse_core_reduce_scatter_v2=true --xla_tpu_use_single_sparse_core_for_all_gather_offload=false --xla_tpu_enable_concurrent_sparse_core_offloading=true --xla_tpu_aggressive_opt_barrier_removal=false --xla_tpu_scoped_vmem_limit_kib=65472 --xla_tpu_enable_sublane_major_scaling_bitcast_fusion=false --xla_tpu_use_single_sparse_core_for_reduce_scatter_offload=false --xla_tpu_dvfs_p_state=7 --xla_tpu_max_hbm_size_mib=84992 --xla_tpu_enable_offloading_copy_to_sparsecore=false"

export PATHWAYS_PROXY_EXTRA_ARGS="--xla_tpu_use_tc_device_shape_on_sc=true --xla_sc_disable_megacore_partitioning=true --xla_tpu_enable_offloading_gather_to_sparsecore=true --xla_tpu_enable_sparse_core_collective_offload_all_gather=true --xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true --xla_tpu_enable_sparse_core_reduce_scatter_v2=true --xla_tpu_use_single_sparse_core_for_all_gather_offload=false --xla_tpu_enable_concurrent_sparse_core_offloading=true --xla_tpu_aggressive_opt_barrier_removal=false --xla_tpu_scoped_vmem_limit_kib=65472 --xla_tpu_enable_sublane_major_scaling_bitcast_fusion=false --xla_tpu_use_single_sparse_core_for_reduce_scatter_offload=false --xla_tpu_dvfs_p_state=7 --xla_tpu_max_hbm_size_mib=84992 --xla_tpu_enable_offloading_copy_to_sparsecore=false"

export TRAINER_EXTRA_ENV="ONEHOT_MOE_PERMUTE_THRESHOLD=2048 ENABLE_MULTI_NUMA=0 TPU_RAIDEN_DATA_NICS=eth0 RAIDEN_BROADCAST_HOST_RATIO=1.0 RAIDEN_BROADCAST_PIPELINE_STAGES=4 LIBTPU_INIT_ARGS='${_TRAINER_LIBTPU_INIT_ARGS}' TPU_RAIDEN_TCP_CONNECT_TIMEOUT_MS=120000 TPU_RAIDEN_TCP_CONNECT_MAX_ATTEMPTS=5 RAIDEN_PARALLELISM=16 RAIDEN_TRANSPORT_COALESCE_WINDOW_BYTES=67108864 RAIDEN_WEIGHT_SYNC_PIPELINE_GROUP_SIZE=16"

export PATHWAYS_WORKER_EXTRA_ENV="LIBTPU_INIT_ARGS=${_TRAINER_LIBTPU_INIT_ARGS} --megascale_port=-1 --xprof_compress_jftrace=true
SKIP_MEGASCALE_PJRT_CLIENT=true
RAIDEN_BROADCAST_HOST_RATIO=1.0
RAIDEN_BROADCAST_PIPELINE_STAGES=4
TPU_RAIDEN_TCP_CONNECT_TIMEOUT_MS=120000
TPU_RAIDEN_TCP_CONNECT_MAX_ATTEMPTS=5
RAIDEN_TRANSPORT_COALESCE_WINDOW_BYTES=67108864
RAIDEN_WEIGHT_SYNC_PIPELINE_GROUP_SIZE=16"
unset _TRAINER_LIBTPU_INIT_ARGS

export ORCHESTRATOR_EXTRA_ENV="WEIGHT_SYNC_BIND_TIMEOUT_S=900 WEIGHT_SYNC_METADATA_TIMEOUT_S=900 WEIGHT_SYNC_H2D_TIMEOUT_S=3600 WEIGHT_SYNC_TRANSFER_TIMEOUT_S=7200 RAIDEN_PARALLELISM=16 TPU_RAIDEN_DATA_NICS=eth0 RAIDEN_BROADCAST_HOST_RATIO=1.0 RAIDEN_BROADCAST_PIPELINE_STAGES=4 TPU_RAIDEN_TCP_CONNECT_TIMEOUT_MS=120000 TPU_RAIDEN_TCP_CONNECT_MAX_ATTEMPTS=5"

export MAXTEXT_EXTRA_FLAGS="custom_mesh_and_rule=cp-as-ep \
use_gdn_kernel=true gdn_cp_mode=seq gdn_cp_state=device gdn_cp_matmul_precision=high gdn_chunk_size=64 \
decoder_layer_input=offload context=device gdn=remat gdn_conv=remat gdn_states=remat \
megablox=true sparse_matmul=true use_tokamax_gmm=true use_gmm_v2=true \
use_gmm_v2_heuristic_tiling=false merge_gating_gmm=false \
wi_tile_fwd_batch_seq=256 wi_tile_fwd_embed_dim=4096 wi_tile_fwd_mlp_dim=1024 \
wi_tile_dlhs_batch_seq=256 wi_tile_dlhs_embed_dim=4096 wi_tile_dlhs_mlp_dim=1024 \
wi_tile_drhs_batch_seq=256 wi_tile_drhs_embed_dim=4096 wi_tile_drhs_mlp_dim=1024 \
wo_tile_fwd_batch_seq=256 wo_tile_fwd_embed_dim=4096 wo_tile_fwd_mlp_dim=1024 \
wo_tile_dlhs_batch_seq=256 wo_tile_dlhs_embed_dim=4096 wo_tile_dlhs_mlp_dim=1024 \
wo_tile_drhs_batch_seq=256 wo_tile_drhs_embed_dim=4096 wo_tile_drhs_mlp_dim=1024 \
use_ring_of_experts=true num_moe_token_chunks=2 moe_chunk_barrier=false \
ring_of_experts_local_routing=true moe_expert_weight_prefetch=forward \
moe_chunk_pipeline=true ring_of_experts_row_major_reduce_scatter=true \
use_ragged_sort=true use_custom_sort_vjp=false ragged_buffer_factor=2.75 \
retry_when_tokens_dropped=true retry_num_moe_token_chunks=8 \
moe_exclude_padding_from_routing=true log_moe_drops=true \
use_tokamax_splash=true use_splash_scheduler=true \
sa_block_q=1024 sa_block_kv=4096 sa_block_kv_compute=512 \
sa_block_q_dkv=2048 sa_block_kv_dkv=2048 sa_block_kv_dkv_compute=512 \
sa_fuse_reciprocal=true sa_use_base2_exp=false dq_reduction_steps=3 \
context_parallel_strategy=all_gather context_parallel_load_balance=false allow_split_physical_axes=false \
context_parallel_attention_load_balance=true \
num_vocab_tiling=16 use_iota_embed=false mu_dtype=float32 grad_dtype=float32 cast_grads_after_all_reduce=true \
checkpoint_storage_concurrent_gb=96 \
packing=True optimizer_memory_host_offload=true \
kv_tp_size=8 rollout_fp8_moe=true fp8_moe_fake_quant=true"

# ==============================================================================
# 6. Rollout (vLLM) Flags, Environment & JAX Cache
# ==============================================================================
export ROLLOUT_FP8="true"
export ROLLOUT_MAXTEXT_ATTENTION="vllm_batched_rpa_long_ctx"

export VLLM_LOGGING_LEVEL="INFO"
export VLLM_MAX_MODEL_LEN="65536"
export VLLM_MAX_NUM_BATCHED_TOKENS="2048"
export VLLM_MAX_NUM_SEQS="64"
export VLLM_GPU_MEMORY_UTILIZATION="0.9"
export VLLM_ENABLE_EXPERT_PARALLEL="true"
export ENABLE_PREFIX_CACHING="true"
export VLLM_PREFIX_CACHE_RETENTION_INTERVAL="0"
export VLLM_MAMBA_CACHE_MODE="align"
export RETURN_ROUTED_EXPERTS="true"
export ROLLOUT_FREE_KV_CACHE="false"
export VLLM_KV_CACHE_DTYPE="bfloat16"
export VLLM_BLOCK_SIZE="256"
export VLLM_ASYNC_SCHEDULING="true"
export VLLM_ENABLE_CHUNKED_PREFILL="true"
export VLLM_LANGUAGE_MODEL_ONLY="true"
export VLLM_REASONING_PARSER="qwen3"
export VLLM_LIMIT_MM_PER_PROMPT='{"image": 0, "video": 0}'

export VLLM_ADDITIONAL_CONFIG='{"sharding":{"sharding_strategy":{"expert_parallelism":8,"tensor_parallelism":1,"enable_dp_attention":false}},"custom_mamba_cache_multiplier":16,"maxtext_config":{"scan_layers":false,"attention":"vllm_batched_rpa_long_ctx","allow_split_physical_axes":true,"use_multimodal":false,"prefuse_moe_weights":true,"per_device_batch_size":0.0,"fp8_moe":true}}'

export ONEHOT_MOE_PERMUTE_THRESHOLD="2048"
export ROLLOUT_SKIP_JAX_PRECOMPILE="0"
export NUM_PRECOMPILE_WORKERS="8"
export NEW_MODEL_DESIGN="1"
export ATTN_BUCKETIZED_NUM_REQS="true"
export ATTN_CUSTOM_NUM_REQS_BUCKETS="4"
export RAGGED_GATHER_REDUCE_VERSION="v3"
export VLLM_MOE_CHUNK_SIZE="256"
export SLICE_ROPE_CACHE="1"
export DP_SCHED_BATCH_PREFILL="false"
export VLLM_ENABLE_V1_MULTIPROCESSING="0"
export TPU_SMALL_ALL_REDUCE="1"
export TPU_MOE_SINGLE_SORT="1"
export TPU_MOE_ROUTER_TOPK="rowmax"
export TPU_MOE_LOWTOK="1"
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="1800"

export LIBTPU_INIT_ARGS=" --xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false --xla_tpu_check_legacy_constraints_in_reduce_scatter_legalizer=false --xla_tpu_dvfs_p_state=7"
export ROLLOUT_EXTRA_ENV="USE_BATCHED_RPA_LONG_CTX_KERNEL=true VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 RAIDEN_BROADCAST_HOST_RATIO=1.0 RAIDEN_BROADCAST_PIPELINE_STAGES=4 TPU_RAIDEN_TCP_CONNECT_TIMEOUT_MS=120000 TPU_RAIDEN_TCP_CONNECT_MAX_ATTEMPTS=5 RAIDEN_PARALLELISM=16 RAIDEN_TRANSPORT_COALESCE_WINDOW_BYTES=67108864 RAIDEN_WEIGHT_SYNC_PIPELINE_GROUP_SIZE=16"

# JAX Compilation Cache
export ROLLOUT_JAX_CACHE_GCS_DIR="${BUCKET}/jax_cache/v7x/qwen3.5-397b-a17b/rollout_2x2x1_ep8_tp1_44402c2102"
export SAVE_JAX_CACHE="true"

# ==============================================================================
# 7. Hyperparameters & DeepSWE Pipeline Configuration
# ==============================================================================
# GBS = BATCH_SIZE (72) * NUM_GENERATIONS (16) = 1152
# LEARNING_RATE = 1e-6 * sqrt(1152 / 256), MAX_GRAD_NORM = 0.125 * sqrt(256 / 1152)
export MAX_STEPS="10"
export BATCH_SIZE="72"
export MINI_BATCH_SIZE="72"
export NUM_GENERATIONS="16"
export TRAIN_MICRO_BATCH_SIZE="64"
export LEARNING_RATE="2.121320343559642e-06"
export MAX_GRAD_NORM="0.05892556509887896"

export INITIAL_DISPATCH_FIRST_STAGGER_S="75"
export INITIAL_DISPATCH_STAGGER_S="75"
export MAX_STALENESS="2"
export ROLLOUT_PRIORITY_SCHEDULING="true"
export PIPELINE_TRAIN_MICROBATCHES="true"
export PIPELINE_STEP_BOUNDARY="1"
export STAGE_NEXT_MICROBATCH="1"
export PREFER_VALID_ROLLOUTS="true"
export TRAJECTORY_GROUP_ORDER="prompt_batch"

# Sequence packing
export MAX_SEQ_TOKEN_PER_TPU="65536"
export MAX_SEGMENTS_PER_PACKED_ROW="16"

# Sampling Parameters
export TEMPERATURE="1.0"
export TOP_P="1.0"
export TOP_K="-1"

# Algorithmic & Loss Hyperparameters
export BETA="0.0"
export EPSILON="0.2"
export EPSILON_HIGH="0.28"
export USE_ROLLOUT_LOGPS="false"
export EXACT_TOKEN_CONTINUITY="true"
export OVERLONG_FILTER="false"
export OVERLONG_LOSS_MASKING="true"
export SEQ_LOGPROB_ERROR_THRESHOLD="2.0"
export TRUNCATED_IMPORTANCE_SAMPLING_TYPE="seq-mask-tis"
export TRUNCATED_IMPORTANCE_SAMPLING_RATIO_MIN="0.999"
export TRUNCATED_IMPORTANCE_SAMPLING_RATIO="1.002"
export ADVANTAGE_ESTIMATOR="grpo-loo"
export LOSS_AGG_MODE="token-mean"
export FLOAT32_GATE_LOGITS="true"
export FLOAT32_LOGITS="true"

# Optimizer Hyperparameters
export ADAM_B1="0.9"
export ADAM_B2="0.999"
export WEIGHT_DECAY="0.0"
export OPT_CHAIN_TYPE="clip_by_global_norm"
export WARMUP_STEPS_FRACTION="0.0"
export LEARNING_RATE_FINAL_FRACTION="1.0"
export SKIP_STEP_ON_SPIKES="false"
export SKIP_STEP_ON_NAN="true"

# DeepSWE Environment & Agent Sandbox
export DATASET_PATH="gs://mlperf_dataset/benchmark-r2e-gym-easy"
export SHUFFLE="false"
export USE_AGENT_SANDBOX="1"
export SCAFFOLD="openhands"
export OPENHANDS_MULTI_TOOL_CALLS="true"
export SANDBOX_NAMESPACE="${K8S_NAMESPACE:-trellis}"
export MAX_WARMPOOL_REPLICAS="16"
export SANDBOX_WAIT_INITIAL="false"
export SANDBOX_IMAGE_PULL_POLICY="IfNotPresent"
export PREPULL_SANDBOX_IMAGES="false"
export PREWARM_IMAGES_FILE="${DATASET_PATH}/sandbox_images.txt"
export PREWARM_MAX_WORKERS="32"
export ROLLOUT_MAX_CONCURRENCY="256"
export MAX_CONCURRENCY="4096"
export STEP_TIMEOUT_SECS="60"
export REWARD_TIMEOUT_SECS="60"
export EPISODE_TIMEOUT_SECS="1800"
export FLUSH_EVERY_N_STEPS="1"
export MAX_TURNS="30"
export MAX_PROMPT_LENGTH="8192"
export MAX_RESPONSE_LENGTH="57344"
export SANDBOX_NODE_SELECTOR_KEY="workload"
export SANDBOX_NODE_SELECTOR_VAL="sandbox"
export SANDBOX_TOLERATIONS='[{"key":"workload","operator":"Equal","value":"sandbox","effect":"NoSchedule"}]'
export IMAGE_REWRITE_PREFIX="us-central1-docker.pkg.dev/cloud-tpu-multipod-dev/tunix/"

# ==============================================================================
# 8. Execution Dispatch
# ==============================================================================
if [ -f "${DIR}/ubench_reporting.sh" ]; then
  ( source "${DIR}/ubench_reporting.sh" "$@" ) ||
    echo "UBENCH_REPORTING: Not recording this run for uBench. The launch continues." >&2
fi

if [ -f "${DIR}/../deepswe_dist/k8s_launcher.sh" ]; then
  LAUNCHER="${DIR}/../deepswe_dist/k8s_launcher.sh"
elif [ -f "${DIR}/tunix/experimental/examples/deepswe_dist/k8s_launcher.sh" ]; then
  LAUNCHER="${DIR}/tunix/experimental/examples/deepswe_dist/k8s_launcher.sh"
elif [ -f "${DIR}/../../../../third_party/py/tunix/experimental/examples/deepswe_dist/k8s_launcher.sh" ]; then
  LAUNCHER="${DIR}/../../../../third_party/py/tunix/experimental/examples/deepswe_dist/k8s_launcher.sh"
elif [ -f "${HOME}/github/tunix_build/tunix/experimental/examples/deepswe_dist/k8s_launcher.sh" ]; then
  LAUNCHER="${HOME}/github/tunix_build/tunix/experimental/examples/deepswe_dist/k8s_launcher.sh"
else
  echo "Error: k8s_launcher.sh not found relative to ${DIR}" >&2
  exit 1
fi

COMMAND="${1:-start}"
shift || true
exec "${LAUNCHER}" --command "${COMMAND}" --image "${TUNIX_IMAGE}" "$@"
