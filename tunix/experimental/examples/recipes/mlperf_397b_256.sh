#!/bin/bash
set -e

# ==============================================================================
# MLPerf DeepSWE recipe: Qwen3.5-397B-A17B
# ==============================================================================
# Derived from mlperf_35b_256.sh, with the trainer/rollout topology and the
# weight-sync settings taken from the verified Qwen3.5-397B GSM8K run
# (10 steps, 0 restarts, reward mean 0.33-0.52 per step).
#
# STATUS OF THE NUMBERS BELOW
#   verified at 397B : trainer mesh, rollout mesh, weight-sync env, RPC timeout,
#                      profiling off, checkpoint-save off, pack_size arithmetic.
#   NOT yet verified : everything DeepSWE-specific at this model size -- the
#                      rollout replica count, the 64k context, and the per-step
#                      concurrency. DeepSWE has only been run at 35B. The two
#                      marked TUNE blocks are where it will most likely need to
#                      change; treat a first run as a capacity experiment.
# ==============================================================================

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Fill these before you run.
export JOB_PREFIX="${JOB_PREFIX:-$USER}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-${JOB_PREFIX}-mlperf-397b}"
export MAXTEXT_OUTPUT_DIR="${MAXTEXT_OUTPUT_DIR:-gs://atwigg-trellis-europe-west4-dev/maxtext/${JOB_PREFIX}}"
export TRAJECTORY_LOG_DIR="${TRAJECTORY_LOG_DIR:-gs://atwigg-trellis-europe-west4-dev/trajectories/${JOB_PREFIX}}"
export ORCHESTRATOR_PORT="${ORCHESTRATOR_PORT:-20000}"
export ROLLOUT_PORT="${ROLLOUT_PORT:-20001}"
export TRAINER_PORT="${TRAINER_PORT:-20002}"
export TUNIX_IMAGE="${TUNIX_IMAGE:-gcr.io/cloud-tpu-multipod-dev/${USER:-atwigg}/trellis-397b:latest}"

# Profiling off. MaxText decided whether to profile from profiler_steps alone
# (default 5), so profiler=ProfilerType.NONE still opened a trace at step 1;
# under Pathways that raises "No profile started" inside fwd_bwd and takes the
# trainer down mid-run.
export PROFILER_STEPS=0
export SKIP_FIRST_N_PROFILER_STEPS=-1

export PROJECT="cloud-tpu-shared-capacity"
export REGION="europe-west4"
export CLUSTER="bodaborg-v5p-nap"
kubectl config use-context "gke_${PROJECT}_${REGION}_${CLUSTER}" || true
kubectl config set-context --current --namespace=trellis || true

export K8S_NAMESPACE="trellis"
export KUEUE_QUEUE="${KUEUE_QUEUE:-multislice-queue}"
export PRIORITY_CLASS="${PRIORITY_CLASS:-medium}"
export SERVICE_ACCOUNT="xpk-sa"
export CPU_MACHINE="n2d-standard-64"

# Pathways & Raiden images and settings
export PATHWAYS_SERVER_IMAGE="${PATHWAYS_SERVER_IMAGE:-us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/datenglin/unsanitized_server:raiden_20260920_v2}"
export PATHWAYS_PROXY_IMAGE="${PATHWAYS_PROXY_IMAGE:-us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/datenglin/unsanitized_proxy_server:raiden_20260920_v2}"
export PATHWAYS_PROXY_MEMORY_LIMIT="160G"
export USER_CONTAINER_MEMORY="260G"
export USER_CONTAINER_MEMORY_LIMIT="${USER_CONTAINER_MEMORY_LIMIT:-260G}"
export RAIDEN_DEVICES_PER_HOST=4
export USE_WEIGHT_CONVERTER="true"
export PREFUSE_MOE_WEIGHTS="true"
export TRAINER_PREFUSE_MOE_WEIGHTS="true"
export ROLLOUT_PREFUSE_MOE_WEIGHTS="true"
export VERIFY_WEIGHTS="true"
export TRAINER_PADDED_MOE_MLP_DIM=""

# WandB configuration
export WANDB_ENTITY="google-trellis"
export WANDB_PROJECT="trellis-deepswe"

# Model configuration
export MODEL_NAME="Qwen3.5-397B-A17B"
export MODEL_ID="Qwen/Qwen3.5-397B-A17B"
export TOKENIZER_PATH="Qwen/Qwen3.5-397B-A17B"
export MAXTEXT_MODEL_NAME="qwen3.5-397b-a17b"
export MAXTEXT_CKPT="${MAXTEXT_CKPT:-gs://sanbao-europe/qwen35_397b/scanned_reshard_fsdp32_tp2/0/items}"
export TRAINABLE_PARAMETERS_MASK='^(?!.*routed_experts/gate/kernel).*'
export EOS_TOKENS="${EOS_TOKENS:-151645,151643}"

# Backend configuration
export TRAINER_BACKEND="maxtext"
export SAMPLER="vllm"
export WEIGHT_SYNC_MODE="raiden"

# ------------------------------------------------------------------ verified --
# Trainer: 256 chips. TRAINER_MESH_EXPERT=2 is required, not a tuning choice:
# at expert=1 the GMM_v2 kernel overflows smem by ~8.6K, and raising the GMM
# tile sizes does not help because T(128) padding rounds s32[592] and s32[552]
# to the same 640.
export TRAINER_JOBSET_YAML="jobset.pathways.qwen3.5-397b.yaml"
export TRAINER_TPU_SLICE="tpuv5p:4x8x8"
export TRAINER_MESH_FSDP=64
export TRAINER_MESH_TP=2
export TRAINER_MESH_EXPERT=2
export TRAINER_BASE_NUM_KV_HEADS=16

# Rollout: 16 chips = 4 hosts per replica. tp * expert must equal the slice.
# tp=2 x expert=8 is NOT an alternative -- it produces incoherent rollouts for
# reasons still unexplained. Use tp=1 x expert=16.
export ROLLOUT_JOBSET_YAML="jobset.tpu.yaml"
export ROLLOUT_TPU_SLICE="tpuv5p:2x2x4"
export ROLLOUT_MESH_FSDP=1
export ROLLOUT_MESH_TP=1
export ROLLOUT_MESH_EXPERT=16

# ---------------------------------------------------------------- TUNE (1) ---
# Replica count is a capacity guess. At 35B this was 16 replicas x 4 chips = 64
# rollout chips; 397B needs 16 chips per replica, so the same 64 chips buys only
# 4 replicas and a quarter of the rollout concurrency. Raise if the step is
# generation-bound and there is quota; 256 trainer + N*16 rollout must fit.
export ROLLOUT_WORKERS="${ROLLOUT_WORKERS:-4}"
export ROLLOUT_REPLICAS="${ROLLOUT_REPLICAS:-4}"

# ==============================================================================
# vLLM Rollout Configuration
# ==============================================================================
export VLLM_LOGGING_LEVEL="INFO"
export VLLM_MAX_NUM_BATCHED_TOKENS=2048
export VLLM_MAX_NUM_SEQS=16
export VLLM_GPU_MEMORY_UTILIZATION="0.9"
export VLLM_DATA_PARALLEL_SIZE=1
export VLLM_ENABLE_EXPERT_PARALLEL="true"
export VLLM_ADDITIONAL_CONFIG='{"sharding":{"sharding_strategy":{"expert_parallelism":16,"tensor_parallelism":1,"enable_dp_attention":true}},"custom_mamba_cache_multiplier":16,"maxtext_config":{"scan_layers":false,"attention":"vllm_rpa","allow_split_physical_axes":true,"use_multimodal":false,"prefuse_moe_weights":true}}'

export ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-false}"
export VLLM_PREFIX_CACHE_RETENTION_INTERVAL="${VLLM_PREFIX_CACHE_RETENTION_INTERVAL:-256}"
export VLLM_MAMBA_CACHE_MODE="${VLLM_MAMBA_CACHE_MODE:-${MAMBA_CACHE_MODE:-none}}"

export ROLLOUT_FREE_KV_CACHE="false"
export VLLM_KV_CACHE_DTYPE="bfloat16"
export VLLM_BLOCK_SIZE=256
export VLLM_ASYNC_SCHEDULING="true"
export VLLM_ENABLE_CHUNKED_PREFILL="true"
export VLLM_LANGUAGE_MODEL_ONLY="true"
export VLLM_REASONING_PARSER="qwen3"
export VLLM_LIMIT_MM_PER_PROMPT='{"image": 0, "video": 0}'

# ==============================================================================
# Rollout Worker Environment Flags
# ==============================================================================
export NUM_PRECOMPILE_WORKERS=8
export NEW_MODEL_DESIGN=1
export ATTN_BUCKETIZED_NUM_REQS=true
export ATTN_CUSTOM_NUM_REQS_BUCKETS=4
export ONEHOT_MOE_PERMUTE_THRESHOLD=131072
export VLLM_MOE_CHUNK_SIZE=256
export SLICE_ROPE_CACHE=1
export DP_SCHED_BATCH_PREFILL=false
export LIBTPU_INIT_ARGS=' --xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false --xla_tpu_check_legacy_constraints_in_reduce_scatter_legalizer=false'
export VLLM_ENABLE_V1_MULTIPROCESSING=0

# Raiden tuning carried over from the verified 397B run.
export ORCHESTRATOR_EXTRA_ENV="${ORCHESTRATOR_EXTRA_ENV:-WEIGHT_SYNC_TIMEOUT_H2D=1800 RAIDEN_PARALLELISM=16}"
export ROLLOUT_EXTRA_ENV="${ROLLOUT_EXTRA_ENV:-RAY_memory_monitor_refresh_ms=0 RAIDEN_TRANSPORT_COALESCE_WINDOW_BYTES=67108864 RAIDEN_WEIGHT_SYNC_PIPELINE_GROUP_SIZE=16 RAIDEN_PARALLELISM=16}"
export TRAINER_EXTRA_ENV="${TRAINER_EXTRA_ENV:-RAIDEN_TRANSPORT_COALESCE_WINDOW_BYTES=67108864 RAIDEN_WEIGHT_SYNC_PIPELINE_GROUP_SIZE=16}"

# ==============================================================================
# Hyperparameters & DeepSWE Pipeline Configuration
# ==============================================================================
export MAX_STEPS=${MAX_STEPS:-50}
export BATCH_SIZE=16
export MINI_BATCH_SIZE=${BATCH_SIZE}
export NUM_GENERATIONS=16
# Must be a multiple of TRAINER_MESH_FSDP * TRAINER_MESH_EXPERT = 128, because
# MaxText binds the activation batch axis to
# ('data','fsdp','fsdp_transpose','expert') and the MoE shard_map rejects a row
# count that is not divisible by their product.
export TRAIN_MICRO_BATCH_SIZE="${TRAIN_MICRO_BATCH_SIZE:-128}"
export CHECKPOINT_SAVE_INTERVAL_STEPS=0
export CHECKPOINT_MAX_TO_KEEP=10
export MAX_STALENESS=0

# Step 0 is a cold single-threaded Pallas lowering of the MoE and GDN kernels
# with the TPUs idle; at 256 chips that ran past the 1800 s default.
export RPC_TIMEOUT_S="${RPC_TIMEOUT_S:-10800}"

export TEMPERATURE="1.0"
export TOP_P="1.0"
export TOP_K="-1"

export BETA=0.0
export EPSILON=0.2
export EPSILON_HIGH=0.28
export USE_ROLLOUT_LOGPS="false"
export OVERLONG_FILTER="true"
export OVERLONG_LOSS_MASKING="true"
export SEQ_LOGPROB_ERROR_THRESHOLD=2.0
export TRUNCATED_IMPORTANCE_SAMPLING_TYPE="seq-mask-tis"
export TRUNCATED_IMPORTANCE_SAMPLING_RATIO_MIN=0.999
export TRUNCATED_IMPORTANCE_SAMPLING_RATIO=1.002
export ADVANTAGE_ESTIMATOR="grpo-loo"
export LOSS_AGG_MODE="token-mean"
export FLOAT32_GATE_LOGITS="true"
export FLOAT32_LOGITS="true"

export LEARNING_RATE="1e-6"
export ADAM_B1=0.9
export ADAM_B2=0.999
export WEIGHT_DECAY=0.0
export MAX_GRAD_NORM="0.125"
export WARMUP_STEPS_FRACTION=0.0
export LEARNING_RATE_FINAL_FRACTION=1.0

export REMAT_POLICY="full"
export TRAINER_MAXTEXT_ATTENTION="flash"
export COMPUTE_LOGPS_CHUNK_SIZE=512

export EPISODE_TIMEOUT_SECS=1800
export DEBUG=${DEBUG:-0}

# DeepSWE Environment & Agent Sandbox
export DATASET_PATH="gs://mlperf_dataset/r2e-gym-easy"
export USE_AGENT_SANDBOX=1
export SANDBOX_NAMESPACE="trellis"
export SANDBOX_NODE_SELECTOR_KEY="cloud.google.com/gke-nodepool"
export SANDBOX_NODE_SELECTOR_VAL="sandbox-cpu-pool"
export IMAGE_REWRITE_PREFIX="${IMAGE_REWRITE_PREFIX:-europe-west4-docker.pkg.dev/cloud-tpu-multipod-dev/tunix/}"
export MAX_WARMPOOL_REPLICAS=2
export STEP_TIMEOUT_SECS=300
export REWARD_TIMEOUT_SECS=180
export FLUSH_EVERY_N_STEPS=1
export MAX_TURNS=30

# ---------------------------------------------------------------- TUNE (2) ---
# Context length and concurrency. The 35B recipe uses a 64k window
# (4096 prompt + 61440 response) with 256-way concurrency. Neither has been
# exercised at 397B, where the rollout has 16 chips per replica rather than 4
# and the KV cache per sequence is far larger. If the rollout OOMs or the KV
# cache will not allocate, reduce MAX_RESPONSE_LENGTH first, then concurrency.
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-61440}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-65536}"
export ROLLOUT_MAX_CONCURRENCY="${ROLLOUT_MAX_CONCURRENCY:-256}"
export MAX_CONCURRENCY="${MAX_CONCURRENCY:-256}"

# ==============================================================================
# Execution Dispatch
# ==============================================================================
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

COMMAND="${1:-start}"
exec "${LAUNCHER}" --command "${COMMAND}" --image "${TUNIX_IMAGE}"
