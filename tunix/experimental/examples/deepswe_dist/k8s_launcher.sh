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

COMMAND=""
TUNIX_IMAGE=${TUNIX_IMAGE:-}

PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" &>/dev/null; then
  PYTHON="python"
fi

export MODEL_NAME=${MODEL_NAME:-Qwen3-1.7B}
export MODEL_ID=${MODEL_ID:-Qwen/Qwen3-1.7B}
# Must be model-specific: vLLM prioritizes non-empty local snapshot directories,
# which can cause stale config/shape mismatches if shared across models.
export MODEL_DIR=${MODEL_DIR:-artifacts/qwen3_dist_deepswe/models/${MODEL_NAME}}
# Defaults to MODEL_ID so AutoTokenizer downloads directly from HuggingFace
# instead of failing on an initially empty local MODEL_DIR.
export TOKENIZER_PATH=${TOKENIZER_PATH:-${MODEL_ID}}

export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1024}
export BATCH_SIZE=${BATCH_SIZE:-1}
export NUM_GENERATIONS=${NUM_GENERATIONS:-2}
export MAX_STEPS=${MAX_STEPS:-1}
export MAX_TURNS=${MAX_TURNS:-3}
export TRAIN_MICRO_BATCH_SIZE=${TRAIN_MICRO_BATCH_SIZE:-1}
export MAX_SEQ_TOKEN_PER_TPU=${MAX_SEQ_TOKEN_PER_TPU:-}
export MAX_SEGMENTS_PER_PACKED_ROW=${MAX_SEGMENTS_PER_PACKED_ROW:-}
export SEGMENT_ALIGNMENT_BOUNDARY=${SEGMENT_ALIGNMENT_BOUNDARY:-}

# Set to tunix to run Tunix's PeftTrainer, and maxtext to run MaxText's MaxTextTrainingEngine
export TRAINER_BACKEND=${TRAINER_BACKEND:-tunix}
export TRAINER_PRECOMPILE=${TRAINER_PRECOMPILE:-true}
export MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-${BATCH_SIZE}}
export EVAL_EVERY_N_STEPS=${EVAL_EVERY_N_STEPS:-1000000}
export LEARNING_RATE=${LEARNING_RATE:-1e-6}
export OPT_CHAIN_TYPE=${OPT_CHAIN_TYPE-clip_by_global_norm}
export MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}
export BETA=${BETA:-0.0}
export EPSILON=${EPSILON:-0.2}
export MAX_STALENESS=${MAX_STALENESS:-0}
export TRAJECTORY_GROUP_ORDER=${TRAJECTORY_GROUP_ORDER:-trajectory_completion}
export LORA_RANK=${LORA_RANK:-64}
export LORA_ALPHA=${LORA_ALPHA:-64.0}
export MODEL_DTYPE=${MODEL_DTYPE:-float32}
export USE_LORA=${USE_LORA:-0}
export DEBUG=${DEBUG:-0}
export USE_ROLLOUT_LOGPS=${USE_ROLLOUT_LOGPS:-true}
export SAMPLER=${SAMPLER:-inprocess_vllm}
export WEIGHT_SYNC_MODE=${WEIGHT_SYNC_MODE:-none}
export IN_FLIGHT_WEIGHT_UPDATES=${IN_FLIGHT_WEIGHT_UPDATES:-false}
# Router replay: rollout captures the MoE experts each token was routed to and
# the MaxText trainer (TRAINER_BACKEND=maxtext) replays them. Set
# ROUTED_EXPERTS_NUM_EXPERTS when the model has 256-1023 experts per layer so
# every batch uses the same encoding.
export RETURN_ROUTED_EXPERTS=${RETURN_ROUTED_EXPERTS:-false}
export ROUTED_EXPERTS_ENCODE_MODE=${ROUTED_EXPERTS_ENCODE_MODE:-auto}
export ROUTED_EXPERTS_NUM_EXPERTS=${ROUTED_EXPERTS_NUM_EXPERTS:-}

# DeepSWE dataset and environment configuration
export DATASET_NAME=${DATASET_NAME:-R2E-Gym/R2E-Gym-Subset}
export DATASET_PATH=${DATASET_PATH:-}
export DATASET_SPLIT=${DATASET_SPLIT:-train}
export DATASET_CACHE_DIR=${DATASET_CACHE_DIR:-artifacts/qwen3_dist_deepswe/dataset_cache}
export SHUFFLE=${SHUFFLE:-true}
export SEED=${SEED:-42}
export ENV_BACKEND=${ENV_BACKEND:-kubernetes}
export SCAFFOLD=${SCAFFOLD:-r2egym}
export K8S_NAMESPACE=${K8S_NAMESPACE:-${NAMESPACE:-default}}
export KUEUE_QUEUE_NAME=${KUEUE_QUEUE_NAME:-${KUEUE_QUEUE:-}}
export USE_AGENT_SANDBOX=${USE_AGENT_SANDBOX:-0}
export SANDBOX_NAMESPACE=${SANDBOX_NAMESPACE:-${K8S_NAMESPACE}}
export SANDBOX_NODE_SELECTOR_KEY=${SANDBOX_NODE_SELECTOR_KEY:-}
export SANDBOX_NODE_SELECTOR_VAL=${SANDBOX_NODE_SELECTOR_VAL:-}
export STEP_TIMEOUT_SECS=${STEP_TIMEOUT_SECS:-1800}
export REWARD_TIMEOUT_SECS=${REWARD_TIMEOUT_SECS:-1800}
export ROLLOUT_MAX_CONCURRENCY=${ROLLOUT_MAX_CONCURRENCY:-64}
export FLUSH_EVERY_N_STEPS=${FLUSH_EVERY_N_STEPS:-1}
export ENABLE_PATHWAYS_PERSISTENCE=${ENABLE_PATHWAYS_PERSISTENCE:-0}
export CKPT_D2H_CONCURRENT_GB=${CKPT_D2H_CONCURRENT_GB:-8}

# MaxText trainer configuration: only consulted when TRAINER_BACKEND=maxtext
export MAXTEXT_MODEL_NAME=${MAXTEXT_MODEL_NAME:-qwen3-1.7b}
export MAXTEXT_CKPT=${MAXTEXT_CKPT:-}
# If TRAINER_BACKEND=maxtext, MAXTEXT_CKPT must be set to the path of an Orbax params-only checkpoint.
if [[ "$TRAINER_BACKEND" == "maxtext" && -z "$MAXTEXT_CKPT" ]]; then
  echo "Error: TRAINER_BACKEND=maxtext requires MAXTEXT_CKPT (Orbax params-only checkpoint)."
  exit 1
fi
export MAXTEXT_OUTPUT_DIR=${MAXTEXT_OUTPUT_DIR:-artifacts/deepswe_dist/maxtext}
export MAXTEXT_EXTRA_FLAGS=${MAXTEXT_EXTRA_FLAGS:-}
export PROFILER_STEPS=${PROFILER_STEPS:-0}
export SKIP_FIRST_N_PROFILER_STEPS=${SKIP_FIRST_N_PROFILER_STEPS:-}
export PROFILER_PERIOD=${PROFILER_PERIOD:-}
export ENABLE_TPU_PROFILING_OPTIONS=${ENABLE_TPU_PROFILING_OPTIONS:-true}
export TPU_PROFILE_CHIPS_PER_TASK=${TPU_PROFILE_CHIPS_PER_TASK:-1}
export TPU_PROFILE_SPARSE_CORES=${TPU_PROFILE_SPARSE_CORES:-1}
export TPU_PROFILE_SPARSE_CORE_TILES=${TPU_PROFILE_SPARSE_CORE_TILES:-1}
export UPLOAD_ALL_PROFILER_RESULTS=${UPLOAD_ALL_PROFILER_RESULTS:-false}
export TRAINER_MESH_TP=${TRAINER_MESH_TP:-1}
export TRAINER_MESH_EXPERT=${TRAINER_MESH_EXPERT:-1}
# Padded MoE MLP intermediate dimension; must match rollout TP padding for MoE models.
export TRAINER_PADDED_MOE_MLP_DIM=${TRAINER_PADDED_MOE_MLP_DIM:-}
export ROLLOUT_MESH_TP=${ROLLOUT_MESH_TP:-2}
export ROLLOUT_MESH_FSDP=${ROLLOUT_MESH_FSDP:-1}
# Optional: enable experimental batched-RPA attention kernel for rollout.
export ROLLOUT_USE_BATCHED_RPA=${ROLLOUT_USE_BATCHED_RPA:-}
export ROLLOUT_MAXTEXT_ATTENTION=${ROLLOUT_MAXTEXT_ATTENTION:-}

# Logs source/destination Raiden tensor checksums on both the trainer and
# rollout sides during weight sync, for cross-verification of a real run.
export VERIFY_WEIGHTS=${VERIFY_WEIGHTS:-false}

export WANDB_PROJECT=${WANDB_PROJECT:-trellis-deepswe}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-}
export WANDB_API_KEY=${WANDB_API_KEY:-}
export LOG_DIR=${LOG_DIR:-}
export TRAJECTORY_LOG_DIR=${TRAJECTORY_LOG_DIR:-}
export RUN_ID=${RUN_ID:-run_$(date -u +%Y%m%d_%H%M%S)_$$}
export TRAJECTORY_STORE_ENABLED=${TRAJECTORY_STORE_ENABLED:-false}
export TRAJECTORY_STORE_BACKEND=${TRAJECTORY_STORE_BACKEND:-file}
export TRAJECTORY_STORE_DIR=${TRAJECTORY_STORE_DIR:-"${LOG_DIR:-artifacts/deepswe_dist}/trajectory_store"}

export ORCHESTRATOR_ID=$USER-orch
export ORCHESTRATOR_PORT=20000

export ROLLOUT_ID=$USER-roll
export ROLLOUT_PORT=20001

export TRAINER_ID=$USER-train
export TRAINER_PORT=20002

export CPU_MACHINE=${CPU_MACHINE:-n2-standard-64}
export GCS_SCRATCH_LOCATION=${GCS_SCRATCH_LOCATION:-gs://cloud-pathways-staging/tmp}

export TRAINER_JOBSET_YAML=${TRAINER_JOBSET_YAML:-jobset.pathways.yaml}
export TRAINER_TPU_SLICE=${TRAINER_TPU_SLICE:-tpuv5:2x2x2}
export TRAINER_MESH_FSDP=${TRAINER_MESH_FSDP:-8}
export ROLLOUT_TPU_SLICE=${ROLLOUT_TPU_SLICE:-tpuv5:2x2x1}

# JAX compilation cache configuration
export LOCAL_JAX_CACHE_DIR=${LOCAL_JAX_CACHE_DIR:-${JAX_CACHE_DIR:-/tmp/jax_cache}}
export JAX_CACHE_GCS_DIR=${JAX_CACHE_GCS_DIR:-}
export ROLLOUT_JAX_CACHE_GCS_DIR=${ROLLOUT_JAX_CACHE_GCS_DIR:-}
export SAVE_JAX_CACHE=${SAVE_JAX_CACHE:-true}
export SKIP_JAX_PRECOMPILE=${SKIP_JAX_PRECOMPILE:-1}
export ROLLOUT_SKIP_JAX_PRECOMPILE=${ROLLOUT_SKIP_JAX_PRECOMPILE:-${SKIP_JAX_PRECOMPILE}}
is_jax_cache_disabled() {
  case "${DISABLE_JAX_CACHE:-0}" in
    1|[tT][rR][uU][eE]|[yY][eE][sS]|[yY]|[tT]) return 0 ;;
    *) return 1 ;;
  esac
}

if ! is_jax_cache_disabled && [[ -z "${ROLLOUT_JAX_CACHE_GCS_DIR}" ]]; then
  ROLLOUT_JAX_CACHE_GCS_DIR=$("$PYTHON" -m tunix.experimental.common.gcs_cache resolve-uri 2>/dev/null || true)
fi
export EVAL_JAX_CACHE_GCS_DIR=${EVAL_JAX_CACHE_GCS_DIR:-${ROLLOUT_JAX_CACHE_GCS_DIR}}

export TRAINER_EXTRA_ENV=${TRAINER_EXTRA_ENV:-}
# Trainer only: host buffers of at least this many bytes (e.g. 16777216) go
# from the trainer to pathways-proxy through a shared tmpfs instead of
# localhost gRPC. Unset keeps gRPC. Each in-flight buffer is also held in the
# trainer container's memory, so raise USER_CONTAINER_MEMORY_LIMIT with it.
export TRAINER_IFRT_PROXY_LARGE_TRANSFER_THRESHOLD_BYTES=${TRAINER_IFRT_PROXY_LARGE_TRANSFER_THRESHOLD_BYTES:-}
export DRY_RUN=${DRY_RUN:-false}

apply_manifest() {
  if [[ "$DRY_RUN" == "true" ]]; then
    echo "---"
    cat
  else
    kubectl apply -f -
  fi
}

if [[ "$BETA" != "0" && "$BETA" != "0.0" ]]; then
  echo "Error: this first DeepSWE distributed launcher only wires trainer+rollout."
  echo "Use BETA=0.0 until the reference inference worker is added."
  exit 1
fi

case "${RETURN_ROUTED_EXPERTS}" in
  1|[tT][rR][uU][eE]|[yY][eE][sS]|[yY]|[tT])
    if [[ "${TRAINER_BACKEND}" != "maxtext" ]]; then
      echo "Error: RETURN_ROUTED_EXPERTS needs TRAINER_BACKEND=maxtext; only the MaxText trainer replays routed experts."
      exit 1
    fi
    ;;
esac

stop_orchestrator() {
  if [[ "$DRY_RUN" == "true" ]]; then
    echo "kubectl delete jobset ${ORCHESTRATOR_ID} -n ${K8S_NAMESPACE}"
  else
    kubectl delete jobset "${ORCHESTRATOR_ID}" -n "${K8S_NAMESPACE}" --ignore-not-found
  fi
}

start_orchestrator() {
  local dataset_args=""
  if [[ -n "${DATASET_PATH}" ]]; then
    dataset_args="--dataset_path=${DATASET_PATH}"
  fi
  local shuffle_arg="--shuffle"
  if [[ "${SHUFFLE}" == "0" || "${SHUFFLE}" == "false" || "${SHUFFLE}" == "False" ]]; then
    shuffle_arg="--no-shuffle"
  fi
  local sandbox_arg=""
  if [[ "${USE_AGENT_SANDBOX}" == "1" || "${USE_AGENT_SANDBOX}" == "true" || "${USE_AGENT_SANDBOX}" == "True" ]]; then
    sandbox_arg="--use_agent_sandbox"
  fi
  "$PYTHON" tunix/experimental/distributed/deployment/yaml_generator.py \
    tunix/experimental/distributed/deployment/yamls/jobset.cpu.yaml \
    --jobset_name="${ORCHESTRATOR_ID}" \
    --namespace="${K8S_NAMESPACE}" \
    ${KUEUE_QUEUE_NAME:+--queue_name="${KUEUE_QUEUE_NAME}"} \
    --cpu_machine=${CPU_MACHINE} \
    --worker_container_image="${TUNIX_IMAGE}" \
    --worker_container_port="${ORCHESTRATOR_PORT}" \
    --worker_startup_command=" \
      ${WANDB_API_KEY:+WANDB_API_KEY=\"${WANDB_API_KEY}\"} \
      ${LOG_DIR:+LOG_DIR=\"${LOG_DIR}\"} \
      ${TRAJECTORY_LOG_DIR:+TRAJECTORY_LOG_DIR=\"${TRAJECTORY_LOG_DIR}\"} \
      WANDB_PROJECT=\"${WANDB_PROJECT}\" \
      WANDB_RUN_NAME=\"${WANDB_RUN_NAME}\" \
      python -m tunix.experimental.distributed.runtime.main \
        --discovery_id=${ORCHESTRATOR_ID} \
        --discovery_port=${ORCHESTRATOR_PORT} \
        --process_main=tunix.experimental.examples.deepswe_dist.run_deepswe_dist.main \
        --model_id=${MODEL_ID} \
        --tokenizer_path=${TOKENIZER_PATH} \
        --batch_size=${BATCH_SIZE} \
        --mini_batch_size=${MINI_BATCH_SIZE} \
        --num_generations=${NUM_GENERATIONS} \
        --max_steps=${MAX_STEPS} \
        --max_turns=${MAX_TURNS} \
        --max_prompt_length=${MAX_PROMPT_LENGTH} \
        --max_response_length=${MAX_RESPONSE_LENGTH} \
        --train_micro_batch_size=${TRAIN_MICRO_BATCH_SIZE} \
        --beta=${BETA} \
        --epsilon=${EPSILON} \
        --max_staleness=${MAX_STALENESS} \
        --trajectory_group_order=${TRAJECTORY_GROUP_ORDER} \
        --dataset_name=${DATASET_NAME} \
        --dataset_split=${DATASET_SPLIT} \
        ${DATASET_CACHE_DIR:+--dataset_cache_dir=${DATASET_CACHE_DIR}} \
        --seed=${SEED} \
        --env_backend=${ENV_BACKEND} \
        --scaffold=${SCAFFOLD} \
        --step_timeout_secs=${STEP_TIMEOUT_SECS} \
        --reward_timeout_secs=${REWARD_TIMEOUT_SECS} \
        --flush_every_n_steps=${FLUSH_EVERY_N_STEPS} \
        --wandb_project=\"${WANDB_PROJECT}\" \
        --wandb_run_name=\"${WANDB_RUN_NAME}\" \
        --weight_sync_mode=${WEIGHT_SYNC_MODE} \
        --trainer_precompile=${TRAINER_PRECOMPILE} \
        --routed_experts_encode_mode=${ROUTED_EXPERTS_ENCODE_MODE} \
        ${ROUTED_EXPERTS_NUM_EXPERTS:+--routed_experts_num_experts=${ROUTED_EXPERTS_NUM_EXPERTS}} \
        --stop_workers_on_exit \
        --enable_trajectory_store=${TRAJECTORY_STORE_ENABLED} \
        --trajectory_store_backend=${TRAJECTORY_STORE_BACKEND} \
        --trajectory_store_dir=\"${TRAJECTORY_STORE_DIR}\" \
        --run_id=\"${RUN_ID}\" \
        $([[ "${USE_ROLLOUT_LOGPS}" == "false" || "${USE_ROLLOUT_LOGPS}" == "False" || "${USE_ROLLOUT_LOGPS}" == "0" ]] && echo --no-use_rollout_logps || echo --use_rollout_logps) \
        ${dataset_args} \
        ${shuffle_arg} \
        ${sandbox_arg} \
        ${LOG_DIR:+--log_dir=\"${LOG_DIR}\"} \
        ${TRAJECTORY_LOG_DIR:+--trajectory_log_dir=\"${TRAJECTORY_LOG_DIR}\"} \
        ${MAX_SEQ_TOKEN_PER_TPU:+--max_seq_token_per_tpu=${MAX_SEQ_TOKEN_PER_TPU}} \
        ${MAX_SEGMENTS_PER_PACKED_ROW:+--max_segments_per_packed_row=${MAX_SEGMENTS_PER_PACKED_ROW}} \
        ${SEGMENT_ALIGNMENT_BOUNDARY:+--segment_alignment_boundary=${SEGMENT_ALIGNMENT_BOUNDARY}} \
        --num_data_shards=$(( ${TRAINER_MESH_FSDP:-1} * ${TRAINER_MESH_EXPERT:-1} )) \
        ${DEBUG:+--debug} \
    " \
    | apply_manifest
}

stop_trainer() {
  if [[ "$DRY_RUN" == "true" ]]; then
    echo "kubectl delete jobset ${TRAINER_ID} -n ${K8S_NAMESPACE}"
  else
    kubectl delete jobset "${TRAINER_ID}" -n "${K8S_NAMESPACE}" --ignore-not-found
  fi
}

start_trainer() {
  local maxtext_args=""
  local profiler_flags=""
  if [[ -n "$PROFILER_STEPS" ]]; then
    profiler_flags+=" --profiler_steps=${PROFILER_STEPS}"
  fi
  if [[ -n "$SKIP_FIRST_N_PROFILER_STEPS" ]]; then
    profiler_flags+=" --skip_first_n_profiler_steps=${SKIP_FIRST_N_PROFILER_STEPS}"
  fi
  if [[ -n "$PROFILER_PERIOD" ]]; then
    profiler_flags+=" --profiler_period=${PROFILER_PERIOD}"
  fi
  if [[ -n "$ENABLE_TPU_PROFILING_OPTIONS" ]]; then
    profiler_flags+=" --enable_tpu_profiling_options=${ENABLE_TPU_PROFILING_OPTIONS}"
  fi
  if [[ -n "$TPU_PROFILE_CHIPS_PER_TASK" ]]; then
    profiler_flags+=" --tpu_num_chips_to_profile_per_task=${TPU_PROFILE_CHIPS_PER_TASK}"
  fi
  if [[ -n "$TPU_PROFILE_SPARSE_CORES" ]]; then
    profiler_flags+=" --tpu_num_sparse_cores_to_trace=${TPU_PROFILE_SPARSE_CORES}"
  fi
  if [[ -n "$TPU_PROFILE_SPARSE_CORE_TILES" ]]; then
    profiler_flags+=" --tpu_num_sparse_core_tiles_to_trace=${TPU_PROFILE_SPARSE_CORE_TILES}"
  fi
  if [[ -n "$UPLOAD_ALL_PROFILER_RESULTS" ]]; then
    profiler_flags+=" --upload_all_profiler_results=${UPLOAD_ALL_PROFILER_RESULTS}"
  fi
  if [[ "${TRAINER_BACKEND}" == "maxtext" ]]; then
    maxtext_args=" \
      --maxtext_model_name=${MAXTEXT_MODEL_NAME} \
      ${TRAINER_PADDED_MOE_MLP_DIM:+--maxtext_padded_moe_mlp_dim=${TRAINER_PADDED_MOE_MLP_DIM}} \
      --maxtext_ckpt_path=${MAXTEXT_CKPT} \
      --maxtext_output_directory=${MAXTEXT_OUTPUT_DIR} \
      --mesh_tp=${TRAINER_MESH_TP} \
      --mesh_expert=${TRAINER_MESH_EXPERT} \
      ${ROLLOUT_MESH_TP:+--rollout_mesh_tp=${ROLLOUT_MESH_TP}} \
      ${USE_WEIGHT_CONVERTER:+--use_weight_converter=${USE_WEIGHT_CONVERTER}} \
    "
  fi
  local lora_args=""
  if [[ "${USE_LORA}" == "1" || "${USE_LORA}" == "true" || "${USE_LORA}" == "True" ]]; then
    lora_args="--use_lora"
  fi
  local raiden_env=""
  if [[ "${WEIGHT_SYNC_MODE}" == "raiden" ]]; then
    if [[ "${TRAINER_JOBSET_YAML}" == "jobset.pathways.yaml" ]]; then
      raiden_env+=" RAIDEN_USE_FFI=1"
    fi
  fi
  local opt_chain_flags=""
  if [[ -n "${OPT_CHAIN_TYPE}" ]]; then
    opt_chain_flags="--optimizer_opt_chain_type=\"${OPT_CHAIN_TYPE}\" --optimizer_chain_kwargs=\"{'max_norm': ${MAX_GRAD_NORM}}\""
  fi
  "$PYTHON" tunix/experimental/distributed/deployment/yaml_generator.py \
    tunix/experimental/distributed/deployment/yamls/${TRAINER_JOBSET_YAML} \
    --jobset_name="${TRAINER_ID}" \
    --namespace="${K8S_NAMESPACE}" \
    ${KUEUE_QUEUE_NAME:+--queue_name="${KUEUE_QUEUE_NAME}"} \
    --tpu_slice=${TRAINER_TPU_SLICE} \
    --cpu_machine=${CPU_MACHINE} \
    ${PATHWAYS_SERVER_IMAGE:+--pathways_server_image="${PATHWAYS_SERVER_IMAGE}"} \
    ${PATHWAYS_PROXY_IMAGE:+--pathways_proxy_server_image="${PATHWAYS_PROXY_IMAGE}"} \
    ${PATHWAYS_PROXY_MEMORY_LIMIT:+--pathways_proxy_memory_limit="${PATHWAYS_PROXY_MEMORY_LIMIT}"} \
    ${PATHWAYS_PROXY_MEMORY:+--pathways_proxy_memory="${PATHWAYS_PROXY_MEMORY}"} \
    ${PATHWAYS_RM_MEMORY:+--pathways_rm_memory="${PATHWAYS_RM_MEMORY}"} \
    ${USER_CONTAINER_MEMORY:+--user_container_memory="${USER_CONTAINER_MEMORY}"} \
    ${USER_CONTAINER_MEMORY_LIMIT:+--user_container_memory_limit="${USER_CONTAINER_MEMORY_LIMIT}"} \
    ${PATHWAYS_WORKER_MEMORY:+--pathways_worker_memory="${PATHWAYS_WORKER_MEMORY}"} \
    ${TRAINER_IFRT_PROXY_LARGE_TRANSFER_THRESHOLD_BYTES:+--ifrt_proxy_large_transfer_threshold_bytes="${TRAINER_IFRT_PROXY_LARGE_TRANSFER_THRESHOLD_BYTES}"} \
    --pathways_gcs_scratch_location=${GCS_SCRATCH_LOCATION} \
    --worker_container_image="${TUNIX_IMAGE}" \
    --worker_container_port="${TRAINER_PORT}" \
    --worker_startup_command=" \
      ${HF_TOKEN:+HF_TOKEN=\"${HF_TOKEN}\"} VERIFY_WEIGHTS=${VERIFY_WEIGHTS} ENABLE_PATHWAYS_PERSISTENCE=${ENABLE_PATHWAYS_PERSISTENCE}${CKPT_D2H_CONCURRENT_GB:+ CKPT_D2H_CONCURRENT_GB=${CKPT_D2H_CONCURRENT_GB}}${MAXTEXT_EXTRA_FLAGS:+ MAXTEXT_EXTRA_FLAGS=\"${MAXTEXT_EXTRA_FLAGS}\"}${raiden_env}${TRAINER_EXTRA_ENV:+ ${TRAINER_EXTRA_ENV}} python -m tunix.experimental.distributed.runtime.main \
        --discovery_addrs=${ORCHESTRATOR_ID}:${ORCHESTRATOR_PORT} \
        --process_executor=tunix.experimental.distributed.runtime.executor.K8sExecutor \
        --process_main=tunix.experimental.examples.common.run_trainer_node.main \
        --worker_id=${TRAINER_ID} \
        --port=${TRAINER_PORT} \
        --mesh_fsdp=${TRAINER_MESH_FSDP} \
        --mesh_tp=${TRAINER_MESH_TP} \
        --trainer_backend=${TRAINER_BACKEND} \
        --model_name=${MODEL_NAME} \
        --model_id=${MODEL_ID} \
        --model_dir=${MODEL_DIR} \
        --tokenizer_path=${TOKENIZER_PATH} \
        --max_prompt_length=${MAX_PROMPT_LENGTH} \
        --max_response_length=${MAX_RESPONSE_LENGTH} \
        --mini_batch_size=${MINI_BATCH_SIZE} \
        --num_generations=${NUM_GENERATIONS} \
        --train_micro_batch_size=${TRAIN_MICRO_BATCH_SIZE} \
        --model_dtype=${MODEL_DTYPE} \
        --eval_every_n_steps=${EVAL_EVERY_N_STEPS} \
        ${opt_chain_flags} \
        --learning_rate=${LEARNING_RATE} \
        --lora_rank=${LORA_RANK} \
        --lora_alpha=${LORA_ALPHA} \
        --sampler=${SAMPLER} \
        ${lora_args} \
        ${maxtext_args} \
        ${profiler_flags} \
        ${DEBUG:+--debug} \
    " \
    | apply_manifest
}

stop_rollout() {
  if [[ "$DRY_RUN" == "true" ]]; then
    echo "kubectl delete jobset ${ROLLOUT_ID} -n ${K8S_NAMESPACE}"
  else
    kubectl delete jobset "${ROLLOUT_ID}" -n "${K8S_NAMESPACE}" --ignore-not-found
  fi
}

start_rollout() {
  local maxtext_args=""
  if [[ "${TRAINER_BACKEND}" == "maxtext" ]]; then
    maxtext_args=" \
      --maxtext_model_name=${MAXTEXT_MODEL_NAME} \
      ${ROLLOUT_MAXTEXT_ATTENTION:+--maxtext_attention=${ROLLOUT_MAXTEXT_ATTENTION}} \
    "
  fi
  local vllm_args=""
  if [[ "$SAMPLER" == "vllm" ]]; then
    vllm_args="\
    --tensor_parallel_size=${ROLLOUT_MESH_TP} \
    ${PREFUSE_MOE_WEIGHTS:+--prefuse_moe_weights=${PREFUSE_MOE_WEIGHTS}} \
    ${ENABLE_PREFIX_CACHING:+--enable_prefix_caching=${ENABLE_PREFIX_CACHING}} \
    "
  fi
  local lora_args=""
  if [[ "${USE_LORA}" == "1" || "${USE_LORA}" == "true" || "${USE_LORA}" == "True" ]]; then
    lora_args="--use_lora"
  fi
  local sandbox_env=""
  if [[ "$USE_AGENT_SANDBOX" == "1" || "$USE_AGENT_SANDBOX" == "true" || "$USE_AGENT_SANDBOX" == "True" ]]; then
    sandbox_env="NAMESPACE=\"${SANDBOX_NAMESPACE}\" ${SANDBOX_NODE_SELECTOR_KEY:+NODE_SELECTOR_KEY=\"${SANDBOX_NODE_SELECTOR_KEY}\"} ${SANDBOX_NODE_SELECTOR_VAL:+NODE_SELECTOR_VAL=\"${SANDBOX_NODE_SELECTOR_VAL}\"}"
  fi
  if ! is_jax_cache_disabled && [[ -n "${ROLLOUT_JAX_CACHE_GCS_DIR}" ]]; then
    echo "[launcher] Rollout JAX cache GCS: ${ROLLOUT_JAX_CACHE_GCS_DIR} (save=${SAVE_JAX_CACHE:-true})" >&2
  fi
  local jax_cache_env=" ROLLOUT_TPU_SLICE=\"${ROLLOUT_TPU_SLICE}\""
  if is_jax_cache_disabled; then
    jax_cache_env+=" DISABLE_JAX_CACHE=1"
  else
    if [[ -n "${ROLLOUT_JAX_CACHE_GCS_DIR}" ]]; then
      jax_cache_env+=" ROLLOUT_JAX_CACHE_GCS_DIR=\"${ROLLOUT_JAX_CACHE_GCS_DIR}\""
    fi
    if [[ -n "${JAX_CACHE_GCS_DIR}" ]]; then
      jax_cache_env+=" JAX_CACHE_GCS_DIR=\"${JAX_CACHE_GCS_DIR}\""
    fi
    if [[ -n "${JAX_CACHE_BUCKET}" ]]; then
      jax_cache_env+=" JAX_CACHE_BUCKET=\"${JAX_CACHE_BUCKET}\""
    fi
    if [[ -n "${BUCKET}" ]]; then
      jax_cache_env+=" BUCKET=\"${BUCKET}\""
    fi
    if [[ -n "${MAXTEXT_OUTPUT_DIR}" ]]; then
      jax_cache_env+=" MAXTEXT_OUTPUT_DIR=\"${MAXTEXT_OUTPUT_DIR}\""
    fi
    if [[ -n "${SAVE_JAX_CACHE}" ]]; then
      jax_cache_env+=" SAVE_JAX_CACHE=\"${SAVE_JAX_CACHE}\""
    fi
  fi
  local ray_head_pre_init_cmd="export JAX_COMPILATION_CACHE_DIR=\"${LOCAL_JAX_CACHE_DIR}\" VLLM_XLA_CACHE_PATH=\"${LOCAL_JAX_CACHE_DIR}\" && mkdir -p \"${LOCAL_JAX_CACHE_DIR}\""
  local ray_worker_pre_init_cmd="${ray_head_pre_init_cmd}"
  if ! is_jax_cache_disabled && [[ -n "${ROLLOUT_JAX_CACHE_GCS_DIR}" ]]; then
    ray_worker_pre_init_cmd+=" && (python3 -m tunix.experimental.common.gcs_cache download \"${LOCAL_JAX_CACHE_DIR}\" \"${ROLLOUT_JAX_CACHE_GCS_DIR}\" || true)"
  fi
  "$PYTHON" tunix/experimental/distributed/deployment/yaml_generator.py \
    tunix/experimental/distributed/deployment/yamls/jobset.tpu.yaml \
    --jobset_name="${ROLLOUT_ID}" \
    --namespace="${K8S_NAMESPACE}" \
    ${KUEUE_QUEUE_NAME:+--queue_name="${KUEUE_QUEUE_NAME}"} \
    --tpu_slice=${ROLLOUT_TPU_SLICE} \
    --worker_container_image="${TUNIX_IMAGE}" \
    --worker_container_port="${ROLLOUT_PORT}" \
    --ray_head_pre_init_command="${ray_head_pre_init_cmd}" \
    --ray_worker_pre_init_command="${ray_worker_pre_init_cmd}" \
    --worker_startup_command=" \
      SKIP_JAX_PRECOMPILE=${ROLLOUT_SKIP_JAX_PRECOMPILE} VERIFY_WEIGHTS=${VERIFY_WEIGHTS} ${sandbox_env} ${ROLLOUT_USE_BATCHED_RPA:+USE_BATCHED_RPA_KERNEL=1}${jax_cache_env} python -m tunix.experimental.distributed.runtime.main \
        --discovery_addrs=${ORCHESTRATOR_ID}:${ORCHESTRATOR_PORT} \
        --process_executor=tunix.experimental.distributed.runtime.executor.K8sExecutor \
        --process_main=tunix.experimental.examples.common.run_rollout_node.main \
        --worker_id=${ROLLOUT_ID} \
        --port=${ROLLOUT_PORT} \
        --model_id=${MODEL_ID} \
        --model_dir=${MODEL_DIR} \
        --model_name=${MODEL_NAME} \
        --tokenizer_path=${TOKENIZER_PATH} \
        --mesh_fsdp=${ROLLOUT_MESH_FSDP} \
        --mesh_tp=${ROLLOUT_MESH_TP} \
        --max_prompt_length=${MAX_PROMPT_LENGTH} \
        --max_response_length=${MAX_RESPONSE_LENGTH} \
        --sampler=${SAMPLER} \
        --lora_rank=${LORA_RANK} \
        --lora_alpha=${LORA_ALPHA} \
        --weight_sync_mode=${WEIGHT_SYNC_MODE} \
        --in_flight_weight_updates=${IN_FLIGHT_WEIGHT_UPDATES} \
        --return_routed_experts=${RETURN_ROUTED_EXPERTS} \
        --routed_experts_encode_mode=${ROUTED_EXPERTS_ENCODE_MODE} \
        ${ROUTED_EXPERTS_NUM_EXPERTS:+--routed_experts_num_experts=${ROUTED_EXPERTS_NUM_EXPERTS}} \
        --registry_module=tunix.experimental.examples.deepswe_dist.deepswe \
        --env_name=deepswe_env \
        --agent_name=deepswe_agent \
        --max_concurrency=${ROLLOUT_MAX_CONCURRENCY} \
        --enable_trajectory_store=${TRAJECTORY_STORE_ENABLED} \
        --trajectory_store_backend=${TRAJECTORY_STORE_BACKEND} \
        --trajectory_store_dir=\"${TRAJECTORY_STORE_DIR}\" \
        --run_id=\"${RUN_ID}\" \
        ${lora_args} \
        ${maxtext_args} \
        ${vllm_args} \
        ${DEBUG:+--debug} \
    " \
    | apply_manifest
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --command)
      COMMAND="$2"
      shift 2
      ;;
    --command=*)
      COMMAND="${1#*=}"
      shift
      ;;
    --image)
      TUNIX_IMAGE="$2"
      shift 2
      ;;
    --image=*)
      TUNIX_IMAGE="${1#*=}"
      shift
      ;;
    --dry-run|--render)
      DRY_RUN=true
      shift
      ;;
    *)
      shift
      ;;
  esac
done

if [[ "$DRY_RUN" != "true" ]]; then
  if [[ -f tunix/experimental/examples/common/enter_kube_context.sh ]]; then
    source tunix/experimental/examples/common/enter_kube_context.sh
  elif [[ -f "$(dirname "${BASH_SOURCE[0]}")/../common/enter_kube_context.sh" ]]; then
    source "$(dirname "${BASH_SOURCE[0]}")/../common/enter_kube_context.sh"
  fi
fi

if [[ -z "$TUNIX_IMAGE" ]]; then
  echo "Error: no image set. Build one with tunix, maxtext, and" \
       "tpu-inference installed, then pass it via TUNIX_IMAGE=... or" \
       "--image=..."
  exit 1
fi

if [[ "$COMMAND" == "start" ]]; then
  stop_orchestrator
  stop_trainer
  stop_rollout
  start_orchestrator
  start_trainer
  start_rollout
elif [[ "$COMMAND" == "stop" ]]; then
  stop_orchestrator
  stop_trainer
  stop_rollout
elif [[ "$COMMAND" == "orchestrator" ]]; then
  stop_orchestrator; start_orchestrator
elif [[ "$COMMAND" == "trainer" ]]; then
  stop_trainer; start_trainer
elif [[ "$COMMAND" == "rollout" ]]; then
  stop_rollout; start_rollout
else
  echo "Error: Invalid command '$COMMAND'. Available commands: 'start', 'stop', 'orchestrator', 'trainer', 'rollout'."
  exit 1
fi
