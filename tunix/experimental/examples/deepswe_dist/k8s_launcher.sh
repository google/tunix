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
TUNIX_IMAGE=${TUNIX_IMAGE:-us-central1-docker.pkg.dev/cloud-tpu-multipod-dev/yangmu/tunix/tunix_base_image:trellis-demo-0813}

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TUNIX_ROOT="$(cd "${DIR}/../../.." && pwd)"
PYTHON_BIN=${PYTHON_BIN:-python3}
if ! command -v "$PYTHON_BIN" &>/dev/null && command -v python &>/dev/null; then
  PYTHON_BIN="python"
fi
YAML_GENERATOR="${YAML_GENERATOR:-${TUNIX_ROOT}/experimental/distributed/deployment/yaml_generator.py}"
YAML_DIR="${YAML_DIR:-${TUNIX_ROOT}/experimental/distributed/deployment/yamls}"

BOOTSTRAP_CMD="${BOOTSTRAP_CMD:-}"

export MODEL_NAME=${MODEL_NAME:-Qwen3-4B}
export MODEL_ID=${MODEL_ID:-Qwen/Qwen3-4B}
# Must be model-specific: vLLM prioritizes non-empty local snapshot directories,
# which can cause stale config/shape mismatches if shared across models.
export MODEL_DIR=${MODEL_DIR:-artifacts/qwen3_dist_deepswe/models/${MODEL_NAME}}
# Defaults to MODEL_ID so AutoTokenizer downloads directly from HuggingFace
# instead of failing on an initially empty local MODEL_DIR.
export TOKENIZER_PATH=${TOKENIZER_PATH:-${MODEL_ID}}

export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}
export BATCH_SIZE=${BATCH_SIZE:-1}
export NUM_GENERATIONS=${NUM_GENERATIONS:-4}
export MAX_STEPS=${MAX_STEPS:-10}
export MAX_TURNS=${MAX_TURNS:-20}
export TRAIN_MICRO_BATCH_SIZE=${TRAIN_MICRO_BATCH_SIZE:-1}
export MAX_SEQ_TOKEN_PER_TPU=${MAX_SEQ_TOKEN_PER_TPU:-}
export MAX_SEGMENTS_PER_PACKED_ROW=${MAX_SEGMENTS_PER_PACKED_ROW:-}
export COMPUTE_LOGPS_CHUNK_SIZE=${COMPUTE_LOGPS_CHUNK_SIZE:-}

# Set to tunix to run Tunix's PeftTrainer, and maxtext to run MaxText's MaxTextTrainingEngine
export TRAINER_BACKEND=${TRAINER_BACKEND:-tunix}
export MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-$BATCH_SIZE}
export EVAL_EVERY_N_STEPS=${EVAL_EVERY_N_STEPS:-1000000}
export OPT_CHAIN_TYPE=${OPT_CHAIN_TYPE-clip_by_global_norm}
export MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}
export ADAM_B1=${ADAM_B1:-0.9}
export ADAM_B2=${ADAM_B2:-0.999}
export ADAM_EPS=${ADAM_EPS:-1.0e-8}
export WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
export LEARNING_RATE=${LEARNING_RATE:-1e-6}
export SCHEDULE_TYPE=${SCHEDULE_TYPE-warmup_cosine_decay_schedule}
export LR_INIT_VALUE=${LR_INIT_VALUE:-0.0}
export LR_PEAK_VALUE=${LR_PEAK_VALUE:-$LEARNING_RATE}
export LR_END_VALUE=${LR_END_VALUE:-0.0}
export LR_DECAY_STEPS=${LR_DECAY_STEPS:-500}
export WARMUP_STEPS=${WARMUP_STEPS:-$(((LR_DECAY_STEPS + 9) / 10))}
export BETA=${BETA:-0.0}
export EPSILON=${EPSILON:-0.2}
export LORA_RANK=${LORA_RANK:-64}
export LORA_ALPHA=${LORA_ALPHA:-64.0}
export USE_LORA=${USE_LORA:-0}
# Generation sampling parameters, passed to the runner and the reference scorer.
export TEMPERATURE=${TEMPERATURE:-1.0}
export TOP_P=${TOP_P:-1.0}
export TOP_K=${TOP_K:--1}
export DEBUG=${DEBUG:-0}
export USE_ROLLOUT_LOGPS=${USE_ROLLOUT_LOGPS:-true}
export EXACT_TOKEN_CONTINUITY=${EXACT_TOKEN_CONTINUITY:-true}
export SAMPLER=${SAMPLER:-inprocess_vllm}
export WEIGHT_SYNC_MODE=${WEIGHT_SYNC_MODE:-none}
export CHECKPOINT_SAVE_INTERVAL_STEPS=${CHECKPOINT_SAVE_INTERVAL_STEPS:-5}
export CHECKPOINT_MAX_TO_KEEP=${CHECKPOINT_MAX_TO_KEEP:-2}
export CHECKPOINT_ROOT_DIRECTORY=${CHECKPOINT_ROOT_DIRECTORY:-checkpoints}
export OVERLONG_FILTER=${OVERLONG_FILTER:-}
export TRAINABLE_PARAMETERS_MASK=${TRAINABLE_PARAMETERS_MASK:-}

# Optional GRPO algorithm options. Empty, or 0 for the boolean, leaves the
# option at the runner's default, so an unset variable changes nothing.
export EPSILON_HIGH=${EPSILON_HIGH:-}
export LOSS_AGG_MODE=${LOSS_AGG_MODE:-}
export ADVANTAGE_ESTIMATOR=${ADVANTAGE_ESTIMATOR:-}
export OVERLONG_LOSS_MASKING=${OVERLONG_LOSS_MASKING:-0}
export SEQ_LOGPROB_ERROR_THRESHOLD=${SEQ_LOGPROB_ERROR_THRESHOLD:-}
export TIS_TYPE=${TIS_TYPE:-${TRUNCATED_IMPORTANCE_SAMPLING_TYPE:-}}
export TIS_RATIO_MIN=${TIS_RATIO_MIN:-${TRUNCATED_IMPORTANCE_SAMPLING_RATIO_MIN:-}}
export TIS_RATIO=${TIS_RATIO:-${TRUNCATED_IMPORTANCE_SAMPLING_RATIO:-}}
export SAMPLER_IS_LENGTH_BUCKETS=${SAMPLER_IS_LENGTH_BUCKETS:-}
export MAX_STALENESS=${MAX_STALENESS:-}
export TRAJECTORY_GROUP_ORDER=${TRAJECTORY_GROUP_ORDER:-arrival}
export PROFILER_STEPS=${PROFILER_STEPS:-0}
export SKIP_FIRST_N_PROFILER_STEPS=${SKIP_FIRST_N_PROFILER_STEPS:-}
export PROFILER_PERIOD=${PROFILER_PERIOD:-}
export ROLLOUT_FREE_KV_CACHE=${ROLLOUT_FREE_KV_CACHE:-false}

# DeepSWE dataset and environment configuration
export DATASET_NAME=${DATASET_NAME:-R2E-Gym/R2E-Gym-Subset}
export DATASET_PATH=${DATASET_PATH:-}
export DATASET_SPLIT=${DATASET_SPLIT:-train}
export DATASET_CACHE_DIR=${DATASET_CACHE_DIR:-artifacts/qwen3_dist_deepswe/dataset_cache}
export SHUFFLE=${SHUFFLE:-true}
export SEED=${SEED:-42}
export ENV_BACKEND=${ENV_BACKEND:-kubernetes}
export SCAFFOLD=${SCAFFOLD:-r2egym}
export USE_AGENT_SANDBOX=${USE_AGENT_SANDBOX:-1}
export SANDBOX_NAMESPACE=${SANDBOX_NAMESPACE:-rl-tunix-swebench}
export SANDBOX_NODE_SELECTOR_KEY=${SANDBOX_NODE_SELECTOR_KEY:-}
export SANDBOX_NODE_SELECTOR_VAL=${SANDBOX_NODE_SELECTOR_VAL:-}
export SANDBOX_TOLERATIONS=${SANDBOX_TOLERATIONS:-}
export IMAGE_REWRITE_PREFIX=${IMAGE_REWRITE_PREFIX:-}
export STEP_TIMEOUT_SECS=${STEP_TIMEOUT_SECS:-1800}
export REWARD_TIMEOUT_SECS=${REWARD_TIMEOUT_SECS:-1800}
export ROLLOUT_MAX_CONCURRENCY=${ROLLOUT_MAX_CONCURRENCY:-64}
export MAX_CONCURRENCY=${MAX_CONCURRENCY:-${ROLLOUT_MAX_CONCURRENCY}}
export FLUSH_EVERY_N_STEPS=${FLUSH_EVERY_N_STEPS:-1}
export MAX_WARMPOOL_REPLICAS=${MAX_WARMPOOL_REPLICAS:-4}
export ENABLE_PATHWAYS_PERSISTENCE=${ENABLE_PATHWAYS_PERSISTENCE:-0}
export CHECKPOINT_ASYNC=${CHECKPOINT_ASYNC:-true}
export CKPT_D2H_CONCURRENT_GB=${CKPT_D2H_CONCURRENT_GB:-8}

export TRAINER_MESH_TP=${TRAINER_MESH_TP:-1}
export TRAINER_MESH_EXPERT=${TRAINER_MESH_EXPERT:-1}
# Context-parallel degree for the trainer; shards the sequence axis.
export TRAINER_MESH_CONTEXT=${TRAINER_MESH_CONTEXT:-1}
# Rollout jobset template. Defaults to the single-host TPU jobset, which is what
# a 4-chip-per-replica rollout wants. A multihost rollout -- e.g. 397B at 16
# chips / 4 hosts per replica -- needs the Ray-backed template instead.
export ROLLOUT_JOBSET_YAML=${ROLLOUT_JOBSET_YAML:-jobset.tpu.yaml}
export ROLLOUT_MESH_TP=${ROLLOUT_MESH_TP:-2}
export ROLLOUT_MESH_FSDP=${ROLLOUT_MESH_FSDP:-1}
export ROLLOUT_MESH_EXPERT=${ROLLOUT_MESH_EXPERT:-1}
# Optional: enable experimental batched-RPA attention kernel for rollout.
export ROLLOUT_USE_BATCHED_RPA=${ROLLOUT_USE_BATCHED_RPA:-}

# MaxText configuration: only consulted when TRAINER_BACKEND=maxtext.
export REMAT_POLICY=${REMAT_POLICY:-decoder}
source "${DIR}/../common/maxtext_config.sh"

# Logs source/destination Raiden tensor checksums on both the trainer and
# rollout sides during weight sync, for cross-verification of a real run.
export VERIFY_WEIGHTS=${VERIFY_WEIGHTS:-false}

export WANDB_PROJECT=${WANDB_PROJECT:-trellis-deepswe}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-}
export WANDB_API_KEY=${WANDB_API_KEY:-}
export WANDB_ENTITY=${WANDB_ENTITY:-}
export LOG_DIR=${LOG_DIR:-}
export TRAJECTORY_LOG_DIR=${TRAJECTORY_LOG_DIR:-}
export RCP_LOGGING=${RCP_LOGGING:-false}
export METRIC_LOGGER_DIR=${METRIC_LOGGER_DIR:-}
export TARGET_ACCURACY=${TARGET_ACCURACY:-0.69}
export TRAJECTORY_STORE_ROOT_DIR=${TRAJECTORY_STORE_ROOT_DIR:-${TRAJECTORY_STORE_ROOT:-}}
export EOS_TOKENS=${EOS_TOKENS:-}

# Rollout Worker environment flags
export NUM_PRECOMPILE_WORKERS=${NUM_PRECOMPILE_WORKERS:-}
export NEW_MODEL_DESIGN=${NEW_MODEL_DESIGN:-}
export ATTN_BUCKETIZED_NUM_REQS=${ATTN_BUCKETIZED_NUM_REQS:-}
export ATTN_CUSTOM_NUM_REQS_BUCKETS=${ATTN_CUSTOM_NUM_REQS_BUCKETS:-}
export ONEHOT_MOE_PERMUTE_THRESHOLD=${ONEHOT_MOE_PERMUTE_THRESHOLD:-}
export VLLM_MOE_CHUNK_SIZE=${VLLM_MOE_CHUNK_SIZE:-}
export SLICE_ROPE_CACHE=${SLICE_ROPE_CACHE:-}
export DP_SCHED_BATCH_PREFILL=${DP_SCHED_BATCH_PREFILL:-}
export LIBTPU_INIT_ARGS=${LIBTPU_INIT_ARGS:-}
export VLLM_ENABLE_V1_MULTIPROCESSING=${VLLM_ENABLE_V1_MULTIPROCESSING:-}
export ROLLOUT_ENV_FLAGS=${ROLLOUT_ENV_FLAGS:-}

JOB_PREFIX=${JOB_PREFIX:-$USER}
export ORCHESTRATOR_ID=${ORCHESTRATOR_ID:-$JOB_PREFIX-orch}
export ORCHESTRATOR_PORT=${ORCHESTRATOR_PORT:-20000}

export ROLLOUT_ID=${ROLLOUT_ID:-$JOB_PREFIX-roll}
export ROLLOUT_PORT=${ROLLOUT_PORT:-20001}

export TRAINER_ID=${TRAINER_ID:-$JOB_PREFIX-train}
export TRAINER_PORT=${TRAINER_PORT:-20002}

export CPU_MACHINE=${CPU_MACHINE:-n2-standard-64}
export GCS_SCRATCH_LOCATION=${GCS_SCRATCH_LOCATION:-gs://cloud-pathways-staging/tmp}
export PATHWAYS_SERVER_IMAGE=${PATHWAYS_SERVER_IMAGE:-us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/datenglin/unsanitized_server:raiden_20260908}
export PATHWAYS_PROXY_IMAGE=${PATHWAYS_PROXY_IMAGE:-us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/datenglin/unsanitized_proxy_server:raiden_20260908}
export RAIDEN_DEVICES_PER_HOST=${RAIDEN_DEVICES_PER_HOST:-4}
export USE_WEIGHT_CONVERTER=${USE_WEIGHT_CONVERTER:-true}
export PREFUSE_MOE_WEIGHTS=${PREFUSE_MOE_WEIGHTS:-true}
export TRAINER_PREFUSE_MOE_WEIGHTS=${TRAINER_PREFUSE_MOE_WEIGHTS:-false}
export ROLLOUT_PREFUSE_MOE_WEIGHTS=${ROLLOUT_PREFUSE_MOE_WEIGHTS:-true}
export ENABLE_PREFIX_CACHING=${ENABLE_PREFIX_CACHING:-false}
export RETURN_ROUTED_EXPERTS=${RETURN_ROUTED_EXPERTS:-false}

export TRAINER_JOBSET_YAML=${TRAINER_JOBSET_YAML:-jobset.pathways.yaml}
export TRAINER_TPU_SLICE=${TRAINER_TPU_SLICE:-tpuv5:2x2x2}
export TRAINER_MESH_FSDP=${TRAINER_MESH_FSDP:-8}
export ROLLOUT_TPU_SLICE=${ROLLOUT_TPU_SLICE:-tpuv5:2x2x1}
export ROLLOUT_REPLICAS=${ROLLOUT_REPLICAS:-1}
export KUEUE_QUEUE=${KUEUE_QUEUE:-}
export K8S_NAMESPACE=${K8S_NAMESPACE:-default}
export KUEUE_QUEUE_NAME=${KUEUE_QUEUE_NAME:-${KUEUE_QUEUE:-${QUEUE_NAME:-}}}
export PRIORITY_CLASS=${PRIORITY_CLASS:-medium}

export TRAINER_EXTRA_ENV=${TRAINER_EXTRA_ENV:-}
export ORCHESTRATOR_EXTRA_ENV=${ORCHESTRATOR_EXTRA_ENV:-}
export ROLLOUT_EXTRA_ENV=${ROLLOUT_EXTRA_ENV:-}
export DRY_RUN=${DRY_RUN:-false}

apply_manifest() {
  local priority_sed="s/priorityClassName: [a-zA-Z0-9_-]\+/priorityClassName: ${PRIORITY_CLASS:-medium}/g"
  local filter
  if [[ -n "${KUEUE_QUEUE}" ]]; then
    filter=(sed -e "${priority_sed}" -e "s|^metadata:|metadata:\n  labels:\n    kueue.x-k8s.io/queue-name: ${KUEUE_QUEUE}|")
  else
    filter=(sed -e "${priority_sed}")
  fi

  if [[ "$DRY_RUN" == "true" ]]; then
    echo "---"
    "${filter[@]}"
  else
    "${filter[@]}" | kubectl apply -f -
  fi
}

if [[ "$BETA" != "0" && "$BETA" != "0.0" ]]; then
  echo "Error: this first DeepSWE distributed launcher only wires trainer+rollout."
  echo "Use BETA=0.0 until the reference inference worker is added."
  exit 1
fi

stop_orchestrator() {
  if [[ "$DRY_RUN" == "true" ]]; then
    echo "kubectl delete jobset ${ORCHESTRATOR_ID} -n ${K8S_NAMESPACE}"
    echo "kubectl delete workload -l jobset.sigs.k8s.io/jobset-name=${ORCHESTRATOR_ID} -n ${K8S_NAMESPACE}"
    if [[ "${USE_AGENT_SANDBOX}" == "1" || "${USE_AGENT_SANDBOX}" == "true" || "${USE_AGENT_SANDBOX}" == "True" ]]; then
      echo "kubectl delete sandboxwarmpools -n ${SANDBOX_NAMESPACE} -l app.kubernetes.io/created-by=${ORCHESTRATOR_ID} --ignore-not-found=true"
      echo "kubectl delete sandboxtemplates -n ${SANDBOX_NAMESPACE} -l app.kubernetes.io/created-by=${ORCHESTRATOR_ID} --ignore-not-found=true"
      echo "kubectl delete sandboxclaims -n ${SANDBOX_NAMESPACE} -l app.kubernetes.io/created-by=${ORCHESTRATOR_ID} --ignore-not-found=true"
      echo "kubectl delete pods -n ${SANDBOX_NAMESPACE} -l app.kubernetes.io/created-by=${ORCHESTRATOR_ID} --force --grace-period=0 --ignore-not-found=true"
    fi
  else
    kubectl delete jobset "${ORCHESTRATOR_ID}" -n "${K8S_NAMESPACE}" --ignore-not-found=true
    kubectl delete workload -l "jobset.sigs.k8s.io/jobset-name=${ORCHESTRATOR_ID}" -n "${K8S_NAMESPACE}" --ignore-not-found=true 2>/dev/null || true
    if [[ "${USE_AGENT_SANDBOX}" == "1" || "${USE_AGENT_SANDBOX}" == "true" || "${USE_AGENT_SANDBOX}" == "True" ]]; then
      echo "Cleaning up sandboxes and warmpools for ${ORCHESTRATOR_ID} in ${SANDBOX_NAMESPACE}..."
      kubectl delete sandboxwarmpools -n "${SANDBOX_NAMESPACE}" -l "app.kubernetes.io/created-by=${ORCHESTRATOR_ID}" --ignore-not-found=true 2>/dev/null || true
      kubectl delete sandboxtemplates -n "${SANDBOX_NAMESPACE}" -l "app.kubernetes.io/created-by=${ORCHESTRATOR_ID}" --ignore-not-found=true 2>/dev/null || true
      kubectl delete sandboxclaims -n "${SANDBOX_NAMESPACE}" -l "app.kubernetes.io/created-by=${ORCHESTRATOR_ID}" --ignore-not-found=true 2>/dev/null || true
      kubectl delete pods -n "${SANDBOX_NAMESPACE}" -l "app.kubernetes.io/created-by=${ORCHESTRATOR_ID}" --force --grace-period=0 --ignore-not-found=true 2>/dev/null || true
    fi
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
  local sandbox_env=""
  local sandbox_arg=""
  if [[ "${USE_AGENT_SANDBOX}" == "1" || "${USE_AGENT_SANDBOX}" == "true" || "${USE_AGENT_SANDBOX}" == "True" ]]; then
    sandbox_env="NAMESPACE=\"${SANDBOX_NAMESPACE}\" ${SANDBOX_NODE_SELECTOR_KEY:+NODE_SELECTOR_KEY=\"${SANDBOX_NODE_SELECTOR_KEY}\"} ${SANDBOX_NODE_SELECTOR_VAL:+NODE_SELECTOR_VAL=\"${SANDBOX_NODE_SELECTOR_VAL}\"} ${IMAGE_REWRITE_PREFIX:+IMAGE_REWRITE_PREFIX=\"${IMAGE_REWRITE_PREFIX}\"} ${JOB_PREFIX:+JOB_PREFIX=\"${JOB_PREFIX}\"} ${POOL_NAME_FORMAT:+POOL_NAME_FORMAT=\"${POOL_NAME_FORMAT}\"} ${TEMPLATE_NAME_PREFIX:+TEMPLATE_NAME_PREFIX=\"${TEMPLATE_NAME_PREFIX}\"} ${SANDBOX_TOLERATIONS:+SANDBOX_TOLERATIONS=\"${SANDBOX_TOLERATIONS//\"/\\\"}\"}"
    sandbox_arg="--use_agent_sandbox"
  elif [[ -n "${IMAGE_REWRITE_PREFIX}" ]]; then
    sandbox_env="IMAGE_REWRITE_PREFIX=\"${IMAGE_REWRITE_PREFIX}\""
  fi
  local overlong_arg=""
  if [[ "${OVERLONG_LOSS_MASKING}" == "1" || "${OVERLONG_LOSS_MASKING}" == "true" || "${OVERLONG_LOSS_MASKING}" == "True" ]]; then
    overlong_arg="--overlong_loss_masking"
  fi
  local overlong_filter_arg=""
  if [[ "${OVERLONG_FILTER}" == "1" || "${OVERLONG_FILTER}" == "true" || "${OVERLONG_FILTER}" == "True" ]]; then
    overlong_filter_arg="--overlong_filter"
  elif [[ "${OVERLONG_FILTER}" == "0" || "${OVERLONG_FILTER}" == "false" || "${OVERLONG_FILTER}" == "False" ]]; then
    overlong_filter_arg="--no-overlong_filter"
  fi
  local debug_arg=""
  if [[ "${DEBUG}" == "1" || "${DEBUG}" == "true" || "${DEBUG}" == "True" ]]; then
    debug_arg="--debug"
  fi
  local rcp_arg=""
  if [[ "${RCP_LOGGING}" == "1" || "${RCP_LOGGING}" == "true" || "${RCP_LOGGING}" == "True" ]]; then
    rcp_arg="--rcp_logging"
  fi

  "$PYTHON_BIN" "$YAML_GENERATOR" \
    "${YAML_DIR}/jobset.cpu.yaml" \
    --jobset_name="${ORCHESTRATOR_ID}" \
    --namespace="${K8S_NAMESPACE}" \
    ${KUEUE_QUEUE_NAME:+--queue_name="${KUEUE_QUEUE_NAME}"} \
    --cpu_machine=${CPU_MACHINE} \
    --worker_container_image="${TUNIX_IMAGE}" \
    --worker_container_port="${ORCHESTRATOR_PORT}" \
    --worker_startup_command=" \
      ORCHESTRATOR_ID=\"${ORCHESTRATOR_ID}\" \
      ${sandbox_env} \
      ${SCAFFOLD:+SCAFFOLD=\"${SCAFFOLD}\"} \
      ${HF_TOKEN:+HF_TOKEN=\"${HF_TOKEN}\"} \
      ${WANDB_API_KEY:+WANDB_API_KEY=\"${WANDB_API_KEY}\"} \
      ${WANDB_ENTITY:+WANDB_ENTITY=\"${WANDB_ENTITY}\"} \
      WANDB_PROJECT=\"${WANDB_PROJECT}\" \
      WANDB_RUN_NAME=\"${WANDB_RUN_NAME}\" \
      ROLLOUT_WORKERS=\"${ROLLOUT_WORKERS:-${ROLLOUT_REPLICAS:-1}}\" \
      EPISODE_TIMEOUT_SECS=\"${EPISODE_TIMEOUT_SECS:-5400}\" \
      ${LOG_DIR:+LOG_DIR=\"${LOG_DIR}\"} \
      ${TRAJECTORY_LOG_DIR:+TRAJECTORY_LOG_DIR=\"${TRAJECTORY_LOG_DIR}\"} \
      PYTHONUNBUFFERED=1 \
      TUNIX_IS_INTERNAL_ENV=false \
      ${BOOTSTRAP_CMD} \
      ${ORCHESTRATOR_EXTRA_ENV:+${ORCHESTRATOR_EXTRA_ENV} }\
      python -m tunix.experimental.distributed.runtime.main \
        --discovery_id=${ORCHESTRATOR_ID} \
        --discovery_port=${ORCHESTRATOR_PORT} \
        --process_main=tunix.experimental.examples.deepswe_dist.run_deepswe_dist.main \
        --model_id=${MODEL_ID} \
        --tokenizer_path=${TOKENIZER_PATH} \
        --batch_size=${BATCH_SIZE} \
        --mini_batch_size=${MINI_BATCH_SIZE} \
        --num_generations=${NUM_GENERATIONS} \
        --rollout_replicas=${ROLLOUT_WORKERS:-${ROLLOUT_REPLICAS:-1}} \
        --temperature=${TEMPERATURE} \
        --top_p=${TOP_P} \
        --top_k=${TOP_K} \
        --max_steps=${MAX_STEPS} \
        --max_turns=${MAX_TURNS} \
        --max_prompt_length=${MAX_PROMPT_LENGTH} \
        --max_response_length=${MAX_RESPONSE_LENGTH} \
        --train_micro_batch_size=${TRAIN_MICRO_BATCH_SIZE} \
        --beta=${BETA} \
        --epsilon=${EPSILON} \
        ${EPSILON_HIGH:+--epsilon_high=${EPSILON_HIGH}} \
        ${LOSS_AGG_MODE:+--loss_agg_mode=${LOSS_AGG_MODE}} \
        ${ADVANTAGE_ESTIMATOR:+--advantage_estimator=${ADVANTAGE_ESTIMATOR}} \
        ${overlong_arg} \
        ${overlong_filter_arg} \
        ${SEQ_LOGPROB_ERROR_THRESHOLD:+--seq_logprob_error_threshold=${SEQ_LOGPROB_ERROR_THRESHOLD}} \
        ${TIS_TYPE:+--truncated_importance_sampling_type=${TIS_TYPE}} \
        ${TIS_RATIO_MIN:+--truncated_importance_sampling_ratio_min=${TIS_RATIO_MIN}} \
        ${TIS_RATIO:+--truncated_importance_sampling_ratio=${TIS_RATIO}} \
        ${SAMPLER_IS_LENGTH_BUCKETS:+--sampler_is_length_buckets=${SAMPLER_IS_LENGTH_BUCKETS}} \
        --dataset_name=${DATASET_NAME} \
        --dataset_split=${DATASET_SPLIT} \
        ${DATASET_CACHE_DIR:+--dataset_cache_dir=${DATASET_CACHE_DIR}} \
        --seed=${SEED} \
        --env_backend=${ENV_BACKEND} \
        --scaffold=${SCAFFOLD} \
        --step_timeout_secs=${STEP_TIMEOUT_SECS} \
        --reward_timeout_secs=${REWARD_TIMEOUT_SECS} \
        ${EPISODE_TIMEOUT_SECS:+--episode_timeout_secs=${EPISODE_TIMEOUT_SECS}} \
        ${LOG_DIR:+--log_dir=\"${LOG_DIR}\"} \
        ${TRAJECTORY_LOG_DIR:+--trajectory_log_dir=\"${TRAJECTORY_LOG_DIR}\"} \
        ${TRAJECTORY_STORE_ROOT_DIR:+--trajectory_store_root_dir=\"${TRAJECTORY_STORE_ROOT_DIR}\"} \
        --flush_every_n_steps=${FLUSH_EVERY_N_STEPS} \
        --wandb_project=\"${WANDB_PROJECT}\" \
        --wandb_run_name=\"${WANDB_RUN_NAME}\" \
        --weight_sync_mode=${WEIGHT_SYNC_MODE} \
        --stop_workers_on_exit \
        ${MAX_WARMPOOL_REPLICAS:+--max_warmpool_replicas=${MAX_WARMPOOL_REPLICAS}} \
        ${MAX_CONCURRENCY:+--max_concurrency=${MAX_CONCURRENCY}} \
        ${MAX_STALENESS:+--max_staleness=${MAX_STALENESS}} \
        ${TRAJECTORY_GROUP_ORDER:+--trajectory_group_order=${TRAJECTORY_GROUP_ORDER}} \
        $([[ "${USE_ROLLOUT_LOGPS}" == "false" || "${USE_ROLLOUT_LOGPS}" == "False" || "${USE_ROLLOUT_LOGPS}" == "0" ]] && echo --no-use_rollout_logps || echo --use_rollout_logps) \
        $([[ "${EXACT_TOKEN_CONTINUITY}" == "false" || "${EXACT_TOKEN_CONTINUITY}" == "False" || "${EXACT_TOKEN_CONTINUITY}" == "0" ]] && echo --no-exact_token_continuity || echo --exact_token_continuity) \
        ${dataset_args} \
        ${shuffle_arg} \
        ${sandbox_arg} \
        ${IMAGE_REWRITE_PREFIX:+--image_rewrite_prefix=${IMAGE_REWRITE_PREFIX}} \
        ${MAX_SEQ_TOKEN_PER_TPU:+--max_seq_token_per_tpu=${MAX_SEQ_TOKEN_PER_TPU}} \
        ${MAX_SEGMENTS_PER_PACKED_ROW:+--max_segments_per_packed_row=${MAX_SEGMENTS_PER_PACKED_ROW}} \
        ${TRAINER_MESH_FSDP:+--trainer_fsdp=${TRAINER_MESH_FSDP}} \
        $( [[ "${TRAINER_MESH_EXPERT:-1}" -gt 1 ]] && echo "--trainer_expert=${TRAINER_MESH_EXPERT}" ) \
        ${RPC_TIMEOUT_S:+--rpc_timeout_s=${RPC_TIMEOUT_S}} \
        ${TRAINABLE_PARAMETERS_MASK:+--trainable_parameters_mask='${TRAINABLE_PARAMETERS_MASK}'} \
        --eval_every_n_steps=${EVAL_EVERY_N_STEPS} \
        --learning_rate=${LEARNING_RATE} \
        --b1=${ADAM_B1} \
        --b2=${ADAM_B2} \
        --weight_decay=${WEIGHT_DECAY} \
        --max_grad_norm=${MAX_GRAD_NORM} \
        --train_mesh_tp=${TRAINER_MESH_TP} \
        --train_mesh_expert=${TRAINER_MESH_EXPERT} \
        --rollout_mesh_tp=${ROLLOUT_MESH_TP} \
        --rollout_mesh_expert=${ROLLOUT_MESH_EXPERT:-1} \
        --rollout_engine=${SAMPLER} \
        --tpu_topology="${TRAINER_TPU_SLICE}+${ROLLOUT_TPU_SLICE}" \
        --target_accuracy=${TARGET_ACCURACY} \
        ${METRIC_LOGGER_DIR:+--metric_logger_dir="${METRIC_LOGGER_DIR}"} \
        ${rcp_arg} \
        ${debug_arg} \
    " \
    | apply_manifest
}

stop_trainer() {
  if [[ "$DRY_RUN" == "true" ]]; then
    echo "kubectl delete jobset ${TRAINER_ID} -n ${K8S_NAMESPACE}"
    echo "kubectl delete workload -l jobset.sigs.k8s.io/jobset-name=${TRAINER_ID} -n ${K8S_NAMESPACE}"
  else
    kubectl delete jobset "${TRAINER_ID}" -n "${K8S_NAMESPACE}" --ignore-not-found=true
    kubectl delete workload -l "jobset.sigs.k8s.io/jobset-name=${TRAINER_ID}" -n "${K8S_NAMESPACE}" --ignore-not-found=true 2>/dev/null || true
  fi
}

start_trainer() {
  maxtext_require_ckpt
  local maxtext_args
  maxtext_args="$(maxtext_trainer_flags)"
  local opt_chain_args=""
  if [[ -n "${OPT_CHAIN_TYPE}" ]]; then
    opt_chain_args=" \
      --optimizer_opt_chain_type=${OPT_CHAIN_TYPE} \
      --optimizer_chain_kwargs=\"{'max_norm': ${MAX_GRAD_NORM}}\" \
    "
  fi
  local lora_args=""
  if [[ "${USE_LORA}" == "1" || "${USE_LORA}" == "true" || "${USE_LORA}" == "True" ]]; then
    lora_args="--use_lora"
  fi
  local debug_arg=""
  if [[ "${DEBUG}" == "1" || "${DEBUG}" == "true" || "${DEBUG}" == "True" ]]; then
    debug_arg="--debug"
  fi
  local raiden_env=""
  if [[ "${WEIGHT_SYNC_MODE}" == "raiden" ]]; then
    if [[ "${TRAINER_JOBSET_YAML}" == "jobset.pathways.yaml" ]]; then
      raiden_env+=" RAIDEN_USE_FFI=1"
    fi
  fi
  "$PYTHON_BIN" "$YAML_GENERATOR" \
    "${YAML_DIR}/${TRAINER_JOBSET_YAML}" \
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
    --pathways_gcs_scratch_location=${GCS_SCRATCH_LOCATION} \
    --worker_container_image="${TUNIX_IMAGE}" \
    --worker_container_port="${TRAINER_PORT}" \
    --worker_startup_command=" \
      PYTHONUNBUFFERED=1 \
      TUNIX_IS_INTERNAL_ENV=false \
      ${BOOTSTRAP_CMD} \
      ${HF_TOKEN:+HF_TOKEN=\"${HF_TOKEN}\"} \
      ENABLE_PATHWAYS_PERSISTENCE=${ENABLE_PATHWAYS_PERSISTENCE} \
      ${CHECKPOINT_ASYNC:+CHECKPOINT_ASYNC=${CHECKPOINT_ASYNC}} \
      ${CKPT_D2H_CONCURRENT_GB:+CKPT_D2H_CONCURRENT_GB=${CKPT_D2H_CONCURRENT_GB}} \
      ${TRAINER_MAXTEXT_ATTENTION:+TRAINER_MAXTEXT_ATTENTION=\"${TRAINER_MAXTEXT_ATTENTION}\"} \
      ${raiden_env} \
      ${MAXTEXT_EXTRA_FLAGS:+MAXTEXT_EXTRA_FLAGS=\"${MAXTEXT_EXTRA_FLAGS}\"} \
      ${TRAINER_EXTRA_ENV:+${TRAINER_EXTRA_ENV}} \
      RAIDEN_DEVICES_PER_HOST=${RAIDEN_DEVICES_PER_HOST} \
      USE_WEIGHT_CONVERTER=${USE_WEIGHT_CONVERTER} \
      PREFUSE_MOE_WEIGHTS=${TRAINER_PREFUSE_MOE_WEIGHTS:-false} \
      ROLLOUT_PREFUSE_MOE_WEIGHTS=${ROLLOUT_PREFUSE_MOE_WEIGHTS} \
      ${ROLLOUT_MESH_TP:+ROLLOUT_MESH_TP=${ROLLOUT_MESH_TP}} \
      ${ROLLOUT_MESH_TP:+ROLLOUT_TENSOR_PARALLEL_SIZE=${ROLLOUT_MESH_TP}} \
      FLOAT32_GATE_LOGITS=${FLOAT32_GATE_LOGITS:-true} \
      FLOAT32_LOGITS=${FLOAT32_LOGITS:-true} \
      MAX_GRAD_NORM=${MAX_GRAD_NORM:-0.125} \
      VERIFY_WEIGHTS=${VERIFY_WEIGHTS} python -m tunix.experimental.distributed.runtime.main \
        --discovery_addrs=${ORCHESTRATOR_ID}:${ORCHESTRATOR_PORT} \
        --process_executor=tunix.experimental.distributed.runtime.executor.K8sExecutor \
        --process_main=tunix.experimental.examples.common.run_trainer_node.main \
        --worker_id=${TRAINER_ID} \
        --port=${TRAINER_PORT} \
        --mesh_fsdp=${TRAINER_MESH_FSDP} \
        --mesh_tp=${TRAINER_MESH_TP} \
        --mesh_expert=${TRAINER_MESH_EXPERT} \
      $( [[ "${TRAINER_MESH_CONTEXT:-1}" -gt 1 ]] && echo "--mesh_context=${TRAINER_MESH_CONTEXT}" ) \
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
        ${COMPUTE_LOGPS_CHUNK_SIZE:+--compute_logps_chunk_size=${COMPUTE_LOGPS_CHUNK_SIZE}} \
        --eval_every_n_steps=${EVAL_EVERY_N_STEPS} \
        --optimizer_b1=${ADAM_B1} \
        --optimizer_b2=${ADAM_B2} \
        --optimizer_eps=${ADAM_EPS} \
        --optimizer_weight_decay=${WEIGHT_DECAY} \
        --optimizer_learning_rate=${LEARNING_RATE} \
        --optimizer_schedule_type=${SCHEDULE_TYPE} \
        --optimizer_init_value=${LR_INIT_VALUE} \
        --optimizer_peak_value=${LR_PEAK_VALUE} \
        --optimizer_end_value=${LR_END_VALUE} \
        --optimizer_warmup_steps=${WARMUP_STEPS} \
        --optimizer_decay_steps=${LR_DECAY_STEPS} \
        --lora_rank=${LORA_RANK} \
        --lora_alpha=${LORA_ALPHA} \
        --sampler_type=${SAMPLER} \
        --checkpoint_save_interval_steps=${CHECKPOINT_SAVE_INTERVAL_STEPS} \
        --checkpoint_max_to_keep=${CHECKPOINT_MAX_TO_KEEP} \
        --checkpoint_root_directory=${CHECKPOINT_ROOT_DIRECTORY} \
        --prefuse_moe_weights=${TRAINER_PREFUSE_MOE_WEIGHTS:-false} \
        ${opt_chain_args} \
        ${lora_args} \
        ${maxtext_args} \
        ${TRAINABLE_PARAMETERS_MASK:+--trainable_parameters_mask='${TRAINABLE_PARAMETERS_MASK}'} \
        ${debug_arg} \
    " \
    | apply_manifest
}

stop_rollout() {
  local replicas=${ROLLOUT_WORKERS:-${ROLLOUT_REPLICAS:-1}}
  if [[ "$DRY_RUN" == "true" ]]; then
    echo "kubectl delete jobset ${ROLLOUT_ID} -n ${K8S_NAMESPACE}"
    kubectl get workload -n "${K8S_NAMESPACE}" -o name 2>/dev/null | grep -E "jobset-${ROLLOUT_ID}-[a-f0-9]+" | xargs -r echo kubectl delete -n "${K8S_NAMESPACE}"
    if [[ ${replicas} -gt 1 ]]; then
      echo "kubectl delete jobset $(seq -f "${ROLLOUT_ID}-%g" 0 $((replicas - 1))) -n ${K8S_NAMESPACE}"
      kubectl get workload -n "${K8S_NAMESPACE}" -o name 2>/dev/null | grep -E "jobset-${ROLLOUT_ID}-[0-9]+-[a-f0-9]+" | xargs -r echo kubectl delete -n "${K8S_NAMESPACE}"
    fi
  else
    kubectl delete jobset "${ROLLOUT_ID}" -n "${K8S_NAMESPACE}" --ignore-not-found=true
    kubectl get workload -n "${K8S_NAMESPACE}" -o name 2>/dev/null | grep -E "jobset-${ROLLOUT_ID}-[a-f0-9]+" | xargs -r kubectl delete -n "${K8S_NAMESPACE}" --ignore-not-found=true --wait=false 2>/dev/null || true
    if [[ ${replicas} -gt 1 ]]; then
      kubectl delete jobset $(seq -f "${ROLLOUT_ID}-%g" 0 $((replicas - 1))) -n "${K8S_NAMESPACE}" --ignore-not-found=true 2>/dev/null || true
      kubectl get workload -n "${K8S_NAMESPACE}" -o name 2>/dev/null | grep -E "jobset-${ROLLOUT_ID}-[0-9]+-[a-f0-9]+" | xargs -r kubectl delete -n "${K8S_NAMESPACE}" --ignore-not-found=true --wait=false 2>/dev/null || true
    fi
  fi
}

start_rollout() {
  local maxtext_args
  maxtext_args="$(maxtext_rollout_flags)"
  local vllm_args=""
  if [[ "$SAMPLER" == "vllm" || "$SAMPLER" == "inprocess_vllm" ]]; then
    local vllm_json=""
    if [[ -n "${VLLM_CONFIG_JSON:-}" || -n "${ROLLOUT_VLLM_CONFIG_JSON:-}" || -n "${VLLM_MAX_NUM_BATCHED_TOKENS:-}" || -n "${VLLM_MAX_NUM_SEQS:-}" || -n "${VLLM_GPU_MEMORY_UTILIZATION:-}" || -n "${VLLM_ADDITIONAL_CONFIG:-}" || -n "${VLLM_MAX_MODEL_LEN:-}" || -n "${VLLM_BLOCK_SIZE:-}" ]]; then
      vllm_json=$("${PYTHON_BIN:-python3}" -c '
import json, os

cfg = {}
raw = os.getenv("VLLM_CONFIG_JSON") or os.getenv("ROLLOUT_VLLM_CONFIG_JSON")
if raw:
  try:
    cfg = json.loads(raw)
  except Exception:
    cfg = raw

if isinstance(cfg, dict):
  mapping = {
      "VLLM_MAX_MODEL_LEN": ("max_model_len", int),
      "VLLM_MAX_NUM_BATCHED_TOKENS": ("max_num_batched_tokens", int),
      "VLLM_MAX_NUM_SEQS": ("max_num_seqs", int),
      "VLLM_GPU_MEMORY_UTILIZATION": ("gpu_memory_utilization", float),
      "VLLM_DATA_PARALLEL_SIZE": ("data_parallel_size", int),
      "VLLM_ENABLE_EXPERT_PARALLEL": ("enable_expert_parallel", lambda v: v.lower() in ("true", "1")),
      "VLLM_PREFIX_CACHE_RETENTION_INTERVAL": ("prefix_cache_retention_interval", int),
      "VLLM_MAMBA_CACHE_MODE": ("mamba_cache_mode", str),
      "VLLM_KV_CACHE_DTYPE": ("kv_cache_dtype", str),
      "VLLM_BLOCK_SIZE": ("block_size", int),
      "VLLM_ASYNC_SCHEDULING": ("async_scheduling", lambda v: v.lower() in ("true", "1")),
      "VLLM_ENABLE_CHUNKED_PREFILL": ("enable_chunked_prefill", lambda v: v.lower() in ("true", "1")),
      "VLLM_LANGUAGE_MODEL_ONLY": ("language_model_only", lambda v: v.lower() in ("true", "1")),
      "VLLM_ENABLE_AUTO_TOOL_CHOICE": ("enable_auto_tool_choice", lambda v: v.lower() in ("true", "1")),
      "VLLM_TOOL_CALL_PARSER": ("tool_call_parser", str),
      "VLLM_REASONING_PARSER": ("reasoning_parser", str),
  }
  for env_k, (cfg_k, parse_fn) in mapping.items():
    val = os.getenv(env_k)
    if val is not None and val != "" and cfg_k not in cfg:
      try:
        cfg[cfg_k] = parse_fn(val)
      except Exception:
        cfg[cfg_k] = val

  for env_k, cfg_k in [
      ("VLLM_ADDITIONAL_CONFIG", "additional_config"),
      ("VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS", "default_chat_template_kwargs"),
      ("VLLM_LIMIT_MM_PER_PROMPT", "limit_mm_per_prompt"),
  ]:
    val = os.getenv(env_k)
    if val and cfg_k not in cfg:
      try:
        cfg[cfg_k] = json.loads(val)
      except Exception:
        cfg[cfg_k] = val

if cfg:
  print(json.dumps(cfg) if isinstance(cfg, dict) else cfg)
' 2>/dev/null || true)
    fi

    vllm_args="\
    --tensor_parallel_size=${ROLLOUT_MESH_TP} \
    ${vllm_json:+--vllm_config_json='${vllm_json}'} \
    ${ROLLOUT_EXTRA_ARGS} \
    "
  fi
  local lora_args=""
  if [[ "${USE_LORA}" == "1" || "${USE_LORA}" == "true" || "${USE_LORA}" == "True" ]]; then
    lora_args="--use_lora"
  fi
  local sandbox_env=""
  if [[ "$USE_AGENT_SANDBOX" == "1" || "$USE_AGENT_SANDBOX" == "true" || "$USE_AGENT_SANDBOX" == "True" ]]; then
    sandbox_env="NAMESPACE=\"${SANDBOX_NAMESPACE}\" ${SANDBOX_NODE_SELECTOR_KEY:+NODE_SELECTOR_KEY=\"${SANDBOX_NODE_SELECTOR_KEY}\"} ${SANDBOX_NODE_SELECTOR_VAL:+NODE_SELECTOR_VAL=\"${SANDBOX_NODE_SELECTOR_VAL}\"} ${IMAGE_REWRITE_PREFIX:+IMAGE_REWRITE_PREFIX=\"${IMAGE_REWRITE_PREFIX}\"} ${JOB_PREFIX:+JOB_PREFIX=\"${JOB_PREFIX}\"} ${POOL_NAME_FORMAT:+POOL_NAME_FORMAT=\"${POOL_NAME_FORMAT}\"} ${TEMPLATE_NAME_PREFIX:+TEMPLATE_NAME_PREFIX=\"${TEMPLATE_NAME_PREFIX}\"} ${SANDBOX_TOLERATIONS:+SANDBOX_TOLERATIONS=\"${SANDBOX_TOLERATIONS//\"/\\\"}\"}"
  elif [[ -n "${IMAGE_REWRITE_PREFIX}" ]]; then
    sandbox_env="IMAGE_REWRITE_PREFIX=\"${IMAGE_REWRITE_PREFIX}\" ${JOB_PREFIX:+JOB_PREFIX=\"${JOB_PREFIX}\"} ${POOL_NAME_FORMAT:+POOL_NAME_FORMAT=\"${POOL_NAME_FORMAT}\"} ${TEMPLATE_NAME_PREFIX:+TEMPLATE_NAME_PREFIX=\"${TEMPLATE_NAME_PREFIX}\"}"
  fi
  local dynamic_slicing_single_host=false
  if [[ "${ROLLOUT_TPU_SLICE}" =~ ^(tpu7x|tpu-v7x-slice):2x2x1 ]]; then
    if [[ "${USE_DYNAMIC_SLICING}" == "true" || "${USE_DYNAMIC_SLICING}" == "1" || -z "${USE_DYNAMIC_SLICING}" ]]; then
      dynamic_slicing_single_host=true
    fi
  fi

  for i in $(seq ${ROLLOUT_START_INDEX:-0} $((ROLLOUT_REPLICAS - 1))); do
    local replica_id="${ROLLOUT_ID}"
    local worker_id="${ROLLOUT_ID}"
    if [[ ${ROLLOUT_REPLICAS} -gt 1 ]]; then
      replica_id="${ROLLOUT_ID}-${i}"
      worker_id="${ROLLOUT_ID}-${i}"
    fi

    local extra_generator_flags=()
    if [[ "$dynamic_slicing_single_host" == "true" ]]; then
      extra_generator_flags+=(--omit_slice_topology)
    fi

    "$PYTHON_BIN" "$YAML_GENERATOR" \
      "${YAML_DIR}/${ROLLOUT_JOBSET_YAML}" \
      --jobset_name="${replica_id}" \
      --namespace="${K8S_NAMESPACE}" \
      ${KUEUE_QUEUE_NAME:+--queue_name="${KUEUE_QUEUE_NAME}"} \
      --tpu_slice=${ROLLOUT_TPU_SLICE} \
      --worker_container_image="${TUNIX_IMAGE}" \
      --worker_container_port="${ROLLOUT_PORT}" \
      "${extra_generator_flags[@]}" \
      --worker_startup_command=" \
        PYTHONUNBUFFERED=1 \
        TUNIX_IS_INTERNAL_ENV=false \
        EPISODE_TIMEOUT_SECS="${EPISODE_TIMEOUT_SECS:-5400}" \
        ${SCAFFOLD:+SCAFFOLD=\"${SCAFFOLD}\"} \
        ${BOOTSTRAP_CMD} \
        USE_RAIDEN_FFI=false RAIDEN_USE_FFI=0 \
        RAIDEN_DEVICES_PER_HOST=${RAIDEN_DEVICES_PER_HOST} \
        ROLLOUT_PREFUSE_MOE_WEIGHTS=${ROLLOUT_PREFUSE_MOE_WEIGHTS} \
        ROLLOUT_MESH_TP=${ROLLOUT_MESH_TP} \
        ROLLOUT_TENSOR_PARALLEL_SIZE=${ROLLOUT_MESH_TP} \
        PREFUSE_MOE_WEIGHTS=${ROLLOUT_PREFUSE_MOE_WEIGHTS} \
        ENABLE_PREFIX_CACHING=${ENABLE_PREFIX_CACHING} \
        ROLLOUT_FREE_KV_CACHE=${ROLLOUT_FREE_KV_CACHE} \
        VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-8} \
        VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.9} \
        ${NUM_PRECOMPILE_WORKERS:+NUM_PRECOMPILE_WORKERS=${NUM_PRECOMPILE_WORKERS}} \
        ${NEW_MODEL_DESIGN:+NEW_MODEL_DESIGN=${NEW_MODEL_DESIGN}} \
        ${ATTN_BUCKETIZED_NUM_REQS:+ATTN_BUCKETIZED_NUM_REQS=${ATTN_BUCKETIZED_NUM_REQS}} \
        ${ATTN_CUSTOM_NUM_REQS_BUCKETS:+ATTN_CUSTOM_NUM_REQS_BUCKETS=${ATTN_CUSTOM_NUM_REQS_BUCKETS}} \
        ${ONEHOT_MOE_PERMUTE_THRESHOLD:+ONEHOT_MOE_PERMUTE_THRESHOLD=${ONEHOT_MOE_PERMUTE_THRESHOLD}} \
        ${VLLM_MOE_CHUNK_SIZE:+VLLM_MOE_CHUNK_SIZE=${VLLM_MOE_CHUNK_SIZE}} \
        ${SLICE_ROPE_CACHE:+SLICE_ROPE_CACHE=${SLICE_ROPE_CACHE}} \
        ${DP_SCHED_BATCH_PREFILL:+DP_SCHED_BATCH_PREFILL=${DP_SCHED_BATCH_PREFILL}} \
        ${LIBTPU_INIT_ARGS:+LIBTPU_INIT_ARGS=\"${LIBTPU_INIT_ARGS}\"} \
        ${VLLM_ENABLE_V1_MULTIPROCESSING:+VLLM_ENABLE_V1_MULTIPROCESSING=${VLLM_ENABLE_V1_MULTIPROCESSING}} \
        ${VLLM_LOGGING_LEVEL:+VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL}} \
        ${ROLLOUT_ENV_FLAGS} \
        ${ROLLOUT_EXTRA_ENV:+${ROLLOUT_EXTRA_ENV} }\
        ${HF_TOKEN:+HF_TOKEN=\"${HF_TOKEN}\"} \
        SKIP_JAX_PRECOMPILE=1 VERIFY_WEIGHTS=${VERIFY_WEIGHTS} ${sandbox_env} ${ROLLOUT_USE_BATCHED_RPA:+USE_BATCHED_RPA_KERNEL=1} python -m tunix.experimental.distributed.runtime.main \
          --discovery_addrs=${ORCHESTRATOR_ID}:${ORCHESTRATOR_PORT} \
          --process_executor=tunix.experimental.distributed.runtime.executor.K8sExecutor \
          --process_main=tunix.experimental.examples.common.run_rollout_node.main \
          --worker_id=${worker_id} \
          --port=${ROLLOUT_PORT} \
          --model_id=${MODEL_ID} \
          --model_dir=${MODEL_DIR} \
          --model_name=${MODEL_NAME} \
          --tokenizer_path=${TOKENIZER_PATH} \
          --mesh_fsdp=${ROLLOUT_MESH_FSDP} \
          --mesh_tp=${ROLLOUT_MESH_TP} \
          --max_prompt_length=${MAX_PROMPT_LENGTH} \
          --max_response_length=${MAX_RESPONSE_LENGTH} \
          ${EOS_TOKENS:+--eos_tokens=\"${EOS_TOKENS}\"} \
          --sampler=${SAMPLER} \
          --lora_rank=${LORA_RANK} \
          --lora_alpha=${LORA_ALPHA} \
          --weight_sync_mode=${WEIGHT_SYNC_MODE} \
          --prefuse_moe_weights=${ROLLOUT_PREFUSE_MOE_WEIGHTS} \
          --enable_prefix_caching=${ENABLE_PREFIX_CACHING} \
          --free_kv_cache_during_weight_sync=${ROLLOUT_FREE_KV_CACHE} \
          --return_routed_experts=${RETURN_ROUTED_EXPERTS} \
          --registry_module=tunix.experimental.examples.deepswe_dist.deepswe \
          --env_name=deepswe_env \
          --agent_name=deepswe_agent \
          --max_concurrency=${ROLLOUT_MAX_CONCURRENCY} \
          ${lora_args} \
          ${maxtext_args} \
          ${vllm_args} \
          ${DEBUG:+--debug} \
      " \
      | apply_manifest

    if [[ "$dynamic_slicing_single_host" == "true" && "$DRY_RUN" != "true" ]]; then
      local slice_topo="${ROLLOUT_TPU_SLICE#*:}"
      echo "Applying single-host dynamic slicing patch for ${replica_id} (${slice_topo})..."
      kubectl patch jobset "${replica_id}" -n "${K8S_NAMESPACE}" --type='json' \
        -p="[{\"op\": \"add\", \"path\": \"/spec/replicatedJobs/0/template/spec/template/metadata/annotations/cloud.google.com~1gke-tpu-slice-topology\", \"value\": \"${slice_topo}\"}]"
    fi
  done

  if [[ "$dynamic_slicing_single_host" == "true" && "$DRY_RUN" != "true" ]]; then
    local replicas=${ROLLOUT_WORKERS:-${ROLLOUT_REPLICAS:-1}}
    echo "Waiting for Kueue to create initial workloads before recycling..."
    sleep 3
    for ((i=0; i<replicas; i++)); do
      local replica_id="${ROLLOUT_ID}"
      if [[ ${replicas} -gt 1 ]]; then
        replica_id="${ROLLOUT_ID}-${i}"
      fi
      local wl_name
      wl_name=$(kubectl get workload -n "${K8S_NAMESPACE}" -o name 2>/dev/null | grep -E "jobset-${replica_id}-[a-f0-9]+" | head -n 1 | sed 's|^workload.*/||' || true)
      if [[ -n "${wl_name}" ]]; then
        local has_topo
        has_topo=$(kubectl get workload "${wl_name}" -n "${K8S_NAMESPACE}" -o jsonpath='{.spec.podSets[0].template.metadata.annotations.cloud\.google\.com/gke-tpu-slice-topology}' 2>/dev/null || true)
        if [[ -z "${has_topo}" ]]; then
          echo "Recycling workload ${wl_name} for ${replica_id} to apply dynamic slicing..."
          kubectl delete workload "${wl_name}" -n "${K8S_NAMESPACE}" --ignore-not-found=true --wait=false 2>/dev/null || true
        fi
      fi
    done
  fi
}

start_mock_trainer() {
  "$PYTHON_BIN" "$YAML_GENERATOR" \
    "${YAML_DIR}/jobset.cpu.yaml" \
    --jobset_name="${TRAINER_ID}" \
    --namespace="${K8S_NAMESPACE}" \
    ${KUEUE_QUEUE_NAME:+--queue_name="${KUEUE_QUEUE_NAME}"} \
    --cpu_machine="${CPU_MACHINE}" \
    --worker_container_image="${TUNIX_IMAGE}" \
    --worker_container_port="${TRAINER_PORT}" \
    --worker_startup_command=" \
      TUNIX_IS_INTERNAL_ENV=false \
      python -m tunix.experimental.distributed.runtime.main \
        --discovery_addrs=${ORCHESTRATOR_ID}:${ORCHESTRATOR_PORT} \
        --process_executor=tunix.experimental.distributed.runtime.executor.K8sExecutor \
        --process_main=tunix.experimental.examples.common.run_mock_trainer_node.main \
        --worker_id=${TRAINER_ID} \
        --port=${TRAINER_PORT} \
    " \
    | apply_manifest
}

start_mock_rollout() {
  "$PYTHON_BIN" "$YAML_GENERATOR" \
    "${YAML_DIR}/jobset.cpu.yaml" \
    --jobset_name="${ROLLOUT_ID}" \
    --namespace="${K8S_NAMESPACE}" \
    ${KUEUE_QUEUE_NAME:+--queue_name="${KUEUE_QUEUE_NAME}"} \
    --cpu_machine="${CPU_MACHINE}" \
    --worker_container_image="${TUNIX_IMAGE}" \
    --worker_container_port="${ROLLOUT_PORT}" \
    --worker_startup_command=" \
      TUNIX_IS_INTERNAL_ENV=false \
      python -m tunix.experimental.distributed.runtime.main \
        --discovery_addrs=${ORCHESTRATOR_ID}:${ORCHESTRATOR_PORT} \
        --process_executor=tunix.experimental.distributed.runtime.executor.K8sExecutor \
        --process_main=tunix.experimental.examples.common.run_mock_rollout_node.main \
        --worker_id=${ROLLOUT_ID} \
        --port=${ROLLOUT_PORT} \
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
    --dry-run|--dry_run|--render)
      DRY_RUN=true
      shift
      ;;
    --vllm_config_json)
      VLLM_CONFIG_JSON="$2"
      shift 2
      ;;
    --vllm_config_json=*)
      VLLM_CONFIG_JSON="${1#*=}"
      shift
      ;;
    --image_rewrite_prefix)
      IMAGE_REWRITE_PREFIX="$2"
      shift 2
      ;;
    --image_rewrite_prefix=*)
      IMAGE_REWRITE_PREFIX="${1#*=}"
      shift
      ;;
    --namespace)
      K8S_NAMESPACE="$2"
      shift 2
      ;;
    --namespace=*)
      K8S_NAMESPACE="${1#*=}"
      shift
      ;;
    --queue)
      KUEUE_QUEUE_NAME="$2"
      shift 2
      ;;
    --queue=*)
      KUEUE_QUEUE_NAME="${1#*=}"
      shift
      ;;
    --scratch|--gcs-scratch)
      GCS_SCRATCH_LOCATION="$2"
      shift 2
      ;;
    --scratch=*|--gcs-scratch=*)
      GCS_SCRATCH_LOCATION="${1#*=}"
      shift
      ;;
    --debug)
      DEBUG=1
      shift
      ;;
    --no-debug)
      DEBUG=0
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [start|stop|orchestrator|trainer|rollout|test_orchestrator|mock_trainer|mock_rollout|start_rollout_only|eval|stop_eval] [options]"
      echo "Options:"
      echo "  --command <cmd>          Command to run"
      echo "  --namespace <ns>         Kubernetes namespace (default: default)"
      echo "  --queue <name>           Kueue local queue name (optional)"
      echo "  --image <image>          Container image to use"
      echo "  --dry-run, --render      Print generated YAMLs without applying"
      echo "  --scratch, --gcs-scratch GCS scratch location"
      echo "  --debug, --no-debug      Toggle debug logging (default: disabled)"
      echo "  --vllm_config_json <js>  vLLM engine config overrides"
      echo "  --image_rewrite_prefix   Registry prefix for sandbox images"
      exit 0
      ;;
    start|stop|orchestrator|trainer|rollout|test_orchestrator|mock_trainer|mock_rollout|start_rollout_only|eval|stop_eval)
      COMMAND="$1"
      shift
      ;;
    *)
      shift
      ;;
  esac
done

start_eval() {
  if [[ -z "${MAXTEXT_CKPT:-}" ]]; then
    local eval_yaml
    eval_yaml="$(dirname "${BASH_SOURCE[0]}")/eval_qwen35_gke.yaml"
    if [[ "$DRY_RUN" == "true" ]]; then
      cat "$eval_yaml"
    else
      kubectl apply -f "$eval_yaml"
    fi
    return
  fi

  local eval_name="${EVAL_JOBSET_NAME:-${JOB_PREFIX}-eval}"
  local eval_port="${ROLLOUT_PORT:-20001}"
  local max_model_len="${VLLM_MAX_MODEL_LEN:-65536}"
  local max_context_limit="${MAX_CONTEXT_LIMIT:-$((max_model_len - ${MAX_PROMPT_LENGTH:-4096}))}"
  local output_dir="${EVAL_OUTPUT_DIR:-${TRAJECTORY_LOG_DIR:-eval_results}}"
  local sandbox_env=""
  if [[ "${USE_AGENT_SANDBOX}" == "1" || "${USE_AGENT_SANDBOX}" == "true" || "${USE_AGENT_SANDBOX}" == "True" ]]; then
    sandbox_env="NAMESPACE=\"${SANDBOX_NAMESPACE}\" ${SANDBOX_NODE_SELECTOR_KEY:+NODE_SELECTOR_KEY=\"${SANDBOX_NODE_SELECTOR_KEY}\"} ${SANDBOX_NODE_SELECTOR_VAL:+NODE_SELECTOR_VAL=\"${SANDBOX_NODE_SELECTOR_VAL}\"} ${SANDBOX_TOLERATIONS:+SANDBOX_TOLERATIONS=\"${SANDBOX_TOLERATIONS//\"/\\\"}\"} ${IMAGE_REWRITE_PREFIX:+IMAGE_REWRITE_PREFIX=\"${IMAGE_REWRITE_PREFIX}\"} ORCHESTRATOR_ID=\"${JOB_PREFIX}\" ${JOB_PREFIX:+JOB_PREFIX=\"${JOB_PREFIX}\"} ${POOL_NAME_FORMAT:+POOL_NAME_FORMAT=\"${POOL_NAME_FORMAT}\"} ${TEMPLATE_NAME_PREFIX:+TEMPLATE_NAME_PREFIX=\"${TEMPLATE_NAME_PREFIX}\"}"
  elif [[ -n "${IMAGE_REWRITE_PREFIX}" ]]; then
    sandbox_env="IMAGE_REWRITE_PREFIX=\"${IMAGE_REWRITE_PREFIX}\""
  fi

  local worker_addrs="localhost:${eval_port}"
  if [[ ${ROLLOUT_REPLICAS} -gt 1 ]]; then
    for ((j=1; j<ROLLOUT_REPLICAS; j++)); do
      worker_addrs="${worker_addrs} ${eval_name}-${j}-proc-0-0.${eval_name}-${j}:${eval_port}"
    done
  fi

  for i in $(seq ${ROLLOUT_START_INDEX:-0} $((ROLLOUT_REPLICAS - 1))); do
    local replica_id="${eval_name}"
    if [[ ${ROLLOUT_REPLICAS} -gt 1 ]]; then
      replica_id="${eval_name}-${i}"
    fi
    local eval_cmd="tunix/experimental/examples/deepswe_dist/eval_launcher.py"
    local role_arg=""
    if [[ ${i} -gt 0 ]]; then
      eval_cmd="tunix/experimental/examples/deepswe_dist/eval_deepswe.py"
      role_arg="--role=worker"
    fi

    "$PYTHON_BIN" "$YAML_GENERATOR" \
      "${YAML_DIR}/${ROLLOUT_JOBSET_YAML:-jobset.pathways.yaml}" \
      --jobset_name="${replica_id}" \
      --namespace="${K8S_NAMESPACE}" \
      ${KUEUE_QUEUE_NAME:+--queue_name="${KUEUE_QUEUE_NAME}"} \
      --tpu_slice="${ROLLOUT_TPU_SLICE:-tpuv5:2x2x1}" \
      --cpu_machine="${CPU_MACHINE}" \
      ${PATHWAYS_SERVER_IMAGE:+--pathways_server_image="${PATHWAYS_SERVER_IMAGE}"} \
      ${PATHWAYS_PROXY_IMAGE:+--pathways_proxy_server_image="${PATHWAYS_PROXY_IMAGE}"} \
      ${PATHWAYS_PROXY_MEMORY_LIMIT:+--pathways_proxy_memory_limit="${PATHWAYS_PROXY_MEMORY_LIMIT}"} \
      ${PATHWAYS_PROXY_MEMORY:+--pathways_proxy_memory="${PATHWAYS_PROXY_MEMORY}"} \
      ${PATHWAYS_RM_MEMORY:+--pathways_rm_memory="${PATHWAYS_RM_MEMORY}"} \
      ${USER_CONTAINER_MEMORY:+--user_container_memory="${USER_CONTAINER_MEMORY}"} \
      ${USER_CONTAINER_MEMORY_LIMIT:+--user_container_memory_limit="${USER_CONTAINER_MEMORY_LIMIT}"} \
      ${PATHWAYS_WORKER_MEMORY:+--pathways_worker_memory="${PATHWAYS_WORKER_MEMORY}"} \
      --pathways_gcs_scratch_location="${GCS_SCRATCH_LOCATION}" \
      --worker_container_image="${TUNIX_IMAGE}" \
      --worker_container_port="${eval_port}" \
      --worker_startup_command=" \
        PYTHONUNBUFFERED=1 \
        TUNIX_IS_INTERNAL_ENV=false \
        VLLM_TPU_USING_PATHWAYS=1 \
        ${sandbox_env} \
        ${SCAFFOLD:+SCAFFOLD=\"${SCAFFOLD}\"} \
        ${BOOTSTRAP_CMD} \
        ${HF_TOKEN:+HF_TOKEN=\"${HF_TOKEN}\"} \
        ENABLE_PATHWAYS_PERSISTENCE=${ENABLE_PATHWAYS_PERSISTENCE} \
        PREFUSE_MOE_WEIGHTS=${ROLLOUT_PREFUSE_MOE_WEIGHTS} \
        ROLLOUT_PREFUSE_MOE_WEIGHTS=${ROLLOUT_PREFUSE_MOE_WEIGHTS} \
        ENABLE_PREFIX_CACHING=${ENABLE_PREFIX_CACHING} \
        ROLLOUT_FREE_KV_CACHE=${ROLLOUT_FREE_KV_CACHE} \
        VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-16} \
        VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.9} \
        ${VLLM_ASYNC_SCHEDULING:+VLLM_ASYNC_SCHEDULING=${VLLM_ASYNC_SCHEDULING}} \
        ${VLLM_ENABLE_EXPERT_PARALLEL:+VLLM_ENABLE_EXPERT_PARALLEL=${VLLM_ENABLE_EXPERT_PARALLEL}} \
        ${VLLM_LANGUAGE_MODEL_ONLY:+VLLM_LANGUAGE_MODEL_ONLY=${VLLM_LANGUAGE_MODEL_ONLY}} \
        ${VLLM_ENABLE_CHUNKED_PREFILL:+VLLM_ENABLE_CHUNKED_PREFILL=${VLLM_ENABLE_CHUNKED_PREFILL}} \
        ${VLLM_KV_CACHE_DTYPE:+VLLM_KV_CACHE_DTYPE=${VLLM_KV_CACHE_DTYPE}} \
        ${VLLM_BLOCK_SIZE:+VLLM_BLOCK_SIZE=${VLLM_BLOCK_SIZE}} \
        ${VLLM_MAMBA_CACHE_MODE:+VLLM_MAMBA_CACHE_MODE=${VLLM_MAMBA_CACHE_MODE}} \
        ${VLLM_LIMIT_MM_PER_PROMPT:+VLLM_LIMIT_MM_PER_PROMPT='${VLLM_LIMIT_MM_PER_PROMPT}'} \
        ${NUM_PRECOMPILE_WORKERS:+NUM_PRECOMPILE_WORKERS=${NUM_PRECOMPILE_WORKERS}} \
        ${NEW_MODEL_DESIGN:+NEW_MODEL_DESIGN=${NEW_MODEL_DESIGN}} \
        ${ATTN_BUCKETIZED_NUM_REQS:+ATTN_BUCKETIZED_NUM_REQS=${ATTN_BUCKETIZED_NUM_REQS}} \
        ${ATTN_CUSTOM_NUM_REQS_BUCKETS:+ATTN_CUSTOM_NUM_REQS_BUCKETS=${ATTN_CUSTOM_NUM_REQS_BUCKETS}} \
        ${ONEHOT_MOE_PERMUTE_THRESHOLD:+ONEHOT_MOE_PERMUTE_THRESHOLD=${ONEHOT_MOE_PERMUTE_THRESHOLD}} \
        ${VLLM_MOE_CHUNK_SIZE:+VLLM_MOE_CHUNK_SIZE=${VLLM_MOE_CHUNK_SIZE}} \
        ${SLICE_ROPE_CACHE:+SLICE_ROPE_CACHE=${SLICE_ROPE_CACHE}} \
        ${DP_SCHED_BATCH_PREFILL:+DP_SCHED_BATCH_PREFILL=${DP_SCHED_BATCH_PREFILL}} \
        ${LIBTPU_INIT_ARGS:+LIBTPU_INIT_ARGS=\"${LIBTPU_INIT_ARGS}\"} \
        ${VLLM_ENABLE_V1_MULTIPROCESSING:+VLLM_ENABLE_V1_MULTIPROCESSING=${VLLM_ENABLE_V1_MULTIPROCESSING}} \
        ${VLLM_LOGGING_LEVEL:+VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL}} \
        ${ROLLOUT_ENV_FLAGS} \
        ${ROLLOUT_EXTRA_ENV:+${ROLLOUT_EXTRA_ENV} }\
        SKIP_JAX_PRECOMPILE=1 python3 -u ${eval_cmd} \
          ${role_arg} \
          --worker_addresses ${worker_addrs} \
          --port=${eval_port} \
          --model_id=${MODEL_ID} \
          --tokenizer_path=${TOKENIZER_PATH} \
          --model_absolute_path=${MAXTEXT_CKPT} \
          --maxtext_model_name=${MAXTEXT_MODEL_NAME} \
          --mesh_fsdp=${ROLLOUT_MESH_FSDP:-2} \
          --mesh_tp=${ROLLOUT_MESH_TP:-2} \
          --vllm_utilization=${VLLM_GPU_MEMORY_UTILIZATION:-0.9} \
          --max_model_len=${max_model_len} \
          --max_context_limit=${max_context_limit} \
          --max_response_length=${MAX_RESPONSE_LENGTH} \
          --max_steps=${MAX_TURNS} \
          --max_concurrent=${MAX_CONCURRENCY} \
          --batch_size=${BATCH_SIZE:-16} \
          --vllm_max_num_seqs=${VLLM_MAX_NUM_SEQS:-16} \
          --vllm_max_num_batched_tokens=${VLLM_MAX_NUM_BATCHED_TOKENS:-2048} \
          --timeout=${EPISODE_TIMEOUT_SECS:-1800} \
          --reward_timeout=${REWARD_TIMEOUT_SECS} \
          --step_timeout=${STEP_TIMEOUT_SECS} \
          --temperature=${TEMPERATURE} \
          --top_p=${TOP_P} \
          --top_k=${TOP_K} \
          --seed=${SEED} \
          --enable_thinking=${ENABLE_THINKING:-false} \
          --enable_prefix_caching=${ENABLE_PREFIX_CACHING} \
          --checkpoint_storage_use_ocdbt=${CHECKPOINT_STORAGE_USE_OCDBT:-true} \
          --checkpoint_storage_use_zarr3=${CHECKPOINT_STORAGE_USE_ZARR3:-false} \
          --dataset_name=${DATASET_NAME} \
          --dataset_split=${DATASET_SPLIT} \
          ${DATASET_PATH:+--dataset_path=${DATASET_PATH}} \
          ${TASKS_LIMIT:+--tasks_limit=${TASKS_LIMIT}} \
          --num_rollouts_per_instance=${NUM_GENERATIONS} \
          --scaffold=${SCAFFOLD} \
          --use_agent_sandbox=${USE_AGENT_SANDBOX} \
          --max_warmpool_size=${MAX_WARMPOOL_REPLICAS} \
          --output_dir=${output_dir} \
      " \
      | apply_manifest
  done
}

stop_eval() {
  local eval_name="${EVAL_JOBSET_NAME:-${JOB_PREFIX}-eval}"
  local eval_ns="${EVAL_NAMESPACE:-${K8S_NAMESPACE:-trellis}}"
  if [[ "$DRY_RUN" == "true" ]]; then
    echo "[DRY RUN] Would delete jobset ${eval_name} in namespace ${eval_ns}"
    if [[ ${ROLLOUT_REPLICAS} -gt 1 ]]; then
      echo "kubectl delete jobset $(seq -f "${eval_name}-%g" 0 $((ROLLOUT_REPLICAS - 1))) -n ${eval_ns}"
      for ((i=0; i<ROLLOUT_REPLICAS; i++)); do
        echo "kubectl delete workload -l jobset.sigs.k8s.io/jobset-name=${eval_name}-${i} -n ${eval_ns}"
      done
    fi
    if [[ "${USE_AGENT_SANDBOX}" == "1" || "${USE_AGENT_SANDBOX}" == "true" || "${USE_AGENT_SANDBOX}" == "True" ]]; then
      echo "kubectl delete sandboxwarmpools -n ${SANDBOX_NAMESPACE} -l app.kubernetes.io/created-by=${JOB_PREFIX} --ignore-not-found=true"
      echo "kubectl delete sandboxtemplates -n ${SANDBOX_NAMESPACE} -l app.kubernetes.io/created-by=${JOB_PREFIX} --ignore-not-found=true"
      echo "kubectl delete sandboxclaims -n ${SANDBOX_NAMESPACE} -l app.kubernetes.io/created-by=${JOB_PREFIX} --ignore-not-found=true"
      echo "kubectl delete pods -n ${SANDBOX_NAMESPACE} -l app.kubernetes.io/created-by=${JOB_PREFIX} --force --grace-period=0 --ignore-not-found=true"
    fi
  else
    kubectl delete jobset "${eval_name}" -n "${eval_ns}" --ignore-not-found=true || true
    kubectl delete workload -l "jobset.sigs.k8s.io/jobset-name=${eval_name}" -n "${eval_ns}" --ignore-not-found=true 2>/dev/null || true
    if [[ ${ROLLOUT_REPLICAS} -gt 1 ]]; then
      kubectl delete jobset $(seq -f "${eval_name}-%g" 0 $((ROLLOUT_REPLICAS - 1))) -n "${eval_ns}" --ignore-not-found=true 2>/dev/null || true
      for ((i=0; i<ROLLOUT_REPLICAS; i++)); do
        kubectl delete workload -l "jobset.sigs.k8s.io/jobset-name=${eval_name}-${i}" -n "${eval_ns}" --ignore-not-found=true 2>/dev/null || true
      done
    fi
    if [[ "${USE_AGENT_SANDBOX}" == "1" || "${USE_AGENT_SANDBOX}" == "true" || "${USE_AGENT_SANDBOX}" == "True" ]]; then
      echo "Cleaning up sandboxes and warmpools for ${JOB_PREFIX} in ${SANDBOX_NAMESPACE}..."
      kubectl delete sandboxwarmpools -n "${SANDBOX_NAMESPACE}" -l "app.kubernetes.io/created-by=${JOB_PREFIX}" --ignore-not-found=true 2>/dev/null || true
      kubectl delete sandboxtemplates -n "${SANDBOX_NAMESPACE}" -l "app.kubernetes.io/created-by=${JOB_PREFIX}" --ignore-not-found=true 2>/dev/null || true
      kubectl delete sandboxclaims -n "${SANDBOX_NAMESPACE}" -l "app.kubernetes.io/created-by=${JOB_PREFIX}" --ignore-not-found=true 2>/dev/null || true
      kubectl delete pods -n "${SANDBOX_NAMESPACE}" -l "app.kubernetes.io/created-by=${JOB_PREFIX}" --force --grace-period=0 --ignore-not-found=true 2>/dev/null || true
    fi
  fi
}

if [[ "$DRY_RUN" != "true" ]]; then
  if [[ -z "${KUBECONFIG:-}" ]]; then
    if [[ -f "$HOME/.kube/config" ]]; then
      export KUBECONFIG="$HOME/.kube/config"
    else
      export KUBECONFIG="$HOME/.kube/config.cloud-tpu-multipod-dev.us-central1.trellis-demo-0810"
    fi
  fi
  if ! kubectl get nodes &>/dev/null; then
    if [[ -f tunix/experimental/examples/common/enter_kube_context.sh ]]; then
      source tunix/experimental/examples/common/enter_kube_context.sh || true
    elif [[ -f "$(dirname "${BASH_SOURCE[0]}")/../common/enter_kube_context.sh" ]]; then
      source "$(dirname "${BASH_SOURCE[0]}")/../common/enter_kube_context.sh" || true
    fi
  fi
fi

if [[ "$COMMAND" != "eval" && "$COMMAND" != "stop_eval" && -z "$TUNIX_IMAGE" ]]; then
  echo "Error: no image set. Build one with tunix, maxtext, and" \
       "tpu-inference installed, then pass it via TUNIX_IMAGE=... or" \
       "--image=..."
  exit 1
fi

if [[ "$COMMAND" == "start" ]]; then
  if [[ "${USE_AGENT_SANDBOX}" == "1" || "${USE_AGENT_SANDBOX}" == "true" || "${USE_AGENT_SANDBOX}" == "True" ]]; then
    echo "Ensuring RBAC permissions for default:xpk-sa in namespace '${SANDBOX_NAMESPACE:-trellis}'..."
    kubectl create rolebinding xpk-sa-default-pod-exec -n "${SANDBOX_NAMESPACE:-trellis}" --role=pod-exec --serviceaccount=default:xpk-sa --dry-run=client -o yaml | kubectl apply -f - || true
    kubectl create rolebinding xpk-sa-default-power-users -n "${SANDBOX_NAMESPACE:-trellis}" --clusterrole=power-users --serviceaccount=default:xpk-sa --dry-run=client -o yaml | kubectl apply -f - || true
  fi
  stop_orchestrator
  stop_trainer
  stop_rollout
  start_orchestrator
  start_trainer
  start_rollout
elif [[ "$COMMAND" == "test_orchestrator" ]]; then
  stop_orchestrator
  stop_trainer
  stop_rollout
  start_orchestrator
  start_mock_trainer
  start_mock_rollout
elif [[ "$COMMAND" == "stop" ]]; then
  stop_orchestrator
  stop_trainer
  stop_rollout
elif [[ "$COMMAND" == "orchestrator" ]]; then
  stop_orchestrator; start_orchestrator
elif [[ "$COMMAND" == "trainer" ]]; then
  stop_trainer; start_trainer
elif [[ "$COMMAND" == "mock_trainer" ]]; then
  stop_trainer; start_mock_trainer
elif [[ "$COMMAND" == "rollout" ]]; then
  stop_rollout; start_rollout
elif [[ "$COMMAND" == "mock_rollout" ]]; then
  stop_rollout; start_mock_rollout
elif [[ "$COMMAND" == "start_rollout_only" ]]; then
  start_rollout
elif [[ "$COMMAND" == "eval" ]]; then
  stop_eval
  start_eval
elif [[ "$COMMAND" == "stop_eval" ]]; then
  stop_eval
else
  echo "Error: Invalid command '$COMMAND'. Available commands: 'start', 'test_orchestrator', 'stop', 'orchestrator', 'trainer', 'mock_trainer', 'rollout', 'mock_rollout', 'eval', 'stop_eval'."
  exit 1
fi
