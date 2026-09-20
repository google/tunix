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
export SAMPLER=${SAMPLER:-inprocess_vllm}
export WEIGHT_SYNC_MODE=${WEIGHT_SYNC_MODE:-none}
export CHECKPOINT_SAVE_INTERVAL_STEPS=${CHECKPOINT_SAVE_INTERVAL_STEPS:-5}
export CHECKPOINT_MAX_TO_KEEP=${CHECKPOINT_MAX_TO_KEEP:-2}
export REMAT_POLICY=${REMAT_POLICY:-decoder}
export LEARNING_RATE_FINAL_FRACTION=${LEARNING_RATE_FINAL_FRACTION:-}
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
export PROFILER_STEPS=${PROFILER_STEPS:-0}
export SKIP_FIRST_N_PROFILER_STEPS=${SKIP_FIRST_N_PROFILER_STEPS:-}
export PROFILER_PERIOD=${PROFILER_PERIOD:-}

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
export STEP_TIMEOUT_SECS=${STEP_TIMEOUT_SECS:-1800}
export REWARD_TIMEOUT_SECS=${REWARD_TIMEOUT_SECS:-1800}
export ROLLOUT_MAX_CONCURRENCY=${ROLLOUT_MAX_CONCURRENCY:-64}
export MAX_CONCURRENCY=${MAX_CONCURRENCY:-${ROLLOUT_MAX_CONCURRENCY}}
export FLUSH_EVERY_N_STEPS=${FLUSH_EVERY_N_STEPS:-1}
export MAX_WARMPOOL_REPLICAS=${MAX_WARMPOOL_REPLICAS:-4}
export ENABLE_PATHWAYS_PERSISTENCE=${ENABLE_PATHWAYS_PERSISTENCE:-0}
export CKPT_D2H_CONCURRENT_GB=${CKPT_D2H_CONCURRENT_GB:-8}

# MaxText trainer configuration: only consulted when TRAINER_BACKEND=maxtext
export MAXTEXT_MODEL_NAME=${MAXTEXT_MODEL_NAME:-qwen3-4b}
export MAXTEXT_CKPT=${MAXTEXT_CKPT:-}
# If TRAINER_BACKEND=maxtext, MAXTEXT_CKPT must be set to the path of an Orbax params-only checkpoint.
if [[ "$TRAINER_BACKEND" == "maxtext" && -z "$MAXTEXT_CKPT" ]]; then
  echo "Error: TRAINER_BACKEND=maxtext requires MAXTEXT_CKPT (Orbax params-only checkpoint)."
  exit 1
fi
export MAXTEXT_OUTPUT_DIR=${MAXTEXT_OUTPUT_DIR:-artifacts/deepswe_dist/maxtext}
export TRAINER_MESH_TP=${TRAINER_MESH_TP:-1}
export TRAINER_MESH_EXPERT=${TRAINER_MESH_EXPERT:-1}
# Padded MoE MLP intermediate dimension; must match rollout TP padding for MoE models.
export TRAINER_PADDED_MOE_MLP_DIM=${TRAINER_PADDED_MOE_MLP_DIM:-}
export TRAINER_BASE_NUM_KV_HEADS=${TRAINER_BASE_NUM_KV_HEADS:-${BASE_NUM_KV_HEADS:-}}
export ROLLOUT_MESH_TP=${ROLLOUT_MESH_TP:-2}
export ROLLOUT_MESH_FSDP=${ROLLOUT_MESH_FSDP:-1}
# Optional: enable experimental batched-RPA attention kernel for rollout.
export ROLLOUT_USE_BATCHED_RPA=${ROLLOUT_USE_BATCHED_RPA:-}
export ROLLOUT_MAXTEXT_ATTENTION=${ROLLOUT_MAXTEXT_ATTENTION:-}
export TRAINER_MAXTEXT_ATTENTION=${TRAINER_MAXTEXT_ATTENTION:-}

# Logs source/destination Raiden tensor checksums on both the trainer and
# rollout sides during weight sync, for cross-verification of a real run.
export VERIFY_WEIGHTS=${VERIFY_WEIGHTS:-false}

export WANDB_PROJECT=${WANDB_PROJECT:-trellis-deepswe}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-}
export WANDB_API_KEY=${WANDB_API_KEY:-}
export WANDB_ENTITY=${WANDB_ENTITY:-}
export TRAJECTORY_LOG_DIR=${TRAJECTORY_LOG_DIR:-}
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

export ORCHESTRATOR_ID=$USER-orch
export ORCHESTRATOR_PORT=20000

export ROLLOUT_ID=$USER-roll
export ROLLOUT_PORT=20001

export TRAINER_ID=$USER-train
export TRAINER_PORT=20002

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

export TRAINER_JOBSET_YAML=${TRAINER_JOBSET_YAML:-jobset.pathways.yaml}
export TRAINER_TPU_SLICE=${TRAINER_TPU_SLICE:-tpuv5:2x2x2}
export TRAINER_MESH_FSDP=${TRAINER_MESH_FSDP:-8}
export ROLLOUT_TPU_SLICE=${ROLLOUT_TPU_SLICE:-tpuv5:2x2x1}
export ROLLOUT_REPLICAS=${ROLLOUT_REPLICAS:-1}
export KUEUE_QUEUE=${KUEUE_QUEUE:-}
export K8S_NAMESPACE=${K8S_NAMESPACE:-default}

export TRAINER_EXTRA_ENV=${TRAINER_EXTRA_ENV:-}
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
  else
    kubectl delete jobset "${ORCHESTRATOR_ID}" -n "${K8S_NAMESPACE}" --ignore-not-found=true
    kubectl delete workload -l "jobset.sigs.k8s.io/jobset-name=${ORCHESTRATOR_ID}" -n "${K8S_NAMESPACE}" --ignore-not-found=true 2>/dev/null || true
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
    sandbox_env="NAMESPACE=\"${SANDBOX_NAMESPACE}\" ${SANDBOX_NODE_SELECTOR_KEY:+NODE_SELECTOR_KEY=\"${SANDBOX_NODE_SELECTOR_KEY}\"} ${SANDBOX_NODE_SELECTOR_VAL:+NODE_SELECTOR_VAL=\"${SANDBOX_NODE_SELECTOR_VAL}\"}"
    sandbox_arg="--use_agent_sandbox"
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

  "$PYTHON_BIN" "$YAML_GENERATOR" \
    "${YAML_DIR}/jobset.cpu.yaml" \
    --jobset_name="${ORCHESTRATOR_ID}" \
    --cpu_machine=${CPU_MACHINE} \
    --worker_container_image="${TUNIX_IMAGE}" \
    --worker_container_port="${ORCHESTRATOR_PORT}" \
    --worker_startup_command=" \
      ${sandbox_env} \
      ${SCAFFOLD:+SCAFFOLD=\"${SCAFFOLD}\"} \
      ${WANDB_API_KEY:+WANDB_API_KEY=\"${WANDB_API_KEY}\"} \
      ${WANDB_ENTITY:+WANDB_ENTITY=\"${WANDB_ENTITY}\"} \
      WANDB_PROJECT=\"${WANDB_PROJECT}\" \
      WANDB_RUN_NAME=\"${WANDB_RUN_NAME}\" \
      ROLLOUT_WORKERS=\"${ROLLOUT_WORKERS:-${ROLLOUT_REPLICAS:-1}}\" \
      EPISODE_TIMEOUT_SECS=\"${EPISODE_TIMEOUT_SECS:-5400}\" \
      ${TRAJECTORY_LOG_DIR:+TRAJECTORY_LOG_DIR=\"${TRAJECTORY_LOG_DIR}\"} \
      PYTHONUNBUFFERED=1 \
      TUNIX_IS_INTERNAL_ENV=false \
      ${BOOTSTRAP_CMD} \
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
        ${TRAJECTORY_LOG_DIR:+--trajectory_log_dir=\"${TRAJECTORY_LOG_DIR}\"} \
        --flush_every_n_steps=${FLUSH_EVERY_N_STEPS} \
        --wandb_project=\"${WANDB_PROJECT}\" \
        --wandb_run_name=\"${WANDB_RUN_NAME}\" \
        --weight_sync_mode=${WEIGHT_SYNC_MODE} \
        --stop_workers_on_exit \
        ${MAX_WARMPOOL_REPLICAS:+--max_warmpool_replicas=${MAX_WARMPOOL_REPLICAS}} \
        ${MAX_CONCURRENCY:+--max_concurrency=${MAX_CONCURRENCY}} \
        ${MAX_STALENESS:+--max_staleness=${MAX_STALENESS}} \
        $([[ "${USE_ROLLOUT_LOGPS}" == "false" || "${USE_ROLLOUT_LOGPS}" == "False" || "${USE_ROLLOUT_LOGPS}" == "0" ]] && echo --no-use_rollout_logps || echo --use_rollout_logps) \
        ${dataset_args} \
        ${shuffle_arg} \
        ${sandbox_arg} \
        ${MAX_SEQ_TOKEN_PER_TPU:+--max_seq_token_per_tpu=${MAX_SEQ_TOKEN_PER_TPU}} \
        ${MAX_SEGMENTS_PER_PACKED_ROW:+--max_segments_per_packed_row=${MAX_SEGMENTS_PER_PACKED_ROW}} \
        ${TRAINER_MESH_FSDP:+--trainer_fsdp=${TRAINER_MESH_FSDP}} \
        ${TRAINABLE_PARAMETERS_MASK:+--trainable_parameters_mask='${TRAINABLE_PARAMETERS_MASK}'} \
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
  local maxtext_args=""
  if [[ "${TRAINER_BACKEND}" == "maxtext" ]]; then
    maxtext_args=" \
      --maxtext_model_name=${MAXTEXT_MODEL_NAME} \
      ${TRAINER_PADDED_MOE_MLP_DIM:+--maxtext_padded_moe_mlp_dim=${TRAINER_PADDED_MOE_MLP_DIM}} \
      --maxtext_ckpt_path=${MAXTEXT_CKPT} \
      --maxtext_output_directory=${MAXTEXT_OUTPUT_DIR} \
      --mesh_expert=${TRAINER_MESH_EXPERT} \
      ${ROLLOUT_MESH_TP:+--rollout_mesh_tp=${ROLLOUT_MESH_TP}} \
      ${TRAINER_BASE_NUM_KV_HEADS:+--base_num_kv_heads=${TRAINER_BASE_NUM_KV_HEADS}} \
      ${TRAINER_MAXTEXT_ATTENTION:+--maxtext_attention=${TRAINER_MAXTEXT_ATTENTION}} \
      ${REMAT_POLICY:+--remat_policy=${REMAT_POLICY}} \
      ${LEARNING_RATE_FINAL_FRACTION:+--learning_rate_final_fraction=${LEARNING_RATE_FINAL_FRACTION}} \
    "
  fi
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
  local profiler_args="--profiler_steps=${PROFILER_STEPS:-0}"
  if [[ -n "${SKIP_FIRST_N_PROFILER_STEPS:-}" ]]; then
    profiler_args+=" --skip_first_n_profiler_steps=${SKIP_FIRST_N_PROFILER_STEPS}"
  fi
  if [[ -n "${PROFILER_PERIOD:-}" ]]; then
    profiler_args+=" --profiler_period=${PROFILER_PERIOD}"
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
      ${CKPT_D2H_CONCURRENT_GB:+CKPT_D2H_CONCURRENT_GB=${CKPT_D2H_CONCURRENT_GB}} \
      ${TRAINER_MAXTEXT_ATTENTION:+TRAINER_MAXTEXT_ATTENTION=\"${TRAINER_MAXTEXT_ATTENTION}\"} \
      ${raiden_env} \
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
        ${WARMUP_STEPS_FRACTION:+--maxtext_warmup_steps_fraction=${WARMUP_STEPS_FRACTION}} \
        --lora_rank=${LORA_RANK} \
        --lora_alpha=${LORA_ALPHA} \
        --sampler_type=${SAMPLER} \
        --checkpoint_save_interval_steps=${CHECKPOINT_SAVE_INTERVAL_STEPS} \
        --checkpoint_max_to_keep=${CHECKPOINT_MAX_TO_KEEP} \
        --prefuse_moe_weights=${TRAINER_PREFUSE_MOE_WEIGHTS:-false} \
        --use_weight_converter=${USE_WEIGHT_CONVERTER} \
        ${MAX_SEQ_TOKEN_PER_TPU:+--max_seq_token_per_tpu=${MAX_SEQ_TOKEN_PER_TPU}} \
        ${COMPUTE_LOGPS_CHUNK_SIZE:+--compute_logps_chunk_size=${COMPUTE_LOGPS_CHUNK_SIZE}} \
        ${opt_chain_args} \
        ${lora_args} \
        ${maxtext_args} \
        ${profiler_args} \
        ${TRAINABLE_PARAMETERS_MASK:+--trainable_parameters_mask='${TRAINABLE_PARAMETERS_MASK}'} \
        ${debug_arg} \
    " \
    | apply_manifest
}

stop_rollout() {
  if [[ "$DRY_RUN" == "true" ]]; then
    echo "kubectl delete jobset ${ROLLOUT_ID} -n ${K8S_NAMESPACE}"
    echo "kubectl delete workload -l jobset.sigs.k8s.io/jobset-name=${ROLLOUT_ID} -n ${K8S_NAMESPACE}"
    if [[ ${ROLLOUT_REPLICAS} -gt 1 ]]; then
      echo "kubectl delete jobset $(seq -f "${ROLLOUT_ID}-%g" 0 $((ROLLOUT_REPLICAS - 1))) -n ${K8S_NAMESPACE}"
      for ((i=0; i<ROLLOUT_REPLICAS; i++)); do
        echo "kubectl delete workload -l jobset.sigs.k8s.io/jobset-name=${ROLLOUT_ID}-${i} -n ${K8S_NAMESPACE}"
      done
    fi
  else
    kubectl delete jobset "${ROLLOUT_ID}" -n "${K8S_NAMESPACE}" --ignore-not-found=true
    kubectl delete workload -l "jobset.sigs.k8s.io/jobset-name=${ROLLOUT_ID}" -n "${K8S_NAMESPACE}" --ignore-not-found=true 2>/dev/null || true
    if [[ ${ROLLOUT_REPLICAS} -gt 1 ]]; then
      kubectl delete jobset $(seq -f "${ROLLOUT_ID}-%g" 0 $((ROLLOUT_REPLICAS - 1))) -n "${K8S_NAMESPACE}" --ignore-not-found=true 2>/dev/null || true
      for ((i=0; i<ROLLOUT_REPLICAS; i++)); do
        kubectl delete workload -l "jobset.sigs.k8s.io/jobset-name=${ROLLOUT_ID}-${i}" -n "${K8S_NAMESPACE}" --ignore-not-found=true 2>/dev/null || true
      done
    fi
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
    sandbox_env="NAMESPACE=\"${SANDBOX_NAMESPACE}\" ${SANDBOX_NODE_SELECTOR_KEY:+NODE_SELECTOR_KEY=\"${SANDBOX_NODE_SELECTOR_KEY}\"} ${SANDBOX_NODE_SELECTOR_VAL:+NODE_SELECTOR_VAL=\"${SANDBOX_NODE_SELECTOR_VAL}\"}"
  fi
  for i in $(seq ${ROLLOUT_START_INDEX:-0} $((ROLLOUT_REPLICAS - 1))); do
    local replica_id="${ROLLOUT_ID}"
    local worker_id="${ROLLOUT_ID}"
    if [[ ${ROLLOUT_REPLICAS} -gt 1 ]]; then
      replica_id="${ROLLOUT_ID}-${i}"
      worker_id="${ROLLOUT_ID}-${i}"
    fi
    "$PYTHON_BIN" "$YAML_GENERATOR" \
      "${YAML_DIR}/jobset.tpu.yaml" \
      --jobset_name="${replica_id}" \
      --tpu_slice=${ROLLOUT_TPU_SLICE} \
      --worker_container_image="${TUNIX_IMAGE}" \
      --worker_container_port="${ROLLOUT_PORT}" \
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
  done
}

start_mock_trainer() {
  "$PYTHON_BIN" "$YAML_GENERATOR" \
    "${YAML_DIR}/jobset.cpu.yaml" \
    --jobset_name="${TRAINER_ID}" \
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
    --dry-run|--render)
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
    start|stop|orchestrator|trainer|rollout|test_orchestrator|mock_trainer|mock_rollout|start_rollout_only)
      COMMAND="$1"
      shift
      ;;
    *)
      shift
      ;;
  esac
done

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

if [[ -z "$TUNIX_IMAGE" ]]; then
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
else
  echo "Error: Invalid command '$COMMAND'. Available commands: 'start', 'test_orchestrator', 'stop', 'orchestrator', 'trainer', 'mock_trainer', 'rollout', 'mock_rollout'."
  exit 1
fi

