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

# Available options: 'start', 'stop', 'orchestrator', 'trainer', 'rollout'.
COMMAND=""
TUNIX_IMAGE=${TUNIX_IMAGE:-}

LAUNCHER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" &>/dev/null; then
  PYTHON="python"
fi

YAML_GEN=${YAML_GEN:-"${LAUNCHER_DIR}/../../distributed/deployment/yaml_generator.py"}
YAML_DIR=${YAML_DIR:-"${LAUNCHER_DIR}/../../distributed/deployment/yamls"}

export MODEL_NAME=${MODEL_NAME:-Qwen3.5-35B-A3B}
export MODEL_ID=${MODEL_ID:-Qwen/Qwen3.5-35B-A3B}
export MODEL_DIR=${MODEL_DIR:-artifacts/frozenlake_dist/models/${MODEL_NAME}}
export TOKENIZER_PATH=${TOKENIZER_PATH:-${MODEL_ID}}

export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
export MAX_TURNS=${MAX_TURNS:-8}
export DATASET_SIZE=${DATASET_SIZE:-10000}
export IS_SLIPPERY=${IS_SLIPPERY:-0}
export USE_MULTISTEP_PROMPT=${USE_MULTISTEP_PROMPT:-1}
export EOS_TOKENS=${EOS_TOKENS-'151645,151643'}
export BATCH_SIZE=${BATCH_SIZE:-16}
export MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-${BATCH_SIZE}}
export NUM_GENERATIONS=${NUM_GENERATIONS:-16}
export MAX_STEPS=${MAX_STEPS:-50}
export TRAIN_MICRO_BATCH_SIZE=${TRAIN_MICRO_BATCH_SIZE:-4}
export MAX_SEQ_TOKEN_PER_TPU=${MAX_SEQ_TOKEN_PER_TPU:-}
export MAX_SEGMENTS_PER_PACKED_ROW=${MAX_SEGMENTS_PER_PACKED_ROW:-}

export TRAINER_BACKEND=${TRAINER_BACKEND:-maxtext}
export COMPUTE_LOGPS_CHUNK_SIZE=${COMPUTE_LOGPS_CHUNK_SIZE:-0}
export EVAL_EVERY_N_STEPS=${EVAL_EVERY_N_STEPS:-1000000}
export OPT_CHAIN_TYPE=${OPT_CHAIN_TYPE-clip_by_global_norm}
export MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}
export ADAM_B1=${ADAM_B1:-0.9}
export ADAM_B2=${ADAM_B2:-0.999}
export ADAM_EPS=${ADAM_EPS:-1.0e-8}
export WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}
export LEARNING_RATE=${LEARNING_RATE:-1e-6}
export SCHEDULE_TYPE=${SCHEDULE_TYPE-warmup_cosine_decay_schedule}
export LR_INIT_VALUE=${LR_INIT_VALUE:-0.0}
export LR_PEAK_VALUE=${LR_PEAK_VALUE:-$LEARNING_RATE}
export LR_END_VALUE=${LR_END_VALUE:-0.0}
export LR_DECAY_STEPS=${LR_DECAY_STEPS:-500}
export WARMUP_STEPS=${WARMUP_STEPS:-$(((LR_DECAY_STEPS + 9) / 10))}
export LORA_RANK=${LORA_RANK:-16}
export LORA_ALPHA=${LORA_ALPHA:-16.0}
export USE_LORA=${USE_LORA:-0}
export BETA=${BETA:-0}
export EPSILON=${EPSILON:-0.2}
export EPSILON_HIGH=${EPSILON_HIGH:-0.28}
export LOSS_ALGO=${LOSS_ALGO:-gspo-token}
export LOSS_AGG_MODE=${LOSS_AGG_MODE:-sequence-mean-token-mean}
export ADVANTAGE_ESTIMATOR=${ADVANTAGE_ESTIMATOR:-rloo}
export TEMPERATURE=${TEMPERATURE:-0.7}
export TOP_P=${TOP_P:-1.0}
export TOP_K=${TOP_K:-0}
export DEBUG=${DEBUG:-0}
export SAMPLER=${SAMPLER:-vllm}
export WEIGHT_SYNC_MODE=${WEIGHT_SYNC_MODE:-raiden}
export USE_ROLLOUT_LOGPS=${USE_ROLLOUT_LOGPS:-true}
export CHAT_PARSER=${CHAT_PARSER:-raw}
export CHECKPOINT_SAVE_INTERVAL_STEPS=${CHECKPOINT_SAVE_INTERVAL_STEPS:-0}
export CHECKPOINT_MAX_TO_KEEP=${CHECKPOINT_MAX_TO_KEEP:-10}
export CHECKPOINT_ROOT_DIRECTORY=${CHECKPOINT_ROOT_DIRECTORY:-checkpoints}
export ENABLE_PATHWAYS_PERSISTENCE=${ENABLE_PATHWAYS_PERSISTENCE:-0}
export CKPT_D2H_CONCURRENT_GB=${CKPT_D2H_CONCURRENT_GB:-8}

# MaxText trainer configuration: only consulted when TRAINER_BACKEND=maxtext
export MAXTEXT_MODEL_NAME=${MAXTEXT_MODEL_NAME:-qwen3.5-35b-a3b}
export MAXTEXT_CKPT=${MAXTEXT_CKPT:-}
export MAXTEXT_OUTPUT_DIR=${MAXTEXT_OUTPUT_DIR:-artifacts/frozenlake_dist/maxtext}
export TRAINER_PADDED_MOE_MLP_DIM=${TRAINER_PADDED_MOE_MLP_DIM:-}
export ROLLOUT_USE_BATCHED_RPA=${ROLLOUT_USE_BATCHED_RPA:-}
export ROLLOUT_MAXTEXT_ATTENTION=${ROLLOUT_MAXTEXT_ATTENTION:-}

# MoE & Weight Sync Flags
export PREFUSE_MOE_WEIGHTS=${PREFUSE_MOE_WEIGHTS:-true}
export USE_WEIGHT_CONVERTER=${USE_WEIGHT_CONVERTER:-true}
export ENABLE_PREFIX_CACHING=${ENABLE_PREFIX_CACHING:-false}
export VERIFY_WEIGHTS=${VERIFY_WEIGHTS:-false}

export WANDB_PROJECT=${WANDB_PROJECT:-trellis-frozenlake}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-}
export WANDB_API_KEY=${WANDB_API_KEY:-}
export LOG_DIR=${LOG_DIR:-}
export TRAJECTORY_LOG_DIR=${TRAJECTORY_LOG_DIR:-}
export FLUSH_METRICS_EVERY_N_STEPS=${FLUSH_METRICS_EVERY_N_STEPS:-1}

# Rollout Registry and Environment Settings
export ROLLOUT_REGISTRY_MODULE=${ROLLOUT_REGISTRY_MODULE:-tunix.experimental.examples.frozenlake_dist.frozenlake}
export ROLLOUT_ENV_NAME=${ROLLOUT_ENV_NAME:-frozenlake_env}
export ROLLOUT_AGENT_NAME=${ROLLOUT_AGENT_NAME:-frozenlake_agent}

export ORCHESTRATOR_ID=${ORCHESTRATOR_ID:-$USER-orch}
export ORCHESTRATOR_PORT=${ORCHESTRATOR_PORT:-20000}

export ROLLOUT_ID=${ROLLOUT_ID:-$USER-roll}
export ROLLOUT_PORT=${ROLLOUT_PORT:-20001}
export ROLLOUT_REPLICAS=${ROLLOUT_REPLICAS:-16}

export TRAINER_ID=${TRAINER_ID:-$USER-train}
export TRAINER_PORT=${TRAINER_PORT:-20002}

export CPU_MACHINE=${CPU_MACHINE:-n2-standard-64}
export GCS_SCRATCH_LOCATION=${GCS_SCRATCH_LOCATION:-gs://cloud-pathways-staging/tmp}

export TRAINER_JOBSET_YAML=${TRAINER_JOBSET_YAML:-jobset.pathways.yaml}
export TRAINER_TPU_SLICE=${TRAINER_TPU_SLICE:-tpuv5:4x4x4}
export TRAINER_MESH_FSDP=${TRAINER_MESH_FSDP:-32}
export TRAINER_MESH_TP=${TRAINER_MESH_TP:-2}
export TRAINER_MESH_EXPERT=${TRAINER_MESH_EXPERT:-1}

export PATHWAYS_SERVER_IMAGE=${PATHWAYS_SERVER_IMAGE:-us-docker.pkg.dev/cloud-tpu-v2-images/pathways/server:latest}
export PATHWAYS_PROXY_IMAGE=${PATHWAYS_PROXY_IMAGE:-us-docker.pkg.dev/cloud-tpu-v2-images/pathways/proxy_server:latest}
export PATHWAYS_PROXY_MEMORY_LIMIT=${PATHWAYS_PROXY_MEMORY_LIMIT:-190G}
export PATHWAYS_PROXY_MEMORY=${PATHWAYS_PROXY_MEMORY:-16G}
export PATHWAYS_RM_MEMORY=${PATHWAYS_RM_MEMORY:-4G}
export USER_CONTAINER_MEMORY=${USER_CONTAINER_MEMORY:-48G}
export USER_CONTAINER_MEMORY_LIMIT=${USER_CONTAINER_MEMORY_LIMIT:-70G}
export PATHWAYS_WORKER_MEMORY=${PATHWAYS_WORKER_MEMORY:-100G}
export TRAINER_EXTRA_ENV=${TRAINER_EXTRA_ENV:-}

export ROLLOUT_JOBSET_YAML=${ROLLOUT_JOBSET_YAML:-jobset.tpu.yaml}
export ROLLOUT_TPU_SLICE=${ROLLOUT_TPU_SLICE:-tpuv5:2x2x1}
export ROLLOUT_MESH_FSDP=${ROLLOUT_MESH_FSDP:-1}
export ROLLOUT_MESH_TP=${ROLLOUT_MESH_TP:-1}

# Kubernetes Cluster & Scheduling Options
export K8S_NAMESPACE=${K8S_NAMESPACE:-${NAMESPACE:-default}}
export KUEUE_QUEUE_NAME=${KUEUE_QUEUE_NAME:-${KUEUE_QUEUE:-${QUEUE_NAME:-}}}
export DRY_RUN=${DRY_RUN:-false}

apply_manifest() {
  if [[ "$DRY_RUN" == "true" ]]; then
    echo "---"
    cat
  else
    kubectl apply -f -
  fi
}

stop_orchestrator() {
  if [[ "$DRY_RUN" == "true" ]]; then
    echo "kubectl delete jobset ${ORCHESTRATOR_ID} -n ${K8S_NAMESPACE}"
    echo "kubectl delete workload -l jobset.sigs.k8s.io/jobset-name=${ORCHESTRATOR_ID} -n ${K8S_NAMESPACE}"
  else
    kubectl delete jobset "${ORCHESTRATOR_ID}" -n "${K8S_NAMESPACE}" --ignore-not-found --wait=true
    kubectl delete workload -l "jobset.sigs.k8s.io/jobset-name=${ORCHESTRATOR_ID}" -n "${K8S_NAMESPACE}" --ignore-not-found=true 2>/dev/null || true
    while kubectl get jobset "${ORCHESTRATOR_ID}" -n "${K8S_NAMESPACE}" &>/dev/null; do
      sleep 2
    done
  fi
}

start_orchestrator() {
  local debug_flag=""
  if [[ "${DEBUG}" == "1" || "${DEBUG}" == "true" || "${DEBUG}" == "True" ]]; then
    debug_flag="--debug"
  fi

  local overlong_loss_masking_arg=""
  if [[ "${OVERLONG_LOSS_MASKING}" == "1" || "${OVERLONG_LOSS_MASKING}" == "true" || "${OVERLONG_LOSS_MASKING}" == "True" ]]; then
    overlong_loss_masking_arg="--overlong_loss_masking"
  fi
  local is_slippery_arg=""
  if [[ "${IS_SLIPPERY}" == "1" || "${IS_SLIPPERY}" == "true" || "${IS_SLIPPERY}" == "True" ]]; then
    is_slippery_arg="--is_slippery"
  else
    is_slippery_arg="--no-is_slippery"
  fi
  local use_multistep_prompt_arg=""
  if [[ "${USE_MULTISTEP_PROMPT}" == "0" || "${USE_MULTISTEP_PROMPT}" == "false" || "${USE_MULTISTEP_PROMPT}" == "False" ]]; then
    use_multistep_prompt_arg="--no-use_multistep_prompt"
  else
    use_multistep_prompt_arg="--use_multistep_prompt"
  fi
  local tis_type="${TIS_TYPE:-${TRUNCATED_IMPORTANCE_SAMPLING_TYPE:-}}"
  local tis_ratio_min="${TIS_RATIO_MIN:-${TRUNCATED_IMPORTANCE_SAMPLING_RATIO_MIN:-}}"
  local tis_ratio="${TIS_RATIO:-${TRUNCATED_IMPORTANCE_SAMPLING_RATIO:-}}"
  local shuffle_arg=""
  if [[ "${SHUFFLE}" == "0" || "${SHUFFLE}" == "false" || "${SHUFFLE}" == "False" ]]; then
    shuffle_arg="--no-shuffle"
  elif [[ "${SHUFFLE}" == "1" || "${SHUFFLE}" == "true" || "${SHUFFLE}" == "True" ]]; then
    shuffle_arg="--shuffle"
  fi

  "$PYTHON" "$YAML_GEN" \
    "$YAML_DIR/jobset.cpu.yaml" \
    --jobset_name="${ORCHESTRATOR_ID}" \
    --namespace="${K8S_NAMESPACE}" \
    ${KUEUE_QUEUE_NAME:+--queue_name="${KUEUE_QUEUE_NAME}"} \
    --cpu_machine=${CPU_MACHINE} \
    --worker_container_image="${TUNIX_IMAGE}" \
    --worker_container_port="${ORCHESTRATOR_PORT}" \
    --worker_startup_command=" \
      (python3 -c 'import gymnasium' 2>/dev/null || pip install --no-cache-dir gymnasium) && \
      ((echo 'cCA9ICIvYXBwL3R1bml4L2V4cGVyaW1lbnRhbC9leGFtcGxlcy9mcm96ZW5sYWtlX2Rpc3QvcnVuX2Zyb3plbmxha2VfZGlzdC5weSIKdHJ5OgogIGMgPSBvcGVuKHApLnJlYWQoKQogIGlmICItLXJvbGxvdXRfcmVwbGljYXMiIG5vdCBpbiBjOgogICAgYyA9IGMucmVwbGFjZSgKICAgICAgICAiZGF0YXR5cGVzLlJvbGUuUk9MTE9VVDogMSwiLAogICAgICAgICJkYXRhdHlwZXMuUm9sZS5ST0xMT1VUOiBnZXRhdHRyKGFyZ3MsIFwicm9sbG91dF9yZXBsaWNhc1wiLCAxKSwiLAogICAgKS5yZXBsYWNlKAogICAgICAgICJwYXJzZXIuYWRkX2FyZ3VtZW50KFwiLS1iYXRjaF9zaXplXCIiLAogICAgICAgICJwYXJzZXIuYWRkX2FyZ3VtZW50KFwiLS1yb2xsb3V0X3JlcGxpY2FzXCIsIHR5cGU9aW50LCBkZWZhdWx0PTEpXG4gIHBhcnNlci5hZGRfYXJndW1lbnQoXCItLWJhdGNoX3NpemVcIiIsCiAgICApCiAgICBvcGVuKHAsICJ3Iikud3JpdGUoYykKZXhjZXB0IEV4Y2VwdGlvbjoKICBwYXNz' | base64 -d | python3) 2>/dev/null || true) && \
      ${HF_TOKEN:+HF_TOKEN=\"${HF_TOKEN}\"} \
      ${WANDB_API_KEY:+WANDB_API_KEY=\"${WANDB_API_KEY}\"} \
      ${LOG_DIR:+LOG_DIR=\"${LOG_DIR}\"} \
      ${TRAJECTORY_LOG_DIR:+TRAJECTORY_LOG_DIR=\"${TRAJECTORY_LOG_DIR}\"} \
      WANDB_PROJECT=\"${WANDB_PROJECT}\" \
      WANDB_RUN_NAME=\"${WANDB_RUN_NAME}\" \
      ${ORCHESTRATOR_EXTRA_ENV:+${ORCHESTRATOR_EXTRA_ENV} }python -m tunix.experimental.distributed.runtime.main \
        --discovery_id=${ORCHESTRATOR_ID} \
        --discovery_port=${ORCHESTRATOR_PORT} \
        --process_main=tunix.experimental.examples.frozenlake_dist.run_frozenlake_dist.main \
        --model_id=${MODEL_ID} \
        --tokenizer_path=${TOKENIZER_PATH} \
        --batch_size=${BATCH_SIZE} \
        --mini_batch_size=${MINI_BATCH_SIZE} \
        --num_generations=${NUM_GENERATIONS} \
        --max_steps=${MAX_STEPS} \
        --max_turns=${MAX_TURNS} \
        --dataset_size=${DATASET_SIZE} \
        --max_prompt_length=${MAX_PROMPT_LENGTH} \
        --max_response_length=${MAX_RESPONSE_LENGTH} \
        --train_micro_batch_size=${TRAIN_MICRO_BATCH_SIZE} \
        --rollout_replicas=${ROLLOUT_REPLICAS} \
        --wandb_project=\"${WANDB_PROJECT}\" \
        --wandb_run_name=\"${WANDB_RUN_NAME}\" \
        --flush_every_n_steps=${FLUSH_METRICS_EVERY_N_STEPS} \
        --weight_sync_mode=${WEIGHT_SYNC_MODE} \
        --stop_workers_on_exit \
        $([[ "${USE_ROLLOUT_LOGPS}" == "false" || "${USE_ROLLOUT_LOGPS}" == "False" || "${USE_ROLLOUT_LOGPS}" == "0" ]] && echo --no-use_rollout_logps || echo --use_rollout_logps) \
        ${LOG_DIR:+--log_dir=\"${LOG_DIR}\"} \
        ${TRAJECTORY_LOG_DIR:+--trajectory_log_dir=\"${TRAJECTORY_LOG_DIR}\"} \
        ${MAX_SEQ_TOKEN_PER_TPU:+--max_seq_token_per_tpu=${MAX_SEQ_TOKEN_PER_TPU}} \
        ${MAX_SEGMENTS_PER_PACKED_ROW:+--max_segments_per_packed_row=${MAX_SEGMENTS_PER_PACKED_ROW}} \
        ${TRAINER_MESH_FSDP:+--trainer_fsdp=${TRAINER_MESH_FSDP}} \
        ${is_slippery_arg} \
        ${use_multistep_prompt_arg} \
        ${LOSS_ALGO:+--loss_algo=${LOSS_ALGO}} \
        ${ADVANTAGE_ESTIMATOR:+--advantage_estimator=${ADVANTAGE_ESTIMATOR}} \
        ${BETA:+--beta=${BETA}} \
        ${EPSILON:+--epsilon=${EPSILON}} \
        ${EPSILON_HIGH:+--epsilon_high=${EPSILON_HIGH}} \
        ${TEMPERATURE:+--temperature=${TEMPERATURE}} \
        ${TOP_P:+--top_p=${TOP_P}} \
        ${TOP_K:+--top_k=${TOP_K}} \
        ${MAX_STALENESS:+--offpolicy=${MAX_STALENESS}} \
        ${SEED:+--seed=${SEED}} \
        ${shuffle_arg} \
        ${ORCHESTRATOR_EXTRA_ARGS:+${ORCHESTRATOR_EXTRA_ARGS} }${debug_flag} \
    " \
    | apply_manifest
}

stop_trainer() {
  if [[ "$DRY_RUN" == "true" ]]; then
    echo "kubectl delete jobset ${TRAINER_ID} -n ${K8S_NAMESPACE}"
    echo "kubectl delete workload -l jobset.sigs.k8s.io/jobset-name=${TRAINER_ID} -n ${K8S_NAMESPACE}"
  else
    kubectl delete jobset "${TRAINER_ID}" -n "${K8S_NAMESPACE}" --ignore-not-found --wait=true
    kubectl delete workload -l "jobset.sigs.k8s.io/jobset-name=${TRAINER_ID}" -n "${K8S_NAMESPACE}" --ignore-not-found=true 2>/dev/null || true
    while kubectl get jobset "${TRAINER_ID}" -n "${K8S_NAMESPACE}" &>/dev/null; do
      sleep 2
    done
  fi
}

start_trainer() {
  local extra_flags=""
  local debug_flag=""
  local profiler_flags=""
  if [[ "${DEBUG}" == "1" || "${DEBUG}" == "true" || "${DEBUG}" == "True" ]]; then
    debug_flag="--debug"
  fi

  if [[ "${TRAINER_JOBSET_YAML}" == "jobset.pathways.yaml" ]]; then
    echo "Trainer Pathways images: server=${PATHWAYS_SERVER_IMAGE} proxy=${PATHWAYS_PROXY_IMAGE}"
  fi

  if [[ -n "$PROFILER_STEPS" ]]; then
    profiler_flags+=" --profiler_steps=${PROFILER_STEPS}"
  fi
  if [[ -n "$SKIP_FIRST_N_PROFILER_STEPS" ]]; then
    profiler_flags+=" --skip_first_n_profiler_steps=${SKIP_FIRST_N_PROFILER_STEPS}"
  fi
  if [[ -n "$PROFILER_PERIOD" ]]; then
    profiler_flags+=" --profiler_period=${PROFILER_PERIOD}"
  fi

  if [[ "${TRAINER_BACKEND}" == "maxtext" ]]; then
    if [[ -z "${MAXTEXT_CKPT}" ]]; then
      if [[ "${DRY_RUN}" == "true" ]]; then
        echo "DRY_RUN: MAXTEXT_CKPT is empty, using dummy checkpoint path for manifest rendering."
        MAXTEXT_CKPT="gs://dummy-checkpoint/path"
      else
        echo "Error: MAXTEXT_CKPT must be set when TRAINER_BACKEND=maxtext."
        exit 1
      fi
    fi
    extra_flags+="\
      --maxtext_model_name=${MAXTEXT_MODEL_NAME} \
      --maxtext_load_parameters_path=${MAXTEXT_CKPT} \
      --maxtext_checkpoint_dir=${MAXTEXT_OUTPUT_DIR} \
      ${TRAINER_PADDED_MOE_MLP_DIM:+--intermediate_dim=${TRAINER_PADDED_MOE_MLP_DIM}} \
      --use_weight_converter=${USE_WEIGHT_CONVERTER} \
      ${MAX_SEQ_TOKEN_PER_TPU:+--max_seq_token_per_tpu=${MAX_SEQ_TOKEN_PER_TPU}} \
    "
    if [[ "${TRAINER_EXTRA_ARGS:-}" != *"--base_num_kv_heads"* && -n "${TRAINER_BASE_NUM_KV_HEADS:-}" ]]; then
      extra_flags+=" --base_num_kv_heads=${TRAINER_BASE_NUM_KV_HEADS}"
    fi
    if [[ "${TRAINER_EXTRA_ARGS:-}" != *"--maxtext_attention"* && -n "${TRAINER_MAXTEXT_ATTENTION:-}" ]]; then
      extra_flags+=" --maxtext_attention=${TRAINER_MAXTEXT_ATTENTION}"
    fi
    if [[ "${TRAINER_EXTRA_ARGS:-}" != *"--remat_policy"* && -n "${REMAT_POLICY:-}" ]]; then
      extra_flags+=" --remat_policy=${REMAT_POLICY}"
    fi
    if [[ "${TRAINER_EXTRA_ARGS:-}" != *"--learning_rate_final_fraction"* && -n "${LEARNING_RATE_FINAL_FRACTION:-}" ]]; then
      extra_flags+=" --learning_rate_final_fraction=${LEARNING_RATE_FINAL_FRACTION}"
    fi
    if [[ "${TRAINER_EXTRA_ARGS:-}" != *"--maxtext_warmup_steps_fraction"* && -n "${WARMUP_STEPS_FRACTION:-}" ]]; then
      extra_flags+=" --maxtext_warmup_steps_fraction=${WARMUP_STEPS_FRACTION}"
    fi
    if [[ "${TRAINER_EXTRA_ARGS:-}" != *"--trainable_parameters_mask"* && -n "${TRAINABLE_PARAMETERS_MASK:-}" ]]; then
      extra_flags+=" --trainable_parameters_mask='${TRAINABLE_PARAMETERS_MASK}'"
    fi
    if [[ "${TRAINER_EXTRA_ARGS:-}" != *"--prefuse_moe_weights"* && -n "${TRAINER_PREFUSE_MOE_WEIGHTS:-}" ]]; then
      extra_flags+=" --prefuse_moe_weights=${TRAINER_PREFUSE_MOE_WEIGHTS}"
    fi
  fi

  local raiden_env=""
  if [[ "${WEIGHT_SYNC_MODE}" == "raiden" ]]; then
    raiden_env+=" RAIDEN_USE_FFI=1"
    if [[ -n "${RAIDEN_DEVICES_PER_HOST:-}" ]]; then
      raiden_env+=" RAIDEN_DEVICES_PER_HOST=${RAIDEN_DEVICES_PER_HOST}"
    fi
  fi

  "$PYTHON" "$YAML_GEN" \
    "$YAML_DIR/${TRAINER_JOBSET_YAML}" \
    --jobset_name="${TRAINER_ID}" \
    --namespace="${K8S_NAMESPACE}" \
    ${KUEUE_QUEUE_NAME:+--queue_name="${KUEUE_QUEUE_NAME}"} \
    --tpu_slice="${TRAINER_TPU_SLICE}" \
    --pathways_server_image="${PATHWAYS_SERVER_IMAGE}" \
    --pathways_proxy_server_image="${PATHWAYS_PROXY_IMAGE}" \
    --pathways_proxy_memory_limit="${PATHWAYS_PROXY_MEMORY_LIMIT}" \
    --pathways_proxy_memory="${PATHWAYS_PROXY_MEMORY}" \
    --pathways_rm_memory="${PATHWAYS_RM_MEMORY}" \
    --user_container_memory="${USER_CONTAINER_MEMORY}" \
    --user_container_memory_limit="${USER_CONTAINER_MEMORY_LIMIT}" \
    --pathways_worker_memory="${PATHWAYS_WORKER_MEMORY}" \
    --pathways_gcs_scratch_location=${GCS_SCRATCH_LOCATION} \
    --worker_container_image="${TUNIX_IMAGE}" \
    --worker_container_port="${TRAINER_PORT}" \
    --worker_startup_command=" \
      ${HF_TOKEN:+HF_TOKEN=\"${HF_TOKEN}\"} \
      VERIFY_WEIGHTS=${VERIFY_WEIGHTS}${raiden_env} \
      ${TRAINER_EXTRA_ENV:+${TRAINER_EXTRA_ENV} }python -m tunix.experimental.distributed.runtime.main \
        --discovery_addrs=${ORCHESTRATOR_ID}:${ORCHESTRATOR_PORT} \
        --process_executor=tunix.experimental.distributed.runtime.executor.K8sExecutor \
        --process_main=tunix.experimental.examples.common.run_trainer_node.main \
        --worker_id=${TRAINER_ID} \
        --port=${TRAINER_PORT} \
        --trainer_backend=${TRAINER_BACKEND} \
        --mesh_fsdp=${TRAINER_MESH_FSDP} \
        --mesh_tp=${TRAINER_MESH_TP} \
        --mesh_expert=${TRAINER_MESH_EXPERT} \
        --model_name=${MODEL_NAME} \
        --model_id=${MODEL_ID} \
        --tokenizer_path=${TOKENIZER_PATH} \
        --max_prompt_length=${MAX_PROMPT_LENGTH} \
        --max_response_length=${MAX_RESPONSE_LENGTH} \
        --lora_rank=${LORA_RANK} \
        --lora_alpha=${LORA_ALPHA} \
        --opt_chain_type=${OPT_CHAIN_TYPE} \
        --max_grad_norm=${MAX_GRAD_NORM} \
        --adam_b1=${ADAM_B1} \
        --adam_b2=${ADAM_B2} \
        --adam_eps=${ADAM_EPS} \
        --weight_decay=${WEIGHT_DECAY} \
        --learning_rate=${LEARNING_RATE} \
        --schedule_type=${SCHEDULE_TYPE} \
        --lr_init_value=${LR_INIT_VALUE} \
        --lr_peak_value=${LR_PEAK_VALUE} \
        --lr_end_value=${LR_END_VALUE} \
        --lr_decay_steps=${LR_DECAY_STEPS} \
        --warmup_steps=${WARMUP_STEPS} \
        --compute_logps_chunk_size=${COMPUTE_LOGPS_CHUNK_SIZE} \
        --weight_sync_mode=${WEIGHT_SYNC_MODE} \
        --checkpoint_save_interval_steps=${CHECKPOINT_SAVE_INTERVAL_STEPS} \
        --checkpoint_max_to_keep=${CHECKPOINT_MAX_TO_KEEP} \
        --checkpoint_root_directory=${CHECKPOINT_ROOT_DIRECTORY} \
        --enable_pathways_persistence=${ENABLE_PATHWAYS_PERSISTENCE} \
        --ckpt_d2h_concurrent_gb=${CKPT_D2H_CONCURRENT_GB} \
        ${extra_flags} \
        ${profiler_flags} \
        ${TRAINER_EXTRA_ARGS:+${TRAINER_EXTRA_ARGS} }${debug_flag} \
    " \
    | apply_manifest
}

stop_rollout_instance() {
  local target_id="$1"
  if [[ "${ROLLOUT_JOBSET_YAML}" == "leaderworkerset.mcjax.ray.yaml" ]]; then
    if [[ "$DRY_RUN" == "true" ]]; then
      echo "kubectl delete leaderworkerset ${target_id} -n ${K8S_NAMESPACE}"
    else
      kubectl delete leaderworkerset "${target_id}" -n "${K8S_NAMESPACE}" --ignore-not-found --wait=true
      while kubectl get leaderworkerset "${target_id}" -n "${K8S_NAMESPACE}" &>/dev/null; do
        sleep 2
      done
    fi
  else
    if [[ "$DRY_RUN" == "true" ]]; then
      echo "kubectl delete jobset ${target_id} -n ${K8S_NAMESPACE}"
      echo "kubectl delete workload -l jobset.sigs.k8s.io/jobset-name=${target_id} -n ${K8S_NAMESPACE}"
    else
      kubectl delete jobset "${target_id}" -n "${K8S_NAMESPACE}" --ignore-not-found --wait=true
      kubectl delete workload -l "jobset.sigs.k8s.io/jobset-name=${target_id}" -n "${K8S_NAMESPACE}" --ignore-not-found=true 2>/dev/null || true
      while kubectl get jobset "${target_id}" -n "${K8S_NAMESPACE}" &>/dev/null; do
        sleep 2
      done
    fi
  fi
}

stop_rollout() {
  stop_rollout_instance "${ROLLOUT_ID}"
  for ((i = 0; i < ROLLOUT_REPLICAS; i++)); do
    stop_rollout_instance "${ROLLOUT_ID}-${i}"
  done
}

start_rollout_instance() {
  local target_id="$1"
  local extra_flags=""
  local debug_flag=""
  if [[ "${DEBUG}" == "1" || "${DEBUG}" == "true" || "${DEBUG}" == "True" ]]; then
    debug_flag="--debug"
  fi

  if [[ "${ROLLOUT_JOBSET_YAML}" == "jobset.pathways.yaml" ]]; then
    echo "Rollout Pathways images: server=${PATHWAYS_SERVER_IMAGE} proxy=${PATHWAYS_PROXY_IMAGE}"
  fi

  if [[ "${TRAINER_BACKEND}" == "maxtext" ]]; then
    extra_flags+="\
      --maxtext_model_name=${MAXTEXT_MODEL_NAME} \
      ${ROLLOUT_MAXTEXT_ATTENTION:+--maxtext_attention=${ROLLOUT_MAXTEXT_ATTENTION}} \
    "
  fi

  local vllm_args=""
  if [[ "$SAMPLER" == "vllm" || "$SAMPLER" == "inprocess_vllm" ]]; then
    local vllm_json=""
    if [[ -n "${VLLM_CONFIG_JSON:-}" || -n "${ROLLOUT_VLLM_CONFIG_JSON:-}" || -n "${VLLM_MAX_NUM_BATCHED_TOKENS:-}" || -n "${VLLM_MAX_NUM_SEQS:-}" || -n "${VLLM_GPU_MEMORY_UTILIZATION:-}" || -n "${VLLM_ADDITIONAL_CONFIG:-}" || -n "${VLLM_MAX_MODEL_LEN:-}" || -n "${VLLM_BLOCK_SIZE:-}" ]]; then
      vllm_json=$("${PYTHON:-python3}" -c '
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

    if [[ -n "${vllm_json}" && "${ROLLOUT_EXTRA_ARGS:-}" != *"--vllm_config_json"* ]]; then
      vllm_args+=" --vllm_config_json='${vllm_json}'"
    fi
    if [[ "${ROLLOUT_EXTRA_ARGS:-}" != *"--tensor_parallel_size"* && -n "${ROLLOUT_MESH_TP:-}" ]]; then
      vllm_args+=" --tensor_parallel_size=${ROLLOUT_MESH_TP}"
    fi
  fi

  if [[ "${ROLLOUT_EXTRA_ARGS:-}" != *"--return_routed_experts"* && -n "${RETURN_ROUTED_EXPERTS:-}" ]]; then
    extra_flags+=" --return_routed_experts=${RETURN_ROUTED_EXPERTS}"
  fi
  if [[ "${ROLLOUT_EXTRA_ARGS:-}" != *"--free_kv_cache_during_weight_sync"* && -n "${ROLLOUT_FREE_KV_CACHE:-}" ]]; then
    extra_flags+=" --free_kv_cache_during_weight_sync=${ROLLOUT_FREE_KV_CACHE}"
  fi
  if [[ "${ROLLOUT_EXTRA_ARGS:-}" != *"--max_concurrency"* && -n "${ROLLOUT_MAX_CONCURRENCY:-}" ]]; then
    extra_flags+=" --max_concurrency=${ROLLOUT_MAX_CONCURRENCY}"
  fi

  local raiden_env=""
  if [[ "${WEIGHT_SYNC_MODE}" == "raiden" ]]; then
    raiden_env+=" RAIDEN_USE_FFI=0"
  fi

  "$PYTHON" "$YAML_GEN" \
    "$YAML_DIR/${ROLLOUT_JOBSET_YAML}" \
    --jobset_name="${target_id}" \
    --namespace="${K8S_NAMESPACE}" \
    ${KUEUE_QUEUE_NAME:+--queue_name="${KUEUE_QUEUE_NAME}"} \
    --tpu_slice="${ROLLOUT_TPU_SLICE}" \
    --pathways_server_image="${PATHWAYS_SERVER_IMAGE}" \
    --pathways_proxy_server_image="${PATHWAYS_PROXY_IMAGE}" \
    --pathways_gcs_scratch_location=${GCS_SCRATCH_LOCATION} \
    --worker_container_image="${TUNIX_IMAGE}" \
    --worker_container_port="${ROLLOUT_PORT}" \
    --worker_startup_command=" \
      (python3 -c 'import gymnasium' 2>/dev/null || pip install --no-cache-dir gymnasium) && \
      ${HF_TOKEN:+HF_TOKEN=\"${HF_TOKEN}\"} SKIP_JAX_PRECOMPILE=1 VERIFY_WEIGHTS=${VERIFY_WEIGHTS}${raiden_env}${ROLLOUT_EXTRA_ENV:+ ${ROLLOUT_EXTRA_ENV}} ${ROLLOUT_USE_BATCHED_RPA:+USE_BATCHED_RPA_KERNEL=1} python -m tunix.experimental.distributed.runtime.main \
        --discovery_addrs=${ORCHESTRATOR_ID}:${ORCHESTRATOR_PORT} \
        --process_executor=tunix.experimental.distributed.runtime.executor.K8sExecutor \
        --process_main=tunix.experimental.examples.common.run_rollout_node.main \
        --registry_module=${ROLLOUT_REGISTRY_MODULE} \
        --env_name=${ROLLOUT_ENV_NAME} \
        --agent_name=${ROLLOUT_AGENT_NAME} \
        --worker_id=${target_id} \
        --port=${ROLLOUT_PORT} \
        --mesh_fsdp=${ROLLOUT_MESH_FSDP} \
        --mesh_tp=${ROLLOUT_MESH_TP} \
        --model_name=${MODEL_NAME} \
        --model_id=${MODEL_ID} \
        --model_dir=${MODEL_DIR} \
        --tokenizer_path=${TOKENIZER_PATH} \
        --max_prompt_length=${MAX_PROMPT_LENGTH} \
        --max_response_length=${MAX_RESPONSE_LENGTH} \
        ${EOS_TOKENS:+--eos_tokens=\"${EOS_TOKENS}\"} \
        --sampler=${SAMPLER} \
        --lora_rank=${LORA_RANK} \
        --lora_alpha=${LORA_ALPHA} \
        --weight_sync_mode=${WEIGHT_SYNC_MODE} \
        --chat_parser=${CHAT_PARSER} \
        --prefuse_moe_weights=${PREFUSE_MOE_WEIGHTS} \
        --enable_prefix_caching=${ENABLE_PREFIX_CACHING} \
        ${extra_flags} \
        ${vllm_args} \
        ${ROLLOUT_EXTRA_ARGS:+${ROLLOUT_EXTRA_ARGS} }${debug_flag} \
    " \
    | apply_manifest
}

start_rollout() {
  for ((i = 0; i < ROLLOUT_REPLICAS; i++)); do
    local target_id="${ROLLOUT_ID}"
    if [[ $ROLLOUT_REPLICAS -gt 1 ]]; then
      target_id="${ROLLOUT_ID}-${i}"
    fi
    start_rollout_instance "${target_id}"
  done
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    start|stop|orchestrator|trainer|rollout)
      COMMAND="$1"
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
    --dry-run|--render)
      DRY_RUN=true
      shift
      ;;
    --scratch=*|--gcs-scratch=*)
      GCS_SCRATCH_LOCATION="${1#*=}"
      shift
      ;;
    --scratch|--gcs-scratch)
      GCS_SCRATCH_LOCATION="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [start|stop|orchestrator|trainer|rollout] [options]"
      echo "Options:"
      echo "  --command <cmd>          Command to run (start, stop, orchestrator, trainer, rollout)"
      echo "  --namespace <ns>         Kubernetes namespace (default: default)"
      echo "  --queue <name>           Kueue local queue name (optional)"
      echo "  --image <image>          Container image to use"
      echo "  --dry-run, --render      Print generated YAMLs without applying"
      echo "  --scratch, --gcs-scratch GCS scratch location"
      echo "  --debug, --no-debug      Toggle debug logging (default: disabled)"
      exit 0
      ;;
    *)
      shift
      ;;
  esac
done

if [[ "$DRY_RUN" != "true" ]]; then
  ENTER_KUBE_CONTEXT=${ENTER_KUBE_CONTEXT:-"${LAUNCHER_DIR}/../common/enter_kube_context.sh"}
  source "${ENTER_KUBE_CONTEXT}"
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
