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

export MODEL_NAME=${MODEL_NAME:-Qwen3-1.7B}
export MODEL_ID=${MODEL_ID:-Qwen/Qwen3-1.7B}
# Must be model-specific: vLLM prioritizes non-empty local snapshot directories,
# which can cause stale config/shape mismatches if shared across models.
export MODEL_DIR=${MODEL_DIR:-artifacts/qwen3_dist_gsm8k/models/${MODEL_NAME}}
# Defaults to MODEL_ID so AutoTokenizer downloads directly from HuggingFace
# instead of failing on an initially empty local MODEL_DIR.
export TOKENIZER_PATH=${TOKENIZER_PATH:-${MODEL_ID}}

export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-128}
export BATCH_SIZE=${BATCH_SIZE:-2}
export NUM_GENERATIONS=${NUM_GENERATIONS:-2}
export MAX_STEPS=${MAX_STEPS:-1}
export TRAIN_MICRO_BATCH_SIZE=${TRAIN_MICRO_BATCH_SIZE:-1}
export MAX_SEQ_TOKEN_PER_TPU=${MAX_SEQ_TOKEN_PER_TPU:-}
export MAX_SEGMENTS_PER_PACKED_ROW=${MAX_SEGMENTS_PER_PACKED_ROW:-}

# Set to tunix to run Tunix's PeftTrainer, and maxtext to run MaxText's MaxTextTrainingEngine
export TRAINER_BACKEND=${TRAINER_BACKEND:-tunix}
export MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-$((BATCH_SIZE * NUM_GENERATIONS))}
export EVAL_EVERY_N_STEPS=${EVAL_EVERY_N_STEPS:-1000000}
export MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}
export ADAM_B1=${ADAM_B1:-0.9}
export ADAM_B2=${ADAM_B2:-0.99}
export WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
export LEARNING_RATE=${LEARNING_RATE:-2.0e-7}
export LORA_RANK=${LORA_RANK:-16}
export LORA_ALPHA=${LORA_ALPHA:-16.0}
export USE_LORA=${USE_LORA:-0}
export REWARD_MODE=${REWARD_MODE:-env}
export BETA=${BETA:-0}
export EPSILON=${EPSILON:-0.2}
export DEBUG=${DEBUG:-0}
export SAMPLER=${SAMPLER:-inprocess_vllm}
export WEIGHT_SYNC_MODE=${WEIGHT_SYNC_MODE:-none}
export PREFUSE_MOE_WEIGHTS=${PREFUSE_MOE_WEIGHTS:-true}
export USE_WEIGHT_CONVERTER=${USE_WEIGHT_CONVERTER:-true}
export USE_ROLLOUT_LOGPS=${USE_ROLLOUT_LOGPS:-true}
export CHECKPOINT_SAVE_INTERVAL_STEPS=${CHECKPOINT_SAVE_INTERVAL_STEPS:-1}
export CHECKPOINT_MAX_TO_KEEP=${CHECKPOINT_MAX_TO_KEEP:-10}
export CHECKPOINT_ROOT_DIRECTORY=${CHECKPOINT_ROOT_DIRECTORY:-checkpoints}
export DISABLE_CHECKPOINTING=${DISABLE_CHECKPOINTING:-false}

# MaxText trainer configuration: only consulted when TRAINER_BACKEND=maxtext
export MAXTEXT_MODEL_NAME=${MAXTEXT_MODEL_NAME:-qwen3-1.7b}
export MAXTEXT_CKPT=${MAXTEXT_CKPT:-}
# If TRAINER_BACKEND=maxtext, MAXTEXT_CKPT must be set to the path of an Orbax params-only checkpoint.
if [[ "$TRAINER_BACKEND" == "maxtext" && -z "$MAXTEXT_CKPT" ]]; then
  echo "Error: TRAINER_BACKEND=maxtext requires MAXTEXT_CKPT (Orbax params-only checkpoint)."
  exit 1
fi
export MAXTEXT_OUTPUT_DIR=${MAXTEXT_OUTPUT_DIR:-artifacts/math_gsm8k_dist/maxtext}
# Padded MoE MLP intermediate dimension; must match rollout TP padding for MoE models.
export TRAINER_PADDED_MOE_MLP_DIM=${TRAINER_PADDED_MOE_MLP_DIM:-}
# Optional: enable experimental batched-RPA attention kernel for rollout.
export ROLLOUT_USE_BATCHED_RPA=${ROLLOUT_USE_BATCHED_RPA:-}
export ROLLOUT_MAXTEXT_ATTENTION=${ROLLOUT_MAXTEXT_ATTENTION:-}
export ENABLE_PREFIX_CACHING=${ENABLE_PREFIX_CACHING:-false}

# Logs source/destination Raiden tensor checksums on both the trainer and
# rollout sides during weight sync, for cross-verification of a real run.
export VERIFY_WEIGHTS=${VERIFY_WEIGHTS:-false}
# Number of accelerator devices per host for Raiden FFI initialization.
# Set to 4 for TPU v5e/v5p multi-host slices; set to 8 for 8-device host architectures.
export RAIDEN_DEVICES_PER_HOST=${RAIDEN_DEVICES_PER_HOST:-4}

export WANDB_PROJECT=${WANDB_PROJECT:-trellis-gsm8k}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-}
export WANDB_API_KEY=${WANDB_API_KEY:-}
export TFDS_DATA_DIR=${TFDS_DATA_DIR:-"artifacts/data"}
export TFDS_SPLIT=${TFDS_SPLIT:-train}
export FLUSH_METRICS_EVERY_N_STEPS=${FLUSH_METRICS_EVERY_N_STEPS:-1}

export NAMESPACE=${NAMESPACE:-default}
export QUEUE_NAME=${QUEUE_NAME:-}

export ORCHESTRATOR_ID=$USER-orch
export ORCHESTRATOR_PORT=20000

export ROLLOUT_ID=$USER-roll
export ROLLOUT_PORT=20001
export ROLLOUT_REPLICAS=${ROLLOUT_REPLICAS:-1}

export TRAINER_ID=$USER-train
export TRAINER_PORT=20002

export CPU_MACHINE=${CPU_MACHINE:-n2-standard-64}
export GCS_SCRATCH_LOCATION=${GCS_SCRATCH_LOCATION:-gs://cloud-pathways-staging/tmp}
export DRY_RUN=${DRY_RUN:-false}

export TRAINER_JOBSET_YAML=${TRAINER_JOBSET_YAML:-jobset.pathways.yaml}
export TRAINER_TPU_SLICE=${TRAINER_TPU_SLICE:-tpuv5e:4x4}
export TRAINER_MESH_FSDP=${TRAINER_MESH_FSDP:-16}
export TRAINER_MESH_TP=${TRAINER_MESH_TP:-1}
export TRAINER_MESH_EXPERT=${TRAINER_MESH_EXPERT:-1}

export PATHWAYS_SERVER_IMAGE=${PATHWAYS_SERVER_IMAGE:-us-docker.pkg.dev/cloud-tpu-v2-images/pathways/server:latest}
export PATHWAYS_PROXY_IMAGE=${PATHWAYS_PROXY_IMAGE:-us-docker.pkg.dev/cloud-tpu-v2-images/pathways/proxy_server:latest}

export ROLLOUT_JOBSET_YAML=${ROLLOUT_JOBSET_YAML:-jobset.tpu.yaml}
export ROLLOUT_TPU_SLICE=${ROLLOUT_TPU_SLICE:-tpuv5e:4x4}
export ROLLOUT_MESH_FSDP=${ROLLOUT_MESH_FSDP:-1}
export ROLLOUT_MESH_TP=${ROLLOUT_MESH_TP:-16}

apply_manifest() {
  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "---"
    cat
  else
    kubectl apply -f -
  fi
}


stop_orchestrator() {
  kubectl delete jobset "${ORCHESTRATOR_ID}" --namespace="${NAMESPACE}" --ignore-not-found
}

start_orchestrator() {
  local debug_flag=""
  if [[ "${DEBUG}" == "1" || "${DEBUG}" == "true" || "${DEBUG}" == "True" ]]; then
    debug_flag="--debug"
  fi

  python3 tunix/experimental/distributed/deployment/yaml_generator.py \
    tunix/experimental/distributed/deployment/yamls/jobset.cpu.yaml \
    --jobset_name="${ORCHESTRATOR_ID}" \
    --namespace="${NAMESPACE}" \
    ${QUEUE_NAME:+--queue_name="${QUEUE_NAME}"} \
    --cpu_machine=${CPU_MACHINE} \
    --worker_container_image="${TUNIX_IMAGE}" \
    --worker_container_port="${ORCHESTRATOR_PORT}" \
    --worker_startup_command=" \
      ${HF_TOKEN:+HF_TOKEN=\"${HF_TOKEN}\"} \
      ${WANDB_API_KEY:+WANDB_API_KEY=\"${WANDB_API_KEY}\"} \
      WANDB_PROJECT=\"${WANDB_PROJECT}\" \
      WANDB_RUN_NAME=\"${WANDB_RUN_NAME}\" \
      python -m tunix.experimental.distributed.runtime.main \
        --discovery_id=${ORCHESTRATOR_ID} \
        --discovery_port=${ORCHESTRATOR_PORT} \
        --process_main=tunix.experimental.examples.math_gsm8k_dist.run_gsm8k_dist_grpo.main \
        --model_id=${MODEL_ID} \
        --tokenizer_path=${TOKENIZER_PATH} \
        --batch_size=${BATCH_SIZE} \
        --num_generations=${NUM_GENERATIONS} \
        --max_steps=${MAX_STEPS} \
        --max_prompt_length=${MAX_PROMPT_LENGTH} \
        --max_response_length=${MAX_RESPONSE_LENGTH} \
        --train_micro_batch_size=${TRAIN_MICRO_BATCH_SIZE} \
        --rollout_replicas=${ROLLOUT_REPLICAS} \
        --wandb_project=\"${WANDB_PROJECT}\" \
        --wandb_run_name=\"${WANDB_RUN_NAME}\" \
        --flush_metrics_every_n_steps=${FLUSH_METRICS_EVERY_N_STEPS} \
        --weight_sync_mode=${WEIGHT_SYNC_MODE} \
        --stop_workers_on_exit \
        $([[ "${USE_ROLLOUT_LOGPS}" == "false" || "${USE_ROLLOUT_LOGPS}" == "False" || "${USE_ROLLOUT_LOGPS}" == "0" ]] && echo --no-use_rollout_logps || echo --use_rollout_logps) \
        ${MAX_SEQ_TOKEN_PER_TPU:+--max_seq_token_per_tpu=${MAX_SEQ_TOKEN_PER_TPU}} \
        ${MAX_SEGMENTS_PER_PACKED_ROW:+--max_segments_per_packed_row=${MAX_SEGMENTS_PER_PACKED_ROW}} \
        ${TRAINER_MESH_FSDP:+--trainer_fsdp=${TRAINER_MESH_FSDP}} \
        ${debug_flag} \
    " \
    | apply_manifest
}

stop_trainer() {
  kubectl delete jobset "${TRAINER_ID}" --namespace="${NAMESPACE}" --ignore-not-found
}

start_trainer() {
  local extra_flags=""
  local debug_flag=""
  if [[ "${DEBUG}" == "1" || "${DEBUG}" == "true" || "${DEBUG}" == "True" ]]; then
    debug_flag="--debug"
  fi

  if [[ "${TRAINER_JOBSET_YAML}" == "jobset.pathways.yaml" ]]; then
    echo "Trainer Pathways images: server=${PATHWAYS_SERVER_IMAGE} proxy=${PATHWAYS_PROXY_IMAGE}"
  fi

  if [[ "${TRAINER_BACKEND}" == "maxtext" ]]; then
    extra_flags+=" \
      --maxtext_model_name=${MAXTEXT_MODEL_NAME} \
      ${TRAINER_PADDED_MOE_MLP_DIM:+--maxtext_padded_moe_mlp_dim=${TRAINER_PADDED_MOE_MLP_DIM}} \
      --maxtext_ckpt_path=${MAXTEXT_CKPT} \
      --maxtext_output_directory=${MAXTEXT_OUTPUT_DIR} \
      --mesh_tp=${TRAINER_MESH_TP} \
      --mesh_expert=${TRAINER_MESH_EXPERT} \
      --rollout_mesh_tp=${ROLLOUT_MESH_TP} \
      --prefuse_moe_weights=${PREFUSE_MOE_WEIGHTS} \
      --use_weight_converter=${USE_WEIGHT_CONVERTER} \
    "
  fi

  local trainer_env="RAIDEN_DEVICES_PER_HOST=${RAIDEN_DEVICES_PER_HOST}"

  python3 tunix/experimental/distributed/deployment/yaml_generator.py \
    tunix/experimental/distributed/deployment/yamls/${TRAINER_JOBSET_YAML} \
    --jobset_name="${TRAINER_ID}" \
    --namespace="${NAMESPACE}" \
    ${QUEUE_NAME:+--queue_name="${QUEUE_NAME}"} \
    --tpu_slice=${TRAINER_TPU_SLICE} \
    --cpu_machine=${CPU_MACHINE} \
    --pathways_server_image="${PATHWAYS_SERVER_IMAGE}" \
    --pathways_proxy_server_image="${PATHWAYS_PROXY_IMAGE}" \
    --pathways_gcs_scratch_location=${GCS_SCRATCH_LOCATION} \
    --worker_container_image="${TUNIX_IMAGE}" \
    --worker_container_port="${TRAINER_PORT}" \
    --worker_startup_command=" \
      ${HF_TOKEN:+HF_TOKEN="${HF_TOKEN}"} VERIFY_WEIGHTS=${VERIFY_WEIGHTS} DISABLE_CHECKPOINTING=${DISABLE_CHECKPOINTING} ${trainer_env} python -m tunix.experimental.distributed.runtime.main \
        --discovery_addrs=${ORCHESTRATOR_ID}:${ORCHESTRATOR_PORT} \
        --process_executor=tunix.experimental.distributed.runtime.executor.K8sExecutor \
        --process_main=tunix.experimental.examples.common.run_trainer_node.main \
        --worker_id=${TRAINER_ID} \
        --port=${TRAINER_PORT} \
        --mesh_fsdp=${TRAINER_MESH_FSDP} \
        --trainer_backend=${TRAINER_BACKEND} \
        --model_name=${MODEL_NAME} \
        --model_id=${MODEL_ID} \
        --model_dir=${MODEL_DIR} \
        --sampler_type=${SAMPLER} \
        --tokenizer_path=${TOKENIZER_PATH} \
        --max_prompt_length=${MAX_PROMPT_LENGTH} \
        --max_response_length=${MAX_RESPONSE_LENGTH} \
        --mini_batch_size=${MINI_BATCH_SIZE} \
        --train_micro_batch_size=${TRAIN_MICRO_BATCH_SIZE} \
        --eval_every_n_steps=${EVAL_EVERY_N_STEPS} \
        --max_grad_norm=${MAX_GRAD_NORM} \
        --adam_b1=${ADAM_B1} \
        --adam_b2=${ADAM_B2} \
        --weight_decay=${WEIGHT_DECAY} \
        --learning_rate=${LEARNING_RATE} \
        --lora_rank=${LORA_RANK} \
        --lora_alpha=${LORA_ALPHA} \
        --checkpoint_save_interval_steps=${CHECKPOINT_SAVE_INTERVAL_STEPS} \
        --checkpoint_max_to_keep=${CHECKPOINT_MAX_TO_KEEP} \
        --checkpoint_root_directory=${CHECKPOINT_ROOT_DIRECTORY} \
        ${extra_flags} \
        ${debug_flag} \
    " \
    | apply_manifest
}

stop_rollout() {
  for ((i = 0; i < ROLLOUT_REPLICAS; i++)); do
    local target_id
    if [[ $ROLLOUT_REPLICAS -eq 1 ]]; then
      target_id="${ROLLOUT_ID}"
    else
      target_id="${ROLLOUT_ID}-${i}"
    fi
    if [[ "$ROLLOUT_JOBSET_YAML" =~ ^leaderworkerset ]]; then
      kubectl delete leaderworkerset "${target_id}" --namespace="${NAMESPACE}" --ignore-not-found
    else
      kubectl delete jobset "${target_id}" --namespace="${NAMESPACE}" --ignore-not-found
    fi
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
    extra_flags+=" \
      --maxtext_model_name=${MAXTEXT_MODEL_NAME} \
      ${ROLLOUT_MAXTEXT_ATTENTION:+--maxtext_attention=${ROLLOUT_MAXTEXT_ATTENTION}} \
      --prefuse_moe_weights=${PREFUSE_MOE_WEIGHTS} \
    "
  fi

  local rollout_env="RAIDEN_DEVICES_PER_HOST=${RAIDEN_DEVICES_PER_HOST}"

  python3 tunix/experimental/distributed/deployment/yaml_generator.py \
    tunix/experimental/distributed/deployment/yamls/${ROLLOUT_JOBSET_YAML} \
    --jobset_name="${target_id}" \
    --namespace="${NAMESPACE}" \
    ${QUEUE_NAME:+--queue_name="${QUEUE_NAME}"} \
    --tpu_slice="${ROLLOUT_TPU_SLICE}" \
    --pathways_server_image="${PATHWAYS_SERVER_IMAGE}" \
    --pathways_proxy_server_image="${PATHWAYS_PROXY_IMAGE}" \
    --pathways_gcs_scratch_location=${GCS_SCRATCH_LOCATION} \
    --worker_container_image="${TUNIX_IMAGE}" \
    --worker_container_port="${ROLLOUT_PORT}" \
    --worker_startup_command=" \
      ${HF_TOKEN:+HF_TOKEN="${HF_TOKEN}"} VERIFY_WEIGHTS=${VERIFY_WEIGHTS} ${rollout_env} ${ROLLOUT_USE_BATCHED_RPA:+USE_BATCHED_RPA_KERNEL=1} python -m tunix.experimental.distributed.runtime.main \
        --discovery_addrs=${ORCHESTRATOR_ID}:${ORCHESTRATOR_PORT} \
        --process_executor=tunix.experimental.distributed.runtime.executor.K8sExecutor \
        --process_main=tunix.experimental.examples.common.run_rollout_node.main \
        --worker_id=${target_id} \
        --port=${ROLLOUT_PORT} \
        --mesh_fsdp=${ROLLOUT_MESH_FSDP} \
        --mesh_tp=${ROLLOUT_MESH_TP} \
        --model_id=${MODEL_ID} \
        --model_dir=${MODEL_DIR} \
        --tokenizer_path=${TOKENIZER_PATH} \
        --max_prompt_length=${MAX_PROMPT_LENGTH} \
        --max_response_length=${MAX_RESPONSE_LENGTH} \
        --sampler=${SAMPLER} \
        --lora_rank=${LORA_RANK} \
        --lora_alpha=${LORA_ALPHA} \
        --weight_sync_mode=${WEIGHT_SYNC_MODE} \
        --enable_prefix_caching=${ENABLE_PREFIX_CACHING} \
        ${extra_flags} \
        ${debug_flag} \
    " \
    | apply_manifest
}

start_rollout() {
  for ((i = 0; i < ROLLOUT_REPLICAS; i++)); do
    local target_id
    if [[ $ROLLOUT_REPLICAS -eq 1 ]]; then
      target_id="${ROLLOUT_ID}"
    else
      target_id="${ROLLOUT_ID}-${i}"
    fi
    start_rollout_instance "${target_id}"
  done
}

source tunix/experimental/examples/common/enter_kube_context.sh

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
      NAMESPACE="$2"
      shift 2
      ;;
    --namespace=*)
      NAMESPACE="${1#*=}"
      shift
      ;;
    --queue)
      QUEUE_NAME="$2"
      shift 2
      ;;
    --queue=*)
      QUEUE_NAME="${1#*=}"
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
    *)
      shift
      ;;
  esac
done

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
  stop_orchestrator
  start_orchestrator
elif [[ "$COMMAND" == "trainer" ]]; then
  stop_trainer
  start_trainer
elif [[ "$COMMAND" == "rollout" ]]; then
  stop_rollout
  start_rollout
else
  echo "Error: Invalid command '$COMMAND'. Available commands: 'start', 'stop', 'orchestrator', 'trainer', 'rollout'."
  exit 1
fi
