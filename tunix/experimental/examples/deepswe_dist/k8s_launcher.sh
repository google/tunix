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

set -Ee

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

COMMAND=""
TUNIX_IMAGE=${TUNIX_IMAGE:-}

export MODEL_NAME=${MODEL_NAME:-Qwen3-1.7B}
export MODEL_ID=${MODEL_ID:-Qwen/Qwen3-1.7B}
export MODEL_DIR=${MODEL_DIR:-artifacts/qwen3_dist_deepswe/models/${MODEL_NAME}}
export TOKENIZER_PATH=${TOKENIZER_PATH:-${MODEL_ID}}

export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1024}
export BATCH_SIZE=${BATCH_SIZE:-1}
export NUM_GENERATIONS=${NUM_GENERATIONS:-2}
export MAX_STEPS=${MAX_STEPS:-1}
export MAX_TURNS=${MAX_TURNS:-3}
export TRAIN_MICRO_BATCH_SIZE=${TRAIN_MICRO_BATCH_SIZE:-1}
export MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-$((BATCH_SIZE * NUM_GENERATIONS))}
export EVAL_EVERY_N_STEPS=${EVAL_EVERY_N_STEPS:-1000000}
export BETA=${BETA:-0.0}
export EPSILON=${EPSILON:-0.2}
export SAMPLER=${SAMPLER:-inprocess_vllm}
export WEIGHT_SYNC_MODE=${WEIGHT_SYNC_MODE:-none}
export USE_LORA=${USE_LORA:-0}
export LORA_RANK=${LORA_RANK:-64}
export LORA_ALPHA=${LORA_ALPHA:-64.0}
export DEBUG=${DEBUG:-0}

export DATASET_NAME=${DATASET_NAME:-R2E-Gym/R2E-Gym-V1}
export DATASET_PATH=${DATASET_PATH:-}
export DATASET_SPLIT=${DATASET_SPLIT:-train}
export DATASET_CACHE_DIR=${DATASET_CACHE_DIR:-artifacts/qwen3_dist_deepswe/dataset_cache}
export SHUFFLE=${SHUFFLE:-true}
export SEED=${SEED:-42}
export ENV_BACKEND=${ENV_BACKEND:-kubernetes}
export SCAFFOLD=${SCAFFOLD:-r2egym}
export USE_AGENT_SANDBOX=${USE_AGENT_SANDBOX:-1}
export SANDBOX_NAMESPACE=${SANDBOX_NAMESPACE:-rl-tunix-swebench}
export SANDBOX_NODE_SELECTOR_KEY=${SANDBOX_NODE_SELECTOR_KEY:-cloud.google.com/gke-nodepool}
export SANDBOX_NODE_SELECTOR_VAL=${SANDBOX_NODE_SELECTOR_VAL:-deepswe-cpu-pool}
export SANDBOX_MAX_CONCURRENCY=${SANDBOX_MAX_CONCURRENCY:-1}
export STEP_TIMEOUT_SECS=${STEP_TIMEOUT_SECS:-1800}
export REWARD_TIMEOUT_SECS=${REWARD_TIMEOUT_SECS:-1800}
export ROLLOUT_MAX_CONCURRENCY=${ROLLOUT_MAX_CONCURRENCY:-64}

export WANDB_PROJECT=${WANDB_PROJECT:-trellis-deepswe}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-}
export WANDB_API_KEY=${WANDB_API_KEY:-}
export VERIFY_WEIGHTS=${VERIFY_WEIGHTS:-false}

export ORCHESTRATOR_ID=${ORCHESTRATOR_ID:-$USER-deepswe-orch}
export ORCHESTRATOR_PORT=${ORCHESTRATOR_PORT:-20000}
export TRAINER_ID=${TRAINER_ID:-$USER-deepswe-train}
export TRAINER_PORT=${TRAINER_PORT:-20002}
export ROLLOUT_ID=${ROLLOUT_ID:-$USER-deepswe-roll}
export ROLLOUT_PORT=${ROLLOUT_PORT:-20001}

export CPU_MACHINE=${CPU_MACHINE:-n2-standard-64}
export GCS_SCRATCH_LOCATION=${GCS_SCRATCH_LOCATION:-gs://cloud-pathways-staging/tmp}
export TRAINER_JOBSET_YAML=${TRAINER_JOBSET_YAML:-jobset.pathways.yaml}
export TRAINER_TPU_SLICE=${TRAINER_TPU_SLICE:-tpuv5:2x2x2}
export TRAINER_MESH_FSDP=${TRAINER_MESH_FSDP:-8}
export TRAINER_MESH_TP=${TRAINER_MESH_TP:-1}
export TRAINER_MESH_EXPERT=${TRAINER_MESH_EXPERT:-1}
export ROLLOUT_TPU_SLICE=${ROLLOUT_TPU_SLICE:-tpuv5:2x2x1}
export ROLLOUT_MESH_FSDP=${ROLLOUT_MESH_FSDP:-1}
export ROLLOUT_MESH_TP=${ROLLOUT_MESH_TP:-4}
export ROLLOUT_REPLICAS=${ROLLOUT_REPLICAS:-1}

DEBUG_FLAG=""
if [[ "$DEBUG" == "1" || "$DEBUG" == "true" || "$DEBUG" == "True" ]]; then
  DEBUG_FLAG="--debug"
fi

USE_LORA_FLAG=""
if [[ "$USE_LORA" == "1" || "$USE_LORA" == "true" || "$USE_LORA" == "True" ]]; then
  USE_LORA_FLAG="--use_lora"
fi

USE_AGENT_SANDBOX_FLAG=""
if [[ "$USE_AGENT_SANDBOX" == "1" || "$USE_AGENT_SANDBOX" == "true" || "$USE_AGENT_SANDBOX" == "True" ]]; then
  USE_AGENT_SANDBOX_FLAG="--use_agent_sandbox"
fi

SHUFFLE_FLAG="--shuffle"
if [[ "$SHUFFLE" == "0" || "$SHUFFLE" == "false" || "$SHUFFLE" == "False" ]]; then
  SHUFFLE_FLAG="--no-shuffle"
fi

if [[ "$BETA" != "0" && "$BETA" != "0.0" ]]; then
  echo "Error: this launcher starts actor and rollout workers only."
  echo "Use BETA=0.0 until a reference worker is added."
  exit 1
fi

stop_orchestrator() {
  kubectl delete jobset "${ORCHESTRATOR_ID}" 2>/dev/null || true
}

start_orchestrator() {
  python3 tunix/experimental/distributed/deployment/yaml_generator.py \
    tunix/experimental/distributed/deployment/yamls/jobset.cpu.yaml \
    --jobset_name="${ORCHESTRATOR_ID}" \
    --cpu_machine="${CPU_MACHINE}" \
    --worker_container_image="${TUNIX_IMAGE}" \
    --worker_container_port="${ORCHESTRATOR_PORT}" \
    --worker_startup_command=" \
      ${WANDB_API_KEY:+WANDB_API_KEY=\"${WANDB_API_KEY}\"} \
      WANDB_PROJECT=\"${WANDB_PROJECT}\" \
      WANDB_RUN_NAME=\"${WANDB_RUN_NAME}\" \
      python3 -m tunix.experimental.distributed.runtime.main \
        --discovery_id=${ORCHESTRATOR_ID} \
        --discovery_port=${ORCHESTRATOR_PORT} \
        --process_main=tunix.experimental.examples.deepswe_dist.run_deepswe_dist.main \
        --model_id=${MODEL_ID} \
        --tokenizer_path=${TOKENIZER_PATH} \
        --batch_size=${BATCH_SIZE} \
        --num_generations=${NUM_GENERATIONS} \
        --max_steps=${MAX_STEPS} \
        --max_turns=${MAX_TURNS} \
        --max_prompt_length=${MAX_PROMPT_LENGTH} \
        --max_response_length=${MAX_RESPONSE_LENGTH} \
        --train_micro_batch_size=${TRAIN_MICRO_BATCH_SIZE} \
        --beta=${BETA} \
        --epsilon=${EPSILON} \
        --dataset_name=${DATASET_NAME} \
        --dataset_split=${DATASET_SPLIT} \
        --dataset_cache_dir=${DATASET_CACHE_DIR} \
        --seed=${SEED} \
        --env_backend=${ENV_BACKEND} \
        --scaffold=${SCAFFOLD} \
        --step_timeout_secs=${STEP_TIMEOUT_SECS} \
        --reward_timeout_secs=${REWARD_TIMEOUT_SECS} \
        --weight_sync_mode=${WEIGHT_SYNC_MODE} \
        --stop_workers_on_exit \
        ${DATASET_PATH:+--dataset_path=${DATASET_PATH}} \
        ${USE_AGENT_SANDBOX_FLAG} \
        ${SHUFFLE_FLAG} \
        ${DEBUG_FLAG} \
    " \
    | kubectl apply -f -
}

stop_trainer() {
  kubectl delete jobset "${TRAINER_ID}" 2>/dev/null || true
}

start_trainer() {
  python3 tunix/experimental/distributed/deployment/yaml_generator.py \
    tunix/experimental/distributed/deployment/yamls/${TRAINER_JOBSET_YAML} \
    --jobset_name="${TRAINER_ID}" \
    --tpu_slice="${TRAINER_TPU_SLICE}" \
    --cpu_machine="${CPU_MACHINE}" \
    --pathways_gcs_scratch_location="${GCS_SCRATCH_LOCATION}" \
    --worker_container_image="${TUNIX_IMAGE}" \
    --worker_container_port="${TRAINER_PORT}" \
    --worker_startup_command=" \
      VERIFY_WEIGHTS=${VERIFY_WEIGHTS} \
      python3 -m tunix.experimental.distributed.runtime.main \
        --discovery_addrs=${ORCHESTRATOR_ID}:${ORCHESTRATOR_PORT} \
        --process_executor=tunix.experimental.distributed.runtime.executor.K8sExecutor \
        --process_main=tunix.experimental.examples.common.run_trainer_node.main \
        --worker_id=${TRAINER_ID} \
        --port=${TRAINER_PORT} \
        --mesh_fsdp=${TRAINER_MESH_FSDP} \
        --mesh_tp=${TRAINER_MESH_TP} \
        --mesh_expert=${TRAINER_MESH_EXPERT} \
        --model_name=${MODEL_NAME} \
        --model_id=${MODEL_ID} \
        --model_dir=${MODEL_DIR} \
        --tokenizer_path=${TOKENIZER_PATH} \
        --max_prompt_length=${MAX_PROMPT_LENGTH} \
        --max_response_length=${MAX_RESPONSE_LENGTH} \
        --mini_batch_size=${MINI_BATCH_SIZE} \
        --train_micro_batch_size=${TRAIN_MICRO_BATCH_SIZE} \
        --eval_every_n_steps=${EVAL_EVERY_N_STEPS} \
        --lora_rank=${LORA_RANK} \
        --lora_alpha=${LORA_ALPHA} \
        ${USE_LORA_FLAG} \
        ${DEBUG_FLAG} \
    " \
    | kubectl apply -f -
}

stop_rollout() {
  for i in $(seq 0 $((ROLLOUT_REPLICAS - 1))); do
    kubectl delete jobset "${ROLLOUT_ID}-${i}" 2>/dev/null || true
  done
}

start_rollout() {
  for i in $(seq 0 $((ROLLOUT_REPLICAS - 1))); do
    local replica_id="${ROLLOUT_ID}-${i}"
    python3 tunix/experimental/distributed/deployment/yaml_generator.py \
      tunix/experimental/distributed/deployment/yamls/jobset.tpu.yaml \
      --jobset_name="${replica_id}" \
      --tpu_slice="${ROLLOUT_TPU_SLICE}" \
      --worker_container_image="${TUNIX_IMAGE}" \
      --worker_container_port="${ROLLOUT_PORT}" \
      --worker_startup_command=" \
        SKIP_JAX_PRECOMPILE=1 \
        VERIFY_WEIGHTS=${VERIFY_WEIGHTS} \
        NAMESPACE=${SANDBOX_NAMESPACE} \
        NODE_SELECTOR_KEY=${SANDBOX_NODE_SELECTOR_KEY} \
        NODE_SELECTOR_VAL=${SANDBOX_NODE_SELECTOR_VAL} \
        DATASET_NAME=${DATASET_NAME} \
        DATASET_PATH=${DATASET_PATH} \
        DATASET_SPLIT=${DATASET_SPLIT} \
        DATASET_CACHE_DIR=${DATASET_CACHE_DIR} \
        SHUFFLE=${SHUFFLE} \
        SEED=${SEED} \
        ROLLOUT_MAX_CONCURRENCY=${ROLLOUT_MAX_CONCURRENCY} \
        SANDBOX_MAX_CONCURRENCY=${SANDBOX_MAX_CONCURRENCY} \
        python3 -m tunix.experimental.distributed.runtime.main \
          --discovery_addrs=${ORCHESTRATOR_ID}:${ORCHESTRATOR_PORT} \
          --process_executor=tunix.experimental.distributed.runtime.executor.K8sExecutor \
          --process_main=tunix.experimental.examples.common.run_rollout_node.main \
          --worker_id=${replica_id} \
          --port=${ROLLOUT_PORT} \
          --model_id=${MODEL_ID} \
          --model_dir=${MODEL_DIR} \
          --model_name=${MODEL_NAME} \
          --sampler=${SAMPLER} \
          --sampler_mesh_tp=${ROLLOUT_MESH_TP} \
          --mesh_fsdp=${ROLLOUT_MESH_FSDP} \
          --mesh_tp=${ROLLOUT_MESH_TP} \
          --tokenizer_path=${TOKENIZER_PATH} \
          --max_prompt_length=${MAX_PROMPT_LENGTH} \
          --max_response_length=${MAX_RESPONSE_LENGTH} \
          --lora_rank=${LORA_RANK} \
          --lora_alpha=${LORA_ALPHA} \
          --weight_sync_mode=${WEIGHT_SYNC_MODE} \
          --registry_module=tunix.experimental.examples.deepswe_dist.deepswe \
          --env_name=deepswe_env \
          --agent_name=deepswe_agent \
          --max_concurrency=${ROLLOUT_MAX_CONCURRENCY} \
          ${USE_LORA_FLAG} \
          ${DEBUG_FLAG} \
      " \
      | kubectl apply -f -
  done
}

if [[ -f tunix/experimental/examples/deepswe_dist/enter_kube_context.sh ]]; then
  source tunix/experimental/examples/deepswe_dist/enter_kube_context.sh
elif [[ -f tunix/experimental/examples/math_gsm8k_dist/enter_kube_context.sh ]]; then
  source tunix/experimental/examples/math_gsm8k_dist/enter_kube_context.sh
fi

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
    *)
      shift
      ;;
  esac
done

if [[ -z "$TUNIX_IMAGE" ]]; then
  echo "Error: no image set. Pass TUNIX_IMAGE=... or --image=..."
  exit 1
fi

case "$COMMAND" in
  start)
    stop_orchestrator
    stop_trainer
    stop_rollout
    start_orchestrator
    start_trainer
    start_rollout
    ;;
  stop)
    stop_orchestrator
    stop_trainer
    stop_rollout
    ;;
  orchestrator)
    stop_orchestrator
    start_orchestrator
    ;;
  trainer)
    stop_trainer
    start_trainer
    ;;
  rollout)
    stop_rollout
    start_rollout
    ;;
  *)
    echo "Error: Invalid command '$COMMAND'."
    echo "Available commands: start, stop, orchestrator, trainer, rollout."
    exit 1
    ;;
esac
