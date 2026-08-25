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
# No default: build an image with tunix, maxtext, and tpu-inference baked in
# at pinned commits (see requirements/special_requirements.txt), then pass it
# via TUNIX_IMAGE or --image.
TUNIX_IMAGE=${TUNIX_IMAGE:-}

export MODEL_NAME=${MODEL_NAME:-Qwen3-1.7B}
export MODEL_ID=${MODEL_ID:-Qwen/Qwen3-1.7B}
export MODEL_DIR=${MODEL_DIR:-artifacts/qwen3_dist_gsm8k/models}
export TOKENIZER_PATH=${TOKENIZER_PATH:-${MODEL_DIR}}
# only consulted when TRAINER_BACKEND=maxtext; keep it matching
# MODEL_NAME/MODEL_ID or Raiden weight sync between trainer and rollout won't
# line up
export MAXTEXT_MODEL_NAME=${MAXTEXT_MODEL_NAME:-qwen3-1.7b}

export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-128}
export BATCH_SIZE=${BATCH_SIZE:-2}
export NUM_GENERATIONS=${NUM_GENERATIONS:-2}
export MAX_STEPS=${MAX_STEPS:-1}
export TRAIN_MICRO_BATCH_SIZE=${TRAIN_MICRO_BATCH_SIZE:-1}

# peft runs tunix's PeftTrainer (default, matches this script's prior
# behavior); maxtext runs MaxText's MaxTextTrainingEngine.
export TRAINER_BACKEND=${TRAINER_BACKEND:-peft}
export MAXTEXT_CKPT=${MAXTEXT_CKPT:-}
export MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-$((BATCH_SIZE * NUM_GENERATIONS))}
export EVAL_EVERY_N_STEPS=${EVAL_EVERY_N_STEPS:-1000000}
export LORA_RANK=${LORA_RANK:-16}
export LORA_ALPHA=${LORA_ALPHA:-16.0}
export USE_LORA=${USE_LORA:-0}

# Logs source/destination Raiden tensor checksums on both the trainer and
# rollout sides during weight sync, for cross-verification of a real run.
export VERIFY_WEIGHTS=${VERIFY_WEIGHTS:-false}

export ORCHESTRATOR_ID=$USER-orch
export ORCHESTRATOR_PORT=20000

export ROLLOUT_ID=$USER-roll
export ROLLOUT_PORT=20001

export TRAINER_ID=$USER-train
export TRAINER_PORT=20002

export CPU_MACHINE=${CPU_MACHINE:-n2-standard-64}
export GCS_SCRATCH_LOCATION=${GCS_SCRATCH_LOCATION:-gs://cloud-pathways-staging/tmp}
# Empty by default: yaml_generator.py's own defaults (the public
# us-docker.pkg.dev/cloud-tpu-v2-images/pathways/{server,proxy_server}:latest
# images) apply unless overridden.
export PATHWAYS_SERVER_IMAGE=${PATHWAYS_SERVER_IMAGE:-}
export PATHWAYS_PROXY_IMAGE=${PATHWAYS_PROXY_IMAGE:-}
export TRAINER_JOBSET_YAML=${TRAINER_JOBSET_YAML:-jobset.pathways.yaml}
export TRAINER_TPU_SLICE=${TRAINER_TPU_SLICE:-tpuv5:2x2x2}
export TRAINER_MESH_FSDP=${TRAINER_MESH_FSDP:-8}
export TRAINER_MESH_TP=${TRAINER_MESH_TP:-1}
export TRAINER_MESH_EXPERT=${TRAINER_MESH_EXPERT:-1}
export ROLLOUT_TPU_SLICE=${ROLLOUT_TPU_SLICE:-tpuv5:2x2x1}
export ROLLOUT_TENSOR_PARALLEL_SIZE=${ROLLOUT_TENSOR_PARALLEL_SIZE:-4}
# Set both to try tpu-inference's experimental batched-RPA kernel instead of
# the default v3 ragged_paged_attention kernel. ROLLOUT_USE_BATCHED_RPA sets
# USE_BATCHED_RPA_KERNEL=1 as a pod env var (must be pre-import, since
# attention_interface.py picks the kernel at module-import time -- MaxText's
# own self-set of this var, gated on maxtext_attention, happens too late).
export ROLLOUT_USE_BATCHED_RPA=${ROLLOUT_USE_BATCHED_RPA:-}
export ROLLOUT_MAXTEXT_ATTENTION=${ROLLOUT_MAXTEXT_ATTENTION:-}
# Number of independent rollout replicas (each its own TPU slice reservation
# and JobSet, ${ROLLOUT_ID}-0, ${ROLLOUT_ID}-1, ...) for data-parallel
# rollout. A single multi-host ROLLOUT_TPU_SLICE (e.g. tpuv5:2x2x2) is one
# replica -- its pods share one JAX-distributed session and register under
# one worker_id, so it does NOT need REPLICAS>1. Use REPLICAS>1 to instead
# run N independent single-host (or smaller multi-host) slices, since a
# single multi-host slice reservation can't be split into independent JAX
# sessions (libtpu coordinates the whole physical slice as one session).
export ROLLOUT_REPLICAS=${ROLLOUT_REPLICAS:-1}

stop_orchestrator() {
  kubectl delete jobset "${ORCHESTRATOR_ID}" 2>/dev/null || true
}

start_orchestrator() {
  python tunix/experimental/distributed/deployment/yaml_generator.py \
    tunix/experimental/distributed/deployment/yamls/jobset.cpu.yaml \
    --jobset_name="${ORCHESTRATOR_ID}" \
    --cpu_machine=${CPU_MACHINE} \
    --worker_container_image="${TUNIX_IMAGE}" \
    --worker_container_port="${ORCHESTRATOR_PORT}" \
    --worker_startup_command=" \
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
        --num_rollout_workers=${ROLLOUT_REPLICAS} \
        --sync_weights \
        --stop_workers_on_exit \
    " \
    | kubectl apply -f -
}

stop_trainer() {
  kubectl delete jobset "${TRAINER_ID}" 2>/dev/null || true
}

start_trainer() {
  python tunix/experimental/distributed/deployment/yaml_generator.py \
    tunix/experimental/distributed/deployment/yamls/${TRAINER_JOBSET_YAML} \
    --jobset_name="${TRAINER_ID}" \
    --tpu_slice=${TRAINER_TPU_SLICE} \
    --cpu_machine=${CPU_MACHINE} \
    --pathways_gcs_scratch_location=${GCS_SCRATCH_LOCATION} \
    ${PATHWAYS_SERVER_IMAGE:+--pathways_server_image=${PATHWAYS_SERVER_IMAGE}} \
    ${PATHWAYS_PROXY_IMAGE:+--pathways_proxy_server_image=${PATHWAYS_PROXY_IMAGE}} \
    --worker_container_image="${TUNIX_IMAGE}" \
    --worker_container_port="${TRAINER_PORT}" \
    --worker_startup_command=" \
      VERIFY_WEIGHTS=${VERIFY_WEIGHTS} python -m tunix.experimental.distributed.runtime.main \
        --discovery_addrs=${ORCHESTRATOR_ID}:${ORCHESTRATOR_PORT} \
        --process_executor=tunix.experimental.distributed.runtime.executor.K8sExecutor \
        --process_main=tunix.experimental.examples.math_gsm8k_dist.run_trainer_node.main \
        --worker_id=${TRAINER_ID} \
        --port=${TRAINER_PORT} \
        --mesh_fsdp=${TRAINER_MESH_FSDP} \
        --mesh_tp=${TRAINER_MESH_TP} \
        --mesh_expert=${TRAINER_MESH_EXPERT} \
        --trainer_backend=${TRAINER_BACKEND} \
        ${MAXTEXT_CKPT:+--maxtext_load_parameters_path=${MAXTEXT_CKPT}} \
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
        --maxtext_model_name=${MAXTEXT_MODEL_NAME} \
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
    python tunix/experimental/distributed/deployment/yaml_generator.py \
      tunix/experimental/distributed/deployment/yamls/jobset.tpu.yaml \
      --jobset_name="${replica_id}" \
      --tpu_slice=${ROLLOUT_TPU_SLICE} \
      --worker_container_image="${TUNIX_IMAGE}" \
      --worker_container_port="${ROLLOUT_PORT}" \
      --worker_startup_command=" \
        SKIP_JAX_PRECOMPILE=1 VERIFY_WEIGHTS=${VERIFY_WEIGHTS} ${ROLLOUT_USE_BATCHED_RPA:+USE_BATCHED_RPA_KERNEL=1} python -m tunix.experimental.distributed.runtime.main \
          --discovery_addrs=${ORCHESTRATOR_ID}:${ORCHESTRATOR_PORT} \
          --process_executor=tunix.experimental.distributed.runtime.executor.K8sExecutor \
          --process_main=tunix.experimental.examples.math_gsm8k_dist.run_rollout_node.main \
          --worker_id=${replica_id} \
          --port=${ROLLOUT_PORT} \
          --model_id=${MODEL_ID} \
          --model_dir=${MODEL_DIR} \
          --tokenizer_path=${TOKENIZER_PATH} \
          --max_prompt_length=${MAX_PROMPT_LENGTH} \
          --max_response_length=${MAX_RESPONSE_LENGTH} \
          --lora_rank=${LORA_RANK} \
          --lora_alpha=${LORA_ALPHA} \
          --tensor_parallel_size=${ROLLOUT_TENSOR_PARALLEL_SIZE} \
          --maxtext_model_name=${MAXTEXT_MODEL_NAME} \
          ${ROLLOUT_MAXTEXT_ATTENTION:+--maxtext_attention=${ROLLOUT_MAXTEXT_ATTENTION}} \
      " \
      | kubectl apply -f -
  done
}

source tunix/experimental/examples/math_gsm8k_dist/enter_kube_context.sh

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
