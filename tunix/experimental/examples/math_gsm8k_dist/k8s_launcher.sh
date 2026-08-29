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
# Must be model-specific: it's a local HF snapshot cache, and vLLM's own
# model_config reads config.json straight from here whenever the directory
# is non-empty, bypassing MODEL_ID entirely. A shared path across different
# models silently reuses a stale snapshot -- confirmed root cause of a
# multi-hour KV-cache shape-mismatch chase on Qwen3-30B-A3B: with a leftover
# Qwen3-0.6B snapshot sitting here, vLLM's config (num_key_value_heads=8)
# and MaxText's own model construction (num_key_value_heads=4, correctly
# built from MAXTEXT_MODEL_NAME) silently disagreed.
export MODEL_DIR=${MODEL_DIR:-/app/artifacts/qwen3_dist_gsm8k/models}
# Defaults to MODEL_ID (an HF hub repo id), not MODEL_DIR: MODEL_DIR is a
# per-model-name local directory that starts out empty for any model that
# hasn't been downloaded there before by *something* -- nothing pre-populates
# it before the trainer's own AutoTokenizer.from_pretrained(tokenizer_path)
# call, which then misreads a nonexistent local path as a malformed hub repo
# id and fails HFValidationError. MODEL_ID always works: it's a genuine hub
# id AutoTokenizer downloads directly, independent of any local directory
# state or trainer/rollout pod download-ordering races.
export TOKENIZER_PATH=${TOKENIZER_PATH:-${MODEL_ID}}
# only consulted when TRAINER_BACKEND=maxtext; keep it matching
# MODEL_NAME/MODEL_ID or Raiden weight sync between trainer and rollout won't
# line up
export MAXTEXT_MODEL_NAME=${MAXTEXT_MODEL_NAME:-qwen3-1.7b}

export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-512}
export BATCH_SIZE=${BATCH_SIZE:-8}
export NUM_GENERATIONS=${NUM_GENERATIONS:-4}
export MAX_STEPS=${MAX_STEPS:-100}
export TRAIN_MICRO_BATCH_SIZE=${TRAIN_MICRO_BATCH_SIZE:-8}
export REWARD_MODE=${REWARD_MODE:-exact}
export DATASET_SOURCE=${DATASET_SOURCE:-huggingface}
export LEARNING_RATE=${LEARNING_RATE:-2e-6}
export TEMPERATURE=${TEMPERATURE:-0.7}
export TOP_P=${TOP_P:-0.95}

# peft runs tunix's PeftTrainer; maxtext runs MaxText's MaxTextTrainingEngine.
export TRAINER_BACKEND=${TRAINER_BACKEND:-maxtext}
export MAXTEXT_CKPT=${MAXTEXT_CKPT:-gs://cloud-tpu-multipod-dev-bucket-europe-west4/users/atwigg/checkpoints/qwen3_1.7b_fixed/0/items}
export ROLLOUT_MAXTEXT_CKPT=${ROLLOUT_MAXTEXT_CKPT:-gs://cloud-tpu-multipod-dev-bucket-europe-west4/users/atwigg/checkpoints/qwen3_1.7b_unrolled/0/items}
export MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-$((BATCH_SIZE * NUM_GENERATIONS))}
export EVAL_EVERY_N_STEPS=${EVAL_EVERY_N_STEPS:-1000000}
export LORA_RANK=${LORA_RANK:-16}
export LORA_ALPHA=${LORA_ALPHA:-16.0}
export USE_LORA=${USE_LORA:-0}

# Logs source/destination Raiden tensor checksums on both the trainer and
# rollout sides during weight sync, for cross-verification of a real run.
export VERIFY_WEIGHTS=${VERIFY_WEIGHTS:-false}
export RAIDEN_WEIGHT_SYNC_CHUNKS=${RAIDEN_WEIGHT_SYNC_CHUNKS:-1}
export WEIGHT_SYNC_BACKEND=${WEIGHT_SYNC_BACKEND:-raiden}

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
# Empty by default (no padding). Must match whatever
# maxtext_vllm_adapter.generate_maxtext_config() computes for the rollout's
# own moe_intermediate_size/tensor_parallel_size on the rollout side -- see
# run_trainer_node.py's --maxtext_padded_moe_mlp_dim for why this has to be
# set explicitly for MoE models rather than derived automatically here.
export TRAINER_PADDED_MOE_MLP_DIM=${TRAINER_PADDED_MOE_MLP_DIM:-}
export ROLLOUT_TPU_SLICE=${ROLLOUT_TPU_SLICE:-tpuv5:2x2x1}
export ROLLOUT_TENSOR_PARALLEL_SIZE=${ROLLOUT_TENSOR_PARALLEL_SIZE:-4}
# Default to standard v3 ragged_paged_attention kernel.
# Set ROLLOUT_USE_BATCHED_RPA=1 and ROLLOUT_MAXTEXT_ATTENTION=vllm_batched_rpa
# only if experimenting with the batched-RPA kernel.
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

export SYNC_GIT_BRANCH=${SYNC_GIT_BRANCH:-atwigg/gsm8k-dist}

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
      cd /app && git fetch https://github.com/google/tunix.git ${SYNC_GIT_BRANCH} && git checkout FETCH_HEAD -- tunix examples && \
      WEIGHT_SYNC_BACKEND=${WEIGHT_SYNC_BACKEND} python -m tunix.experimental.distributed.runtime.main \
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
        --reward_mode=${REWARD_MODE} \
        --dataset_source=${DATASET_SOURCE} \
        --temperature=${TEMPERATURE} \
        --top_p=${TOP_P} \
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
      ${SYNC_GIT_BRANCH:+cd /app && git fetch https://github.com/google/tunix.git ${SYNC_GIT_BRANCH} && git checkout FETCH_HEAD -- tunix examples &&} \
      VERIFY_WEIGHTS=${VERIFY_WEIGHTS} ${RAIDEN_WEIGHT_SYNC_CHUNKS:+RAIDEN_WEIGHT_SYNC_CHUNKS=${RAIDEN_WEIGHT_SYNC_CHUNKS}} python -m tunix.experimental.distributed.runtime.main \
        --discovery_addrs=${ORCHESTRATOR_ID}:${ORCHESTRATOR_PORT} \
        --process_executor=tunix.experimental.distributed.runtime.executor.K8sExecutor \
        --process_main=tunix.experimental.examples.math_gsm8k_dist.run_trainer_node.main \
        --worker_id=${TRAINER_ID} \
        --port=${TRAINER_PORT} \
        --mesh_fsdp=${TRAINER_MESH_FSDP} \
        --mesh_tp=${TRAINER_MESH_TP} \
        --mesh_expert=${TRAINER_MESH_EXPERT} \
        --trainer_backend=${TRAINER_BACKEND} \
        ${TRAINER_PADDED_MOE_MLP_DIM:+--maxtext_padded_moe_mlp_dim=${TRAINER_PADDED_MOE_MLP_DIM}} \
        ${MAXTEXT_CKPT:+--maxtext_load_parameters_path=${MAXTEXT_CKPT}} \
        --model_name=${MODEL_NAME} \
        --model_id=${MODEL_ID} \
        --model_dir=${MODEL_DIR} \
        --tokenizer_path=${TOKENIZER_PATH} \
        --max_prompt_length=${MAX_PROMPT_LENGTH} \
        --max_response_length=${MAX_RESPONSE_LENGTH} \
        --mini_batch_size=${MINI_BATCH_SIZE} \
        --train_micro_batch_size=${TRAIN_MICRO_BATCH_SIZE} \
        --learning_rate=${LEARNING_RATE} \
        --log_train_steps \
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
        ${SYNC_GIT_BRANCH:+cd /app && git fetch https://github.com/google/tunix.git ${SYNC_GIT_BRANCH} && git checkout FETCH_HEAD -- tunix examples &&} \
        SKIP_JAX_PRECOMPILE=1 VERIFY_WEIGHTS=${VERIFY_WEIGHTS} ${ROLLOUT_USE_BATCHED_RPA:+USE_BATCHED_RPA_KERNEL=1} python -m tunix.experimental.distributed.runtime.main \
          --discovery_addrs=${ORCHESTRATOR_ID}:${ORCHESTRATOR_PORT} \
          --process_executor=tunix.experimental.distributed.runtime.executor.K8sExecutor \
          --process_main=tunix.experimental.examples.math_gsm8k_dist.run_rollout_node.main \
          --worker_id=${replica_id} \
          --worker_index=$((i + 1)) \
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
          --temperature=${TEMPERATURE} \
          --top_p=${TOP_P} \
          ${ROLLOUT_MAXTEXT_ATTENTION:+--maxtext_attention=${ROLLOUT_MAXTEXT_ATTENTION}} \
          ${ROLLOUT_MAXTEXT_CKPT:+--maxtext_load_parameters_path=${ROLLOUT_MAXTEXT_CKPT}} \
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
