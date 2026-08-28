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
TUNIX_IMAGE="us-central1-docker.pkg.dev/cloud-tpu-multipod-dev/yangmu/tunix/tunix_base_image:trellis-demo-0813"

export MODEL_NAME=${MODEL_NAME:-Qwen3-0.6B}
export MODEL_ID=${MODEL_ID:-Qwen/Qwen3-0.6B}
# Must be model-specific: it's a local HF snapshot cache, and vLLM's own
# model_config reads config.json straight from here whenever the directory
# is non-empty, bypassing MODEL_ID entirely. A shared path across different
# models silently reuses a stale snapshot -- confirmed root cause of a
# multi-hour KV-cache shape-mismatch chase on Qwen3-30B-A3B: with a leftover
# Qwen3-0.6B snapshot sitting here, vLLM's config (num_key_value_heads=8)
# and MaxText's own model construction (num_key_value_heads=4, correctly
# built from MAXTEXT_MODEL_NAME) silently disagreed.
export MODEL_DIR=${MODEL_DIR:-artifacts/qwen3_dist_gsm8k/models/${MODEL_NAME}}
# Defaults to MODEL_ID (an HF hub repo id), not MODEL_DIR: MODEL_DIR is a
# per-model-name local directory that starts out empty for any model that
# hasn't been downloaded there before by *something* -- nothing pre-populates
# it before the trainer's own AutoTokenizer.from_pretrained(tokenizer_path)
# call, which then misreads a nonexistent local path as a malformed hub repo
# id and fails HFValidationError. MODEL_ID always works: it's a genuine hub
# id AutoTokenizer downloads directly, independent of any local directory
# state or trainer/rollout pod download-ordering races.
export TOKENIZER_PATH=${TOKENIZER_PATH:-${MODEL_ID}}

export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-128}
export BATCH_SIZE=${BATCH_SIZE:-2}
export NUM_GENERATIONS=${NUM_GENERATIONS:-2}
export MAX_STEPS=${MAX_STEPS:-1}
export TRAINER_FSDP=${TRAINER_FSDP:-8}

# Cluster hardware. Defaults match trellis-demo-0810; override for other clusters
# (e.g. mlperf-v5p has no n2-standard-64 pool and no 2x2x2 topology).
export CPU_MACHINE=${CPU_MACHINE:-n2-standard-64}
export TPU_SLICE=${TPU_SLICE:-tpuv5:2x2x1}
# Pathways writes its persistent compilation cache here; the default shared bucket
# is not writable from this project, which kills the compilation service.
export GCS_SCRATCH_LOCATION=${GCS_SCRATCH_LOCATION:-gs://cloud-pathways-staging/tmp}

# peft runs tunix's PeftTrainer; maxtext runs MaxText's MaxTextTrainingEngine.
export TRAINER_BACKEND=${TRAINER_BACKEND:-peft}
export MAXTEXT_CKPT=${MAXTEXT_CKPT:-}
if [[ "$TRAINER_BACKEND" == "maxtext" ]]; then
  # MaxText shards the batch dimension of every loss input across the fsdp axis, so the
  # microbatch has to be a multiple of it. The trainer node enforces this too.
  export TRAIN_MICRO_BATCH_SIZE=${TRAIN_MICRO_BATCH_SIZE:-$TRAINER_FSDP}
  if [[ -z "$MAXTEXT_CKPT" ]]; then
    echo "Error: TRAINER_BACKEND=maxtext requires MAXTEXT_CKPT (Orbax params-only checkpoint)."
    exit 1
  fi
else
  export TRAIN_MICRO_BATCH_SIZE=${TRAIN_MICRO_BATCH_SIZE:-1}
fi
export MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-$((BATCH_SIZE * NUM_GENERATIONS))}
export EVAL_EVERY_N_STEPS=${EVAL_EVERY_N_STEPS:-1000000}
export LORA_RANK=${LORA_RANK:-16}
export LORA_ALPHA=${LORA_ALPHA:-16.0}
export USE_LORA=${USE_LORA:-0}

export ORCHESTRATOR_ID=$USER-orch
export ORCHESTRATOR_PORT=20000

export ROLLOUT_ID=$USER-roll
export ROLLOUT_PORT=20001

export TRAINER_ID=$USER-train
export TRAINER_PORT=20002

stop_orchestrator() {
  kubectl delete jobset "${ORCHESTRATOR_ID}"
}

start_orchestrator() {
  python tunix/experimental/distributed/deployment/yaml_generator.py \
    tunix/experimental/distributed/deployment/yamls/jobset.cpu.yaml \
    --jobset_name="${ORCHESTRATOR_ID}" \
    --cpu_machine="${CPU_MACHINE}" \
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
        --stop_workers_on_exit \
    " \
    | kubectl apply -f -
}

stop_trainer() {
  kubectl delete jobset "${TRAINER_ID}"
}

start_trainer() {
  python tunix/experimental/distributed/deployment/yaml_generator.py \
    tunix/experimental/distributed/deployment/yamls/jobset.pathways.yaml \
    --jobset_name="${TRAINER_ID}" \
    --tpu_slice="${TPU_SLICE}" \
    --pathways_gcs_scratch_location="${GCS_SCRATCH_LOCATION}" \
    --worker_container_image="${TUNIX_IMAGE}" \
    --worker_container_port="${TRAINER_PORT}" \
    --worker_startup_command=" \
      python -m tunix.experimental.distributed.runtime.main \
        --discovery_addrs=${ORCHESTRATOR_ID}:${ORCHESTRATOR_PORT} \
        --process_executor=tunix.experimental.distributed.runtime.executor.K8sExecutor \
        --process_main=tunix.experimental.examples.math_gsm8k_dist.run_trainer_node.main \
        --worker_id=${TRAINER_ID} \
        --port=${TRAINER_PORT} \
        --mesh_fsdp=${TRAINER_FSDP} \
        --trainer_backend=${TRAINER_BACKEND} \
        --maxtext_load_parameters_path=${MAXTEXT_CKPT} \
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
    " \
    | kubectl apply -f -

}

stop_rollout() {
  kubectl delete jobset "${ROLLOUT_ID}"
}

start_rollout() {
  python tunix/experimental/distributed/deployment/yaml_generator.py \
    tunix/experimental/distributed/deployment/yamls/jobset.tpu.yaml \
    --jobset_name="${ROLLOUT_ID}" \
    --tpu_slice="${TPU_SLICE}" \
    --pathways_gcs_scratch_location="${GCS_SCRATCH_LOCATION}" \
    --worker_container_image="${TUNIX_IMAGE}" \
    --worker_container_port="${ROLLOUT_PORT}" \
    --worker_startup_command=" \
      SKIP_JAX_PRECOMPILE=1 python -m tunix.experimental.distributed.runtime.main \
        --discovery_addrs=${ORCHESTRATOR_ID}:${ORCHESTRATOR_PORT} \
        --process_executor=tunix.experimental.distributed.runtime.executor.K8sExecutor \
        --process_main=tunix.experimental.examples.math_gsm8k_dist.run_rollout_node.main \
        --worker_id=${ROLLOUT_ID} \
        --port=${ROLLOUT_PORT} \
        --model_id=${MODEL_ID} \
        --model_dir=${MODEL_DIR} \
        --tokenizer_path=${TOKENIZER_PATH} \
        --max_prompt_length=${MAX_PROMPT_LENGTH} \
        --max_response_length=${MAX_RESPONSE_LENGTH} \
        --lora_rank=${LORA_RANK} \
        --lora_alpha=${LORA_ALPHA} \
    " \
    | kubectl apply -f -
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
