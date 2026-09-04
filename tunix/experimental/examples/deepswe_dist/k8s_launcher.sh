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

# Set to tunix to run Tunix's PeftTrainer, and maxtext to run MaxText's MaxTextTrainingEngine
export TRAINER_BACKEND=${TRAINER_BACKEND:-tunix}
export MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-$((BATCH_SIZE * NUM_GENERATIONS))}
export EVAL_EVERY_N_STEPS=${EVAL_EVERY_N_STEPS:-1000000}
export BETA=${BETA:-0.0}
export EPSILON=${EPSILON:-0.2}
export LORA_RANK=${LORA_RANK:-64}
export LORA_ALPHA=${LORA_ALPHA:-64.0}
export USE_LORA=${USE_LORA:-0}
export DEBUG=${DEBUG:-0}
export SAMPLER=${SAMPLER:-inprocess_vllm}
export WEIGHT_SYNC_MODE=${WEIGHT_SYNC_MODE:-none}

# DeepSWE dataset and environment configuration
export DATASET_NAME=${DATASET_NAME:-R2E-Gym/R2E-Gym-Subset}
export DATASET_PATH=${DATASET_PATH:-}
export DATASET_SPLIT=${DATASET_SPLIT:-train}
export DATASET_CACHE_DIR=${DATASET_CACHE_DIR:-artifacts/qwen3_dist_deepswe/dataset_cache}
export SHUFFLE=${SHUFFLE:-true}
export SEED=${SEED:-42}
export ENV_BACKEND=${ENV_BACKEND:-kubernetes}
export SCAFFOLD=${SCAFFOLD:-r2egym}
export USE_AGENT_SANDBOX=${USE_AGENT_SANDBOX:-0}
export SANDBOX_NAMESPACE=${SANDBOX_NAMESPACE:-rl-tunix-swebench}
export SANDBOX_NODE_SELECTOR_KEY=${SANDBOX_NODE_SELECTOR_KEY:-}
export SANDBOX_NODE_SELECTOR_VAL=${SANDBOX_NODE_SELECTOR_VAL:-}
export STEP_TIMEOUT_SECS=${STEP_TIMEOUT_SECS:-1800}
export REWARD_TIMEOUT_SECS=${REWARD_TIMEOUT_SECS:-1800}
export ROLLOUT_MAX_CONCURRENCY=${ROLLOUT_MAX_CONCURRENCY:-64}
export FLUSH_EVERY_N_STEPS=${FLUSH_EVERY_N_STEPS:-1}

# MaxText trainer configuration: only consulted when TRAINER_BACKEND=maxtext
export MAXTEXT_MODEL_NAME=${MAXTEXT_MODEL_NAME:-qwen3-1.7b}
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

if [[ "$BETA" != "0" && "$BETA" != "0.0" ]]; then
  echo "Error: this first DeepSWE distributed launcher only wires trainer+rollout."
  echo "Use BETA=0.0 until the reference inference worker is added."
  exit 1
fi

stop_orchestrator() {
  kubectl delete jobset "${ORCHESTRATOR_ID}"
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

  python tunix/experimental/distributed/deployment/yaml_generator.py \
    tunix/experimental/distributed/deployment/yamls/jobset.cpu.yaml \
    --jobset_name="${ORCHESTRATOR_ID}" \
    --cpu_machine=${CPU_MACHINE} \
    --worker_container_image="${TUNIX_IMAGE}" \
    --worker_container_port="${ORCHESTRATOR_PORT}" \
    --worker_startup_command=" \
      ${WANDB_API_KEY:+WANDB_API_KEY=\"${WANDB_API_KEY}\"} \
      WANDB_PROJECT=\"${WANDB_PROJECT}\" \
      WANDB_RUN_NAME=\"${WANDB_RUN_NAME}\" \
      python -m tunix.experimental.distributed.runtime.main \
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
        --stop_workers_on_exit \
        ${dataset_args} \
        ${shuffle_arg} \
        ${sandbox_arg} \
        ${DEBUG:+--debug} \
    " \
    | kubectl apply -f -
}

stop_trainer() {
  kubectl delete jobset "${TRAINER_ID}"
}

start_trainer() {
  local maxtext_args=""
  if [[ "${TRAINER_BACKEND}" == "maxtext" ]]; then
    maxtext_args=" \
      --maxtext_model_name=${MAXTEXT_MODEL_NAME} \
      ${TRAINER_PADDED_MOE_MLP_DIM:+--maxtext_padded_moe_mlp_dim=${TRAINER_PADDED_MOE_MLP_DIM}} \
      --maxtext_ckpt_path=${MAXTEXT_CKPT} \
      --maxtext_output_directory=${MAXTEXT_OUTPUT_DIR} \
      --mesh_tp=${TRAINER_MESH_TP} \
      --mesh_expert=${TRAINER_MESH_EXPERT} \
    "
  fi
  local lora_args=""
  if [[ "${USE_LORA}" == "1" || "${USE_LORA}" == "true" || "${USE_LORA}" == "True" ]]; then
    lora_args="--use_lora"
  fi
  python tunix/experimental/distributed/deployment/yaml_generator.py \
    tunix/experimental/distributed/deployment/yamls/${TRAINER_JOBSET_YAML} \
    --jobset_name="${TRAINER_ID}" \
    --tpu_slice=${TRAINER_TPU_SLICE} \
    --cpu_machine=${CPU_MACHINE} \
    --pathways_gcs_scratch_location=${GCS_SCRATCH_LOCATION} \
    --worker_container_image="${TUNIX_IMAGE}" \
    --worker_container_port="${TRAINER_PORT}" \
    --worker_startup_command=" \
      VERIFY_WEIGHTS=${VERIFY_WEIGHTS} python -m tunix.experimental.distributed.runtime.main \
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
        --train_micro_batch_size=${TRAIN_MICRO_BATCH_SIZE} \
        --eval_every_n_steps=${EVAL_EVERY_N_STEPS} \
        --lora_rank=${LORA_RANK} \
        --lora_alpha=${LORA_ALPHA} \
        --sampler=${SAMPLER} \
        ${lora_args} \
        ${maxtext_args} \
        ${DEBUG:+--debug} \
    " \
    | kubectl apply -f -
}

stop_rollout() {
  kubectl delete jobset "${ROLLOUT_ID}"
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
    --sampler_mesh_tp=${ROLLOUT_MESH_TP} \
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
  python tunix/experimental/distributed/deployment/yaml_generator.py \
    tunix/experimental/distributed/deployment/yamls/jobset.tpu.yaml \
    --jobset_name="${ROLLOUT_ID}" \
    --tpu_slice=${ROLLOUT_TPU_SLICE} \
    --worker_container_image="${TUNIX_IMAGE}" \
    --worker_container_port="${ROLLOUT_PORT}" \
    --worker_startup_command=" \
      SKIP_JAX_PRECOMPILE=1 VERIFY_WEIGHTS=${VERIFY_WEIGHTS} ${sandbox_env} ${ROLLOUT_USE_BATCHED_RPA:+USE_BATCHED_RPA_KERNEL=1} python -m tunix.experimental.distributed.runtime.main \
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
        --registry_module=tunix.experimental.examples.deepswe_dist.deepswe \
        --env_name=deepswe_env \
        --agent_name=deepswe_agent \
        --max_concurrency=${ROLLOUT_MAX_CONCURRENCY} \
        ${lora_args} \
        ${maxtext_args} \
        ${vllm_args} \
        ${DEBUG:+--debug} \
    " \
    | kubectl apply -f -
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
