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
TARGET_CLUSTER=""

export PROJECT=${PROJECT:-cloud-tpu-multipod-dev}
export REGION=${REGION:-us-central1}
export ZONE=${ZONE:-us-central1-a}
export CLUSTER=${CLUSTER:-trellis-demo-0810}
export CLUSTER_LOCATION=${CLUSTER_LOCATION:-${REGION}}

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

# Set to tunix to run Tunix's PeftTrainer, and maxtext to run MaxText's MaxTextTrainingEngine
export TRAINER_BACKEND=${TRAINER_BACKEND:-tunix}
export MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-$((BATCH_SIZE * NUM_GENERATIONS))}
export EVAL_EVERY_N_STEPS=${EVAL_EVERY_N_STEPS:-1000000}
export LORA_RANK=${LORA_RANK:-16}
export LORA_ALPHA=${LORA_ALPHA:-16.0}
export USE_LORA=${USE_LORA:-0}
export DEBUG=${DEBUG:-0}
export SAMPLER=${SAMPLER:-inprocess_vllm}
export WEIGHT_SYNC_MODE=${WEIGHT_SYNC_MODE:-none}

# MaxText trainer configuration: only consulted when TRAINER_BACKEND=maxtext
export MAXTEXT_MODEL_NAME=${MAXTEXT_MODEL_NAME:-qwen3-1.7b}
export MAXTEXT_CKPT=${MAXTEXT_CKPT:-}
# If TRAINER_BACKEND=maxtext, MAXTEXT_CKPT must be set to the path of an Orbax params-only checkpoint.
if [[ "$TRAINER_BACKEND" == "maxtext" && -z "$MAXTEXT_CKPT" ]]; then
  echo "Error: TRAINER_BACKEND=maxtext requires MAXTEXT_CKPT (Orbax params-only checkpoint)."
  exit 1
fi
export MAXTEXT_OUTPUT_DIR=${MAXTEXT_OUTPUT_DIR:-artifacts/math_gsm8k_dist/maxtext}
export TRAINER_MESH_TP=${TRAINER_MESH_TP:-1}
export TRAINER_MESH_EXPERT=${TRAINER_MESH_EXPERT:-1}
# Padded MoE MLP intermediate dimension; must match rollout TP padding for MoE models.
export TRAINER_PADDED_MOE_MLP_DIM=${TRAINER_PADDED_MOE_MLP_DIM:-}
export ROLLOUT_MESH_TP=${ROLLOUT_MESH_TP:-4}
# Optional: enable experimental batched-RPA attention kernel for rollout.
export ROLLOUT_USE_BATCHED_RPA=${ROLLOUT_USE_BATCHED_RPA:-}
export ROLLOUT_MAXTEXT_ATTENTION=${ROLLOUT_MAXTEXT_ATTENTION:-}

# Logs source/destination Raiden tensor checksums on both the trainer and
# rollout sides during weight sync, for cross-verification of a real run.
export VERIFY_WEIGHTS=${VERIFY_WEIGHTS:-false}

export WANDB_PROJECT=${WANDB_PROJECT:-trellis-gsm8k}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-}
export WANDB_API_KEY=${WANDB_API_KEY:-}
export WEIGHT_SYNC_BACKEND=${WEIGHT_SYNC_BACKEND:-none}

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

apply_cluster_target() {
  case "$1" in
    multipod-rl-scaffolding)
      PROJECT=cloud-tpu-multipod-dev
      REGION=us-central1
      ZONE=us-central1-a
      CLUSTER_LOCATION=us-central1
      CLUSTER=rl-scaffolding
      TRAINER_TPU_SLICE=tpuv5e:4x4
      TRAINER_MESH_FSDP=16
      ROLLOUT_TPU_SLICE=tpuv5e:4x4
      ;;
    multipod-rl-v5e)
      PROJECT=cloud-tpu-multipod-dev
      REGION=us-central1
      ZONE=us-central1-a
      CLUSTER_LOCATION=us-central1
      CLUSTER=rl-v5e-16-cluster-v2
      TRAINER_TPU_SLICE=tpuv5e:4x4
      TRAINER_MESH_FSDP=16
      ROLLOUT_TPU_SLICE=tpuv5e:4x4
      ;;
    inference-v5e)
      PROJECT=cloud-tpu-inference-test
      REGION=us-west1
      ZONE=us-west1-c
      CLUSTER_LOCATION=us-west1-c
      CLUSTER=lancewang-pw-v5e-4slice
      TRAINER_TPU_SLICE=tpuv5e:4x4
      TRAINER_MESH_FSDP=16
      ROLLOUT_TPU_SLICE=tpuv5e:4x4
      ;;
    "")
      ;;
    *)
      echo "Error: Unknown cluster target '$1'. Available targets: multipod-rl-scaffolding, multipod-rl-v5e, inference-v5e." >&2
      exit 1
      ;;
  esac
}

print_selected_cluster() {
  echo "Cluster target: project=${PROJECT} location=${CLUSTER_LOCATION} region=${REGION} zone=${ZONE} cluster=${CLUSTER}"
}

print_status() {
  print_selected_cluster
  kubectl get jobset "${ORCHESTRATOR_ID}" "${TRAINER_ID}" "${ROLLOUT_ID}" 2>/dev/null || true
  kubectl get pods -n default | grep -E "${ORCHESTRATOR_ID}|${TRAINER_ID}|${ROLLOUT_ID}" || true
}

stop_orchestrator() {
  kubectl delete jobset "${ORCHESTRATOR_ID}"
}

start_orchestrator() {
  local weight_sync_arg=""
  if [[ "${WEIGHT_SYNC_BACKEND}" != "none" ]]; then
    weight_sync_arg="--weight_sync_backend=${WEIGHT_SYNC_BACKEND}"
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
        --process_main=tunix.experimental.examples.math_gsm8k_dist.run_gsm8k_dist_grpo.main \
        --model_id=${MODEL_ID} \
        --tokenizer_path=${TOKENIZER_PATH} \
        --batch_size=${BATCH_SIZE} \
        --num_generations=${NUM_GENERATIONS} \
        --max_steps=${MAX_STEPS} \
        --max_prompt_length=${MAX_PROMPT_LENGTH} \
        --max_response_length=${MAX_RESPONSE_LENGTH} \
        --train_micro_batch_size=${TRAIN_MICRO_BATCH_SIZE} \
        --wandb_project=\"${WANDB_PROJECT}\" \
        --wandb_run_name=\"${WANDB_RUN_NAME}\" \
        --weight_sync_mode=${WEIGHT_SYNC_MODE} \
        --stop_workers_on_exit \
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
        --process_main=tunix.experimental.examples.math_gsm8k_dist.run_trainer_node.main \
        --worker_id=${TRAINER_ID} \
        --port=${TRAINER_PORT} \
        --mesh_fsdp=${TRAINER_MESH_FSDP} \
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
  local rollout_mesh_fsdp=1
  local rollout_mesh_tp=4
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
  python tunix/experimental/distributed/deployment/yaml_generator.py \
    tunix/experimental/distributed/deployment/yamls/jobset.tpu.yaml \
    --jobset_name="${ROLLOUT_ID}" \
    --tpu_slice=${ROLLOUT_TPU_SLICE} \
    --worker_container_image="${TUNIX_IMAGE}" \
    --worker_container_port="${ROLLOUT_PORT}" \
    --worker_startup_command=" \
      SKIP_JAX_PRECOMPILE=1 VERIFY_WEIGHTS=${VERIFY_WEIGHTS} ${ROLLOUT_USE_BATCHED_RPA:+USE_BATCHED_RPA_KERNEL=1} python -m tunix.experimental.distributed.runtime.main \
        --discovery_addrs=${ORCHESTRATOR_ID}:${ORCHESTRATOR_PORT} \
        --process_executor=tunix.experimental.distributed.runtime.executor.K8sExecutor \
        --process_main=tunix.experimental.examples.math_gsm8k_dist.run_rollout_node.main \
        --worker_id=${ROLLOUT_ID} \
        --port=${ROLLOUT_PORT} \
        --mesh_fsdp=${rollout_mesh_fsdp} \
        --mesh_tp=${rollout_mesh_tp} \
        --model_id=${MODEL_ID} \
        --model_dir=${MODEL_DIR} \
        --tokenizer_path=${TOKENIZER_PATH} \
        --max_prompt_length=${MAX_PROMPT_LENGTH} \
        --max_response_length=${MAX_RESPONSE_LENGTH} \
        --lora_rank=${LORA_RANK} \
        --lora_alpha=${LORA_ALPHA} \
        --weight_sync_mode=${WEIGHT_SYNC_MODE} \
        ${maxtext_args} \
        ${vllm_args} \
        ${DEBUG:+--debug} \
    " \
    | kubectl apply -f -
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
    --project)
      PROJECT="$2"
      shift 2
      ;;
    --project=*)
      PROJECT="${1#*=}"
      shift
      ;;
    --region)
      REGION="$2"
      shift 2
      ;;
    --region=*)
      REGION="${1#*=}"
      shift
      ;;
    --zone)
      ZONE="$2"
      shift 2
      ;;
    --zone=*)
      ZONE="${1#*=}"
      shift
      ;;
    --cluster)
      CLUSTER="$2"
      shift 2
      ;;
    --cluster=*)
      CLUSTER="${1#*=}"
      shift
      ;;
    --location)
      CLUSTER_LOCATION="$2"
      shift 2
      ;;
    --location=*)
      CLUSTER_LOCATION="${1#*=}"
      shift
      ;;
    --target)
      TARGET_CLUSTER="$2"
      shift 2
      ;;
    --target=*)
      TARGET_CLUSTER="${1#*=}"
      shift
      ;;
    --print-target)
      COMMAND="print-target"
      shift
      ;;
    *)
      shift
      ;;
  esac
done

apply_cluster_target "$TARGET_CLUSTER"

source tunix/experimental/examples/math_gsm8k_dist/enter_kube_context.sh

if [[ "$COMMAND" =~ ^(start|orchestrator|trainer|rollout)$ ]] && [[ -z "$TUNIX_IMAGE" ]]; then
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
elif [[ "$COMMAND" == "status" ]]; then
  print_status
elif [[ "$COMMAND" == "print-target" ]]; then
  print_selected_cluster
else
  echo "Error: Invalid command '$COMMAND'. Available commands: 'start', 'stop', 'orchestrator', 'trainer', 'rollout', 'status', 'print-target'."
  exit 1
fi
