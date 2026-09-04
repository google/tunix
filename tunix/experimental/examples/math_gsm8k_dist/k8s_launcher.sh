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

# Set to tunix to run Tunix's PeftTrainer, and maxtext to run MaxText's MaxTextTrainingEngine
export TRAINER_BACKEND=${TRAINER_BACKEND:-tunix}
export MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-$((BATCH_SIZE * NUM_GENERATIONS))}
export EVAL_EVERY_N_STEPS=${EVAL_EVERY_N_STEPS:-1000000}
export LORA_RANK=${LORA_RANK:-16}
export LORA_ALPHA=${LORA_ALPHA:-16.0}
export USE_LORA=${USE_LORA:-0}
export BETA=${BETA:-0}
export EPSILON=${EPSILON:-0.2}
# Must default to empty, not 0: ${DEBUG:+--debug} expands whenever DEBUG is
# non-empty, and "0" is non-empty -- so --debug was always passed, which
# run_trainer_node.py rejects.
export DEBUG=${DEBUG:-}
export SAMPLER=${SAMPLER:-inprocess_vllm}
# 'exact' recomputes the GSM8K reward in the orchestrator from the returned
# trajectory text, which is also what enables DEBUG=1 to print the full
# sampler responses; 'env' uses the rollout environment's own rewards.
export REWARD_MODE=${REWARD_MODE:-env}
export WEIGHT_SYNC_MODE=${WEIGHT_SYNC_MODE:-none}

# MaxText trainer configuration: only consulted when TRAINER_BACKEND=maxtext
export MAXTEXT_MODEL_NAME=${MAXTEXT_MODEL_NAME:-qwen3-1.7b}
export MAXTEXT_CKPT=${MAXTEXT_CKPT:-}
# If TRAINER_BACKEND=maxtext, MAXTEXT_CKPT must be set to the path of an Orbax params-only checkpoint.
# Set ALLOW_RANDOM_INIT=1 to skip this and let MaxText random-init instead,
# which an empty load_parameters_path already supports downstream.
if [[ "$TRAINER_BACKEND" == "maxtext" && -z "$MAXTEXT_CKPT" && "$ALLOW_RANDOM_INIT" != "1" ]]; then
  echo "Error: TRAINER_BACKEND=maxtext requires MAXTEXT_CKPT (Orbax params-only checkpoint)."
  exit 1
fi
# Must be absolute: Orbax/tensorstore rejects a relative checkpoint path.
export MAXTEXT_OUTPUT_DIR=${MAXTEXT_OUTPUT_DIR:-/app/artifacts/math_gsm8k_dist/maxtext}
# Padded MoE MLP intermediate dimension; must match rollout TP padding for MoE models.
export TRAINER_PADDED_MOE_MLP_DIM=${TRAINER_PADDED_MOE_MLP_DIM:-}
export TRAINER_BASE_NUM_KV_HEADS=${TRAINER_BASE_NUM_KV_HEADS:-}
# Split TP into tp x attn_dp instead of replicating KV heads; see
# run_rollout_node.py --sampler_dp_attention.
export ROLLOUT_DP_ATTENTION=${ROLLOUT_DP_ATTENTION:-}
# Rollout data parallelism. Total rollout devices are ROLLOUT_MESH_TP x this.
export ROLLOUT_DATA_PARALLEL=${ROLLOUT_DATA_PARALLEL:-1}
# Optional: enable experimental batched-RPA attention kernel for rollout.
export ROLLOUT_USE_BATCHED_RPA=${ROLLOUT_USE_BATCHED_RPA:-}
export ROLLOUT_MAXTEXT_ATTENTION=${ROLLOUT_MAXTEXT_ATTENTION:-}

# Logs source/destination Raiden tensor checksums on both the trainer and
# rollout sides during weight sync, for cross-verification of a real run.
export VERIFY_WEIGHTS=${VERIFY_WEIGHTS:-false}

export WANDB_PROJECT=${WANDB_PROJECT:-trellis-gsm8k}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-}
export WANDB_API_KEY=${WANDB_API_KEY:-}
export TFDS_DATA_DIR=${TFDS_DATA_DIR:-"artifacts/data"}
export TFDS_SPLIT=${TFDS_SPLIT:-train}

# Raiden's num_shards must be the real per-host device count. Pathways cannot
# be asked -- every proxy device reports task_id=0, process_index=0 and no
# local_hardware_id -- and getting it wrong leaves each host's
# SetGlobalShardIndices half -1, so the transfer silently delivers one host's
# shards. Every TPU machine type yaml_generator supports is a `-4t`, which is
# where its own `num_chips // 4` host count comes from.
export RAIDEN_DEVICES_PER_HOST=${RAIDEN_DEVICES_PER_HOST:-4}

export K8S_NAMESPACE=${K8S_NAMESPACE:-default}

export ORCHESTRATOR_ID=$USER-orch
export ORCHESTRATOR_PORT=20000

export ROLLOUT_ID=$USER-roll
# Number of independent rollout workers. Each gets its own ROLLOUT_TPU_SLICE
# and its own worker id, and registers with the orchestrator separately, so
# N replicas x ROLLOUT_MESH_TP is the destination shard count. Keep that equal
# to the trainer's: an unbalanced pair syncs without error but delivers only
# part of every TP-sharded tensor.
export ROLLOUT_REPLICAS=${ROLLOUT_REPLICAS:-1}
export ROLLOUT_PORT=20001

export TRAINER_ID=$USER-train
export TRAINER_PORT=20002

export CPU_MACHINE=${CPU_MACHINE:-n2-standard-64}
export GCS_SCRATCH_LOCATION=${GCS_SCRATCH_LOCATION:-gs://cloud-pathways-staging/tmp}

# Pathways server/proxy must be built for the same JAX as the client image:
# a mismatch surfaces as 'Cache key mismatch between PW client and server'.
export PATHWAYS_SERVER_IMAGE=${PATHWAYS_SERVER_IMAGE:-}
export PATHWAYS_PROXY_SERVER_IMAGE=${PATHWAYS_PROXY_SERVER_IMAGE:-}
export TRAINER_JOBSET_YAML=${TRAINER_JOBSET_YAML:-jobset.pathways.yaml}
export TRAINER_TPU_SLICE=${TRAINER_TPU_SLICE:-tpuv5e:4x4}
export TRAINER_MESH_FSDP=${TRAINER_MESH_FSDP:-16}
export TRAINER_MESH_TP=${TRAINER_MESH_TP:-1}
export TRAINER_MESH_EXPERT=${TRAINER_MESH_EXPERT:-1}

export ROLLOUT_JOBSET_YAML=${ROLLOUT_JOBSET_YAML:-leaderworkerset.mcjax.ray.yaml}
export ROLLOUT_TPU_SLICE=${ROLLOUT_TPU_SLICE:-tpuv5e:4x4}
export ROLLOUT_MESH_FSDP=${ROLLOUT_MESH_FSDP:-1}
export ROLLOUT_MESH_TP=${ROLLOUT_MESH_TP:-16}

# FFI handlers exist only in the Raiden-enabled Pathways runtime, so an FFI
# destination needs a Pathways rollout. Derive the transport from the rollout
# so both ends agree: mixing them fails in the native layer with
# "INVALID_ARGUMENT: Destination out of bounds in batched push".
if [[ -z "${RAIDEN_USE_FFI:-}" ]]; then
  if [[ "${ROLLOUT_JOBSET_YAML}" == *pathways* ]]; then
    RAIDEN_USE_FFI=1
  else
    RAIDEN_USE_FFI=0
  fi
fi
export RAIDEN_USE_FFI
# Per-side overrides on top of that default. Needed for a Pathways trainer
# feeding an mcjax sampler: the source can use FFI (no host copy of the model)
# while the destination must stay legacy, since importing
# weight_synchronizer_ffi registers nothing into jaxlib's FFI registry.
export TRAINER_RAIDEN_USE_FFI=${TRAINER_RAIDEN_USE_FFI:-${RAIDEN_USE_FFI}}
export ROLLOUT_RAIDEN_USE_FFI=${ROLLOUT_RAIDEN_USE_FFI:-${RAIDEN_USE_FFI}}

# The jobset YAMLs hardcode `namespace: default`, which wins over `kubectl -n`,
# so rewrite it on the way through.
apply_jobset() {
  sed "s|^  namespace: .*|  namespace: ${K8S_NAMESPACE}|" \
    | kubectl apply -n "${K8S_NAMESPACE}" -f -
}

stop_orchestrator() {
  kubectl delete jobset -n "${K8S_NAMESPACE}" "${ORCHESTRATOR_ID}"
}

start_orchestrator() {
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
        --reward_mode=${REWARD_MODE} \
        --stop_workers_on_exit \
        ${DEBUG:+--debug} \
    " \
    | apply_jobset
}

stop_trainer() {
  kubectl delete jobset -n "${K8S_NAMESPACE}" "${TRAINER_ID}"
}

start_trainer() {
  local extra_flags=""

  if [[ "${TRAINER_BACKEND}" == "maxtext" ]]; then
    extra_flags+=" \
      --maxtext_model_name=${MAXTEXT_MODEL_NAME} \
      ${TRAINER_PADDED_MOE_MLP_DIM:+--maxtext_padded_moe_mlp_dim=${TRAINER_PADDED_MOE_MLP_DIM}} \
      ${TRAINER_BASE_NUM_KV_HEADS:+--maxtext_base_num_kv_heads=${TRAINER_BASE_NUM_KV_HEADS}} \
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
    ${PATHWAYS_SERVER_IMAGE:+--pathways_server_image=${PATHWAYS_SERVER_IMAGE}} \
    ${PATHWAYS_PROXY_SERVER_IMAGE:+--pathways_proxy_server_image=${PATHWAYS_PROXY_SERVER_IMAGE}} \
    --worker_container_image="${TUNIX_IMAGE}" \
    --worker_container_port="${TRAINER_PORT}" \
    --worker_startup_command=" \
      VERIFY_WEIGHTS=${VERIFY_WEIGHTS} RAIDEN_USE_FFI=${TRAINER_RAIDEN_USE_FFI} RAIDEN_DEVICES_PER_HOST=${RAIDEN_DEVICES_PER_HOST} python -m tunix.experimental.distributed.runtime.main \
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
        --lora_rank=${LORA_RANK} \
        --lora_alpha=${LORA_ALPHA} \
        ${extra_flags} \
        ${DEBUG:+--debug} \
    " \
    | apply_jobset
}

# Replica i's ids: unsuffixed for a single replica, so existing runs are
# unchanged; -0/-1/... when there is more than one.
rollout_id_for() {
  if [[ "${ROLLOUT_REPLICAS}" -le 1 ]]; then echo "${ROLLOUT_ID}"; else echo "${ROLLOUT_ID}-$1"; fi
}

stop_rollout() {
  local _id
  for _replica in $(seq 0 $((ROLLOUT_REPLICAS - 1))); do
    _id="$(rollout_id_for "${_replica}")"
    if [[ "$ROLLOUT_JOBSET_YAML" =~ ^leaderworkerset ]]; then
      kubectl delete leaderworkerset -n "${K8S_NAMESPACE}" "${_id}"
    else
      kubectl delete jobset -n "${K8S_NAMESPACE}" "${_id}"
    fi
  done
}

start_rollout() {
  for _replica in $(seq 0 $((ROLLOUT_REPLICAS - 1))); do
    ROLLOUT_ID="$(rollout_id_for "${_replica}")" _start_one_rollout
  done
}

_start_one_rollout() {
  local extra_flags=""

  if [[ "${TRAINER_BACKEND}" == "maxtext" ]]; then
    extra_flags+="\
      --maxtext_model_name=${MAXTEXT_MODEL_NAME} \
      ${ROLLOUT_MAXTEXT_ATTENTION:+--maxtext_attention=${ROLLOUT_MAXTEXT_ATTENTION}} \
    "
  fi
  # Tensor parallelism reaches the sampler as --mesh_tp; only the axes vLLM
  # alone understands are passed here.
  if [[ "$SAMPLER" == "vllm" ]]; then
    extra_flags+="\
      --sampler_data_parallel=${ROLLOUT_DATA_PARALLEL} \
      ${ROLLOUT_DP_ATTENTION:+--sampler_dp_attention} \
    "
  fi
  local rollout_pathways_args=""
  if [[ "${ROLLOUT_JOBSET_YAML}" == *pathways* ]]; then
    rollout_pathways_args=" \
      --cpu_machine=${CPU_MACHINE} \
      --pathways_gcs_scratch_location=${GCS_SCRATCH_LOCATION} \
      ${PATHWAYS_SERVER_IMAGE:+--pathways_server_image=${PATHWAYS_SERVER_IMAGE}} \
      ${PATHWAYS_PROXY_SERVER_IMAGE:+--pathways_proxy_server_image=${PATHWAYS_PROXY_SERVER_IMAGE}} \
    "
  fi

  python tunix/experimental/distributed/deployment/yaml_generator.py \
    tunix/experimental/distributed/deployment/yamls/${ROLLOUT_JOBSET_YAML} \
    --jobset_name="${ROLLOUT_ID}" \
    --tpu_slice="${ROLLOUT_TPU_SLICE}" \
    ${rollout_pathways_args} \
    --worker_container_image="${TUNIX_IMAGE}" \
    --worker_container_port="${ROLLOUT_PORT}" \
    --worker_startup_command=" \
      SKIP_JAX_PRECOMPILE=1 SAMPLER=${SAMPLER} RAIDEN_USE_FFI=${ROLLOUT_RAIDEN_USE_FFI} RAIDEN_DEVICES_PER_HOST=${RAIDEN_DEVICES_PER_HOST} ${RAIDEN_H2D_CHUNK_BYTES:+RAIDEN_H2D_CHUNK_BYTES=${RAIDEN_H2D_CHUNK_BYTES}} VERIFY_WEIGHTS=${VERIFY_WEIGHTS} ${ROLLOUT_USE_BATCHED_RPA:+USE_BATCHED_RPA_KERNEL=1} python -m tunix.experimental.distributed.runtime.main \
        --discovery_addrs=${ORCHESTRATOR_ID}:${ORCHESTRATOR_PORT} \
        --process_executor=tunix.experimental.distributed.runtime.executor.K8sExecutor \
        --process_main=tunix.experimental.examples.common.run_rollout_node.main \
        --worker_id=${ROLLOUT_ID} \
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
        ${extra_flags} \
        ${DEBUG:+--debug} \
    " \
    | apply_jobset
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
