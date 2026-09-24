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
LOG_ROOT=${LOG_ROOT:-"${DIR}"}
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON_BIN=${PYTHON_BIN:-python3}
ORCHESTRATOR_ID=${ORCHESTRATOR_ID:-orchestrator}
ORCHESTRATOR_PORT=${ORCHESTRATOR_PORT:-30000}
TRAINER_PORT=${TRAINER_PORT:-20000}
ROLLOUT_PORT=${ROLLOUT_PORT:-20001}
INFERENCE_PORT=${INFERENCE_PORT:-20002}
RUN_INFERENCE_NODE=${RUN_INFERENCE_NODE:-0}
INFERENCE_ADDR=${INFERENCE_ADDR:-}

MODEL_NAME=${MODEL_NAME:-Qwen3-1.7B}
MODEL_ID=${MODEL_ID:-Qwen/Qwen3-1.7B}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-"${REPO_ROOT}/artifacts/qwen3_dist_deepswe"}
MODEL_DIR=${MODEL_DIR:-"${ARTIFACT_ROOT}/models/${MODEL_NAME}"}
TOKENIZER_PATH=${TOKENIZER_PATH:-"${MODEL_DIR}"}

MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1024}
BATCH_SIZE=${BATCH_SIZE:-1}
NUM_GENERATIONS=${NUM_GENERATIONS:-2}
MAX_STEPS=${MAX_STEPS:-1}
MAX_TURNS=${MAX_TURNS:-3}
MAX_STALENESS=${MAX_STALENESS:-0}
TRAJECTORY_GROUP_ORDER=${TRAJECTORY_GROUP_ORDER:-arrival}
TRAIN_MICRO_BATCH_SIZE=${TRAIN_MICRO_BATCH_SIZE:-1}
MAX_SEQ_TOKEN_PER_TPU=${MAX_SEQ_TOKEN_PER_TPU:-}
MAX_SEGMENTS_PER_PACKED_ROW=${MAX_SEGMENTS_PER_PACKED_ROW:-}
COMPUTE_LOGPS_CHUNK_SIZE=${COMPUTE_LOGPS_CHUNK_SIZE:-}
MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-$BATCH_SIZE}
EVAL_EVERY_N_STEPS=${EVAL_EVERY_N_STEPS:-1000000}
OPT_CHAIN_TYPE=${OPT_CHAIN_TYPE-clip_by_global_norm}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}
ADAM_B1=${ADAM_B1:-0.9}
ADAM_B2=${ADAM_B2:-0.999}
ADAM_EPS=${ADAM_EPS:-1.0e-8}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
LEARNING_RATE=${LEARNING_RATE:-1e-6}
# The default is applied with `-` rather than `:-` so that an explicitly empty
# SCHEDULE_TYPE selects the constant learning rate instead of the default.
SCHEDULE_TYPE=${SCHEDULE_TYPE-warmup_cosine_decay_schedule}
LR_INIT_VALUE=${LR_INIT_VALUE:-0.0}
LR_PEAK_VALUE=${LR_PEAK_VALUE:-$LEARNING_RATE}
LR_END_VALUE=${LR_END_VALUE:-0.0}
LR_DECAY_STEPS=${LR_DECAY_STEPS:-500}
WARMUP_STEPS=${WARMUP_STEPS:-$(((LR_DECAY_STEPS + 9) / 10))}
BETA=${BETA:-0.0}
EPSILON=${EPSILON:-0.2}
SAMPLER=${SAMPLER:-inprocess_vllm}
WEIGHT_SYNC_MODE=${WEIGHT_SYNC_MODE:-none}
USE_LORA=${USE_LORA:-0}
LORA_RANK=${LORA_RANK:-64}
LORA_ALPHA=${LORA_ALPHA:-64.0}
# Generation sampling parameters, passed to the runner and the reference scorer.
TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-1.0}
TOP_K=${TOP_K:--1}
EOS_TOKENS=${EOS_TOKENS-}
# DEBUG=1 passes --debug to the runner, which logs full sampler responses.
DEBUG=${DEBUG:-0}
USE_ROLLOUT_LOGPS=${USE_ROLLOUT_LOGPS:-true}
EXACT_TOKEN_CONTINUITY=${EXACT_TOKEN_CONTINUITY:-true}

# Optional GRPO algorithm options. Empty, or 0 for the boolean, leaves the
# option at the runner's default, so an unset variable changes nothing.
EPSILON_HIGH=${EPSILON_HIGH:-}
LOSS_AGG_MODE=${LOSS_AGG_MODE:-}
ADVANTAGE_ESTIMATOR=${ADVANTAGE_ESTIMATOR:-}
OVERLONG_LOSS_MASKING=${OVERLONG_LOSS_MASKING:-0}
SEQ_LOGPROB_ERROR_THRESHOLD=${SEQ_LOGPROB_ERROR_THRESHOLD:-}
TIS_TYPE=${TIS_TYPE:-${TRUNCATED_IMPORTANCE_SAMPLING_TYPE:-}}
TIS_RATIO_MIN=${TIS_RATIO_MIN:-${TRUNCATED_IMPORTANCE_SAMPLING_RATIO_MIN:-}}
TIS_RATIO=${TIS_RATIO:-${TRUNCATED_IMPORTANCE_SAMPLING_RATIO:-}}
SAMPLER_IS_LENGTH_BUCKETS=${SAMPLER_IS_LENGTH_BUCKETS:-}

CHECKPOINT_SAVE_INTERVAL_STEPS=${CHECKPOINT_SAVE_INTERVAL_STEPS:-1}
CHECKPOINT_MAX_TO_KEEP=${CHECKPOINT_MAX_TO_KEEP:-10}
CHECKPOINT_ROOT_DIRECTORY=${CHECKPOINT_ROOT_DIRECTORY:-"${REPO_ROOT}/checkpoints"}

DATASET_NAME=${DATASET_NAME:-R2E-Gym/R2E-Gym-Subset}
DATASET_PATH=${DATASET_PATH:-}
DATASET_SPLIT=${DATASET_SPLIT:-train}
DATASET_CACHE_DIR=${DATASET_CACHE_DIR:-"${ARTIFACT_ROOT}/dataset_cache"}
SHUFFLE=${SHUFFLE:-true}
SEED=${SEED:-42}
ENV_BACKEND=${ENV_BACKEND:-kubernetes}
SCAFFOLD=${SCAFFOLD:-r2egym}
USE_AGENT_SANDBOX=${USE_AGENT_SANDBOX:-0}
SANDBOX_NAMESPACE=${SANDBOX_NAMESPACE:-rl-tunix-swebench}
SANDBOX_NODE_SELECTOR_KEY=${SANDBOX_NODE_SELECTOR_KEY:-}
SANDBOX_NODE_SELECTOR_VAL=${SANDBOX_NODE_SELECTOR_VAL:-}
STEP_TIMEOUT_SECS=${STEP_TIMEOUT_SECS:-1800}
REWARD_TIMEOUT_SECS=${REWARD_TIMEOUT_SECS:-1800}
ROLLOUT_MAX_CONCURRENCY=${ROLLOUT_MAX_CONCURRENCY:-64}

WANDB_PROJECT=${WANDB_PROJECT:-trellis-deepswe}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-}
WANDB_API_KEY=${WANDB_API_KEY:-}
LOG_DIR=${LOG_DIR:-}
TRAJECTORY_LOG_DIR=${TRAJECTORY_LOG_DIR:-}
TRAJECTORY_STORE_ROOT_DIR=${TRAJECTORY_STORE_ROOT_DIR:-${TRAJECTORY_STORE_ROOT:-}}
FLUSH_EVERY_N_STEPS=${FLUSH_EVERY_N_STEPS:-1}
TRAINABLE_PARAMETERS_MASK=${TRAINABLE_PARAMETERS_MASK:-}

TRAINER_TPU_CHIPS=${TRAINER_TPU_CHIPS:-0,1}
TRAINER_FSDP=${TRAINER_FSDP:-1}
TRAINER_TP=${TRAINER_TP:-2}

# tunix runs Tunix's PeftTrainer; maxtext runs MaxText's MaxTextTrainingEngine.
TRAINER_BACKEND=${TRAINER_BACKEND:-tunix}

# MaxText configuration: only consulted when TRAINER_BACKEND=maxtext.
source "${DIR}/../common/maxtext_config.sh"

if [[ "$TRAINER_BACKEND" == "maxtext" ]]; then
  if (( TRAIN_MICRO_BATCH_SIZE % TRAINER_FSDP != 0 )); then
    TRAIN_MICRO_BATCH_SIZE=$TRAINER_FSDP
  fi
elif [[ "$TRAINER_BACKEND" == "tunix" ]]; then
  # Must stay empty on the tunix backend. A non-empty value puts the rollout on
  # MaxText's MaxTextForCausalLM while the trainer still emits tunix/vllm_jax
  # tensor names, and Raiden pairs tensors by exact name, so zero of them match.
  # Export it explicitly to override.
  MAXTEXT_MODEL_NAME=${MAXTEXT_MODEL_NAME-}
else
  echo "Error: Unsupported TRAINER_BACKEND='$TRAINER_BACKEND' (expected 'tunix' or 'maxtext')." >&2
  exit 1
fi

ROLLOUT_TPU_CHIPS=${ROLLOUT_TPU_CHIPS:-2,3}
ROLLOUT_FSDP=${ROLLOUT_FSDP:-1}
ROLLOUT_TP=${ROLLOUT_TP:-2}
INFERENCE_TPU_CHIPS=${INFERENCE_TPU_CHIPS:-}
TPU_CHIPS_PER_HOST_BOUNDS=${TPU_CHIPS_PER_HOST_BOUNDS:-1,2,1}
TPU_HOST_BOUNDS=${TPU_HOST_BOUNDS:-1,1,1}

WAIT_TIMEOUT_SECS=${WAIT_TIMEOUT_SECS:-1800}
WAIT_POLL_SECS=${WAIT_POLL_SECS:-5}
SHUTDOWN_GRACE_SECS=${SHUTDOWN_GRACE_SECS:-60}

TRAINER_LOG="${LOG_ROOT}/trainer.log"
ROLLOUT_LOG="${LOG_ROOT}/rollout.log"
INFERENCE_LOG="${LOG_ROOT}/inference.log"
ORCHESTRATOR_LOG="${LOG_ROOT}/orchestrator.log"

print_command() {
  local label="$1"
  shift
  echo "$label:"
  printf '  %q' "$@"
  echo
}

has_direct_safetensors() {
  [[ -d "$MODEL_DIR" ]] && [[ -n "$(
    find "$MODEL_DIR" -maxdepth 1 -type f -name '*.safetensors' -print -quit 2>/dev/null || true
  )" ]]
}

download_model_dir() {
  echo "Downloading $MODEL_ID to MODEL_DIR: $MODEL_DIR"
  "$PYTHON_BIN" - "$MODEL_ID" "$MODEL_DIR" <<'PY'
import os
import sys

from tunix.oss import utils as oss_utils

model_id, model_dir = sys.argv[1], sys.argv[2]
os.makedirs(model_dir, exist_ok=True)
oss_utils.hf_pipeline(model_id, model_dir)
PY
}

ensure_model_dir() {
  if has_direct_safetensors; then
    return
  fi
  download_model_dir
  if ! has_direct_safetensors; then
    echo "Error: MODEL_DIR has no direct '*.safetensors': $MODEL_DIR"
    exit 1
  fi
}

wait_for_port() {
  local name="$1"
  local port="$2"
  local pid="$3"
  local log_file="$4"
  local elapsed=0
  while true; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "Error: $name process exited before port $port became ready."
      tail -n 80 "$log_file" 2>/dev/null || true
      exit 1
    fi
    if "$PYTHON_BIN" - "$port" <<'PY'
import socket
import sys

port = int(sys.argv[1])
try:
  socket.create_connection(("localhost", port), timeout=1).close()
except OSError:
  sys.exit(1)
PY
    then
      echo "$name port $port is ready after ${elapsed}s."
      return
    fi
    if (( elapsed >= WAIT_TIMEOUT_SECS )); then
      echo "Error: timed out waiting for $name port $port."
      tail -n 80 "$log_file" 2>/dev/null || true
      exit 1
    fi
    echo "Waiting for $name port $port... elapsed=${elapsed}s"
    sleep "$WAIT_POLL_SECS"
    elapsed=$((elapsed + WAIT_POLL_SECS))
  done
}

cleanup() {
  trap - EXIT ERR
  local pids=()
  for pid in "${TRAINER_PID:-}" "${ROLLOUT_PID:-}" "${INFERENCE_PID:-}"; do
    if [[ -n "$pid" ]]; then
      pids+=("$pid")
    fi
  done
  if (( ${#pids[@]} == 0 )); then
    return
  fi
  echo "Cleaning up worker processes: ${pids[*]}"
  kill "${pids[@]}" 2>/dev/null || true
  sleep "$SHUTDOWN_GRACE_SECS" || true
  kill -9 "${pids[@]}" 2>/dev/null || true
  wait "${pids[@]}" 2>/dev/null || true
}

trap cleanup EXIT

echo "=================================================="
echo "Starting distributed DeepSWE GRPO pipeline locally"
echo "  model:          ${MODEL_ID}"
echo "  model dir:      ${MODEL_DIR}"
echo "  tokenizer path: ${TOKENIZER_PATH}"
echo "  dataset:        ${DATASET_PATH:-${DATASET_NAME}:${DATASET_SPLIT}}"
echo "  trajectories:   $((BATCH_SIZE * NUM_GENERATIONS)) per step"
echo "  batch size:     ${BATCH_SIZE}"
echo "  generations:    ${NUM_GENERATIONS}"
echo "  sampling:       temperature=$TEMPERATURE top_p=$TOP_P top_k=$TOP_K"
echo "  max steps:      ${MAX_STEPS}"
echo "  max turns:      ${MAX_TURNS}"
echo "  prompt length:  ${MAX_PROMPT_LENGTH}"
echo "  response len:   ${MAX_RESPONSE_LENGTH}"
echo "  max seq token:  ${MAX_SEQ_TOKEN_PER_TPU:-<unset>}"
echo "  max segments:   ${MAX_SEGMENTS_PER_PACKED_ROW:-<unset>}"
echo "  learning rate:  ${LEARNING_RATE}"
echo "  lr schedule:    ${SCHEDULE_TYPE:-<constant>} (warmup $WARMUP_STEPS, decay $LR_DECAY_STEPS)"
echo "  beta:           ${BETA}"
echo "  epsilon:        ${EPSILON}"
echo "  max staleness:  ${MAX_STALENESS}"
echo "  traj order:     ${TRAJECTORY_GROUP_ORDER}"
echo "  sampler:        ${SAMPLER}"
echo "  weight sync:    ${WEIGHT_SYNC_MODE}"
echo "  trainer backend:${TRAINER_BACKEND}"
echo "  trainer chips:  ${TRAINER_TPU_CHIPS}"
echo "  rollout chips:  ${ROLLOUT_TPU_CHIPS}"
echo "  inference:      ${RUN_INFERENCE_NODE}"
echo "  inference addr: ${INFERENCE_ADDR:-<none>}"
echo "  inference chips:${INFERENCE_TPU_CHIPS:-<unset>}"
echo "  wandb project:  ${WANDB_PROJECT:-<none>}"
echo "  wandb run name: ${WANDB_RUN_NAME:-<auto>}"
echo "  ckpt interval:  ${CHECKPOINT_SAVE_INTERVAL_STEPS}"
echo "  ckpt max keep:  ${CHECKPOINT_MAX_TO_KEEP}"
echo "  ckpt root dir:  ${CHECKPOINT_ROOT_DIRECTORY}"
echo "=================================================="

if [[ "$BETA" != "0" && "$BETA" != "0.0" && -z "$INFERENCE_ADDR" ]]; then
  if [[ "$RUN_INFERENCE_NODE" != "1" &&
        "$RUN_INFERENCE_NODE" != "true" &&
        "$RUN_INFERENCE_NODE" != "True" ]]; then
    echo "Error: BETA=$BETA requires a reference inference worker."
    echo "Set RUN_INFERENCE_NODE=1 with INFERENCE_TPU_CHIPS, pass INFERENCE_ADDR,"
    echo "or use BETA=0 for a trainer+rollout smoke run."
    exit 1
  fi
fi

ensure_model_dir
mkdir -p "$LOG_ROOT" "$ARTIFACT_ROOT"
: > "$TRAINER_LOG"
: > "$ROLLOUT_LOG"
: > "$INFERENCE_LOG"
: > "$ORCHESTRATOR_LOG"

echo "Launching trainer node..."
(
  TRAINER_CMD=(
    "$PYTHON_BIN" -m tunix.experimental.distributed.runtime.main
    --discovery_addrs="${ORCHESTRATOR_ID}:${ORCHESTRATOR_PORT}"
    --process_main=tunix.experimental.examples.common.run_trainer_node.main
    --port="$TRAINER_PORT"
    --mesh_fsdp="$TRAINER_FSDP"
    --mesh_tp="$TRAINER_TP"
    --model_id="$MODEL_ID"
    --model_dir="$MODEL_DIR"
    --model_name="$MODEL_NAME"
    --sampler_type="$SAMPLER"
    --tokenizer_path="$TOKENIZER_PATH"
    --max_prompt_length="$MAX_PROMPT_LENGTH"
    --max_response_length="$MAX_RESPONSE_LENGTH"
    --mini_batch_size="$MINI_BATCH_SIZE"
    --num_generations="$NUM_GENERATIONS"
    --train_micro_batch_size="$TRAIN_MICRO_BATCH_SIZE"
    ${COMPUTE_LOGPS_CHUNK_SIZE:+--compute_logps_chunk_size="$COMPUTE_LOGPS_CHUNK_SIZE"}
    --eval_every_n_steps="$EVAL_EVERY_N_STEPS"
    --optimizer_b1="$ADAM_B1"
    --optimizer_b2="$ADAM_B2"
    --optimizer_eps="$ADAM_EPS"
    --optimizer_weight_decay="$WEIGHT_DECAY"
    --optimizer_learning_rate="$LEARNING_RATE"
    --optimizer_schedule_type="$SCHEDULE_TYPE"
    --optimizer_init_value="$LR_INIT_VALUE"
    --optimizer_peak_value="$LR_PEAK_VALUE"
    --optimizer_end_value="$LR_END_VALUE"
    --optimizer_warmup_steps="$WARMUP_STEPS"
    --optimizer_decay_steps="$LR_DECAY_STEPS"
    --lora_rank="$LORA_RANK"
    --lora_alpha="$LORA_ALPHA"
    --trainer_backend="$TRAINER_BACKEND"
    --checkpoint_save_interval_steps="$CHECKPOINT_SAVE_INTERVAL_STEPS"
    --checkpoint_max_to_keep="$CHECKPOINT_MAX_TO_KEEP"
    --checkpoint_root_directory="$CHECKPOINT_ROOT_DIRECTORY"
  )
  if [[ -n "$OPT_CHAIN_TYPE" ]]; then
    TRAINER_CMD+=(
      --optimizer_opt_chain_type="$OPT_CHAIN_TYPE"
      --optimizer_chain_kwargs="{'max_norm': $MAX_GRAD_NORM}"
    )
  fi

  TRAINER_CMD+=+="$(maxtext_trainer_flags)"
  
  if [[ -n "$MAX_SEQ_TOKEN_PER_TPU" ]]; then
    TRAINER_CMD+=(--max_seq_token_per_tpu="$MAX_SEQ_TOKEN_PER_TPU")
  fi
  if [[ "$USE_LORA" == "1" || "$USE_LORA" == "true" || "$USE_LORA" == "True" ]]; then
    TRAINER_CMD+=(--use_lora)
  fi
  
  if [[ "$DEBUG" == "1" || "$DEBUG" == "true" || "$DEBUG" == "True" ]]; then
    TRAINER_CMD+=(--debug)
  fi
  if [[ -n "$TRAINABLE_PARAMETERS_MASK" ]]; then
    TRAINER_CMD+=(--trainable_parameters_mask="$TRAINABLE_PARAMETERS_MASK")
  fi
  if [[ "${TRAINER_PATHWAYS:-0}" == "1" ]]; then
    export JAX_PLATFORMS=proxy,cpu
    export JAX_BACKEND_TARGET=${JAX_BACKEND_TARGET:-grpc://127.0.0.1:29000}
    export TRAINER_PATHWAYS_LOCAL_INIT=1
    unset TPU_VISIBLE_DEVICES TPU_VISIBLE_CHIPS LIBTPU_INIT_ARGS
  else
    export JAX_PLATFORMS=tpu,cpu
    export TPU_VISIBLE_DEVICES=${TRAINER_TPU_CHIPS}
    export TPU_VISIBLE_CHIPS=${TPU_VISIBLE_DEVICES}
    export TPU_CHIPS_PER_HOST_BOUNDS=${TPU_CHIPS_PER_HOST_BOUNDS}
    export TPU_HOST_BOUNDS=${TPU_HOST_BOUNDS}
    export LIBTPU_INIT_ARGS="--deepsea_chips_per_host_bounds=${TPU_CHIPS_PER_HOST_BOUNDS} --deepsea_host_bounds=${TPU_HOST_BOUNDS}"
  fi
  export PYTHONUNBUFFERED=1
  print_command "Trainer command" "${TRAINER_CMD[@]}"
  exec "${TRAINER_CMD[@]}" > "$TRAINER_LOG" 2>&1
) &
TRAINER_PID=$!

echo "Launching DeepSWE rollout node..."
(
  ROLLOUT_CMD=(
    "$PYTHON_BIN" -m tunix.experimental.distributed.runtime.main
    --discovery_addrs="${ORCHESTRATOR_ID}:${ORCHESTRATOR_PORT}"
    --process_main=tunix.experimental.examples.common.run_rollout_node.main
    --port="$ROLLOUT_PORT"
    --model_id="$MODEL_ID"
    --model_dir="$MODEL_DIR"
    --model_name="$MODEL_NAME"
    --sampler="$SAMPLER"
    --mesh_fsdp="$ROLLOUT_FSDP"
    --mesh_tp="$ROLLOUT_TP"
    --tokenizer_path="$TOKENIZER_PATH"
    --max_prompt_length="$MAX_PROMPT_LENGTH"
    --max_response_length="$MAX_RESPONSE_LENGTH"
    --lora_rank="$LORA_RANK"
    --lora_alpha="$LORA_ALPHA"
    --weight_sync_mode="$WEIGHT_SYNC_MODE"
    --registry_module=tunix.experimental.examples.deepswe_dist.deepswe
    --env_name=deepswe_env
    --agent_name=deepswe_agent
    --max_concurrency="$ROLLOUT_MAX_CONCURRENCY"
  )
  
  ROLLOUT_CMD+="$(maxtext_rollout_flags)"

  if [[ -n "$EOS_TOKENS" ]]; then
    ROLLOUT_CMD+=(--eos_tokens="$EOS_TOKENS")
  fi
  if [[ "$USE_LORA" == "1" || "$USE_LORA" == "true" || "$USE_LORA" == "True" ]]; then
    ROLLOUT_CMD+=(--use_lora)
  fi
  if [[ "$DEBUG" == "1" || "$DEBUG" == "true" || "$DEBUG" == "True" ]]; then
    ROLLOUT_CMD+=(--debug)
  fi
  export JAX_PLATFORMS=tpu,cpu
  export SKIP_JAX_PRECOMPILE=1
  export TPU_VISIBLE_DEVICES=${ROLLOUT_TPU_CHIPS}
  export TPU_VISIBLE_CHIPS=${TPU_VISIBLE_DEVICES}
  export TPU_CHIPS_PER_HOST_BOUNDS=${TPU_CHIPS_PER_HOST_BOUNDS}
  export TPU_HOST_BOUNDS=${TPU_HOST_BOUNDS}
  export LIBTPU_INIT_ARGS="--deepsea_chips_per_host_bounds=${TPU_CHIPS_PER_HOST_BOUNDS} --deepsea_host_bounds=${TPU_HOST_BOUNDS}"
  if [[ "$USE_AGENT_SANDBOX" == "1" || "$USE_AGENT_SANDBOX" == "true" || "$USE_AGENT_SANDBOX" == "True" ]]; then
    export NAMESPACE="$SANDBOX_NAMESPACE"
    if [[ -n "$SANDBOX_NODE_SELECTOR_KEY" && -n "$SANDBOX_NODE_SELECTOR_VAL" ]]; then
      export NODE_SELECTOR_KEY="$SANDBOX_NODE_SELECTOR_KEY"
      export NODE_SELECTOR_VAL="$SANDBOX_NODE_SELECTOR_VAL"
    fi
  fi
  export PYTHONUNBUFFERED=1
  print_command "Rollout command" "${ROLLOUT_CMD[@]}"
  exec "${ROLLOUT_CMD[@]}" > "$ROLLOUT_LOG" 2>&1
) &
ROLLOUT_PID=$!

if [[ "$RUN_INFERENCE_NODE" == "1" || "$RUN_INFERENCE_NODE" == "true" || "$RUN_INFERENCE_NODE" == "True" ]]; then
  if [[ -z "$INFERENCE_TPU_CHIPS" ]]; then
    echo "Error: RUN_INFERENCE_NODE requires INFERENCE_TPU_CHIPS to avoid TPU contention."
    exit 1
  fi
  echo "Launching reference inference node on TPU chips $INFERENCE_TPU_CHIPS..."
  (
    INFERENCE_CMD=(
      "$PYTHON_BIN" -m tunix.experimental.distributed.runtime.main
      --discovery_addrs="${ORCHESTRATOR_ID}:${ORCHESTRATOR_PORT}"
      --process_main=tunix.experimental.examples.common.run_inference_node.main
      --port="$INFERENCE_PORT"
      --model_name="$MODEL_NAME"
      --model_id="$MODEL_ID"
      --model_dir="$MODEL_DIR"
      --tokenizer_path="$TOKENIZER_PATH"
      --compute_logps_micro_batch_size="$TRAIN_MICRO_BATCH_SIZE"
      --max_prompt_length="$MAX_PROMPT_LENGTH"
      --max_response_length="$MAX_RESPONSE_LENGTH"
      # Must match the sampling temperature: this node scores the reference
      # policy for the KL term.
      --temperature="$TEMPERATURE"
    )
    export JAX_PLATFORMS=tpu,cpu
    export TPU_VISIBLE_DEVICES=${INFERENCE_TPU_CHIPS}
    export TPU_VISIBLE_CHIPS=${TPU_VISIBLE_DEVICES}
    export TPU_CHIPS_PER_HOST_BOUNDS=${TPU_CHIPS_PER_HOST_BOUNDS}
    export TPU_HOST_BOUNDS=${TPU_HOST_BOUNDS}
    export LIBTPU_INIT_ARGS="--deepsea_chips_per_host_bounds=${TPU_CHIPS_PER_HOST_BOUNDS} --deepsea_host_bounds=${TPU_HOST_BOUNDS}"
    export PYTHONUNBUFFERED=1
    print_command "Inference command" "${INFERENCE_CMD[@]}"
    exec "${INFERENCE_CMD[@]}" > "$INFERENCE_LOG" 2>&1
  ) &
  INFERENCE_PID=$!
  INFERENCE_ADDR="localhost:$INFERENCE_PORT"
fi

wait_for_port "trainer" "$TRAINER_PORT" "$TRAINER_PID" "$TRAINER_LOG"
wait_for_port "rollout" "$ROLLOUT_PORT" "$ROLLOUT_PID" "$ROLLOUT_LOG"
if [[ -n "${INFERENCE_PID:-}" ]]; then
  wait_for_port "inference" "$INFERENCE_PORT" "$INFERENCE_PID" "$INFERENCE_LOG"
fi

echo "Launching CPU orchestrator..."
(
  ORCHESTRATOR_CMD=(
    "$PYTHON_BIN" -m tunix.experimental.distributed.runtime.main
    --discovery_id="${ORCHESTRATOR_ID}"
    --discovery_port="${ORCHESTRATOR_PORT}"
    --process_main=tunix.experimental.examples.deepswe_dist.run_deepswe_dist.main
    --model_id="$MODEL_ID"
    --tokenizer_path="$TOKENIZER_PATH"
    --batch_size="$BATCH_SIZE"
    --mini_batch_size="$MINI_BATCH_SIZE"
    --num_generations="$NUM_GENERATIONS"
    --temperature="$TEMPERATURE"
    --top_p="$TOP_P"
    --top_k="$TOP_K"
    --max_steps="$MAX_STEPS"
    --max_turns="$MAX_TURNS"
    --max_prompt_length="$MAX_PROMPT_LENGTH"
    --max_response_length="$MAX_RESPONSE_LENGTH"
    --max_staleness="$MAX_STALENESS"
    --trajectory_group_order="$TRAJECTORY_GROUP_ORDER"
    --train_micro_batch_size="$TRAIN_MICRO_BATCH_SIZE"
    --beta="$BETA"
    --epsilon="$EPSILON"
    --dataset_name="$DATASET_NAME"
    --dataset_split="$DATASET_SPLIT"
    --dataset_cache_dir="$DATASET_CACHE_DIR"
    --seed="$SEED"
    --env_backend="$ENV_BACKEND"
    --scaffold="$SCAFFOLD"
    --step_timeout_secs="$STEP_TIMEOUT_SECS"
    --reward_timeout_secs="$REWARD_TIMEOUT_SECS"
    --flush_every_n_steps="$FLUSH_EVERY_N_STEPS"
    --weight_sync_mode="$WEIGHT_SYNC_MODE"
    --stop_workers_on_exit
  )
  if [[ "$DEBUG" == "1" || "$DEBUG" == "true" || "$DEBUG" == "True" ]]; then
    ORCHESTRATOR_CMD+=(--debug)
  fi
  # Explicit if-blocks rather than `[[ -n x ]] && cmd`: this script runs under
  # `set -Ee`, where a false test at the head of an AND-list aborts the
  # launcher. An option left unset has to be a no-op.
  if [[ -n "$EPSILON_HIGH" ]]; then
    ORCHESTRATOR_CMD+=(--epsilon_high="$EPSILON_HIGH")
  fi
  if [[ -n "$LOSS_AGG_MODE" ]]; then
    ORCHESTRATOR_CMD+=(--loss_agg_mode="$LOSS_AGG_MODE")
  fi
  if [[ -n "$ADVANTAGE_ESTIMATOR" ]]; then
    ORCHESTRATOR_CMD+=(--advantage_estimator="$ADVANTAGE_ESTIMATOR")
  fi
  if [[ "$OVERLONG_LOSS_MASKING" == "1" || "$OVERLONG_LOSS_MASKING" == "true" || "$OVERLONG_LOSS_MASKING" == "True" ]]; then
    ORCHESTRATOR_CMD+=(--overlong_loss_masking)
  fi
  if [[ -n "$SEQ_LOGPROB_ERROR_THRESHOLD" ]]; then
    ORCHESTRATOR_CMD+=(--seq_logprob_error_threshold="$SEQ_LOGPROB_ERROR_THRESHOLD")
  fi
  if [[ -n "$TIS_TYPE" ]]; then
    ORCHESTRATOR_CMD+=(--truncated_importance_sampling_type="$TIS_TYPE")
  fi
  if [[ -n "$TIS_RATIO_MIN" ]]; then
    ORCHESTRATOR_CMD+=(--truncated_importance_sampling_ratio_min="$TIS_RATIO_MIN")
  fi
  if [[ -n "$TIS_RATIO" ]]; then
    ORCHESTRATOR_CMD+=(--truncated_importance_sampling_ratio="$TIS_RATIO")
  fi
  if [[ -n "$SAMPLER_IS_LENGTH_BUCKETS" ]]; then
    ORCHESTRATOR_CMD+=(--sampler_is_length_buckets="$SAMPLER_IS_LENGTH_BUCKETS")
  fi
  if [[ -n "$DATASET_PATH" ]]; then
    ORCHESTRATOR_CMD+=(--dataset_path="$DATASET_PATH")
  fi
  if [[ "$SHUFFLE" == "0" || "$SHUFFLE" == "false" || "$SHUFFLE" == "False" ]]; then
    ORCHESTRATOR_CMD+=(--no-shuffle)
  else
    ORCHESTRATOR_CMD+=(--shuffle)
  fi
  if [[ -n "$INFERENCE_ADDR" ]]; then
    ORCHESTRATOR_CMD+=(--inference_addr="$INFERENCE_ADDR")
  fi
  if [[ "$USE_AGENT_SANDBOX" == "1" || "$USE_AGENT_SANDBOX" == "true" || "$USE_AGENT_SANDBOX" == "True" ]]; then
    ORCHESTRATOR_CMD+=(--use_agent_sandbox)
  fi
  if [[ -n "$MAX_SEQ_TOKEN_PER_TPU" ]]; then
    ORCHESTRATOR_CMD+=(--max_seq_token_per_tpu="$MAX_SEQ_TOKEN_PER_TPU")
  fi
  if [[ -n "$MAX_SEGMENTS_PER_PACKED_ROW" ]]; then
    ORCHESTRATOR_CMD+=(--max_segments_per_packed_row="$MAX_SEGMENTS_PER_PACKED_ROW")
  fi
  if [[ -n "$TRAINER_FSDP" ]]; then
    ORCHESTRATOR_CMD+=(--trainer_fsdp="$TRAINER_FSDP")
  fi
  if [[ -n "$LOG_DIR" ]]; then
    ORCHESTRATOR_CMD+=(--log_dir="$LOG_DIR")
  fi
  if [[ -n "$TRAJECTORY_LOG_DIR" ]]; then
    ORCHESTRATOR_CMD+=(--trajectory_log_dir="$TRAJECTORY_LOG_DIR")
  fi
  if [[ -n "$TRAJECTORY_STORE_ROOT_DIR" ]]; then
    ORCHESTRATOR_CMD+=(--trajectory_store_root_dir="$TRAJECTORY_STORE_ROOT_DIR")
  fi
  if [[ -n "$TRAINABLE_PARAMETERS_MASK" ]]; then
    ORCHESTRATOR_CMD+=(--trainable_parameters_mask="$TRAINABLE_PARAMETERS_MASK")
  fi
  if [[ "$USE_ROLLOUT_LOGPS" == "false" || "$USE_ROLLOUT_LOGPS" == "False" || "$USE_ROLLOUT_LOGPS" == "0" ]]; then
    ORCHESTRATOR_CMD+=(--no-use_rollout_logps)
  else
    ORCHESTRATOR_CMD+=(--use_rollout_logps)
  fi
  if [[ "$EXACT_TOKEN_CONTINUITY" == "false" || "$EXACT_TOKEN_CONTINUITY" == "False" || "$EXACT_TOKEN_CONTINUITY" == "0" ]]; then
    ORCHESTRATOR_CMD+=(--no-exact_token_continuity)
  else
    ORCHESTRATOR_CMD+=(--exact_token_continuity)
  fi
  export JAX_PLATFORMS=cpu
  export PYTHONUNBUFFERED=1
  export WANDB_PROJECT="$WANDB_PROJECT"
  export WANDB_RUN_NAME="$WANDB_RUN_NAME"
  export WANDB_API_KEY="$WANDB_API_KEY"
  print_command "Orchestrator command" "${ORCHESTRATOR_CMD[@]}"
  "${ORCHESTRATOR_CMD[@]}" > "$ORCHESTRATOR_LOG" 2>&1
)

echo "Distributed DeepSWE GRPO pipeline finished successfully."
echo "Trainer log:      $TRAINER_LOG"
echo "Rollout log:      $ROLLOUT_LOG"
echo "Orchestrator log: $ORCHESTRATOR_LOG"