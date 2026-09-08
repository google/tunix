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

MODEL_NAME=${MODEL_NAME:-Qwen3-32B}
MODEL_ID=${MODEL_ID:-Qwen/Qwen3-32B}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-"${REPO_ROOT}/artifacts/qwen3_dist_deepswe"}
MODEL_DIR=${MODEL_DIR:-"${ARTIFACT_ROOT}/models/${MODEL_NAME}"}
TOKENIZER_PATH=${TOKENIZER_PATH:-"${MODEL_DIR}"}

MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-8192}
BATCH_SIZE=${BATCH_SIZE:-8}
NUM_GENERATIONS=${NUM_GENERATIONS:-8}
MAX_STEPS=${MAX_STEPS:-50}
MAX_TURNS=${MAX_TURNS:-50}
TRAIN_MICRO_BATCH_SIZE=${TRAIN_MICRO_BATCH_SIZE:-1}
MAX_SEQ_TOKEN_PER_TPU=${MAX_SEQ_TOKEN_PER_TPU:-}
MAX_SEGMENTS_PER_PACKED_ROW=${MAX_SEGMENTS_PER_PACKED_ROW:-}
MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-${BATCH_SIZE}}
EVAL_EVERY_N_STEPS=${EVAL_EVERY_N_STEPS:-1000000}
BETA=${BETA:-0.0}
EPSILON=${EPSILON:-0.2}
EPSILON_HIGH=${EPSILON_HIGH:-0.28}
ADVANTAGE_ESTIMATOR=${ADVANTAGE_ESTIMATOR:-rloo}
LOSS_AGG_MODE=${LOSS_AGG_MODE:-sequence-mean-token-scale}
OFF_POLICY_STEPS=${OFF_POLICY_STEPS:-0}
TOP_P=${TOP_P:-none}
TOP_K=${TOP_K:-none}
LEARNING_RATE=${LEARNING_RATE:-1e-6}
ADAM_B1=${ADAM_B1:-0.9}
ADAM_B2=${ADAM_B2:-0.99}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}
SAMPLER=${SAMPLER:-inprocess_vllm}
WEIGHT_SYNC_MODE=${WEIGHT_SYNC_MODE:-none}
USE_LORA=${USE_LORA:-0}
LORA_RANK=${LORA_RANK:-64}
LORA_ALPHA=${LORA_ALPHA:-64.0}
DEBUG=${DEBUG:-0}
USE_ROLLOUT_LOGPS=${USE_ROLLOUT_LOGPS:-true}

CHECKPOINT_SAVE_INTERVAL_STEPS=${CHECKPOINT_SAVE_INTERVAL_STEPS:-500}
CHECKPOINT_MAX_TO_KEEP=${CHECKPOINT_MAX_TO_KEEP:-4}
CHECKPOINT_ROOT_DIRECTORY=${CHECKPOINT_ROOT_DIRECTORY:-"${REPO_ROOT}/checkpoints"}

DATASET_NAME=${DATASET_NAME:-R2E-Gym/R2E-Gym-V1}
DATASET_PATH=${DATASET_PATH:-}
DATASET_SPLIT=${DATASET_SPLIT:-train}
DATASET_CACHE_DIR=${DATASET_CACHE_DIR:-"${ARTIFACT_ROOT}/dataset_cache"}
SHUFFLE=${SHUFFLE:-true}
SEED=${SEED:-42}
ENV_BACKEND=${ENV_BACKEND:-kubernetes}
SCAFFOLD=${SCAFFOLD:-r2egym}
USE_AGENT_SANDBOX=${USE_AGENT_SANDBOX:-0}
SANDBOX_NAMESPACE=${SANDBOX_NAMESPACE:-rl-tunix-swebench}
SANDBOX_NODE_SELECTOR_KEY=${SANDBOX_NODE_SELECTOR_KEY:-cloud.google.com/gke-nodepool}
SANDBOX_NODE_SELECTOR_VAL=${SANDBOX_NODE_SELECTOR_VAL:-deepswe-cpu-pool}
SANDBOX_MAX_CONCURRENCY=${SANDBOX_MAX_CONCURRENCY:-200}
MAX_WARMPOOL_REPLICAS=${MAX_WARMPOOL_REPLICAS:-}
STEP_TIMEOUT_SECS=${STEP_TIMEOUT_SECS:-1800}
REWARD_TIMEOUT_SECS=${REWARD_TIMEOUT_SECS:-1800}
EPISODE_TIMEOUT_SECS=${EPISODE_TIMEOUT_SECS:-10800}
OVERLONG_FILTER=${OVERLONG_FILTER:-true}
ROLLOUT_MAX_CONCURRENCY=${ROLLOUT_MAX_CONCURRENCY:-200}

WANDB_PROJECT=${WANDB_PROJECT:-trellis-deepswe}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-}
WANDB_API_KEY=${WANDB_API_KEY:-}
FLUSH_EVERY_N_STEPS=${FLUSH_EVERY_N_STEPS:-1}

TRAINER_TPU_CHIPS=${TRAINER_TPU_CHIPS:-0,1}
TRAINER_FSDP=${TRAINER_FSDP:-1}
TRAINER_TP=${TRAINER_TP:-2}
ROLLOUT_TPU_CHIPS=${ROLLOUT_TPU_CHIPS:-2,3}
ROLLOUT_FSDP=${ROLLOUT_FSDP:-1}
ROLLOUT_TP=${ROLLOUT_TP:-2}
TPU_CHIPS_PER_HOST_BOUNDS=${TPU_CHIPS_PER_HOST_BOUNDS:-1,2,1}
TPU_HOST_BOUNDS=${TPU_HOST_BOUNDS:-1,1,1}

WAIT_TIMEOUT_SECS=${WAIT_TIMEOUT_SECS:-1800}
WAIT_POLL_SECS=${WAIT_POLL_SECS:-5}
SHUTDOWN_GRACE_SECS=${SHUTDOWN_GRACE_SECS:-60}

TRAINER_LOG="${LOG_ROOT}/trainer.log"
ROLLOUT_LOG="${LOG_ROOT}/rollout.log"
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
  for pid in "${TRAINER_PID:-}" "${ROLLOUT_PID:-}"; do
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
echo "  mini batch:     ${MINI_BATCH_SIZE} prompt groups/update"
echo "  generations:    ${NUM_GENERATIONS}"
echo "  max steps:      ${MAX_STEPS}"
echo "  max turns:      ${MAX_TURNS}"
echo "  prompt length:  ${MAX_PROMPT_LENGTH}"
echo "  response len:   ${MAX_RESPONSE_LENGTH}"
echo "  max seq token:  ${MAX_SEQ_TOKEN_PER_TPU:-<unset>}"
echo "  max segments:   ${MAX_SEGMENTS_PER_PACKED_ROW:-<unset>}"
echo "  learning rate:  ${LEARNING_RATE}"
echo "  beta:           ${BETA}"
echo "  sampler:        ${SAMPLER}"
echo "  weight sync:    ${WEIGHT_SYNC_MODE}"
echo "  trainer chips:  ${TRAINER_TPU_CHIPS}"
echo "  rollout chips:  ${ROLLOUT_TPU_CHIPS}"
echo "  wandb project:  ${WANDB_PROJECT:-<none>}"
echo "  wandb run name: ${WANDB_RUN_NAME:-<auto>}"
echo "  ckpt interval:  ${CHECKPOINT_SAVE_INTERVAL_STEPS}"
echo "  ckpt max keep:  ${CHECKPOINT_MAX_TO_KEEP}"
echo "  ckpt root dir:  ${CHECKPOINT_ROOT_DIRECTORY}"
echo "=================================================="

if [[ "$BETA" != "0" && "$BETA" != "0.0" ]]; then
  echo "Error: this first DeepSWE distributed launcher only wires trainer+rollout."
  echo "Use BETA=0.0 until the reference inference worker is added."
  exit 1
fi

if (( BATCH_SIZE <= 0 ||
      MINI_BATCH_SIZE <= 0 ||
      NUM_GENERATIONS <= 0 ||
      TRAIN_MICRO_BATCH_SIZE <= 0 )); then
  echo "Error: batch sizes and NUM_GENERATIONS must all be positive."
  echo "  BATCH_SIZE=$BATCH_SIZE MINI_BATCH_SIZE=$MINI_BATCH_SIZE"
  echo "  NUM_GENERATIONS=$NUM_GENERATIONS TRAIN_MICRO_BATCH_SIZE=$TRAIN_MICRO_BATCH_SIZE"
  exit 1
fi
if (( BATCH_SIZE % MINI_BATCH_SIZE != 0 )); then
  echo "Error: BATCH_SIZE must be divisible by MINI_BATCH_SIZE."
  echo "  BATCH_SIZE=$BATCH_SIZE MINI_BATCH_SIZE=$MINI_BATCH_SIZE"
  exit 1
fi
if (( (MINI_BATCH_SIZE * NUM_GENERATIONS) % TRAIN_MICRO_BATCH_SIZE != 0 )); then
  echo "Error: MINI_BATCH_SIZE * NUM_GENERATIONS must be divisible by TRAIN_MICRO_BATCH_SIZE."
  echo "  MINI_BATCH_SIZE=$MINI_BATCH_SIZE NUM_GENERATIONS=$NUM_GENERATIONS TRAIN_MICRO_BATCH_SIZE=$TRAIN_MICRO_BATCH_SIZE"
  exit 1
fi

ensure_model_dir
mkdir -p "$LOG_ROOT" "$ARTIFACT_ROOT"
: > "$TRAINER_LOG"
: > "$ROLLOUT_LOG"
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
    --tokenizer_path="$TOKENIZER_PATH"
    --max_prompt_length="$MAX_PROMPT_LENGTH"
    --max_response_length="$MAX_RESPONSE_LENGTH"
    --mini_batch_size="$MINI_BATCH_SIZE"
    --num_generations="$NUM_GENERATIONS"
    --train_micro_batch_size="$TRAIN_MICRO_BATCH_SIZE"
    --eval_every_n_steps="$EVAL_EVERY_N_STEPS"
    --learning_rate="$LEARNING_RATE"
    --adam_b1="$ADAM_B1"
    --adam_b2="$ADAM_B2"
    --weight_decay="$WEIGHT_DECAY"
    --max_grad_norm="$MAX_GRAD_NORM"
    --lora_rank="$LORA_RANK"
    --lora_alpha="$LORA_ALPHA"
    --sampler_type="$SAMPLER"
    --checkpoint_save_interval_steps="$CHECKPOINT_SAVE_INTERVAL_STEPS"
    --checkpoint_max_to_keep="$CHECKPOINT_MAX_TO_KEEP"
    --checkpoint_root_directory="$CHECKPOINT_ROOT_DIRECTORY"
  )
  if [[ "$USE_LORA" == "1" || "$USE_LORA" == "true" || "$USE_LORA" == "True" ]]; then
    TRAINER_CMD+=(--use_lora)
  fi
  if [[ "$DEBUG" == "1" || "$DEBUG" == "true" || "$DEBUG" == "True" ]]; then
    TRAINER_CMD+=(--debug)
  fi
  export JAX_PLATFORMS=tpu,cpu
  export TPU_VISIBLE_DEVICES=${TRAINER_TPU_CHIPS}
  export TPU_VISIBLE_CHIPS=${TPU_VISIBLE_DEVICES}
  export TPU_CHIPS_PER_HOST_BOUNDS=${TPU_CHIPS_PER_HOST_BOUNDS}
  export TPU_HOST_BOUNDS=${TPU_HOST_BOUNDS}
  export LIBTPU_INIT_ARGS="--deepsea_chips_per_host_bounds=${TPU_CHIPS_PER_HOST_BOUNDS} --deepsea_host_bounds=${TPU_HOST_BOUNDS}"
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
    --enable_thinking
  )
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
    export DATASET_NAME="$DATASET_NAME"
    export DATASET_PATH="$DATASET_PATH"
    export DATASET_SPLIT="$DATASET_SPLIT"
    export DATASET_CACHE_DIR="$DATASET_CACHE_DIR"
    export SHUFFLE="$SHUFFLE"
    export SEED="$SEED"
    export ROLLOUT_MAX_CONCURRENCY="$ROLLOUT_MAX_CONCURRENCY"
    export SANDBOX_MAX_CONCURRENCY="$SANDBOX_MAX_CONCURRENCY"
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

wait_for_port "trainer" "$TRAINER_PORT" "$TRAINER_PID" "$TRAINER_LOG"
wait_for_port "rollout" "$ROLLOUT_PORT" "$ROLLOUT_PID" "$ROLLOUT_LOG"

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
    --max_steps="$MAX_STEPS"
    --max_turns="$MAX_TURNS"
    --max_prompt_length="$MAX_PROMPT_LENGTH"
    --max_response_length="$MAX_RESPONSE_LENGTH"
    --episode_timeout_secs="$EPISODE_TIMEOUT_SECS"
    --train_micro_batch_size="$TRAIN_MICRO_BATCH_SIZE"
    --beta="$BETA"
    --epsilon="$EPSILON"
    --epsilon_high="$EPSILON_HIGH"
    --advantage_estimator="$ADVANTAGE_ESTIMATOR"
    --loss_agg_mode="$LOSS_AGG_MODE"
    --max_staleness="$OFF_POLICY_STEPS"
    --top_p="$TOP_P"
    --top_k="$TOP_K"
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
  if [[ -n "$DATASET_PATH" ]]; then
    ORCHESTRATOR_CMD+=(--dataset_path="$DATASET_PATH")
  fi
  if [[ -n "$MAX_WARMPOOL_REPLICAS" ]]; then
    ORCHESTRATOR_CMD+=(--max_warmpool_replicas="$MAX_WARMPOOL_REPLICAS")
  fi
  if [[ "$SHUFFLE" == "0" || "$SHUFFLE" == "false" || "$SHUFFLE" == "False" ]]; then
    ORCHESTRATOR_CMD+=(--no-shuffle)
  else
    ORCHESTRATOR_CMD+=(--shuffle)
  fi
  if [[ "$OVERLONG_FILTER" == "0" || "$OVERLONG_FILTER" == "false" || "$OVERLONG_FILTER" == "False" ]]; then
    ORCHESTRATOR_CMD+=(--no-overlong_filter)
  else
    ORCHESTRATOR_CMD+=(--overlong_filter)
  fi
  if [[ "$USE_AGENT_SANDBOX" == "1" || "$USE_AGENT_SANDBOX" == "true" || "$USE_AGENT_SANDBOX" == "True" ]]; then
    ORCHESTRATOR_CMD+=(--use_agent_sandbox)
  fi
  if [[ "$DEBUG" == "1" || "$DEBUG" == "true" || "$DEBUG" == "True" ]]; then
    ORCHESTRATOR_CMD+=(--debug)
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
  if [[ "$USE_ROLLOUT_LOGPS" == "false" || "$USE_ROLLOUT_LOGPS" == "False" || "$USE_ROLLOUT_LOGPS" == "0" ]]; then
    ORCHESTRATOR_CMD+=(--no-use_rollout_logps)
  else
    ORCHESTRATOR_CMD+=(--use_rollout_logps)
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
