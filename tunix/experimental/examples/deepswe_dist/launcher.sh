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
TRAIN_MICRO_BATCH_SIZE=${TRAIN_MICRO_BATCH_SIZE:-1}
MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-$((BATCH_SIZE * NUM_GENERATIONS))}
EVAL_EVERY_N_STEPS=${EVAL_EVERY_N_STEPS:-1000000}
BETA=${BETA:-0.0}
EPSILON=${EPSILON:-0.2}
SAMPLER=${SAMPLER:-inprocess_vllm}
WEIGHT_SYNC_MODE=${WEIGHT_SYNC_MODE:-none}
USE_LORA=${USE_LORA:-0}
LORA_RANK=${LORA_RANK:-64}
LORA_ALPHA=${LORA_ALPHA:-64.0}
DEBUG=${DEBUG:-0}

DATASET_NAME=${DATASET_NAME:-R2E-Gym/R2E-Gym-Subset}
DATASET_PATH=${DATASET_PATH:-}
DATASET_SPLIT=${DATASET_SPLIT:-train}
DATASET_CACHE_DIR=${DATASET_CACHE_DIR:-"${ARTIFACT_ROOT}/dataset_cache"}
SHUFFLE=${SHUFFLE:-true}
SEED=${SEED:-42}
ENV_BACKEND=${ENV_BACKEND:-kubernetes}
SCAFFOLD=${SCAFFOLD:-r2egym}
USE_AGENT_SANDBOX=${USE_AGENT_SANDBOX:-0}
STEP_TIMEOUT_SECS=${STEP_TIMEOUT_SECS:-1800}
REWARD_TIMEOUT_SECS=${REWARD_TIMEOUT_SECS:-1800}
ROLLOUT_MAX_CONCURRENCY=${ROLLOUT_MAX_CONCURRENCY:-64}

WANDB_PROJECT=${WANDB_PROJECT:-trellis-deepswe}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-}
WANDB_API_KEY=${WANDB_API_KEY:-}

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
echo "  generations:    ${NUM_GENERATIONS}"
echo "  max steps:      ${MAX_STEPS}"
echo "  max turns:      ${MAX_TURNS}"
echo "  prompt length:  ${MAX_PROMPT_LENGTH}"
echo "  response len:   ${MAX_RESPONSE_LENGTH}"
echo "  beta:           ${BETA}"
echo "  sampler:        ${SAMPLER}"
echo "  weight sync:    ${WEIGHT_SYNC_MODE}"
echo "  trainer chips:  ${TRAINER_TPU_CHIPS}"
echo "  rollout chips:  ${ROLLOUT_TPU_CHIPS}"
echo "=================================================="

if [[ "$BETA" != "0" && "$BETA" != "0.0" ]]; then
  echo "Error: this first DeepSWE distributed launcher only wires trainer+rollout."
  echo "Use BETA=0.0 until the reference inference worker is added."
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
    --process_main=tunix.experimental.examples.math_gsm8k_dist.run_trainer_node.main
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
    --train_micro_batch_size="$TRAIN_MICRO_BATCH_SIZE"
    --eval_every_n_steps="$EVAL_EVERY_N_STEPS"
    --lora_rank="$LORA_RANK"
    --lora_alpha="$LORA_ALPHA"
    --sampler="$SAMPLER"
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
    --process_main=tunix.experimental.examples.deepswe_dist.run_rollout_node.main
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
    --scaffold="$SCAFFOLD"
    --max_concurrency="$ROLLOUT_MAX_CONCURRENCY"
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
    --num_generations="$NUM_GENERATIONS"
    --max_steps="$MAX_STEPS"
    --max_turns="$MAX_TURNS"
    --max_prompt_length="$MAX_PROMPT_LENGTH"
    --max_response_length="$MAX_RESPONSE_LENGTH"
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
    --weight_sync_mode="$WEIGHT_SYNC_MODE"
    --stop_workers_on_exit
  )
  if [[ -n "$DATASET_PATH" ]]; then
    ORCHESTRATOR_CMD+=(--dataset_path="$DATASET_PATH")
  fi
  if [[ "$SHUFFLE" == "0" || "$SHUFFLE" == "false" || "$SHUFFLE" == "False" ]]; then
    ORCHESTRATOR_CMD+=(--no-shuffle)
  else
    ORCHESTRATOR_CMD+=(--shuffle)
  fi
  if [[ "$USE_AGENT_SANDBOX" == "1" || "$USE_AGENT_SANDBOX" == "true" || "$USE_AGENT_SANDBOX" == "True" ]]; then
    ORCHESTRATOR_CMD+=(--use_agent_sandbox)
  fi
  if [[ "$DEBUG" == "1" || "$DEBUG" == "true" || "$DEBUG" == "True" ]]; then
    ORCHESTRATOR_CMD+=(--debug)
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