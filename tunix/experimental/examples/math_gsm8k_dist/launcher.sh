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

#!/bin/bash

set -Ee

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${DIR}/../../../.." && pwd)"
LOG_ROOT=${LOG_ROOT:-"${DIR}"}

ORCHESTRATOR_ID=${ORCHESTRATOR_ID:-orchestrator}
ORCHESTRATOR_PORT=${ORCHESTRATOR_PORT:-30000}
TRAINER_PORT=${TRAINER_PORT:-20000}
ROLLOUT_PORT=${ROLLOUT_PORT:-20001}
INFERENCE_PORT=${INFERENCE_PORT:-20002}
# Comma-separated gRPC ports, one rollout node per port (e.g. "20001,20003,20004").
ROLLOUT_PORTS=${ROLLOUT_PORTS:-$ROLLOUT_PORT}
# Optional py-inference-scheduler sidecar. Either point SCHEDULER_URL at an
# externally managed sidecar (e.g. a k8s sidecar container), or set
# START_SCHEDULER=1 with SCHEDULER_REPO pointing at a py-rl-scheduler checkout
# to have this script launch one locally.
SCHEDULER_URL=${SCHEDULER_URL:-}
START_SCHEDULER=${START_SCHEDULER:-0}
SCHEDULER_REPO=${SCHEDULER_REPO:-}
SCHEDULER_PORT=${SCHEDULER_PORT:-8100}
SCHEDULER_CONFIG=${SCHEDULER_CONFIG:-integration/tunix/examples/scheduler.yaml}
RUN_INFERENCE_NODE=${RUN_INFERENCE_NODE:-0}
INFERENCE_ADDR=${INFERENCE_ADDR:-}
MODEL_NAME=${MODEL_NAME:-Qwen3-1.7B}
MODEL_ID=${MODEL_ID:-Qwen/Qwen3-1.7B}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-"${REPO_ROOT}/artifacts/qwen3_dist_gsm8k"}
MODEL_DIR=${MODEL_DIR:-${MODEL_DOWNLOAD_DIR:-"${ARTIFACT_ROOT}/models"}}
TOKENIZER_PATH=${TOKENIZER_PATH:-$MODEL_DIR}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1024}
BATCH_SIZE=${BATCH_SIZE:-4}
NUM_GENERATIONS=${NUM_GENERATIONS:-8}
MAX_STEPS=${MAX_STEPS:-1}
TRAIN_MICRO_BATCH_SIZE=${TRAIN_MICRO_BATCH_SIZE:-1}
MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-2}
EVAL_EVERY_N_STEPS=${EVAL_EVERY_N_STEPS:-50}
LORA_RANK=${LORA_RANK:-64}
LORA_ALPHA=${LORA_ALPHA:-64.0}
USE_LORA=${USE_LORA:-0}
REWARD_MODE=${REWARD_MODE:-env}
TFDS_DATA_DIR=${TFDS_DATA_DIR:-"${ARTIFACT_ROOT}/data"}
TFDS_SPLIT=${TFDS_SPLIT:-train}
SEED=${SEED:-42}
SHUFFLE=${SHUFFLE:-true}
BETA=${BETA:-0.04}
EPSILON=${EPSILON:-0.2}
WANDB_PROJECT=${WANDB_PROJECT:-trellis-gsm8k}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-}
WANDB_API_KEY=${WANDB_API_KEY:-}
SAMPLER=${SAMPLER:-inprocess_vllm}
WEIGHT_SYNC_MODE=${WEIGHT_SYNC_MODE:-none}
PYTHON_BIN=${PYTHON_BIN:-python3}
WAIT_TIMEOUT_SECS=${WAIT_TIMEOUT_SECS:-1800}
WAIT_POLL_SECS=${WAIT_POLL_SECS:-5}
WAIT_DEBUG_EVERY_POLLS=${WAIT_DEBUG_EVERY_POLLS:-6}
WAIT_LOG_TAIL_LINES=${WAIT_LOG_TAIL_LINES:-40}
TRAINER_SHUTDOWN_GRACE_SECS=${TRAINER_SHUTDOWN_GRACE_SECS:-900}
WORKER_SHUTDOWN_GRACE_SECS=${WORKER_SHUTDOWN_GRACE_SECS:-10}
# Time after the first interrupt. During this time, a second one is gnored.
# So a double-tapped Ctrl+C cannot discard a checkpoint that is still being
# written.
FORCE_ARM_SECS=${FORCE_ARM_SECS:-5}
SHUTDOWN_SIGNAL_COUNT=0
DRAIN_START_SECONDS=0
FORCE_KILL=0

TRAINER_TPU_CHIPS=${TRAINER_TPU_CHIPS:-0,1}
TRAINER_FSDP=${TRAINER_FSDP:-1}
TRAINER_TP=${TRAINER_TP:-2}
ROLLOUT_TPU_CHIPS=${ROLLOUT_TPU_CHIPS:-2,3}
ROLLOUT_FSDP=${ROLLOUT_FSDP:-1}
ROLLOUT_TP=${ROLLOUT_TP:-2}
# Semicolon-separated chip groups aligned with ROLLOUT_PORTS (e.g. "2;3;4" for
# three 1-chip rollout nodes). Defaults to a single node on ROLLOUT_TPU_CHIPS.
ROLLOUT_TPU_CHIPS_LIST=${ROLLOUT_TPU_CHIPS_LIST:-$ROLLOUT_TPU_CHIPS}
INFERENCE_TPU_CHIPS=${INFERENCE_TPU_CHIPS:-}
TPU_CHIPS_PER_HOST_BOUNDS=${TPU_CHIPS_PER_HOST_BOUNDS:-1,2,1}
TPU_HOST_BOUNDS=${TPU_HOST_BOUNDS:-1,1,1}
# If OOM, try the following settings (assume 8 chips per host):
# TRAINER_TPU_CHIPS=${TRAINER_TPU_CHIPS:-0,1,2,3}
# TRAINER_FSDP=${TRAINER_FSDP:-1}
# TRAINER_TP=${TRAINER_TP:-4}
# ROLLOUT_TPU_CHIPS=${ROLLOUT_TPU_CHIPS:-4,5,6,7}
# ROLLOUT_FSDP=${ROLLOUT_FSDP:-1}
# ROLLOUT_TP=${ROLLOUT_TP:-4}
# INFERENCE_TPU_CHIPS=${INFERENCE_TPU_CHIPS:-}
# TPU_CHIPS_PER_HOST_BOUNDS=${TPU_CHIPS_PER_HOST_BOUNDS:-1,4,1}
# TPU_HOST_BOUNDS=${TPU_HOST_BOUNDS:-1,1,1}

TRAINER_LOG="${LOG_ROOT}/trainer.log"
ROLLOUT_LOG="${LOG_ROOT}/rollout.log"
INFERENCE_LOG="${LOG_ROOT}/inference.log"
ORCHESTRATOR_LOG="${LOG_ROOT}/orchestrator.log"
SCHEDULER_LOG="${LOG_ROOT}/scheduler.log"

IFS=',' read -r -a ROLLOUT_PORT_ARR <<< "$ROLLOUT_PORTS"
IFS=';' read -r -a ROLLOUT_CHIPS_ARR <<< "$ROLLOUT_TPU_CHIPS_LIST"
if (( ${#ROLLOUT_PORT_ARR[@]} != ${#ROLLOUT_CHIPS_ARR[@]} )); then
  echo "Error: ROLLOUT_PORTS has ${#ROLLOUT_PORT_ARR[@]} entries but" \
       "ROLLOUT_TPU_CHIPS_LIST has ${#ROLLOUT_CHIPS_ARR[@]} (must match)."
  exit 1
fi
ROLLOUT_PIDS=()
ROLLOUT_LOGS=()
for i in "${!ROLLOUT_PORT_ARR[@]}"; do
  ROLLOUT_LOGS+=("${LOG_ROOT}/rollout_${i}.log")
done

print_section() {
  echo
  echo "================ $1 ================"
}

print_command() {
  local label="$1"
  shift
  echo "$label:"
  printf '  %q' "$@"
  echo
}

print_file_debug() {
  local label="$1"
  local file_path="$2"
  echo "$label log path: $file_path"
  if [[ -f "$file_path" ]]; then
    ls -lh "$file_path" 2>/dev/null || true
    echo "$label log last ${WAIT_LOG_TAIL_LINES} lines:"
    tail -n "$WAIT_LOG_TAIL_LINES" "$file_path" 2>/dev/null || true
  else
    echo "$label log does not exist yet."
  fi
}

print_process_debug() {
  local label="$1"
  local pid="$2"
  echo "$label pid=$pid"
  ps -fp "$pid" 2>/dev/null || true
  ps -o pid,ppid,stat,etime,pcpu,pmem,rss,args -p "$pid" 2>/dev/null || true
}

print_port_debug() {
  local port="$1"
  echo "Port $port listeners:"
  if command -v ss >/dev/null 2>&1; then
    ss -ltnp 2>/dev/null | grep -E "(:${port}\\b|:${port} )" || true
  elif command -v lsof >/dev/null 2>&1; then
    lsof -nP -iTCP:"$port" -sTCP:LISTEN 2>/dev/null || true
  else
    echo "Neither ss nor lsof is available."
  fi
}

print_related_processes() {
  echo "Related processes:"
  pgrep -af "run_trainer_node.py|run_rollout_node.py|run_inference_node.py|run_gsm8k_dist_grpo.py|vllm" 2>/dev/null || true
}

dump_debug_snapshot() {
  print_section "process snapshot"
  if [[ -n "${TRAINER_PID:-}" ]]; then
    print_process_debug "trainer" "$TRAINER_PID"
  fi
  for i in "${!ROLLOUT_PIDS[@]}"; do
    print_process_debug "rollout${i}" "${ROLLOUT_PIDS[$i]}"
  done
  if [[ -n "${INFERENCE_PID:-}" ]]; then
    print_process_debug "inference" "$INFERENCE_PID"
  fi
  print_related_processes
  print_section "port snapshot"
  print_port_debug "$TRAINER_PORT"
  for port in "${ROLLOUT_PORT_ARR[@]}"; do
    print_port_debug "$port"
  done
  print_port_debug "$INFERENCE_PORT"
  print_section "log snapshot"
  print_file_debug "trainer" "$TRAINER_LOG"
  for i in "${!ROLLOUT_LOGS[@]}"; do
    print_file_debug "rollout${i}" "${ROLLOUT_LOGS[$i]}"
  done
  print_file_debug "inference" "$INFERENCE_LOG"
  print_file_debug "orchestrator" "$ORCHESTRATOR_LOG"
}

on_error() {
  local exit_code="$?"
  print_section "launcher error"
  echo "Launcher failed with exit code $exit_code."
  dump_debug_snapshot
  exit "$exit_code"
}

print_safetensors_candidates() {
  local root="$1"
  if [[ ! -d "$root" ]]; then
    echo "MODEL_DIR does not exist yet; no nested safetensors to inspect."
    return
  fi
  local candidates
  candidates="$(
    find "$root" -mindepth 2 -maxdepth 5 -type f -name '*.safetensors' -print 2>/dev/null \
      | sed 's#/[^/]*$##' \
      | sort \
      | uniq -c \
      | sort -nr \
      | head -20 || true
  )"
  if [[ -n "$candidates" ]]; then
    echo "Found nested safetensors directories. The trainer expects MODEL_DIR"
    echo "to be one of these directories, not their parent:"
    echo "$candidates" | sed 's/^/  /'
  else
    echo "No nested safetensors were found under MODEL_DIR either."
  fi
}

has_direct_safetensors() {
  [[ -d "$MODEL_DIR" ]] && [[ -n "$(
    find "$MODEL_DIR" -maxdepth 1 -type f -name '*.safetensors' -print -quit 2>/dev/null || true
  )" ]]
}

download_model_dir() {
  echo "Downloading $MODEL_ID to MODEL_DIR: $MODEL_DIR"
  PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "$PYTHON_BIN" - "$MODEL_ID" "$MODEL_DIR" <<'PY'
import os
import sys

from tunix.oss import utils as oss_utils

model_id, model_dir = sys.argv[1], sys.argv[2]
os.makedirs(model_dir, exist_ok=True)
oss_utils.hf_pipeline(model_id, model_dir)
PY
}

ensure_model_dir() {
  if [[ -e "$MODEL_DIR" && ! -d "$MODEL_DIR" ]]; then
    echo "Error: MODEL_DIR exists but is not a directory: $MODEL_DIR"
    exit 1
  fi

  if has_direct_safetensors; then
    return
  fi

  echo "No '*.safetensors' files found directly in MODEL_DIR: $MODEL_DIR"
  echo "The JAX trainer loader uses a non-recursive '*.safetensors' lookup."
  print_safetensors_candidates "$MODEL_DIR"
  download_model_dir

  if ! has_direct_safetensors; then
    echo "Error: download finished but MODEL_DIR still has no direct '*.safetensors': $MODEL_DIR"
    print_safetensors_candidates "$MODEL_DIR"
    exit 1
  fi
}

dump_logs() {
  print_section "trainer.log tail"
  tail -n 200 "$TRAINER_LOG" 2>/dev/null || true
  for i in "${!ROLLOUT_LOGS[@]}"; do
    print_section "rollout_${i}.log tail"
    tail -n 200 "${ROLLOUT_LOGS[$i]}" 2>/dev/null || true
  done
  print_section "inference.log tail"
  tail -n 200 "$INFERENCE_LOG" 2>/dev/null || true
  print_section "orchestrator.log tail"
  tail -n 200 "$ORCHESTRATOR_LOG" 2>/dev/null || true
}

check_process_alive() {
  local name="$1"
  local pid="$2"
  local log_file="$3"
  local process_state
  # Read state from /proc directly: minimal images (e.g. python:*-slim) ship
  # no `ps`, and an empty result here must mean "dead", not "tool missing".
  process_state="$(read -r _ _ st _ < "/proc/$pid/stat" 2>/dev/null && echo "$st" || true)"
  if [[ -z "$process_state" || "$process_state" == Z* ]]; then
    echo "Error: $name process exited before gRPC port became ready (pid=$pid)."
    print_file_debug "$name" "$log_file"
    dump_debug_snapshot
    exit 1
  fi
}

wait_for_port() {
  local name="$1"
  local port="$2"
  local pid="$3"
  local log_file="$4"
  local elapsed=0
  local poll_count=0
  while true; do
    check_process_alive "$name" "$pid" "$log_file"
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
      echo "Error: timed out waiting ${WAIT_TIMEOUT_SECS}s for $name port $port."
      dump_debug_snapshot
      exit 1
    fi
    echo "Waiting for $name port $port... elapsed=${elapsed}s pid=$pid log=$log_file"
    if (( poll_count == 0 || poll_count % WAIT_DEBUG_EVERY_POLLS == 0 )); then
      print_process_debug "$name" "$pid"
      print_port_debug "$port"
      print_file_debug "$name" "$log_file"
    fi
    sleep "$WAIT_POLL_SECS"
    elapsed=$((elapsed + WAIT_POLL_SECS))
    poll_count=$((poll_count + 1))
  done
}

echo "=================================================="
echo "Starting distributed GSM8K GRPO chain demo locally"
echo "  rollout engine: vLLM"
echo "  model dir:      $MODEL_DIR"
echo "  tokenizer path: $TOKENIZER_PATH"
echo "  python:         $PYTHON_BIN"
echo "  trajectories:   $((BATCH_SIZE * NUM_GENERATIONS)) per step"
echo "  batch size:     $BATCH_SIZE"
echo "  generations:    $NUM_GENERATIONS"
echo "  max steps:      $MAX_STEPS"
echo "  eval interval:  $EVAL_EVERY_N_STEPS"
echo "  prompt length:  $MAX_PROMPT_LENGTH"
echo "  response len:   $MAX_RESPONSE_LENGTH"
echo "  train micro:    $TRAIN_MICRO_BATCH_SIZE"
echo "  mini batch:     $MINI_BATCH_SIZE"
echo "  beta:           $BETA"
echo "  epsilon:        $EPSILON"
echo "  reward mode:    $REWARD_MODE"
echo "  tfds split:     $TFDS_SPLIT"
echo "  tfds data dir:  $TFDS_DATA_DIR"
echo "  shuffle:        $SHUFFLE"
echo "  use lora:       $USE_LORA"
echo "  sampler:        $SAMPLER"
echo "  weight sync:    $WEIGHT_SYNC_MODE"
echo "  trainer chips:  $TRAINER_TPU_CHIPS"
echo "  rollout chips:  $ROLLOUT_TPU_CHIPS"
echo "  inference:      $RUN_INFERENCE_NODE"
echo "  inference addr: ${INFERENCE_ADDR:-<none>}"
echo "  inference chips:${INFERENCE_TPU_CHIPS:-<unset>}"
echo "  chip bounds:    $TPU_CHIPS_PER_HOST_BOUNDS"
echo "  host bounds:    $TPU_HOST_BOUNDS"
echo "  wait timeout:   ${WAIT_TIMEOUT_SECS}s"
echo "  wait poll:      ${WAIT_POLL_SECS}s"
echo "  log tail lines: $WAIT_LOG_TAIL_LINES"
echo "  wandb project:  ${WANDB_PROJECT:-<none>}"
echo "  wandb run name: ${WANDB_RUN_NAME:-<auto>}"
echo "  trainer log:    $TRAINER_LOG"
echo "  rollout logs:   ${ROLLOUT_LOGS[*]}"
echo "  orch log:       $ORCHESTRATOR_LOG"
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
mkdir -p "${LOG_ROOT}"

: > "$TRAINER_LOG"
for log in "${ROLLOUT_LOGS[@]}"; do
  : > "$log"
done
: > "$INFERENCE_LOG"
: > "$SCHEDULER_LOG"
: > "$ORCHESTRATOR_LOG"

print_section "runtime context"
date
pwd
"$PYTHON_BIN" --version 2>&1 || true
git -C "$DIR" rev-parse --short HEAD 2>/dev/null || true
git -C "$DIR" status --short --branch 2>/dev/null || true
echo "MODEL_DIR exists? $(if [[ -d "$MODEL_DIR" ]]; then echo yes; else echo no; fi)"
echo "TOKENIZER_PATH exists? $(if [[ -d "$TOKENIZER_PATH" ]]; then echo yes; else echo no; fi)"
echo "Initial LIBTPU_INIT_ARGS=${LIBTPU_INIT_ARGS:-}"
print_related_processes
print_port_debug "$TRAINER_PORT"
for port in "${ROLLOUT_PORT_ARR[@]}"; do
  print_port_debug "$port"
done
print_port_debug "$INFERENCE_PORT"

echo "Launching trainer node on TPU chips $TRAINER_TPU_CHIPS..."
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
  )
  if [[ "$USE_LORA" == "1" || "$USE_LORA" == "true" || "$USE_LORA" == "True" ]]; then
    TRAINER_CMD+=(--use_lora)
  fi

  if [[ "${TRAINER_PATHWAYS:-0}" == "1" ]]; then
    # Chips 0-3 belong to the local Pathways worker container; the trainer
    # is a proxy client and must not grab them.
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
    export LIBTPU_INIT_ARGS=deepsea_chips_per_host_bounds=${TPU_CHIPS_PER_HOST_BOUNDS},deepsea_host_bounds=${TPU_HOST_BOUNDS}
  fi
  export PYTHONUNBUFFERED=1
  env | egrep 'JAX|TPU'
  print_command "Trainer command" "${TRAINER_CMD[@]}"
  exec "${TRAINER_CMD[@]}" > "$TRAINER_LOG" 2>&1
) &
TRAINER_PID=$!
echo "Trainer pid=$TRAINER_PID log=$TRAINER_LOG"
print_process_debug "trainer" "$TRAINER_PID"

echo "Launching vLLM rollout node on TPU chips $ROLLOUT_TPU_CHIPS..."
ROLLOUT_CMD=(
  "$PYTHON_BIN" "${DIR}/run_rollout_node.py"
  --port="$ROLLOUT_PORT"
  --tpu_chips="$ROLLOUT_TPU_CHIPS"
  --tpu_chips_per_host_bounds="$TPU_CHIPS_PER_HOST_BOUNDS"
  --tpu_host_bounds="$TPU_HOST_BOUNDS"
  --model_id="$MODEL_ID"
  --model_dir="$MODEL_DIR"
  --tokenizer_path="$TOKENIZER_PATH"
  --max_prompt_length="$MAX_PROMPT_LENGTH"
  --max_response_length="$MAX_RESPONSE_LENGTH"
  --lora_rank="$LORA_RANK"
  --lora_alpha="$LORA_ALPHA"
)
if [[ "$USE_LORA" == "1" || "$USE_LORA" == "true" || "$USE_LORA" == "True" ]]; then
  ROLLOUT_CMD+=(--use_lora)
fi
print_command "Rollout command" PYTHONUNBUFFERED=1 "${ROLLOUT_CMD[@]}"
PYTHONUNBUFFERED=1 "${ROLLOUT_CMD[@]}" > "$ROLLOUT_LOG" 2>&1 &
ROLLOUT_PID=$!
echo "Rollout pid=$ROLLOUT_PID log=$ROLLOUT_LOG"
print_process_debug "rollout" "$ROLLOUT_PID"

cleanup() {
  echo "Cleaning up worker processes (PIDs: $TRAINER_PID, $ROLLOUT_PID, ${INFERENCE_PID:-})..."
  kill "$TRAINER_PID" "$ROLLOUT_PID" "${INFERENCE_PID:-}" 2>/dev/null || true
  wait "$TRAINER_PID" "$ROLLOUT_PID" "${INFERENCE_PID:-}" 2>/dev/null || true
  echo "Workers stopped."
}
trap on_error ERR
trap cleanup EXIT

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
      --process_main=tunix.experimental.examples.math_gsm8k_dist.run_inference_node.main

      --port="$INFERENCE_PORT"
      --model_name="$MODEL_NAME"
      --model_id="$MODEL_ID"
      --model_dir="$MODEL_DIR"
      --tokenizer_path="$TOKENIZER_PATH"
      --compute_logps_micro_batch_size="$TRAIN_MICRO_BATCH_SIZE"
      --max_prompt_length="$MAX_PROMPT_LENGTH"
      --max_response_length="$MAX_RESPONSE_LENGTH"
    )

    export JAX_PLATFORMS=tpu,cpu
    export TPU_VISIBLE_DEVICES=${INFERENCE_TPU_CHIPS}
    export TPU_VISIBLE_CHIPS=${TPU_VISIBLE_DEVICES}
    export TPU_CHIPS_PER_HOST_BOUNDS=${TPU_CHIPS_PER_HOST_BOUNDS}
    export TPU_HOST_BOUNDS=${TPU_HOST_BOUNDS}
    export LIBTPU_INIT_ARGS=deepsea_chips_per_host_bounds=${TPU_CHIPS_PER_HOST_BOUNDS},deepsea_host_bounds=${TPU_HOST_BOUNDS}
    export PYTHONUNBUFFERED=1
    env | egrep 'JAX|TPU'
    print_command "Inference command" "${INFERENCE_CMD[@]}"
    exec "${INFERENCE_CMD[@]}" > "$INFERENCE_LOG" 2>&1
  ) &
  INFERENCE_PID=$!
  INFERENCE_ADDR="localhost:$INFERENCE_PORT"
  echo "Inference pid=$INFERENCE_PID log=$INFERENCE_LOG"
  print_process_debug "inference" "$INFERENCE_PID"
fi

echo "Waiting for gRPC servers to bind..."
wait_for_port "trainer" "$TRAINER_PORT" "$TRAINER_PID" "$TRAINER_LOG"
for i in "${!ROLLOUT_PORT_ARR[@]}"; do
  wait_for_port "rollout${i}" "${ROLLOUT_PORT_ARR[$i]}" "${ROLLOUT_PIDS[$i]}" "${ROLLOUT_LOGS[$i]}"
done
if [[ -n "${INFERENCE_PID:-}" ]]; then
  wait_for_port "inference" "$INFERENCE_PORT" "$INFERENCE_PID" "$INFERENCE_LOG"
fi
if [[ -n "${SCHEDULER_PID:-}" ]]; then
  wait_for_port "scheduler" "$SCHEDULER_PORT" "$SCHEDULER_PID" "$SCHEDULER_LOG"
fi
dump_debug_snapshot

echo "Launching CPU orchestrator..."
ROLLOUT_ADDRS=""
for port in "${ROLLOUT_PORT_ARR[@]}"; do
  ROLLOUT_ADDRS+="${ROLLOUT_ADDRS:+,}localhost:${port}"
done
(
  ORCHESTRATOR_CMD=(
    "$PYTHON_BIN" -m tunix.experimental.distributed.runtime.main
    --discovery_id="${ORCHESTRATOR_ID}"
    --discovery_port="${ORCHESTRATOR_PORT}"
    --process_main=tunix.experimental.examples.math_gsm8k_dist.run_gsm8k_dist_grpo.main
    --rollout_addr="$ROLLOUT_ADDRS"

    --model_id="$MODEL_ID"
    --tokenizer_path="$TOKENIZER_PATH"
    --batch_size="$BATCH_SIZE"
    --num_generations="$NUM_GENERATIONS"
    --max_steps="$MAX_STEPS"
    --max_prompt_length="$MAX_PROMPT_LENGTH"
    --max_response_length="$MAX_RESPONSE_LENGTH"
    --train_micro_batch_size="$TRAIN_MICRO_BATCH_SIZE"
    --beta="$BETA"
    --epsilon="$EPSILON"
    --reward_mode="$REWARD_MODE"
    --tfds_data_dir="$TFDS_DATA_DIR"
    --tfds_split="$TFDS_SPLIT"
    --seed="$SEED"
    --weight_sync_mode="$WEIGHT_SYNC_MODE"
    --stop_workers_on_exit
  )
  if [[ "$SHUFFLE" == "0" || "$SHUFFLE" == "false" || "$SHUFFLE" == "False" ]]; then
    ORCHESTRATOR_CMD+=(--no-shuffle)
  else
    ORCHESTRATOR_CMD+=(--shuffle)
  fi
  if [[ -n "$INFERENCE_ADDR" ]]; then
    ORCHESTRATOR_CMD+=(--inference_addr="$INFERENCE_ADDR")
  fi
  if [[ -n "$SCHEDULER_URL" ]]; then
  ORCHESTRATOR_CMD+=(--scheduler_url="$SCHEDULER_URL")
  fi

  export JAX_PLATFORMS=cpu
  export PYTHONUNBUFFERED=1
  export WANDB_PROJECT="$WANDB_PROJECT"
  export WANDB_RUN_NAME="$WANDB_RUN_NAME"
  export WANDB_API_KEY="$WANDB_API_KEY"
  env | egrep 'JAX|TPU'
  print_command "Orchestrator command" "${ORCHESTRATOR_CMD[@]}"
  "${ORCHESTRATOR_CMD[@]}" > "$ORCHESTRATOR_LOG" 2>&1
) || {
  exit_code="$?"
  echo "Error: CPU orchestrator failed with exit code $exit_code."
  print_file_debug "orchestrator" "$ORCHESTRATOR_LOG"
  exit "$exit_code"
}

echo "Distributed GSM8K GRPO chain demo (vLLM) finished successfully."
