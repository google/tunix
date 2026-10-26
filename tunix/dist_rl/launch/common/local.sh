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

# Shared single-host TPU launcher for the distributed RL launchers.
#
# Invoked by `launch.sh` (never directly). Consumes the cluster/deployment
# environment variables exported by `resolve_config.py` and forwards YAML +
# `key=value` arguments directly to `run_trainer_node`, `run_rollout_node`,
# `run_inference_node`, and the distributed orchestrator via
# `tunix.dist_rl.distributed.runtime.main`.

set -Ee

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
require_cluster_target local

CONFIG_ARGS=("$@")

ORCHESTRATOR_ID="${ORCHESTRATOR_ID:-orchestrator}"
LOG_ROOT="${LOG_ROOT:-${ARTIFACT_ROOT}}"
TRAINER_LOG="${LOG_ROOT}/trainer.log"
ROLLOUT_LOG="${LOG_ROOT}/rollout.log"
INFERENCE_LOG="${LOG_ROOT}/inference.log"
ORCHESTRATOR_LOG="${LOG_ROOT}/orchestrator.log"

SHUTDOWN_SIGNAL_COUNT=0
DRAIN_START_SECONDS=0
FORCE_KILL=0

if is_true "${LOCAL_AUTO_DETECT_PCI_TPUS:-false}"; then
  GET_SORTED_TPUS_SCRIPT="${REPO_ROOT}/scripts/get_sorted_tpu_ids.py"
  if [[ -f "$GET_SORTED_TPUS_SCRIPT" && -d /dev/vfio ]]; then
    DETECTED_TRAINER_CHIPS=$("$PYTHON_BIN" "$GET_SORTED_TPUS_SCRIPT" --count 4 2>/dev/null || true)
    DETECTED_ROLLOUT_CHIPS=$("$PYTHON_BIN" "$GET_SORTED_TPUS_SCRIPT" --count 4 --begin 4 2>/dev/null || true)
    if [[ -n "$DETECTED_TRAINER_CHIPS" && -n "$DETECTED_ROLLOUT_CHIPS" ]]; then
      LOCAL_TRAINER_TPU_CHIPS="$DETECTED_TRAINER_CHIPS"
      LOCAL_ROLLOUT_TPU_CHIPS="$DETECTED_ROLLOUT_CHIPS"
      echo "Discovered PCI-sorted TPU chips: Trainer=[$LOCAL_TRAINER_TPU_CHIPS], Rollout=[$LOCAL_ROLLOUT_TPU_CHIPS]"
    fi
  fi
fi

if is_true "${RUN_INFERENCE_NODE}" && [[ -z "${LOCAL_INFERENCE_TPU_CHIPS}" ]]; then
  echo "Error: cluster_config.run_inference_node=true requires cluster_config.local.inference_tpu_chips to avoid TPU contention." >&2
  exit 1
fi

NODE_CONFIG_ARGS=(
  "cluster_config.run_id=${RUN_ID}"
  "${CONFIG_ARGS[@]}"
)

TRAINER_CMD=(
  "$PYTHON_BIN" -m tunix.dist_rl.distributed.runtime.main
  --discovery_addrs="${ORCHESTRATOR_ID}:${ORCHESTRATOR_PORT}"
  --process_main="${TRAINER_MAIN}"
  "${NODE_CONFIG_ARGS[@]}"
)

ROLLOUT_CMD=(
  "$PYTHON_BIN" -m tunix.dist_rl.distributed.runtime.main
  --discovery_addrs="${ORCHESTRATOR_ID}:${ORCHESTRATOR_PORT}"
  --process_main="${ROLLOUT_MAIN}"
  "${NODE_CONFIG_ARGS[@]}"
)

INFERENCE_CMD=(
  "$PYTHON_BIN" -m tunix.dist_rl.distributed.runtime.main
  --discovery_addrs="${ORCHESTRATOR_ID}:${ORCHESTRATOR_PORT}"
  --process_main="${INFERENCE_MAIN}"
  "${NODE_CONFIG_ARGS[@]}"
)

EFFECTIVE_INFERENCE_ADDR="${INFERENCE_ADDR}"
if is_true "${RUN_INFERENCE_NODE}" && [[ -z "${EFFECTIVE_INFERENCE_ADDR}" ]]; then
  EFFECTIVE_INFERENCE_ADDR="localhost:${INFERENCE_PORT}"
fi

ORCHESTRATOR_CMD=(
  "$PYTHON_BIN" -m tunix.dist_rl.distributed.runtime.main
  --discovery_id="${ORCHESTRATOR_ID}"
  --discovery_port="${ORCHESTRATOR_PORT}"
  --process_main="${ORCHESTRATOR_MAIN}"
  "${NODE_CONFIG_ARGS[@]}"
)
if [[ -n "${EFFECTIVE_INFERENCE_ADDR}" ]]; then
  ORCHESTRATOR_CMD+=("cluster_config.inference_addr=${EFFECTIVE_INFERENCE_ADDR}")
fi

print_command() {
  local label="$1"
  shift
  echo "$label:"
  printf '  %q' "$@"
  echo
}

if is_true "${DRY_RUN}"; then
  print_command "Trainer command" "${TRAINER_CMD[@]}"
  print_command "Rollout command" "${ROLLOUT_CMD[@]}"
  if is_true "${RUN_INFERENCE_NODE}"; then
    print_command "Inference command" "${INFERENCE_CMD[@]}"
  fi
  print_command "Orchestrator command" "${ORCHESTRATOR_CMD[@]}"
  exit 0
fi

print_section() {
  echo
  echo "================ $1 ================"
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
  pgrep -af "tunix.dist_rl.distributed.runtime.main|vllm" 2>/dev/null || true
}

dump_debug_snapshot() {
  print_section "process snapshot"
  if [[ -n "${TRAINER_PID:-}" ]]; then
    print_process_debug "trainer" "$TRAINER_PID"
  fi
  if [[ -n "${ROLLOUT_PID:-}" ]]; then
    print_process_debug "rollout" "$ROLLOUT_PID"
  fi
  if [[ -n "${INFERENCE_PID:-}" ]]; then
    print_process_debug "inference" "$INFERENCE_PID"
  fi
  print_related_processes
  print_section "port snapshot"
  print_port_debug "$TRAINER_PORT"
  print_port_debug "$ROLLOUT_PORT"
  print_port_debug "$INFERENCE_PORT"
  print_section "log snapshot"
  print_file_debug "trainer" "$TRAINER_LOG"
  print_file_debug "rollout" "$ROLLOUT_LOG"
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

has_direct_safetensors() {
  local dir="$1"
  [[ -d "$dir" ]] && [[ -n "$(
    find "$dir" -maxdepth 1 -type f -name '*.safetensors' -print -quit 2>/dev/null || true
  )" ]]
}

ensure_model_dir_for() {
  local model_id="$1"
  local model_dir="$2"
  if [[ -e "$model_dir" && ! -d "$model_dir" ]]; then
    echo "Error: MODEL_DIR exists but is not a directory: $model_dir" >&2
    exit 1
  fi
  if has_direct_safetensors "$model_dir"; then
    return
  fi
  echo "Downloading ${model_id} to ${model_dir}"
  "$PYTHON_BIN" - "$model_id" "$model_dir" <<'PY'
import os
import sys

from tunix.oss import utils as oss_utils

model_id, model_dir = sys.argv[1], sys.argv[2]
os.makedirs(model_dir, exist_ok=True)
oss_utils.hf_pipeline(model_id, model_dir)
PY
  if ! has_direct_safetensors "$model_dir"; then
    echo "Error: download finished but MODEL_DIR has no direct '*.safetensors': ${model_dir}" >&2
    exit 1
  fi
}

convert_maxtext_ckpt() {
  if [[ "${ACTOR_TRAINER_BACKEND}" != "maxtext" ]]; then
    return
  fi
  if [[ "${ACTOR_MAXTEXT_CKPT_PATH}" =~ ^gs:// ]]; then
    echo "Using GCS MAXTEXT_CKPT: ${ACTOR_MAXTEXT_CKPT_PATH}"
    return
  fi
  if [[ -d "${ACTOR_MAXTEXT_CKPT_PATH}" ]]; then
    echo "Found existing local MAXTEXT_CKPT: ${ACTOR_MAXTEXT_CKPT_PATH}"
    return
  fi
  local maxtext_model_name
  maxtext_model_name="${MAXTEXT_MODEL_NAME:-$(printf '%s' "${ACTOR_MODEL_NAME}" | tr '[:upper:]' '[:lower:]')}"
  local ckpt_base
  ckpt_base="$(dirname "$(dirname "${ACTOR_MAXTEXT_CKPT_PATH}")")"
  echo "Converting HF checkpoint ${ACTOR_MODEL_DIR} to MaxText Orbax checkpoint at ${ckpt_base}..."
  mkdir -p "$ckpt_base"
  JAX_PLATFORMS=cpu "$PYTHON_BIN" \
    -m maxtext.checkpoint_conversion.to_maxtext \
    model_name="${maxtext_model_name}" \
    --hf_model_path="${ACTOR_MODEL_DIR}" \
    base_output_directory="${ckpt_base}" \
    scan_layers=True \
    skip_jax_distributed_system=True \
    checkpoint_storage_use_zarr3=false \
    checkpoint_storage_use_ocdbt=false
  if [[ ! -d "${ACTOR_MAXTEXT_CKPT_PATH}" ]]; then
    echo "Error: MaxText checkpoint conversion did not produce expected directory: ${ACTOR_MAXTEXT_CKPT_PATH}" >&2
    exit 1
  fi
}

check_process_alive() {
  local name="$1"
  local pid="$2"
  local log_file="$3"
  local process_state
  process_state="$(ps -o stat= -p "$pid" 2>/dev/null || true)"
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

on_shutdown_signal() {
  SHUTDOWN_SIGNAL_COUNT=$((SHUTDOWN_SIGNAL_COUNT + 1))
  if (( SHUTDOWN_SIGNAL_COUNT == 1 )); then
    DRAIN_START_SECONDS=$SECONDS
    print_section "graceful shutdown"
    echo "Received $1. Draining workers so the trainer can finalize its"
    echo "checkpoint. Press Ctrl+C again (after ${LOCAL_FORCE_ARM_SECS}s) to ABANDON"
    echo "the in-flight checkpoint and kill workers immediately."
    exit 130
  elif (( SECONDS - DRAIN_START_SECONDS < LOCAL_FORCE_ARM_SECS )); then
    echo "Ignoring rapid $1 (within the ${LOCAL_FORCE_ARM_SECS}s grace period)."
    echo "Press Ctrl+C again if you really want to abandon the checkpoint."
  else
    echo "Received $1. Killing workers immediately."
    FORCE_KILL=1
  fi
}

cleanup() {
  trap - EXIT ERR
  # Entering cleanup starts the drain. Mark it so a Ctrl+C during the drain
  # takes the grace-period / force-kill path in `on_shutdown_signal` instead of
  # `exit 130`, which would abort cleanup and orphan the workers.
  if (( SHUTDOWN_SIGNAL_COUNT == 0 )); then
    SHUTDOWN_SIGNAL_COUNT=1
    DRAIN_START_SECONDS=$SECONDS
  fi
  local trainer_pids=()
  local worker_pids=()
  local all_pids=()

  if [[ -n "${TRAINER_PID:-}" ]]; then
    trainer_pids+=("$TRAINER_PID")
    all_pids+=("$TRAINER_PID")
  fi
  for pid in "${ROLLOUT_PID:-}" "${INFERENCE_PID:-}"; do
    if [[ -n "$pid" ]]; then
      worker_pids+=("$pid")
      all_pids+=("$pid")
    fi
  done

  if (( ${#all_pids[@]} == 0 )); then
    return
  fi

  echo "Cleaning up worker processes: ${all_pids[*]}"
  echo "Waiting worker processes shutdown (grace period: ${LOCAL_TRAINER_SHUTDOWN_GRACE_SECS} seconds)"
  kill "${all_pids[@]}" 2>/dev/null || true
  local waited=0
  local poll_interval=2
  while true; do
    if (( FORCE_KILL == 1 )); then
      echo "Aborting after ${waited}s at user request."
      break
    fi
    local alive=0
    for pid in "${worker_pids[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        if (( waited >= LOCAL_WORKER_SHUTDOWN_GRACE_SECS )); then
          kill -9 "$pid" 2>/dev/null || true
        else
          alive=1
        fi
      fi
    done
    for pid in "${trainer_pids[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        if (( waited >= LOCAL_TRAINER_SHUTDOWN_GRACE_SECS )); then
          kill -9 "$pid" 2>/dev/null || true
        else
          alive=1
        fi
      fi
    done
    if (( alive == 0 )); then
      break
    fi
    sleep "$poll_interval" || true
    waited=$((waited + poll_interval))
  done
  for pid in "${all_pids[@]}"; do
    kill -9 "$pid" 2>/dev/null || true
  done
  wait "${all_pids[@]}" 2>/dev/null || true
  echo "Workers stopped after ${waited}s of graceful drain."
}

trap on_error ERR
trap cleanup EXIT
trap 'on_shutdown_signal SIGINT' INT
trap 'on_shutdown_signal SIGTERM' TERM

ensure_model_dir_for "${ACTOR_MODEL_ID}" "${ACTOR_MODEL_DIR}"
if [[ "${ROLLOUT_MODEL_DIR}" != "${ACTOR_MODEL_DIR}" ]]; then
  ensure_model_dir_for "${ROLLOUT_MODEL_ID}" "${ROLLOUT_MODEL_DIR}"
fi
convert_maxtext_ckpt

mkdir -p "$LOG_ROOT" "$ARTIFACT_ROOT"
: > "$TRAINER_LOG"
: > "$ROLLOUT_LOG"
: > "$INFERENCE_LOG"
: > "$ORCHESTRATOR_LOG"

echo "Launching trainer node on TPU chips ${LOCAL_TRAINER_TPU_CHIPS}..."
(
  if is_true "${LOCAL_TRAINER_PATHWAYS}"; then
    export JAX_PLATFORMS=proxy,cpu
    export JAX_BACKEND_TARGET="${LOCAL_JAX_BACKEND_TARGET}"
    export TRAINER_PATHWAYS_LOCAL_INIT=1
    unset TPU_VISIBLE_DEVICES TPU_VISIBLE_CHIPS LIBTPU_INIT_ARGS
  else
    export JAX_PLATFORMS=tpu,cpu
    export TPU_VISIBLE_DEVICES="${LOCAL_TRAINER_TPU_CHIPS}"
    export TPU_VISIBLE_CHIPS="${TPU_VISIBLE_DEVICES}"
    export TPU_CHIPS_PER_HOST_BOUNDS="${LOCAL_TPU_CHIPS_PER_HOST_BOUNDS}"
    export TPU_HOST_BOUNDS="${LOCAL_TPU_HOST_BOUNDS}"
    export LIBTPU_INIT_ARGS="--deepsea_chips_per_host_bounds=${LOCAL_TPU_CHIPS_PER_HOST_BOUNDS} --deepsea_host_bounds=${LOCAL_TPU_HOST_BOUNDS}"
  fi
  if is_true "${LOCAL_ALLOW_MULTIPLE_LIBTPU_LOAD}"; then
    export ALLOW_MULTIPLE_LIBTPU_LOAD=1
  fi
  export VERIFY_WEIGHTS="${WEIGHT_SYNC_VERIFY_WEIGHTS}"
  export CKPT_D2H_CONCURRENT_GB="${CKPT_D2H_CONCURRENT_GB}"
  export PYTHONUNBUFFERED=1
  print_command "Trainer command" "${TRAINER_CMD[@]}"
  exec "${TRAINER_CMD[@]}" > "$TRAINER_LOG" 2>&1
) &
TRAINER_PID=$!
echo "Trainer pid=$TRAINER_PID log=$TRAINER_LOG"

echo "Launching rollout node on TPU chips ${LOCAL_ROLLOUT_TPU_CHIPS}..."
(
  export JAX_PLATFORMS=tpu,cpu
  export SKIP_JAX_PRECOMPILE=1
  export TPU_VISIBLE_DEVICES="${LOCAL_ROLLOUT_TPU_CHIPS}"
  export TPU_VISIBLE_CHIPS="${TPU_VISIBLE_DEVICES}"
  export TPU_CHIPS_PER_HOST_BOUNDS="${LOCAL_TPU_CHIPS_PER_HOST_BOUNDS}"
  export TPU_HOST_BOUNDS="${LOCAL_TPU_HOST_BOUNDS}"
  export LIBTPU_INIT_ARGS="--deepsea_chips_per_host_bounds=${LOCAL_TPU_CHIPS_PER_HOST_BOUNDS} --deepsea_host_bounds=${LOCAL_TPU_HOST_BOUNDS}"
  if is_true "${LOCAL_ALLOW_MULTIPLE_LIBTPU_LOAD}"; then
    export ALLOW_MULTIPLE_LIBTPU_LOAD=1
  fi
  if [[ -n "${VLLM_ADDITIONAL_CONFIG_JSON}" ]]; then
    export VLLM_ADDITIONAL_CONFIG_JSON
  fi
  if [[ -n "${SANDBOX_NAMESPACE}" ]]; then
    export NAMESPACE="${SANDBOX_NAMESPACE}"
    if [[ -n "${SANDBOX_NODE_SELECTOR_KEY}" && -n "${SANDBOX_NODE_SELECTOR_VAL}" ]]; then
      export NODE_SELECTOR_KEY="${SANDBOX_NODE_SELECTOR_KEY}"
      export NODE_SELECTOR_VAL="${SANDBOX_NODE_SELECTOR_VAL}"
    fi
  fi
  export VERIFY_WEIGHTS="${WEIGHT_SYNC_VERIFY_WEIGHTS}"
  export PYTHONUNBUFFERED=1
  print_command "Rollout command" "${ROLLOUT_CMD[@]}"
  exec "${ROLLOUT_CMD[@]}" > "$ROLLOUT_LOG" 2>&1
) &
ROLLOUT_PID=$!
echo "Rollout pid=$ROLLOUT_PID log=$ROLLOUT_LOG"

if is_true "${RUN_INFERENCE_NODE}"; then
  echo "Launching reference inference node on TPU chips ${LOCAL_INFERENCE_TPU_CHIPS}..."
  (
    export JAX_PLATFORMS=tpu,cpu
    export TPU_VISIBLE_DEVICES="${LOCAL_INFERENCE_TPU_CHIPS}"
    export TPU_VISIBLE_CHIPS="${TPU_VISIBLE_DEVICES}"
    export TPU_CHIPS_PER_HOST_BOUNDS="${LOCAL_TPU_CHIPS_PER_HOST_BOUNDS}"
    export TPU_HOST_BOUNDS="${LOCAL_TPU_HOST_BOUNDS}"
    export LIBTPU_INIT_ARGS="--deepsea_chips_per_host_bounds=${LOCAL_TPU_CHIPS_PER_HOST_BOUNDS} --deepsea_host_bounds=${LOCAL_TPU_HOST_BOUNDS}"
    if is_true "${LOCAL_ALLOW_MULTIPLE_LIBTPU_LOAD}"; then
      export ALLOW_MULTIPLE_LIBTPU_LOAD=1
    fi
    export PYTHONUNBUFFERED=1
    print_command "Inference command" "${INFERENCE_CMD[@]}"
    exec "${INFERENCE_CMD[@]}" > "$INFERENCE_LOG" 2>&1
  ) &
  INFERENCE_PID=$!
  echo "Inference pid=$INFERENCE_PID log=$INFERENCE_LOG"
fi

echo "Waiting for gRPC servers to bind..."
wait_for_port "trainer" "$TRAINER_PORT" "$TRAINER_PID" "$TRAINER_LOG"
wait_for_port "rollout" "$ROLLOUT_PORT" "$ROLLOUT_PID" "$ROLLOUT_LOG"
if is_true "${RUN_INFERENCE_NODE}"; then
  wait_for_port "inference" "$INFERENCE_PORT" "$INFERENCE_PID" "$INFERENCE_LOG"
fi

echo "Launching CPU orchestrator..."
(
  export JAX_PLATFORMS=cpu
  export PYTHONUNBUFFERED=1
  print_command "Orchestrator command" "${ORCHESTRATOR_CMD[@]}"
  "${ORCHESTRATOR_CMD[@]}" > "$ORCHESTRATOR_LOG" 2>&1
)

echo "Distributed RL pipeline finished successfully."
echo "Trainer log:      $TRAINER_LOG"
echo "Rollout log:      $ROLLOUT_LOG"
if is_true "${RUN_INFERENCE_NODE}"; then
  echo "Inference log:    $INFERENCE_LOG"
fi
echo "Orchestrator log: $ORCHESTRATOR_LOG"
