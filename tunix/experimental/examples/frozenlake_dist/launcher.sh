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

set -Eeuo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${DIR}/../../../.." && pwd)"
LOG_ROOT=${LOG_ROOT:-"${DIR}"}
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON_BIN=${PYTHON_BIN:-python3}
ORCHESTRATOR_ID=${ORCHESTRATOR_ID:-orchestrator}
ORCHESTRATOR_PORT=${ORCHESTRATOR_PORT:-30000}
TRAINER_PORT=${TRAINER_PORT:-20000}
ROLLOUT_PORT=${ROLLOUT_PORT:-20001}

MODEL_NAME=${MODEL_NAME:-Qwen3-8B}
MODEL_ID=${MODEL_ID:-Qwen/Qwen3-8B}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-"${REPO_ROOT}/artifacts/qwen3_dist_frozenlake"}
MODEL_DIR=${MODEL_DIR:-"${ARTIFACT_ROOT}/models/${MODEL_NAME}"}
TOKENIZER_PATH=${TOKENIZER_PATH:-"${MODEL_DIR}"}

BATCH_SIZE=${BATCH_SIZE:-64}
MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-64}
NUM_GENERATIONS=${NUM_GENERATIONS:-8}
MAX_STEPS=${MAX_STEPS:-450}
MAX_TURNS=${MAX_TURNS:-8}
DATASET_SIZE=${DATASET_SIZE:-10000}
NUM_BATCHES=${NUM_BATCHES:-150}
NUM_EPOCHS=${NUM_EPOCHS:-3}
EVAL_DATASET_SIZE=${EVAL_DATASET_SIZE:-100}
EVAL_EVERY_N_STEPS=${EVAL_EVERY_N_STEPS:-10}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
TRAIN_MICRO_BATCH_SIZE=${TRAIN_MICRO_BATCH_SIZE:-4}
LEARNING_RATE=${LEARNING_RATE:-1e-6}
ADAM_B1=${ADAM_B1:-0.9}
ADAM_B2=${ADAM_B2:-0.95}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-100.0}
PARAM_DTYPE=${PARAM_DTYPE:-float32}
REMAT_POLICY=${REMAT_POLICY:-decoder}
FLASH_ATTENTION_BLOCK_SIZE=${FLASH_ATTENTION_BLOCK_SIZE:-256}
TEMPERATURE=${TEMPERATURE:-0.7}
TOP_P=${TOP_P:-1.0}
TOP_K=${TOP_K:-0}
BETA=${BETA:-0.0}
EPSILON=${EPSILON:-0.003}
EPSILON_HIGH=${EPSILON_HIGH:-0.005}
LOSS_ALGO=${LOSS_ALGO:-gspo-token}
LOSS_AGG_MODE=${LOSS_AGG_MODE:-sequence-mean-token-mean}
KL_LOSS_MODE=${KL_LOSS_MODE:-low_var_kl}
ADVANTAGE_ESTIMATOR=${ADVANTAGE_ESTIMATOR:-rloo}
OFF_POLICY_STEPS=${OFF_POLICY_STEPS:-0}
EPISODE_TIMEOUT_SECS=${EPISODE_TIMEOUT_SECS:-600}
SEED=${SEED:-42}
SHUFFLE=${SHUFFLE:-1}
IS_SLIPPERY=${IS_SLIPPERY:-0}
USE_MULTISTEP_PROMPT=${USE_MULTISTEP_PROMPT:-1}
USE_ROLLOUT_LOGPS=${USE_ROLLOUT_LOGPS:-1}
SAMPLER_IS=${SAMPLER_IS:-token}
SAMPLER_IS_THRESHOLD=${SAMPLER_IS_THRESHOLD:-2.0}
ROLLOUT_MAX_CONCURRENCY=${ROLLOUT_MAX_CONCURRENCY:-256}
SAMPLER=${SAMPLER:-inprocess_vllm}
WEIGHT_SYNC_MODE=${WEIGHT_SYNC_MODE:-raiden}
VLLM_HBM_UTILIZATION=${VLLM_HBM_UTILIZATION:-0.20}
VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-64}
VLLM_MAX_NUM_BATCHED_TOKENS=${VLLM_MAX_NUM_BATCHED_TOKENS:-32768}
VLLM_MODEL_LEN_MARGIN=${VLLM_MODEL_LEN_MARGIN:-256}
CHECKPOINT_SAVE_INTERVAL_STEPS=${CHECKPOINT_SAVE_INTERVAL_STEPS:-1000000000}
CHECKPOINT_MAX_TO_KEEP=${CHECKPOINT_MAX_TO_KEEP:-1}
CHECKPOINT_ROOT_DIRECTORY=${CHECKPOINT_ROOT_DIRECTORY:-"${REPO_ROOT}/checkpoints/frozenlake"}
WANDB_PROJECT=${WANDB_PROJECT:-trellis-frozenlake}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-}
WANDB_API_KEY=${WANDB_API_KEY:-}
DEBUG=${DEBUG:-0}

# Qwen3-8B defaults target an 8-chip host split between trainer and rollout.
TRAINER_TPU_CHIPS=${TRAINER_TPU_CHIPS:-0,1,2,3}
TRAINER_FSDP=${TRAINER_FSDP:-1}
TRAINER_TP=${TRAINER_TP:-4}
ROLLOUT_TPU_CHIPS=${ROLLOUT_TPU_CHIPS:-4,5,6,7}
ROLLOUT_FSDP=${ROLLOUT_FSDP:-1}
ROLLOUT_TP=${ROLLOUT_TP:-4}
TPU_CHIPS_PER_HOST_BOUNDS=${TPU_CHIPS_PER_HOST_BOUNDS:-1,4,1}
TPU_HOST_BOUNDS=${TPU_HOST_BOUNDS:-1,1,1}
WAIT_TIMEOUT_SECS=${WAIT_TIMEOUT_SECS:-1800}
WAIT_POLL_SECS=${WAIT_POLL_SECS:-5}
SHUTDOWN_GRACE_SECS=${SHUTDOWN_GRACE_SECS:-30}

TRAINER_LOG="${LOG_ROOT}/trainer.log"
ROLLOUT_LOG="${LOG_ROOT}/rollout.log"
ORCHESTRATOR_LOG="${LOG_ROOT}/orchestrator.log"

is_true() {
  [[ "$1" == "1" || "$1" == "true" || "$1" == "True" ]]
}

has_direct_safetensors() {
  [[ -d "$MODEL_DIR" ]] && [[ -n "$(
    find "$MODEL_DIR" -maxdepth 1 -type f -name '*.safetensors' -print -quit 2>/dev/null || true
  )" ]]
}

ensure_model_dir() {
  if has_direct_safetensors; then
    return
  fi
  echo "Downloading ${MODEL_ID} to ${MODEL_DIR}"
  "$PYTHON_BIN" - "$MODEL_ID" "$MODEL_DIR" <<'PY'
import os
import sys

from tunix.oss import utils as oss_utils

model_id, model_dir = sys.argv[1], sys.argv[2]
os.makedirs(model_dir, exist_ok=True)
oss_utils.hf_pipeline(model_id, model_dir)
PY
  if ! has_direct_safetensors; then
    echo "Error: MODEL_DIR has no direct '*.safetensors': ${MODEL_DIR}"
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
      echo "Error: ${name} exited before port ${port} became ready."
      tail -n 80 "$log_file" 2>/dev/null || true
      exit 1
    fi
    if "$PYTHON_BIN" - "$port" <<'PY'
import socket
import sys

try:
  socket.create_connection(("localhost", int(sys.argv[1])), timeout=1).close()
except OSError:
  sys.exit(1)
PY
    then
      return
    fi
    if (( elapsed >= WAIT_TIMEOUT_SECS )); then
      echo "Error: timed out waiting for ${name} on port ${port}."
      tail -n 80 "$log_file" 2>/dev/null || true
      exit 1
    fi
    sleep "$WAIT_POLL_SECS"
    elapsed=$((elapsed + WAIT_POLL_SECS))
  done
}

cleanup() {
  trap - EXIT
  local pid
  for pid in "${TRAINER_PID:-}" "${ROLLOUT_PID:-}"; do
    if [[ -n "$pid" ]]; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  sleep "$SHUTDOWN_GRACE_SECS" || true
  for pid in "${TRAINER_PID:-}" "${ROLLOUT_PID:-}"; do
    if [[ -n "$pid" ]]; then
      kill -9 "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT

if [[ "$BETA" != "0" && "$BETA" != "0.0" ]]; then
  echo "Error: launcher.sh starts actor and rollout workers only; use BETA=0.0."
  exit 1
fi
if (( BATCH_SIZE <= 0 || MINI_BATCH_SIZE <= 0 || NUM_GENERATIONS <= 1 || TRAIN_MICRO_BATCH_SIZE <= 0 )); then
  echo "Error: invalid batch or generation configuration."
  exit 1
fi
if (( BATCH_SIZE % MINI_BATCH_SIZE != 0 )); then
  echo "Error: BATCH_SIZE must be divisible by MINI_BATCH_SIZE."
  exit 1
fi
if (( (MINI_BATCH_SIZE * NUM_GENERATIONS) % TRAIN_MICRO_BATCH_SIZE != 0 )); then
  echo "Error: MINI_BATCH_SIZE * NUM_GENERATIONS must be divisible by TRAIN_MICRO_BATCH_SIZE."
  exit 1
fi

ensure_model_dir
mkdir -p "$LOG_ROOT" "$ARTIFACT_ROOT"
: > "$TRAINER_LOG"
: > "$ROLLOUT_LOG"
: > "$ORCHESTRATOR_LOG"

echo "Starting distributed FrozenLake with ${MODEL_ID}: full batch ${BATCH_SIZE}, mini batch ${MINI_BATCH_SIZE}, generations ${NUM_GENERATIONS}."

(
  cmd=(
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
    --learning_rate="$LEARNING_RATE"
    --adam_b1="$ADAM_B1"
    --adam_b2="$ADAM_B2"
    --weight_decay="$WEIGHT_DECAY"
    --max_grad_norm="$MAX_GRAD_NORM"
    --param_dtype="$PARAM_DTYPE"
    --enable_remat
    --remat_policy="$REMAT_POLICY"
    --use_flash_attention
    --flash_attention_block_size="$FLASH_ATTENTION_BLOCK_SIZE"
    --sampler_type="$SAMPLER"
    --checkpoint_save_interval_steps="$CHECKPOINT_SAVE_INTERVAL_STEPS"
    --checkpoint_max_to_keep="$CHECKPOINT_MAX_TO_KEEP"
    --checkpoint_root_directory="$CHECKPOINT_ROOT_DIRECTORY"
  )
  is_true "$DEBUG" && cmd+=(--debug)
  export JAX_PLATFORMS=tpu,cpu
  export TPU_VISIBLE_DEVICES="$TRAINER_TPU_CHIPS"
  export TPU_VISIBLE_CHIPS="$TPU_VISIBLE_DEVICES"
  export TPU_CHIPS_PER_HOST_BOUNDS
  export TPU_HOST_BOUNDS
  export LIBTPU_INIT_ARGS="--deepsea_chips_per_host_bounds=${TPU_CHIPS_PER_HOST_BOUNDS} --deepsea_host_bounds=${TPU_HOST_BOUNDS}"
  export PYTHONUNBUFFERED=1
  exec "${cmd[@]}" > "$TRAINER_LOG" 2>&1
) &
TRAINER_PID=$!

(
  cmd=(
    "$PYTHON_BIN" -m tunix.experimental.distributed.runtime.main
    --discovery_addrs="${ORCHESTRATOR_ID}:${ORCHESTRATOR_PORT}"
    --process_main=tunix.experimental.examples.common.run_rollout_node.main
    --port="$ROLLOUT_PORT"
    --model_id="$MODEL_ID"
    --model_dir="$MODEL_DIR"
    --model_name="$MODEL_NAME"
    --tokenizer_path="$TOKENIZER_PATH"
    --sampler="$SAMPLER"
    --mesh_fsdp="$ROLLOUT_FSDP"
    --mesh_tp="$ROLLOUT_TP"
    --max_prompt_length="$MAX_PROMPT_LENGTH"
    --max_response_length="$MAX_RESPONSE_LENGTH"
    --weight_sync_mode="$WEIGHT_SYNC_MODE"
    --registry_module=tunix.experimental.examples.frozenlake_dist.frozenlake
    --env_name=frozenlake_env
    --agent_name=frozenlake_agent
    --max_concurrency="$ROLLOUT_MAX_CONCURRENCY"
    --vllm_hbm_utilization="$VLLM_HBM_UTILIZATION"
    --no-vllm_async_scheduling
    --no-vllm_enable_prefix_caching
    --vllm_max_num_seqs="$VLLM_MAX_NUM_SEQS"
    --vllm_max_num_batched_tokens="$VLLM_MAX_NUM_BATCHED_TOKENS"
    --vllm_model_len_margin="$VLLM_MODEL_LEN_MARGIN"
    --vllm_server_mode
    --no-enable_thinking
  )
  is_true "$DEBUG" && cmd+=(--debug)
  export JAX_PLATFORMS=tpu,cpu
  export SKIP_JAX_PRECOMPILE=1
  export TPU_VISIBLE_DEVICES="$ROLLOUT_TPU_CHIPS"
  export TPU_VISIBLE_CHIPS="$TPU_VISIBLE_DEVICES"
  export TPU_CHIPS_PER_HOST_BOUNDS
  export TPU_HOST_BOUNDS
  export LIBTPU_INIT_ARGS="--deepsea_chips_per_host_bounds=${TPU_CHIPS_PER_HOST_BOUNDS} --deepsea_host_bounds=${TPU_HOST_BOUNDS}"
  export PYTHONUNBUFFERED=1
  exec "${cmd[@]}" > "$ROLLOUT_LOG" 2>&1
) &
ROLLOUT_PID=$!

wait_for_port trainer "$TRAINER_PORT" "$TRAINER_PID" "$TRAINER_LOG"
wait_for_port rollout "$ROLLOUT_PORT" "$ROLLOUT_PID" "$ROLLOUT_LOG"

cmd=(
  "$PYTHON_BIN" -m tunix.experimental.distributed.runtime.main
  --discovery_id="$ORCHESTRATOR_ID"
  --discovery_port="$ORCHESTRATOR_PORT"
  --process_main=tunix.experimental.examples.frozenlake_dist.run_frozenlake_dist.main
  --model_id="$MODEL_ID"
  --tokenizer_path="$TOKENIZER_PATH"
  --batch_size="$BATCH_SIZE"
  --mini_batch_size="$MINI_BATCH_SIZE"
  --num_generations="$NUM_GENERATIONS"
  --max_steps="$MAX_STEPS"
  --max_turns="$MAX_TURNS"
  --dataset_size="$DATASET_SIZE"
  --num_batches="$NUM_BATCHES"
  --num_epochs="$NUM_EPOCHS"
  --eval_dataset_size="$EVAL_DATASET_SIZE"
  --eval_every_n_steps="$EVAL_EVERY_N_STEPS"
  --max_prompt_length="$MAX_PROMPT_LENGTH"
  --max_response_length="$MAX_RESPONSE_LENGTH"
  --train_micro_batch_size="$TRAIN_MICRO_BATCH_SIZE"
  --temperature="$TEMPERATURE"
  --top_p="$TOP_P"
  --top_k="$TOP_K"
  --beta="$BETA"
  --epsilon="$EPSILON"
  --epsilon_high="$EPSILON_HIGH"
  --loss_algo="$LOSS_ALGO"
  --loss_agg_mode="$LOSS_AGG_MODE"
  --kl_loss_mode="$KL_LOSS_MODE"
  --advantage_estimator="$ADVANTAGE_ESTIMATOR"
  --max_staleness="$OFF_POLICY_STEPS"
  --episode_timeout_secs="$EPISODE_TIMEOUT_SECS"
  --sampler_is="$SAMPLER_IS"
  --sampler_is_threshold="$SAMPLER_IS_THRESHOLD"
  --seed="$SEED"
  --weight_sync_mode="$WEIGHT_SYNC_MODE"
  --trainer_fsdp="$TRAINER_FSDP"
  --stop_workers_on_exit
)
is_true "$SHUFFLE" && cmd+=(--shuffle) || cmd+=(--no-shuffle)
is_true "$IS_SLIPPERY" && cmd+=(--is_slippery) || cmd+=(--no-is_slippery)
is_true "$USE_MULTISTEP_PROMPT" && cmd+=(--use_multistep_prompt) || cmd+=(--no-use_multistep_prompt)
is_true "$USE_ROLLOUT_LOGPS" && cmd+=(--use_rollout_logps) || cmd+=(--no-use_rollout_logps)
is_true "$DEBUG" && cmd+=(--debug)

export JAX_PLATFORMS=cpu
export PYTHONUNBUFFERED=1
export WANDB_PROJECT WANDB_RUN_NAME WANDB_API_KEY
"${cmd[@]}" > "$ORCHESTRATOR_LOG" 2>&1

echo "Distributed FrozenLake finished. Logs: ${TRAINER_LOG}, ${ROLLOUT_LOG}, ${ORCHESTRATOR_LOG}"
