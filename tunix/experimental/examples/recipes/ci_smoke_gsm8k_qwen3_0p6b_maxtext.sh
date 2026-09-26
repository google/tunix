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

# ==============================================================================
# CI Smoke Test Recipe: Qwen3-0.6B on GSM8K with MaxText + Raiden Weight Sync
# ==============================================================================
# Single-host TPU (v6e-8) end-to-end smoke test for distributed RL with MaxText:
# - Trainer: MaxTextTrainingEngine (backend="maxtext") on chips 0,1,2,3 (TP=4)
# - Rollout: vllm (MaxTextForCausalLM) on chips 4,5,6,7 (TP=4)
# - Weight Sync: Raiden native TPU sync with weight converter & checksum verification
# - Reference model: skipped (BETA=0)
# - Steps: MAX_STEPS=2, BATCH_SIZE=2, NUM_GENERATIONS=2, TRAIN_MICRO_BATCH_SIZE=1
# - Checkpointing: CHECKPOINT_SAVE_INTERVAL_STEPS=1 (saves step 1 checkpoint)
# ==============================================================================

set -Ee

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Model Configuration
export MODEL_NAME="${MODEL_NAME:-Qwen3-0.6B}"
export MODEL_ID="${MODEL_ID:-Qwen/Qwen3-0.6B}"
export MAXTEXT_MODEL_NAME="${MAXTEXT_MODEL_NAME:-qwen3-0.6b}"

# Scope MODEL_DIR and TOKENIZER_PATH by MODEL_NAME to reuse already downloaded
# HF model and TFDS dataset from the Tunix smoke test run.
export ARTIFACT_ROOT="${ARTIFACT_ROOT:-${DIR}/../../../../artifacts/qwen3_dist_gsm8k}"
export MODEL_DIR="${MODEL_DIR:-${ARTIFACT_ROOT}/models/${MODEL_NAME}}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-$MODEL_DIR}"
export MAXTEXT_CKPT="${MAXTEXT_CKPT:-${ARTIFACT_ROOT}/maxtext_models/${MAXTEXT_MODEL_NAME}/0/items}"
export CHECKPOINT_ROOT_DIRECTORY="${CHECKPOINT_ROOT_DIRECTORY:-${ARTIFACT_ROOT}/checkpoints_maxtext}"

# Stack Configuration: MaxText trainer + vLLM rollout + Raiden weight converter
export TRAINER_BACKEND="${TRAINER_BACKEND:-maxtext}"
export SAMPLER="${SAMPLER:-vllm}"
export WEIGHT_SYNC_MODE="${WEIGHT_SYNC_MODE:-raiden}"
export USE_WEIGHT_CONVERTER="${USE_WEIGHT_CONVERTER:-true}"
export VERIFY_WEIGHTS="${VERIFY_WEIGHTS:-false}"
export CHAT_PARSER="${CHAT_PARSER:-auto}"
# MaxTextTrainingEngine does not currently implement standalone per_token_logps.
# Disabling USE_ROLLOUT_LOGPS skips the diagnostic sampler/trainer agreement pass.
export USE_ROLLOUT_LOGPS="${USE_ROLLOUT_LOGPS:-false}"

# Chip allocation on a single 8-chip host (e.g. TPU v6e-8)
# Auto-detect TPU chips sorted by PCI topology order if running on a host with /dev/vfio.
if [[ -z "${TRAINER_TPU_CHIPS:-}" || -z "${ROLLOUT_TPU_CHIPS:-}" ]]; then
  GET_SORTED_TPUS_SCRIPT="${DIR}/../../../../scripts/get_sorted_tpu_ids.py"
  if [[ -f "$GET_SORTED_TPUS_SCRIPT" && -d /dev/vfio ]]; then
    DETECTED_TRAINER_CHIPS=$(python3 "$GET_SORTED_TPUS_SCRIPT" --count 4 2>/dev/null || true)
    DETECTED_ROLLOUT_CHIPS=$(python3 "$GET_SORTED_TPUS_SCRIPT" --count 4 --begin 4 2>/dev/null || true)
    if [[ -n "$DETECTED_TRAINER_CHIPS" && -n "$DETECTED_ROLLOUT_CHIPS" ]]; then
      export TRAINER_TPU_CHIPS="${TRAINER_TPU_CHIPS:-$DETECTED_TRAINER_CHIPS}"
      export ROLLOUT_TPU_CHIPS="${ROLLOUT_TPU_CHIPS:-$DETECTED_ROLLOUT_CHIPS}"
      echo "Discovered PCI-sorted TPU chips: Trainer=[$TRAINER_TPU_CHIPS], Rollout=[$ROLLOUT_TPU_CHIPS]"
    fi
  fi
fi

export TRAINER_TPU_CHIPS="${TRAINER_TPU_CHIPS:-0,1,2,3}"
export TRAINER_FSDP="${TRAINER_FSDP:-1}"
export TRAINER_TP="${TRAINER_TP:-4}"

export ROLLOUT_TPU_CHIPS="${ROLLOUT_TPU_CHIPS:-4,5,6,7}"
export ROLLOUT_FSDP="${ROLLOUT_FSDP:-1}"
export ROLLOUT_TP="${ROLLOUT_TP:-4}"

# TPU v6e-8 physical host bounds are 2,4,1. When partitioned into 4-chip slices
# (chips 0..3 and 4..7), each slice forms a 2x2 grid with bounds 2,2,1.
export TPU_CHIPS_PER_HOST_BOUNDS="${TPU_CHIPS_PER_HOST_BOUNDS:-2,2,1}"
export TPU_HOST_BOUNDS="${TPU_HOST_BOUNDS:-1,1,1}"
export ALLOW_MULTIPLE_LIBTPU_LOAD="${ALLOW_MULTIPLE_LIBTPU_LOAD:-1}"

# Hyperparameters: Full GSM8K GRPO run aligned with Qwen3-0.6B recipe
export MAX_STEPS="${MAX_STEPS:-200}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
export MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-16}"
export NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
export TRAIN_MICRO_BATCH_SIZE="${TRAIN_MICRO_BATCH_SIZE:-1}"
export LEARNING_RATE="${LEARNING_RATE:-3e-6}"
export WARMUP_STEPS="${WARMUP_STEPS:-10}"
export LR_DECAY_STEPS="${LR_DECAY_STEPS:-500}"
export BETA="${BETA:-0.0}"
export EPSILON="${EPSILON:-0.2}"

# Dataset & sequence lengths (full GSM8K train split, shuffled)
export TFDS_SPLIT="${TFDS_SPLIT:-train}"
export SHUFFLE="${SHUFFLE:-true}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"

# Checkpointing & Evaluation
export CHECKPOINT_SAVE_INTERVAL_STEPS="${CHECKPOINT_SAVE_INTERVAL_STEPS:-50}"
export CHECKPOINT_MAX_TO_KEEP="${CHECKPOINT_MAX_TO_KEEP:-4}"

# WandB logging and evaluation interval
export WANDB_PROJECT="${WANDB_PROJECT:-tunix-gsm8k-vtc}"
export EVAL_EVERY_N_STEPS="${EVAL_EVERY_N_STEPS:-50}"

# Timeouts
export WAIT_TIMEOUT_SECS="${WAIT_TIMEOUT_SECS:-1800}"

# Delegate to base launcher located in math_gsm8k_dist
exec "${DIR}/../math_gsm8k_dist/launcher.sh" "$@"
