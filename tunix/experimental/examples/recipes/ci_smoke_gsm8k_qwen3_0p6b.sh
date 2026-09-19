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
# CI Smoke Test Recipe: Qwen3-0.6B on GSM8K with Raiden Weight Sync
# ==============================================================================
# Single-host TPU (v6e-8) end-to-end smoke test for distributed RL with Raiden:
# - Trainer: Native Tunix PeftTrainer (backend="tunix") on chips 0,1,2,3 (TP=4)
# - Rollout: inprocess_vllm (vllm_jax) on chips 4,5,6,7 (TP=4)
# - Weight Sync: Raiden native TPU sync (WEIGHT_SYNC_MODE="raiden")
# - Reference model: skipped (BETA=0)
# - Steps: MAX_STEPS=2, BATCH_SIZE=2, NUM_GENERATIONS=2, TRAIN_MICRO_BATCH_SIZE=1
# - Checkpointing: CHECKPOINT_SAVE_INTERVAL_STEPS=1 (saves step 1 checkpoint)
# ==============================================================================

set -Ee

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Model Configuration
export MODEL_NAME="${MODEL_NAME:-Qwen3-0.6B}"
export MODEL_ID="${MODEL_ID:-Qwen/Qwen3-0.6B}"

# Scope MODEL_DIR and TOKENIZER_PATH by MODEL_NAME to avoid loading weights from
# mismatched models (e.g. Qwen3-1.7B) if previously downloaded to artifacts/
export ARTIFACT_ROOT="${ARTIFACT_ROOT:-${DIR}/../../../../artifacts/qwen3_dist_gsm8k}"
export MODEL_DIR="${MODEL_DIR:-${ARTIFACT_ROOT}/models/${MODEL_NAME}}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-$MODEL_DIR}"

# Stack Configuration: Tunix native stack + vLLM-JAX + Raiden
export TRAINER_BACKEND="${TRAINER_BACKEND:-tunix}"
export SAMPLER="${SAMPLER:-inprocess_vllm}"
export WEIGHT_SYNC_MODE="${WEIGHT_SYNC_MODE:-raiden}"

# Chip allocation on a single 8-chip host (e.g. TPU v6e-8)
# Auto-detect TPU chips sorted by PCI topology order if running on a host with /dev/vfio.
# On multi-tray hosts (e.g. TPU v5e-8 or v6e-8), /dev/vfio device minor IDs can be
# enumerated out of order across PCIe switches. Slicing into 2x2 meshes requires
# chips on the same physical tray (same PCIe root complex) to be grouped together.
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

# Chip allocation on a single 8-chip host (e.g. TPU v6e-8)
# For this lightweight 0.6B smoke test, Trainer defaults to TP=4 (FSDP=1) to
# match the rollout vLLM tensor-parallel layout exactly for direct 1:1 Raiden
# weight sync and lean batch sizes.
# TODO (tunix-dev): explore cross-mesh FSDP configurations.
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
# Allow multiple concurrent libtpu loads across trainer and rollout processes on the same host
export ALLOW_MULTIPLE_LIBTPU_LOAD="${ALLOW_MULTIPLE_LIBTPU_LOAD:-1}"

# Hyperparameters: 2-step run (validates rollout generation on step-1 synced weights)
export MAX_STEPS="${MAX_STEPS:-2}"
export BATCH_SIZE="${BATCH_SIZE:-2}"
export MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-2}"
export NUM_GENERATIONS="${NUM_GENERATIONS:-2}"
# Gradient accumulation steps = (MINI_BATCH_SIZE * NUM_GENERATIONS) /
# TRAIN_MICRO_BATCH_SIZE = (2 * 2) / 1 = 4. Keep this > 1: at exactly 1,
# PeftTrainer v2 takes the single-microstep fast path, which allocates a
# non-persistent gradient accumulator that the default cache_nnx_graph=True
# discards across the jit boundary, failing with "The gradient accumulator is
# empty".
export TRAIN_MICRO_BATCH_SIZE="${TRAIN_MICRO_BATCH_SIZE:-1}"
export BETA="${BETA:-0}"

# Dataset & sequence lengths
export TFDS_SPLIT="${TFDS_SPLIT:-train[:16]}"
export SHUFFLE="${SHUFFLE:-false}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-512}"

# Checkpointing: verify step 1 checkpoint save
export CHECKPOINT_SAVE_INTERVAL_STEPS="${CHECKPOINT_SAVE_INTERVAL_STEPS:-1}"
export CHECKPOINT_MAX_TO_KEEP="${CHECKPOINT_MAX_TO_KEEP:-2}"

# Disable WandB logging and evaluation overhead
export WANDB_PROJECT="${WANDB_PROJECT:-ci-smoke-gsm8k}"
export EVAL_EVERY_N_STEPS="${EVAL_EVERY_N_STEPS:-999}"

# Timeouts: 15-minute safety ceiling (run expected in ~2-3 min)
export WAIT_TIMEOUT_SECS="${WAIT_TIMEOUT_SECS:-900}"

# Delegate to base launcher located in math_gsm8k_dist
exec "${DIR}/../math_gsm8k_dist/launcher.sh" "$@"
