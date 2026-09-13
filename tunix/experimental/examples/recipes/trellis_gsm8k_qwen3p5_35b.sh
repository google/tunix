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
# Trellis Multi-Host Distributed RL Recipe: Qwen3.5-35B-A3B on GSM8K
# ==============================================================================
# Architecture:
# - Trainer: MaxText on Pathways (TPU v5p 2x2x2, 8 chips across 2 hosts)
#   with Raiden FFI enabled (RAIDEN_USE_FFI=1, 4 devices/host).
# - Rollout: mcJax vLLM (TPU v5p 2x2x2, 8 chips across 2 hosts)
#   with 2 replicas, TCP transport (RAIDEN_USE_FFI=0).
# - Weight Sync: Raiden with MoE weight prefusion and weight converter.
# ==============================================================================

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Model Configuration
export MODEL_NAME="Qwen3.5-35B-A3B"
export MODEL_ID="Qwen/Qwen3.5-35B-A3B"
export MAXTEXT_MODEL_NAME="qwen3.5-35b-a3b"

# Training & Sampling Backends
export TRAINER_BACKEND="maxtext"
export SAMPLER="vllm"
export WEIGHT_SYNC_MODE="raiden"

# Hyperparameters
export BATCH_SIZE=4
export NUM_GENERATIONS=2
export MAX_STEPS=2
export TRAIN_MICRO_BATCH_SIZE=8  # Must be a multiple of TRAINER_MESH_FSDP (4)

# Weight Sync & MoE Settings
export PREFUSE_MOE_WEIGHTS="true"
export USE_WEIGHT_CONVERTER="true"
export VERIFY_WEIGHTS="true"
export ENABLE_PREFIX_CACHING="false"

# Topologies
export TRAINER_JOBSET_YAML="jobset.pathways.yaml"
export TRAINER_TPU_SLICE="tpuv5:2x2x2"
export TRAINER_MESH_FSDP=4
export TRAINER_MESH_TP=2
export TRAINER_MESH_EXPERT=1

export ROLLOUT_JOBSET_YAML="jobset.tpu.yaml"
export ROLLOUT_TPU_SLICE="tpuv5:2x2x2"
export ROLLOUT_MESH_FSDP=1
export ROLLOUT_MESH_TP=8
export ROLLOUT_REPLICAS=2

# Checkpoint paths (require explicit or configurable Orbax params checkpoint)
if [[ -z "${MAXTEXT_CKPT:-}" ]]; then
  echo "Note: MAXTEXT_CKPT is not set. For MaxText training, provide:"
  echo "    export MAXTEXT_CKPT=gs://<your-bucket>/checkpoints/<path-to-scanned-items>"
fi
export MAXTEXT_OUTPUT_DIR="${MAXTEXT_OUTPUT_DIR:-/app/artifacts/math_gsm8k_dist/maxtext}"

# Project & Monitoring
export WANDB_PROJECT="trellis-gsm8k"

# Delegate to base launcher located in the math_gsm8k_dist folder
exec "${DIR}/../math_gsm8k_dist/k8s_launcher.sh" "$@"
