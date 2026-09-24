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
# Trellis Multi-Host Distributed RL Recipe: Qwen3.5-397B-A17B on GSM8K
# ==============================================================================
# Architecture:
# - Trainer: MaxText on Pathways, v5p 4x8x8 (256 chips), fsdp=64 x tp=2 x
#   expert=2, Raiden FFI.
# - Rollout: vLLM on Ray multihost, v5p 2x2x4 (16 chips = 4 hosts),
#   tp=1 x expert=16, single replica.
# - Weight sync: Raiden, MoE prefusion and the weight converter.
#
# Usage:
#   export TUNIX_IMAGE=gcr.io/.../runner@sha256:<digest>
#   export MAXTEXT_CKPT=gs://.../qwen35_397b/scanned_reshard_fsdp32_tp2/0/items
#   export USER=<unique-run-id>          # names the jobsets
#   ./trellis_gsm8k_qwen3p5_397b.sh start
# ==============================================================================

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Model Configuration
export MODEL_NAME="Qwen3.5-397B-A17B"
export MODEL_ID="Qwen/Qwen3.5-397B-A17B"
export TOKENIZER_PATH="${TOKENIZER_PATH:-Qwen/Qwen3.5-397B-A17B}"
export MAXTEXT_MODEL_NAME="qwen3.5-397b-a17b"
export NEW_MODEL_DESIGN=1

# Training & Sampling Backends
export TRAINER_BACKEND="maxtext"
export SAMPLER="vllm"
export WEIGHT_SYNC_MODE="raiden"

# Weight Sync & MoE Settings
export PREFUSE_MOE_WEIGHTS="true"
export USE_WEIGHT_CONVERTER="true"
export VERIFY_WEIGHTS="true"
export ENABLE_PREFIX_CACHING="false"
export VLLM_ENABLE_V1_MULTIPROCESSING=0

# Topologies.
# TRAINER_MESH_EXPERT=2 is required, not a tuning choice: at expert=1 the GMM_v2
# kernel overflows smem by ~8.6K. Raising the GMM tile sizes does not help,
# because T(128) padding rounds s32[592] and s32[552] to the same 640.
export TRAINER_JOBSET_YAML="jobset.pathways.qwen3.5-397b.yaml"
export TRAINER_TPU_SLICE="tpuv5p:4x8x8"          # 256 chips
export TRAINER_MESH_FSDP=64
export TRAINER_MESH_TP=2                         # matches the ckpt; kv heads pad to 16
export TRAINER_MESH_EXPERT=2

# ROLLOUT_MESH_TP * ROLLOUT_MESH_EXPERT must equal the slice (16).
# tp=2 x expert=8 is NOT a working alternative -- it produces incoherent
# rollouts for reasons that are still unexplained. Use tp=1 x expert=16.
export ROLLOUT_JOBSET_YAML="jobset.mcjax.ray.yaml"
export ROLLOUT_TPU_SLICE="tpuv5p:2x2x4"          # 16 chips = 4 hosts
export ROLLOUT_MESH_FSDP=1
export ROLLOUT_MESH_TP=1
export ROLLOUT_MESH_EXPERT=16
export ROLLOUT_REPLICAS=1

# Hyperparameters.
# TRAIN_MICRO_BATCH_SIZE must be a multiple of TRAINER_MESH_FSDP *
# TRAINER_MESH_EXPERT = 128: MaxText binds the activation batch axis to
# ('data','fsdp','fsdp_transpose','expert'), so the packed row count has to be
# divisible by the product of all four.
export MAX_STEPS="${MAX_STEPS:-10}"
export BATCH_SIZE=16
export MINI_BATCH_SIZE=16
export NUM_GENERATIONS=8                         # 16 x 8 = 128 rollouts/step
export TRAIN_MICRO_BATCH_SIZE=128
export MAX_PROMPT_LENGTH=3072                    # 3k prefill + 1k generation
export MAX_RESPONSE_LENGTH=1024
export MAX_SEQ_TOKEN_PER_TPU=4096
export MAX_STALENESS=0
export USE_ROLLOUT_LOGPS="false"
export REWARD_MODE="env"

# Step 0 is a cold, single-threaded Pallas lowering of the MoE and GDN kernels
# with the TPUs idle; on 256 chips at seq 4096 that ran past the 1800 s default
# and killed the run on DEADLINE_EXCEEDED. Steady state is ~11 min/step.
export RPC_TIMEOUT_S="${RPC_TIMEOUT_S:-10800}"

# Checkpoint saving OOMs the trainer at this size; restore is unaffected.
export CHECKPOINT_SAVE_INTERVAL_STEPS=0

# Profiling off. MaxText decided whether to profile from profiler_steps alone
# (default 5), so profiler=ProfilerType.NONE still opened a trace at step 1 and
# Pathways then raised "No profile started" from inside fwd_bwd, taking the
# trainer down mid-run.
export PROFILER_STEPS=0
export SKIP_FIRST_N_PROFILER_STEPS=-1

# Per-role container env. The H2D weight-sync timeout belongs on the
# orchestrator, which drives the transfer.
export TRAINER_EXTRA_ENV="${TRAINER_EXTRA_ENV:-ONEHOT_MOE_PERMUTE_THRESHOLD=131072 RAIDEN_TRANSPORT_COALESCE_WINDOW_BYTES=67108864 RAIDEN_WEIGHT_SYNC_PIPELINE_GROUP_SIZE=16}"
export ORCHESTRATOR_EXTRA_ENV="${ORCHESTRATOR_EXTRA_ENV:-WEIGHT_SYNC_H2D_TIMEOUT_S=1800}"
export ROLLOUT_EXTRA_ENV="${ROLLOUT_EXTRA_ENV:-ONEHOT_MOE_PERMUTE_THRESHOLD=131072 RAY_memory_monitor_refresh_ms=0 RAIDEN_TRANSPORT_COALESCE_WINDOW_BYTES=67108864 RAIDEN_WEIGHT_SYNC_PIPELINE_GROUP_SIZE=16 RAIDEN_PARALLELISM=16}"

# Scheduling. Without the reservation pin NAP fails scale-up with "Invalid
# Reservation"; without a priority class Kueue never admits the workload.
export K8S_NAMESPACE="${K8S_NAMESPACE:-trellis}"
export KUEUE_QUEUE_NAME="${KUEUE_QUEUE_NAME:-multislice-queue}"
export KUEUE_PRIORITY_CLASS="${KUEUE_PRIORITY_CLASS:-medium}"
export CPU_MACHINE="${CPU_MACHINE:-n2d-standard-64}"

if [[ -z "${MAXTEXT_CKPT:-}" ]]; then
  echo "Note: MAXTEXT_CKPT is not set. For MaxText training, provide:"
  echo "    export MAXTEXT_CKPT=gs://<bucket>/qwen35_397b/scanned_reshard_fsdp32_tp2/0/items"
fi
export MAXTEXT_OUTPUT_DIR="${MAXTEXT_OUTPUT_DIR:-/app/artifacts/math_gsm8k_dist/maxtext}"

# Project & Monitoring
export WANDB_PROJECT="${WANDB_PROJECT:-trellis-gsm8k}"

# Delegate to base launcher located in the math_gsm8k_dist folder
exec "${DIR}/../math_gsm8k_dist/k8s_launcher.sh" "$@"
