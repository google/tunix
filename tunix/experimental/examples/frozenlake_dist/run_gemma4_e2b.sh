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

export MODEL_NAME=${MODEL_NAME:-gemma-4-e2b}
export MODEL_ID=${MODEL_ID:-google/gemma-4-E2B-it}
export ARTIFACT_ROOT=${ARTIFACT_ROOT:-"${REPO_ROOT}/artifacts/gemma4_e2b_dist_frozenlake"}
export TRAINER_TPU_CHIPS=${TRAINER_TPU_CHIPS:-0,1}
export TRAINER_FSDP=${TRAINER_FSDP:-2}
export TRAINER_TP=${TRAINER_TP:-1}
export ROLLOUT_TPU_CHIPS=${ROLLOUT_TPU_CHIPS:-2,3}
export ROLLOUT_FSDP=${ROLLOUT_FSDP:-1}
export ROLLOUT_TP=${ROLLOUT_TP:-2}
export TPU_CHIPS_PER_HOST_BOUNDS=${TPU_CHIPS_PER_HOST_BOUNDS:-1,2,1}
export TRAIN_MICRO_BATCH_SIZE=${TRAIN_MICRO_BATCH_SIZE:-2}
export COMPUTE_LOGPS_MICRO_BATCH_SIZE=${COMPUTE_LOGPS_MICRO_BATCH_SIZE:-2}
export COMPUTE_LOGPS_CHUNK_SIZE=${COMPUTE_LOGPS_CHUNK_SIZE:-2048}
export ROLLOUT_MAX_CONCURRENCY=${ROLLOUT_MAX_CONCURRENCY:-512}
export VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-32}
export VLLM_MAX_NUM_BATCHED_TOKENS=${VLLM_MAX_NUM_BATCHED_TOKENS:-8192}

exec "${DIR}/launcher.sh"
