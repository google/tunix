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
# Benchmark Recipe: Gemma4 E2B on FrozenLake on a Single 4-Chip TPU Host
# ==============================================================================
# Layered on `../frozenlake_dist/launcher.sh`
# (`../frozenlake_dist/frozenlake.yaml`). The distributed-vs-agentic RL
# efficiency benchmark runs this recipe with its own overrides.
# - Trainer: FSDP 2 on chips 0,1
# - Rollout: TP 2 on chips 2,3
# ==============================================================================
# Usage:
#   bash benchmark_frozenlake_gemma4_e2b.sh [--dry-run] [section.key=value ...]

set -Eeuo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec "${DIR}/../frozenlake_dist/launcher.sh" \
  model_config.model_name="gemma-4-e2b" \
  model_config.model_id="google/gemma-4-E2B-it" \
  tokenizer_config.tokenizer_path="google/gemma-4-E2B-it" \
  actor_model_config.mesh.shape="(2,1)" \
  actor_model_config.mesh.axis_names="('fsdp','tp')" \
  rollout_model_config.mesh.shape="(1,2)" \
  rollout_model_config.mesh.axis_names="('fsdp','tp')" \
  rl_training_config.train_micro_batch_size=2 \
  rl_training_config.compute_logps_micro_batch_size=2 \
  rl_training_config.compute_logps_chunk_size=2048 \
  agentic_grpo_config.max_concurrency=512 \
  cluster_config.artifact_root="artifacts/gemma4_e2b_dist_frozenlake" \
  cluster_config.local.trainer_tpu_chips="0,1" \
  cluster_config.local.rollout_tpu_chips="2,3" \
  cluster_config.local.tpu_chips_per_host_bounds="1,2,1" \
  "$@"
