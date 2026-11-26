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
# Recipe: Qwen3.5 0.8B (TP=1) on FrozenLake on a >=2-Chip TPU Host
# ==============================================================================
# Layered on `../frozenlake_dist/launcher.sh`
# (`../frozenlake_dist/frozenlake.yaml`). Uses symmetric 1-chip host bounds
# (`tpu_chips_per_host_bounds="1,1,1"`) so both processes share the same local
# TPU topology contract while keeping rollout at `TP=1` (`mapping_vllm_jax`).
# - Trainer: 1x1 on chip 0 (`float32` parameter load, `bfloat16` compute)
# - Rollout: TP 1 `inprocess_vllm` on chip 1 (`bfloat16`)
#
# To run `Qwen3.5-2B`, append:
#   model_config.model_name=Qwen3.5-2B \
#   model_config.model_id=Qwen/Qwen3.5-2B \
#   tokenizer_config.tokenizer_path=Qwen/Qwen3.5-2B
# ==============================================================================
# Usage:
#   bash frozenlake_qwen3p5_0p8b.sh [--dry-run] [section.key=value ...]

set -Eeuo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec "${DIR}/../frozenlake_dist/launcher.sh" \
  model_config.model_name="Qwen3.5-0.8B" \
  model_config.model_id="Qwen/Qwen3.5-0.8B" \
  tokenizer_config.tokenizer_path="Qwen/Qwen3.5-0.8B" \
  model_config.dtype="bfloat16" \
  model_config.load_dtype="float32" \
  actor_model_config.mesh.shape="(1,1)" \
  actor_model_config.mesh.axis_names="('fsdp','tp')" \
  rollout_model_config.mesh.shape="(1,1)" \
  rollout_model_config.mesh.axis_names="('fsdp','tp')" \
  vllm_config.hbm_utilization=0.5 \
  rl_training_config.train_micro_batch_size=2 \
  rl_training_config.compute_logps_micro_batch_size=2 \
  rl_training_config.compute_logps_chunk_size=2048 \
  rl_training_config.actor_optimizer_config.chain_kwargs.max_norm=1.0 \
  cluster_config.artifact_root="artifacts/qwen3p5_0p8b_dist_frozenlake" \
  cluster_config.local.trainer_devices="0" \
  cluster_config.local.rollout_devices="1" \
  cluster_config.local.tpu_chips_per_host_bounds="1,1,1" \
  "$@"
