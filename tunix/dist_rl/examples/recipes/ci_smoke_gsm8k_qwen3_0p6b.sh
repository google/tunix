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
# Layered on `../math_gsm8k_dist/launcher.sh`
# (`../math_gsm8k_dist/gsm8k.yaml`).
# Single-host TPU (v6e-8) end-to-end smoke test for distributed RL with Raiden:
# - Trainer: Native Tunix PeftTrainer (model_source="huggingface") on chips 0..3 (TP=4)
# - Rollout: inprocess_vllm (vllm_jax) on chips 4..7 (TP=4)
# - Weight Sync: Raiden native TPU sync (weight_sync_config.mode="raiden")
# ==============================================================================
# Usage: bash ci_smoke_gsm8k_qwen3_0p6b.sh [--dry-run] [section.key=value ...]

set -Eeuo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Trainer and rollout both use TP=4 (FSDP=1) on 4 chips each, matching the
# rollout vLLM tensor-parallel layout for direct 1:1 Raiden weight sync.
# TODO(tunix-dev): explore cross-mesh FSDP configurations.
#
# Keep gradient accumulation (mini_batch_size * num_generations /
# train_micro_batch_size = 4) above 1: at exactly 1, PeftTrainer v2 takes the
# single-microstep fast path, whose gradient accumulator is discarded across
# the jit boundary ("The gradient accumulator is empty").
exec "${DIR}/../math_gsm8k_dist/launcher.sh" \
  model_config.model_name="Qwen3-0.6B" \
  model_config.model_source="huggingface" \
  model_config.model_id="Qwen/Qwen3-0.6B" \
  tokenizer_config.tokenizer_path="Qwen/Qwen3-0.6B" \
  model_config.mesh.shape="(1,4)" \
  model_config.mesh.axis_names="('fsdp','tp')" \
  rollout_engine="inprocess_vllm" \
  rollout_config.max_prompt_length=512 \
  rollout_config.total_generation_steps=512 \
  weight_sync_config.mode="raiden" \
  weight_sync_config.in_flight_weight_updates=false \
  data_config.split="train[:16]" \
  data_config.shuffle=false \
  batch_size=2 \
  rl_training_config.mini_batch_size=2 \
  rl_training_config.train_micro_batch_size=1 \
  rl_training_config.max_steps=2 \
  rl_training_config.eval_every_n_steps=999 \
  rl_training_config.checkpointing_options.save_interval_steps=1 \
  rl_training_config.checkpointing_options.max_to_keep=2 \
  rl_training_config.metrics_logging_options.project_name="ci-smoke-gsm8k" \
  agentic_grpo_config.num_generations=2 \
  agentic_grpo_config.beta=0.0 \
  agentic_grpo_config.off_policy_steps=0 \
  agentic_grpo_config.trajectory_group_order="trajectory_completion" \
  cluster_config.artifact_root="artifacts/qwen3_dist_gsm8k" \
  cluster_config.target="local" \
  cluster_config.wait_timeout_secs=900 \
  cluster_config.local.auto_detect_devices=true \
  cluster_config.local.trainer_devices="0,1,2,3" \
  cluster_config.local.rollout_devices="4,5,6,7" \
  cluster_config.local.tpu_chips_per_host_bounds="2,2,1" \
  cluster_config.local.tpu_host_bounds="1,1,1" \
  cluster_config.local.allow_multiple_libtpu_load=true \
  "$@"
