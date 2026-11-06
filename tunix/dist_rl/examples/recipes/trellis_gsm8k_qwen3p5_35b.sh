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
# Trellis Multi-Host Distributed RL Recipe: Qwen3.5-35B-A3B on GSM8K (GKE)
# ==============================================================================
# Layered on `../math_gsm8k_dist/k8s_launcher.sh`
# (`../math_gsm8k_dist/gsm8k.yaml` + the GKE cluster overrides; sequence
# lengths come from `gsm8k.yaml`).
# Architecture:
# - Trainer: MaxText on Pathways (TPU v5p 2x2x2, 8 chips across 2 hosts)
#   with Raiden FFI enabled (RAIDEN_USE_FFI=1).
# - Rollout: mcJax vLLM (TPU v5p 2x2x2, 8 chips across 2 hosts)
#   with 2 replicas, TCP transport (RAIDEN_USE_FFI=0).
# - Weight Sync: Raiden with MoE weight prefusion and weight converter.
# ==============================================================================
# Usage: bash trellis_gsm8k_qwen3p5_35b.sh [start|stop|orchestrator|trainer|rollout|otel|stop-otel] [--dry-run] [section.key=value ...]

set -Eeuo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec "${DIR}/../math_gsm8k_dist/k8s_launcher.sh" \
  model_config.model_name="Qwen3.5-35B-A3B" \
  model_config.model_id="Qwen/Qwen3.5-35B-A3B" \
  tokenizer_config.tokenizer_path="Qwen/Qwen3.5-35B-A3B" \
  model_config.kwargs.maxtext_model_name="qwen3.5-35b-a3b" \
  model_config.kwargs.maxtext_output_dir="/app/artifacts/math_gsm8k_dist/maxtext" \
  model_config.model_source="maxtext" \
  actor_model_config.model_path="/app/artifacts/math_gsm8k_dist/checkpoints/qwen3.5-35b-a3b/0/items" \
  actor_model_config.mesh.shape="(4,2,1)" \
  actor_model_config.mesh.axis_names="('fsdp','tp','expert')" \
  rollout_model_config.mesh.shape="(1,8)" \
  rollout_model_config.mesh.axis_names="('fsdp','tp')" \
  rollout_engine="vllm" \
  vllm_config.enable_prefix_caching=false \
  weight_sync_config.mode="raiden" \
  weight_sync_config.use_weight_converter=true \
  weight_sync_config.prefuse_moe_weights=true \
  weight_sync_config.verify_weights=true \
  weight_sync_config.in_flight_weight_updates=false \
  weight_sync_config.raiden_use_ffi=1 \
  batch_size=4 \
  rl_training_config.mini_batch_size=4 \
  rl_training_config.train_micro_batch_size=8 \
  rl_training_config.max_steps=2 \
  rl_training_config.metrics_logging_options.project_name="trellis-gsm8k" \
  agentic_grpo_config.num_generations=2 \
  agentic_grpo_config.beta=0.0 \
  agentic_grpo_config.off_policy_steps=0 \
  agentic_grpo_config.trajectory_group_order="trajectory_completion" \
  cluster_config.target="k8s" \
  cluster_config.k8s.trainer.jobset_yaml="jobset.pathways.yaml" \
  cluster_config.k8s.trainer.tpu_slice="tpuv5:2x2x2" \
  cluster_config.k8s.rollout.jobset_yaml="jobset.tpu.yaml" \
  cluster_config.k8s.rollout.tpu_slice="tpuv5:2x2x2" \
  cluster_config.rollout_replicas=2 \
  "$@"
