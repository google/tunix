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
# Distributed RL Recipe: Qwen3-0.6B on OpenMathInstruct-2 (TPU v5p, 64 chips)
# ==============================================================================
# Strict on-policy reproduction of AI-Hypercomputer/tpu-recipes Qwen3-0.6B GRPO
# layered on `../openmath_dist/k8s_launcher.sh` (`../openmath_dist/openmath.yaml`):
# - Trainer: MaxText on Pathways (`tpuv5:2x2x4`, 16 chips / 16 devices, FSDP=16,
#   TP=1) with sequence packing (12288 tok/row) and Raiden FFI (`RAIDEN_USE_FFI=1`).
# - Rollout: 12 single-host vLLM replicas (`12 x tpuv5:2x2x1`, 48 chips / 48
#   devices, DP=4, TP=1 per replica -> 48 concurrent DP workers) with Raiden TCP
#   (`RAIDEN_USE_FFI=0`), round-robin DP routing, and fixed-depth continue-decode.
# ==============================================================================
# Usage:
#   bash v5p_openmath_qwen3_0p6b.sh [start|stop|orchestrator|trainer|rollout|otel|stop-otel] \
#     [--dry-run] \
#     cluster_config.k8s.tunix_image=<image> \
#     actor_model_config.model_path=gs://<bucket>/checkpoints/qwen3-0.6b/0/items \
#     [section.key=value ...]

set -Eeuo pipefail

readonly DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec "${DIR}/../openmath_dist/k8s_launcher.sh" \
  cluster_config.k8s.service_account="xpk-sa" \
  cluster_config.k8s.priority_class="medium" \
  cluster_config.k8s.cpu_machine="n2-standard-64" \
  cluster_config.roles.orchestrator.memory.request="240G" \
  cluster_config.roles.trainer.memory.request="120G" \
  cluster_config.roles.trainer.memory.limit="160G" \
  cluster_config.pathways.proxy_memory.request="16G" \
  cluster_config.pathways.proxy_memory.limit="160G" \
  cluster_config.pathways.server_memory.request="100G" \
  cluster_config.pathways.rm_memory.request="4G" \
  'cluster_config.roles.orchestrator.env.RAIDEN_BROADCAST_HOST_RATIO="1.0"' \
  'cluster_config.roles.orchestrator.env.RAIDEN_BROADCAST_PIPELINE_STAGES="4"' \
  'cluster_config.roles.trainer.env.HF_HOME="/dev/shm/hf"' \
  'cluster_config.roles.trainer.env.RAIDEN_TRANSPORT_COALESCE_WINDOW_BYTES="67108864"' \
  'cluster_config.roles.trainer.env.RAIDEN_WEIGHT_SYNC_PIPELINE_GROUP_SIZE="16"' \
  'cluster_config.roles.trainer.env.RAIDEN_BROADCAST_HOST_RATIO="1.0"' \
  'cluster_config.roles.trainer.env.RAIDEN_BROADCAST_PIPELINE_STAGES="4"' \
  'cluster_config.roles.trainer.env.RAIDEN_FFI_USE_DIRECT_DEVICE_BUFFER="0"' \
  'cluster_config.roles.rollout.env.HF_HOME="/dev/shm/hf"' \
  'cluster_config.roles.rollout.env.RAIDEN_DEVICES_PER_HOST="4"' \
  'cluster_config.roles.rollout.env.DP_SCHED_ROUTING="round_robin"' \
  'cluster_config.roles.rollout.env.CONTINUE_DECODE_EOS_CHECK_INTERVAL="0"' \
  'cluster_config.roles.rollout.env.RAIDEN_TRANSPORT_COALESCE_WINDOW_BYTES="67108864"' \
  'cluster_config.roles.rollout.env.RAIDEN_WEIGHT_SYNC_PIPELINE_GROUP_SIZE="16"' \
  'cluster_config.roles.rollout.env.RAIDEN_PARALLELISM="16"' \
  'cluster_config.roles.rollout.env.RAIDEN_BROADCAST_HOST_RATIO="1.0"' \
  'cluster_config.roles.rollout.env.RAIDEN_BROADCAST_PIPELINE_STAGES="4"' \
  'cluster_config.roles.rollout.env.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="1800"' \
  "$@"
