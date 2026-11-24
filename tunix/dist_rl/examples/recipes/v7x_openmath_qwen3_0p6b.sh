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
# Distributed RL Recipe: Qwen3-0.6B on OpenMathInstruct-2 (TPU v7x, 64 chips)
# ==============================================================================
# High-throughput strict on-policy GRPO recipe (<40s/step) on Ironwood TPU v7x
# layered on `../openmath_dist/k8s_launcher.sh` (`../openmath_dist/openmath.yaml`):
# - Trainer: MaxText on Pathways (`tpu7x:2x2x4`, 16 chips / 32 devices, FSDP=32,
#   TP=1) with 17408-token sequence packing, Tokamax Splash Attention (1024-token
#   tiles + diagonal mask skip), no activation remat (`remat_policy="none"`), and
#   single-pass on-policy logps (`use_rollout_logps: false` in `openmath.yaml`).
# - Rollout: 12 single-host vLLM replicas (`12 x tpu7x:2x2x1`, 48 chips / 96
#   devices, intra-chip TP=2, DP=4 per replica -> 48 concurrent DP workers) with
#   batched RPA kernel, parallel AOT precompilation, top-k prefiltering, and
#   least-loaded DP routing.
# ==============================================================================
# Usage:
#   bash v7x_openmath_qwen3_0p6b.sh [start|stop|orchestrator|trainer|rollout|otel|stop-otel] \
#     [--dry-run] \
#     cluster_config.k8s.tunix_image=<image> \
#     actor_model_config.model_path=gs://<bucket>/checkpoints/qwen3-0.6b/0/items \
#     [section.key=value ...]

set -Eeuo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly DIR

exec "${DIR}/../openmath_dist/k8s_launcher.sh" \
  cluster_config.k8s.service_account="xpk-sa" \
  cluster_config.k8s.priority_class="high" \
  cluster_config.k8s.use_dynamic_slicing=true \
  cluster_config.k8s.cpu_machine="c3d-standard-60" \
  cluster_config.roles.orchestrator.memory.request="200G" \
  cluster_config.roles.trainer.memory.request="120G" \
  cluster_config.roles.trainer.memory.limit="160G" \
  cluster_config.pathways.proxy_memory.request="16G" \
  cluster_config.pathways.proxy_memory.limit="160G" \
  cluster_config.pathways.server_memory.request="100G" \
  cluster_config.pathways.rm_memory.request="4G" \
  cluster_config.k8s.trainer.tpu_slice="tpu7x:2x2x4" \
  actor_model_config.mesh.shape="(32,1,1)" \
  cluster_config.k8s.rollout.tpu_slice="tpu7x:2x2x1" \
  rollout_model_config.mesh.shape="(4,2)" \
  rl_training_config.max_seq_token_per_tpu=17408 \
  rl_training_config.profiler_options.profiler_steps=0 \
  dist_training_config.maxtext_extra_config.use_tokamax_splash=true \
  dist_training_config.maxtext_extra_config.sa_qk_diag_skip=true \
  dist_training_config.maxtext_extra_config.sa_sv_diag_skip=true \
  dist_training_config.maxtext_extra_config.sa_use_base2_exp=true \
  dist_training_config.maxtext_extra_config.remat_policy="none" \
  dist_training_config.maxtext_extra_config.sa_block_q=1024 \
  dist_training_config.maxtext_extra_config.sa_block_q_dkv=1024 \
  dist_training_config.maxtext_extra_config.sa_block_q_dq=1024 \
  dist_training_config.maxtext_extra_config.sa_block_kv=1024 \
  dist_training_config.maxtext_extra_config.sa_block_kv_compute=1024 \
  dist_training_config.maxtext_extra_config.sa_block_kv_dkv=1024 \
  dist_training_config.maxtext_extra_config.sa_block_kv_dkv_compute=1024 \
  dist_training_config.maxtext_extra_config.sa_block_kv_dq=1024 \
  vllm_config.skip_jax_precompile=false \
  vllm_config.max_num_seqs=48 \
  vllm_config.max_num_batched_tokens=8192 \
  vllm_config.engine_kwargs.additional_config.max_decode_steps=256 \
  'cluster_config.roles.orchestrator.env.RAIDEN_BROADCAST_HOST_RATIO="1.0"' \
  'cluster_config.roles.orchestrator.env.RAIDEN_BROADCAST_PIPELINE_STAGES="4"' \
  'cluster_config.roles.orchestrator.env.RAIDEN_PARALLELISM="16"' \
  'cluster_config.roles.orchestrator.env.TPU_RAIDEN_CONTROL_PLANE_BACKEND="tcp"' \
  'cluster_config.roles.orchestrator.env.TPU_RAIDEN_TCP_CONNECT_TIMEOUT_MS="120000"' \
  'cluster_config.roles.orchestrator.env.TPU_RAIDEN_TCP_CONNECT_MAX_ATTEMPTS="5"' \
  'cluster_config.roles.trainer.env.HF_HOME="/dev/shm/hf"' \
  'cluster_config.roles.trainer.env.RAIDEN_TRANSPORT_COALESCE_WINDOW_BYTES="67108864"' \
  'cluster_config.roles.trainer.env.RAIDEN_WEIGHT_SYNC_PIPELINE_GROUP_SIZE="16"' \
  'cluster_config.roles.trainer.env.RAIDEN_BROADCAST_HOST_RATIO="1.0"' \
  'cluster_config.roles.trainer.env.RAIDEN_BROADCAST_PIPELINE_STAGES="4"' \
  'cluster_config.roles.trainer.env.RAIDEN_FFI_USE_DIRECT_DEVICE_BUFFER="0"' \
  'cluster_config.roles.trainer.env.TPU_RAIDEN_CONTROL_PLANE_BACKEND="tcp"' \
  'cluster_config.roles.trainer.env.TPU_RAIDEN_TCP_CONNECT_TIMEOUT_MS="120000"' \
  'cluster_config.roles.trainer.env.TPU_RAIDEN_TCP_CONNECT_MAX_ATTEMPTS="5"' \
  'cluster_config.roles.rollout.env.HF_HOME="/dev/shm/hf"' \
  'cluster_config.roles.rollout.env.RAIDEN_DEVICES_PER_HOST="8"' \
  'cluster_config.roles.rollout.env.DP_SCHED_ROUTING="least_loaded"' \
  'cluster_config.roles.rollout.env.CONTINUE_DECODE_EOS_CHECK_INTERVAL="0"' \
  'cluster_config.roles.rollout.env.NUM_PRECOMPILE_WORKERS="4"' \
  'cluster_config.roles.rollout.env.ENABLE_TOPK_PREFILTER="1"' \
  'cluster_config.roles.rollout.env.USE_BATCHED_RPA_KERNEL="1"' \
  'cluster_config.roles.rollout.env.RAIDEN_TRANSPORT_COALESCE_WINDOW_BYTES="67108864"' \
  'cluster_config.roles.rollout.env.RAIDEN_WEIGHT_SYNC_PIPELINE_GROUP_SIZE="16"' \
  'cluster_config.roles.rollout.env.RAIDEN_PARALLELISM="16"' \
  'cluster_config.roles.rollout.env.RAIDEN_BROADCAST_HOST_RATIO="1.0"' \
  'cluster_config.roles.rollout.env.RAIDEN_BROADCAST_PIPELINE_STAGES="4"' \
  'cluster_config.roles.rollout.env.TPU_RAIDEN_CONTROL_PLANE_BACKEND="tcp"' \
  'cluster_config.roles.rollout.env.TPU_RAIDEN_TCP_CONNECT_TIMEOUT_MS="120000"' \
  'cluster_config.roles.rollout.env.TPU_RAIDEN_TCP_CONNECT_MAX_ATTEMPTS="5"' \
  'cluster_config.roles.rollout.env.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="1800"' \
  "$@"
