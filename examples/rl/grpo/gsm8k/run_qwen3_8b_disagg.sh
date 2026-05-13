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
#
# Agentic GSM8K GRPO launcher for Qwen3 8B using
# tunix/cli/base_agentic_config.yaml plus explicit CLI overrides.
#
# Usage:
#   checkpoint_dir="" bash /examples/rl/grpo/gsm8k/run_qwen3_8b.sh
#
# Run from the tunix repo root.

set -euo pipefail

export SKIP_JAX_PRECOMPILE=true

model_name="${model_name:-Qwen3-8B}"
model_id="${model_id:-Qwen/Qwen3-8B}"
tokenizer_path="${tokenizer_path:-$model_id}"

batch_size="${batch_size:-8}"
num_batches="${num_batches:-934}"
num_train_epochs="${num_train_epochs:-1}"
train_fraction="${train_fraction:-1.0}"
warmup_ratio="${warmup_ratio:-0.1}"

mini_batch_size="${mini_batch_size:-8}"
train_micro_batch_size="${train_micro_batch_size:-1}"
rollout_micro_batch_size="${rollout_micro_batch_size:-8}"
compute_logps_micro_batch_size="${compute_logps_micro_batch_size:-1}"

num_generations="${num_generations:-4}"
total_tpus="${total_tpus:-16}"
train_mesh="${train_mesh:-(8,1)}"
rollout_mesh="${rollout_mesh:-(1,8)}"

source "$(dirname "$0")/../../../tpu_utils.sh"
validate_mesh_allocation "$total_tpus" "$train_mesh" "$rollout_mesh" "null" || exit 1

checkpoint_dir="${checkpoint_dir:-gs://tunix/rl/checkpoints/gsm8k/qwen3/01}"
checkpoint_suffix="${checkpoint_suffix:-$(printf '%04d' "$((RANDOM % 10000))")}"
if [[ -n "$checkpoint_dir" && "$checkpoint_dir" != "null" ]]; then
  checkpoint_dir="${checkpoint_dir}_${checkpoint_suffix}"
fi

max_steps=$(awk "BEGIN {
  value = $num_batches * $num_train_epochs * $train_fraction;
  if (value < 1) value = 1;
  printf \"%.0f\", value;
}")
warmup_steps=$(awk "BEGIN {
  value = $warmup_ratio * $max_steps;
  if (value < 1) value = 1;
  printf \"%.0f\", value;
}")
vllm_max_num_seqs=$(awk "BEGIN {
  value = $rollout_micro_batch_size * $num_generations;
  if (value < 1) value = 1;
  printf \"%.0f\", value;
}")

python -m tunix.cli.grpo_main \
  tunix/cli/base_agentic_config.yaml \
  override_config_file=examples/rl/grpo/gsm8k/configs/qwen3_disagg.yaml \
  model_config.model_name="$model_name" \
  model_config.model_id="$model_id" \
  model_config.model_source="huggingface" \
  model_config.model_download_path="/tmp/models/${model_name}" \
  tokenizer_config.tokenizer_path="$tokenizer_path" \
  actor_model_config.mesh.shape="$train_mesh" \
  rollout_model_config.mesh.shape="$rollout_mesh" \
  batch_size="$batch_size" \
  num_batches="$num_batches" \
  num_train_epochs="$num_train_epochs" \
  train_fraction="$train_fraction" \
  vllm_config.max_num_seqs="$vllm_max_num_seqs" \
  agentic_grpo_config.num_generations="$num_generations" \
  rl_training_config.actor_optimizer_config.warmup_ratio="$warmup_ratio" \
  rl_training_config.actor_optimizer_config.warmup_steps="$warmup_steps" \
  rl_training_config.actor_optimizer_config.decay_steps="$max_steps" \
  rl_training_config.max_steps="$max_steps" \
  rl_training_config.mini_batch_size="$mini_batch_size" \
  rl_training_config.train_micro_batch_size="$train_micro_batch_size" \
  rl_training_config.rollout_micro_batch_size="$rollout_micro_batch_size" \
  rl_training_config.compute_logps_micro_batch_size="$compute_logps_micro_batch_size" \
  rl_training_config.checkpoint_root_directory="$checkpoint_dir" \
  rl_training_config.metrics_logging_options.log_dir="/tmp/tensorboard/gsm8k_qwen3_8b" \
  "$@"
