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
#   checkpoint_dir="" bash /examples/rl/grpo/gsm8k/run_qwen3_8b_maxtext.sh
#
# Run from the tunix repo root.

set -euo pipefail

export SKIP_JAX_PRECOMPILE=true

model_name="${model_name:-Qwen3-8B}"
model_id="${model_id:-/tmp/maxtext_qwen3_8b}"
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

train_mesh="${train_mesh:-(8,1)}"
rollout_mesh="${rollout_mesh:-(1,8)}"

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
  \
  `# -- Model ------------------------------------------------------------` \
  model_config.model_name="$model_name" \
  model_config.model_id="$model_id" \
  model_config.model_source="maxtext" \
  model_config.rng_seed=42 \
  model_config.model_display=false \
  model_config.remat_config=3 \
  actor_model_config.mesh.shape="$train_mesh" \
  actor_model_config.mesh.axis_names="('fsdp','tp')" \
  reference_model_config.mesh=null \
  reference_model_config.same_mesh_as="actor" \
  rollout_model_config.mesh.shape="$rollout_mesh" \
  rollout_model_config.mesh.axis_names="('fsdp','tp')" \
  \
  `# -- Data -------------------------------------------------------------` \
  data_source="huggingface" \
  dataset_name="openai/gsm8k:main" \
  prompt_key="question" \
  \
  `# -- Training loop ----------------------------------------------------` \
  training_mode="agentic_grpo" \
  batch_size="$batch_size" \
  num_batches="$num_batches" \
  num_test_batches=100 \
  num_train_epochs="$num_train_epochs" \
  train_fraction="$train_fraction" \
  reward_functions=["tunix/cli/reward_fn/gsm8k.py"] \
  verl_compatible=false \
  \
  `# -- Rollout engine (vanilla | vllm | sglang_jax) ---------------------` \
  rollout_engine="vllm" \
  offload_to_cpu=false \
  \
  `# -- Rollout config ---------------------------------------------------` \
  rollout_config.max_prompt_length=256 \
  rollout_config.total_generation_steps=768 \
  rollout_config.max_tokens_to_generate=768 \
  rollout_config.temperature=0.9 \
  rollout_config.top_p=1.0 \
  rollout_config.top_k=50 \
  rollout_config.return_logprobs=true \
  \
  `# -- vLLM (used when rollout_engine=vllm) -----------------------------` \
  vllm_config.hbm_utilization=0.4 \
  vllm_config.tpu_backend_type="jax" \
  vllm_config.server_mode=true \
  vllm_config.async_scheduling=true \
  vllm_config.max_num_seqs="$vllm_max_num_seqs" \
  vllm_config.kwargs.kv_cache_metrics=true \
  vllm_config.kwargs.disable_log_stats=false \
  vllm_config.kwargs.enable_prefix_caching=true \
  \
  `# -- Tokenizer / chat parsing ----------------------------------------` \
  chat_parser_config.type="qwen" \
  tokenizer_config.tokenizer_type="huggingface" \
  tokenizer_config.tokenizer_path="$tokenizer_path" \
  tokenizer_config.add_bos=false \
  tokenizer_config.add_eos=false \
  \
  `# -- GRPO algorithm ---------------------------------------------------` \
  agentic_grpo_config.num_generations="$num_generations" \
  agentic_grpo_config.num_iterations=1 \
  agentic_grpo_config.beta=0.08 \
  agentic_grpo_config.epsilon=0.2 \
  agentic_grpo_config.system_prompt="You are given a grade school math problem. Think step by step and respond using <reasoning>...</reasoning> followed by <answer>...</answer> with only the final numeric answer inside <answer>." \
  agentic_grpo_config.max_concurrency=128 \
  agentic_grpo_config.max_response_length=768 \
  agentic_grpo_config.max_turns=1 \
  agentic_grpo_config.context_ratio=1 \
  \
  `# -- Optimizer --------------------------------------------------------` \
  rl_training_config.actor_optimizer_config.opt_type="adamw" \
  rl_training_config.actor_optimizer_config.learning_rate=3e-6 \
  rl_training_config.actor_optimizer_config.schedule_type="warmup_cosine_decay_schedule" \
  rl_training_config.actor_optimizer_config.init_value=0.0 \
  rl_training_config.actor_optimizer_config.peak_value=3e-6 \
  rl_training_config.actor_optimizer_config.end_value=0.0 \
  rl_training_config.actor_optimizer_config.warmup_ratio="$warmup_ratio" \
  rl_training_config.actor_optimizer_config.warmup_steps="$warmup_steps" \
  rl_training_config.actor_optimizer_config.decay_steps="$max_steps" \
  rl_training_config.actor_optimizer_config.b1=0.9 \
  rl_training_config.actor_optimizer_config.b2=0.99 \
  rl_training_config.actor_optimizer_config.weight_decay=0.1 \
  rl_training_config.actor_optimizer_config.max_grad_norm=0.1 \
  \
  `# -- RL training ------------------------------------------------------` \
  rl_training_config.eval_every_n_steps=10 \
  rl_training_config.max_steps="$max_steps" \
  rl_training_config.mini_batch_size="$mini_batch_size" \
  rl_training_config.train_micro_batch_size="$train_micro_batch_size" \
  rl_training_config.rollout_micro_batch_size="$rollout_micro_batch_size" \
  rl_training_config.compute_logps_micro_batch_size="$compute_logps_micro_batch_size" \
  rl_training_config.checkpoint_root_directory="$checkpoint_dir" \
  rl_training_config.checkpointing_options.save_interval_steps=250 \
  rl_training_config.checkpointing_options.max_to_keep=4 \
  rl_training_config.metrics_logging_options.log_dir="/tmp/tensorboard/gsm8k_qwen3_8b_maxtext" \
  rl_training_config.metrics_logging_options.flush_every_n_steps=20 \
  \
  "$@"
