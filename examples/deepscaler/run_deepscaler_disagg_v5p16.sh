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
# DeepScaleR training launcher using tunix/cli/base_config.yaml plus explicit
# CLI overrides.
#
# Usage:
#   checkpoint_dir="" bash examples/deepscaler/run_deepscaler_disagg.sh
#
# Run from the tunix repo root.

set -euo pipefail

export SKIP_JAX_PRECOMPILE=true

num_batches="${num_batches:-312}"
num_train_epochs="${num_train_epochs:-3}"
train_fraction="${train_fraction:-1.0}"
warmup_ratio="${warmup_ratio:-0.1}"
batch_size="${batch_size:-128}"
mini_batch_size="${mini_batch_size:-128}"
max_response_length="${max_response_length:-8192}"
trainer_mesh="${trainer_mesh:-(4,1)}"
rollout_mesh="${rollout_mesh:-(4,1)}"

checkpoint_dir="${checkpoint_dir:-gs://tunix/rl/checkpoints/01}"
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

python -m tunix.cli.grpo_main \
  tunix/cli/base_agentic_config.yaml \
  \
  `# ── Model ────────────────────────────────────────────────────────────` \
  model_config.model_name="deepseek_r1_distill_qwen_1_5b" \
  model_config.model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
  model_config.model_source="huggingface" \
  model_config.rng_seed=42 \
  model_config.model_display=false \
  model_config.remat_config=3 \
  actor_model_config.mesh.shape="$trainer_mesh" \
  actor_model_config.mesh.axis_names="('fsdp','tp')" \
  reference_model_config.mesh=null \
  reference_model_config.same_mesh_as="actor" \
  rollout_model_config.mesh.shape="$rollout_mesh" \
  rollout_model_config.mesh.axis_names="('fsdp','tp')" \
  \
  `# ── Data ─────────────────────────────────────────────────────────────` \
  data_module="tunix.cli.recipes.deepscaler_data" \
  data_config.train_data_path="gs://tunix/data/DeepScaleR-Preview-Dataset/deepscaler.json" \
  data_config.eval_data_path="gs://tunix/data/HuggingFaceH4/aime_2024/train-00000-of-00001.parquet" \
  data_config.shuffle=true \
  data_config.seed=42 \
  prompt_key="prompts" \
  \
  `# ── Training loop ────────────────────────────────────────────────────` \
  training_config.profiler_options.log_dir="gs://lancewang-dev-supercomputer-testing/tunix/rl/grpo/profiler" \
  training_config.profiler_options.profiler_steps=3 \
  training_mode="agentic_grpo" \
  batch_size="$batch_size" \
  num_batches="$num_batches" \
  num_train_epochs="$num_train_epochs" \
  train_fraction="$train_fraction" \
  reward_functions=["tunix/utils/math_rewards.py"] \
  verl_compatible=false \
  \
  `# ── Rollout engine (vanilla | vllm | sglang_jax) ─────────────────────` \
  rollout_engine="vllm" \
  offload_to_cpu=false \
  \
  `# ── Rollout config ───────────────────────────────────────────────────` \
  rollout_config.max_prompt_length=2048 \
  rollout_config.total_generation_steps="$max_response_length" \
  rollout_config.max_tokens_to_generate="$max_response_length" \
  rollout_config.temperature=0.6 \
  rollout_config.top_p=null \
  rollout_config.top_k=null \
  rollout_config.return_logprobs=true \
  \
  `# ── SGLang-JAX (used when rollout_engine=sglang_jax) ─────────────────` \
  sglang_jax_config.mem_fraction_static=0.8 \
  sglang_jax_config.init_with_random_weights=true \
  sglang_jax_config.disable_radix_cache=true \
  sglang_jax_config.enable_deterministic_sampling=false \
  sglang_jax_config.chunked_prefill_size=2048 \
  sglang_jax_config.page_size=128 \
  sglang_jax_config.use_sort_for_toppk_minp=false \
  \
  `# ── vLLM (used when rollout_engine=vllm) ─────────────────────────────` \
  vllm_config.hbm_utilization=0.4 \
  vllm_config.tpu_backend_type="jax" \
  vllm_config.server_mode=true \
  vllm_config.async_scheduling=true \
  vllm_config.max_num_seqs=768 \
  vllm_config.kwargs.kv_cache_metrics=true \
  vllm_config.kwargs.disable_log_stats=false \
  vllm_config.kwargs.enable_prefix_caching=true \
  \
  `# ── Chat / agent wiring ───────────────────────────────────────────────` \
  chat_parser_config.type="default" \
  tokenizer_config.tokenizer_type="huggingface" \
  tokenizer_config.tokenizer_path="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
  tokenizer_config.add_bos=false \
  tokenizer_config.add_eos=false \
  \
  `# ── GRPO algorithm ───────────────────────────────────────────────────` \
  agentic_grpo_config.num_generations=8 \
  agentic_grpo_config.num_iterations=1 \
  agentic_grpo_config.beta=0.0 \
  agentic_grpo_config.epsilon=0.2 \
  agentic_grpo_config.epsilon_high=0.28 \
  agentic_grpo_config.system_prompt="" \
  agentic_grpo_config.max_concurrency=1024 \
  agentic_grpo_config.max_response_length="$max_response_length" \
  agentic_grpo_config.off_policy_steps=0 \
  agentic_grpo_config.loss_agg_mode="token-mean" \
  agentic_grpo_config.kl_loss_mode="low_var_kl" \
  agentic_grpo_config.max_turns=1 \
  agentic_grpo_config.context_ratio=1 \
  \
  `# ── Optimizer ────────────────────────────────────────────────────────` \
  rl_training_config.actor_optimizer_config.opt_type="adamw" \
  rl_training_config.actor_optimizer_config.learning_rate=1e-6 \
  rl_training_config.actor_optimizer_config.schedule_type="cosine_decay_schedule" \
  rl_training_config.actor_optimizer_config.init_value=1e-6 \
  rl_training_config.actor_optimizer_config.end_value=0.0 \
  rl_training_config.actor_optimizer_config.warmup_ratio="$warmup_ratio" \
  rl_training_config.actor_optimizer_config.warmup_steps="$warmup_steps" \
  rl_training_config.actor_optimizer_config.decay_steps="$max_steps" \
  rl_training_config.actor_optimizer_config.b1=0.9 \
  rl_training_config.actor_optimizer_config.b2=0.99 \
  rl_training_config.actor_optimizer_config.weight_decay=0.01 \
  rl_training_config.actor_optimizer_config.max_grad_norm=1.0 \
  \
  `# ── RL training ──────────────────────────────────────────────────────` \
  rl_training_config.eval_every_n_steps=1000 \
  rl_training_config.max_steps="$max_steps" \
  rl_training_config.mini_batch_size="$mini_batch_size" \
  rl_training_config.train_micro_batch_size=2 \
  rl_training_config.checkpoint_root_directory="$checkpoint_dir" \
  rl_training_config.checkpointing_options.save_interval_steps=500 \
  rl_training_config.checkpointing_options.max_to_keep=4 \
  rl_training_config.metrics_logging_options.log_dir="/tmp/tensorboard/deepscaler" \
  rl_training_config.metrics_logging_options.flush_every_n_steps=20 \
  \
  "$@"
