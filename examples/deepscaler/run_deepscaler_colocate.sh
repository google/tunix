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
# DeepScaleR colocate training launcher.
#
# Colocate mode: rollout engine and trainer share ALL devices on one mesh
# (split_mesh_config.enabled=false).  The agentic training loop blocks the
# consumer (trainer) until the producer (rollout engine) has finished ALL
# rollouts for the current batch, preventing resource contention.
#
# Mesh layout (colocate, single mesh):
#   All N devices → fsdp × tp  (no device split between rollout and trainer)
#
# Usage:
#   bash examples/deepscaler/run_deepscaler_colocate.sh
#
# Run from the tunix repo root.

set -euo pipefail

CONFIG="examples/deepscaler/configs/deepseek_r1_distill_qwen_1.5b_colocate.yaml"

python -m tunix.cli.grpo_main "${CONFIG}" \
  \
  `# ── Model ────────────────────────────────────────────────────────────` \
  `# model_config.model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"` \
  `# model_config.model_source="huggingface"` \
  `# model_config.model_path="gs://tunix/models/DeepSeek-R1-Distill-Qwen-1.5B"` \
  `# model_config.rng_seed=42` \
  \
  `# ── Colocate mesh (shared by all roles) ──────────────────────────────` \
  `# actor_model_config.mesh.shape="(4,1)"` \
  `# actor_model_config.mesh.axis_names="('fsdp','tp')"` \
  `# rollout_model_config.mesh.shape="(4,1)"` \
  `# rollout_model_config.mesh.axis_names="('fsdp','tp')"` \
  `# split_mesh_config.enabled=false` \
  \
  `# ── Data ─────────────────────────────────────────────────────────────` \
  `# data_config.train_data_path="gs://tunix/data/DeepScaleR-Preview-Dataset/deepscaler.json"` \
  `# data_config.eval_data_path="gs://tunix/data/HuggingFaceH4/aime_2024/train-00000-of-00001.parquet"` \
  `# data_config.shuffle=false` \
  `# data_config.seed=42` \
  \
  `# ── Training loop ────────────────────────────────────────────────────` \
  `# batch_size=128` \
  `# num_batches=312` \
  `# num_train_epochs=3` \
  `# train_fraction=0.8` \
  \
  `# ── Rollout engine (vanilla | vllm | sglang_jax) ─────────────────────` \
  `# rollout_engine="sglang_jax"` \
  \
  `# ── Rollout config ───────────────────────────────────────────────────` \
  `# rollout_config.max_prompt_length=2048` \
  `# rollout_config.total_generation_steps=8192` \
  `# rollout_config.temperature=0.8` \
  `# rollout_config.top_p=0.95` \
  `# rollout_config.top_k=null` \
  \
  `# ── SGLang-JAX (used when rollout_engine=sglang_jax) ─────────────────` \
  `# sglang_jax_config.mem_fraction_static=0.8` \
  `# sglang_jax_config.init_with_random_weights=true` \
  `# sglang_jax_config.disable_radix_cache=true` \
  `# sglang_jax_config.chunked_prefill_size=2048` \
  `# sglang_jax_config.page_size=128` \
  \
  `# ── vLLM (used when rollout_engine=vllm) ─────────────────────────────` \
  `# vllm_config.hbm_utilization=0.4` \
  `# vllm_config.max_num_seqs=768` \
  \
  `# ── Agentic GRPO algorithm ───────────────────────────────────────────` \
  `# agentic_grpo_config.num_generations=8` \
  `# agentic_grpo_config.num_iterations=1` \
  `# agentic_grpo_config.beta=0.0` \
  `# agentic_grpo_config.epsilon=0.2` \
  `# agentic_grpo_config.epsilon_high=0.28` \
  `# agentic_grpo_config.max_concurrency=768` \
  `# agentic_grpo_config.off_policy_steps=0` \
  `# agentic_grpo_config.loss_agg_mode="token-mean"` \
  `# agentic_grpo_config.kl_loss_mode="low_var_kl"` \
  \
  \
  `# ── Optimizer ────────────────────────────────────────────────────────` \
  `# rl_training_config.actor_optimizer_config.learning_rate=1e-6` \
  `# rl_training_config.actor_optimizer_config.b1=0.9` \
  `# rl_training_config.actor_optimizer_config.b2=0.99` \
  `# rl_training_config.actor_optimizer_config.weight_decay=0.01` \
  `# rl_training_config.actor_optimizer_config.max_grad_norm=1.0` \
  `# rl_training_config.actor_optimizer_config.warmup_ratio=0.1` \
  \
  `# ── RL training ──────────────────────────────────────────────────────` \
  `# rl_training_config.eval_every_n_steps=1000` \
  `# rl_training_config.mini_batch_size=128` \
  `# rl_training_config.train_micro_batch_size=2` \
  `# rl_training_config.checkpoint_root_directory="/tmp/tunix/checkpoints/deepscaler_colocate"` \
  `# rl_training_config.checkpointing_options.save_interval_steps=500` \
  `# rl_training_config.checkpointing_options.max_to_keep=4` \
  `# rl_training_config.metrics_logging_options.log_dir="/tmp/tensorboard/deepscaler_colocate"` \
  `# rl_training_config.metrics_logging_options.flush_every_n_steps=20` \
  \
  "$@"
