# Copyright 2025 Google LLC
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


set -x # Enable xtrace

batch_size=${batch_size:-8}
num_batches=${num_batches:-3738}
num_train_epochs=${num_train_epochs:-1}
warmup_ratio=${warmup_ratio:-0.1}
train_fraction=${train_fraction:-1.0} 

echo "Using parameters:"
echo "  Batch Size: $batch_size"
echo "  Num Batches: $num_batches"
echo "  Num Epochs: $num_train_epochs"
echo "  Warmup Ratio: $warmup_ratio" 
echo "  Train Fraction: $train_fraction"


max_steps_float=$(awk "BEGIN {print $batch_size * $num_batches * $num_train_epochs * $train_fraction}")

max_steps=$(printf "%.0f" "$max_steps_float")


warmup_steps=$(awk "BEGIN {printf \"%.0f\", $warmup_ratio * $max_steps}")

echo "Max steps: $max_steps"
echo "Rounded warmup steps: $warmup_steps"

python3 -m tunix.cli.grpo_main \
  base_config.yaml \
  reference_model_config.model_name="gemma-7b-it" \
  reference_model_config.model_id="google/gemma/flax/7b-it" \
  reference_model_config.model_source="kaggle" \
  reference_model_config.model_download_path="/tmp/models/gemma-7b" \
  reference_model_config.intermediate_ckpt_dir="/tmp/intermediate_ckpt/1" \
  reference_model_config.mesh.shape="(2,4)" \
  reference_model_config.mesh.axis_names="('fsdp','tp')" \
  reference_model_config.rng_seed=42 \
  actor_model_config.lora_config.rank=64 \
  actor_model_config.lora_config.alpha=64.0 \
  actor_model_config.lora_config.module_path=".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj|.*attn_vec_einsum" \
  actor_model_config.mesh.shape="(2,4)" \
  actor_model_config.mesh.axis_names="('fsdp','tp')" \
  rollout_model_config.mesh.shape="(2,4)" \
  rollout_model_config.mesh.axis_names="('fsdp','tp')" \
  tokenizer_config.tokenizer_path="/tmp/models/gemma-7b/models/google/gemma/flax/7b-it/2/tokenizer.model" \
  tokenizer_config.tokenizer_type="sentencepiece" \
  tokenizer_config.add_bos=false \
  dataset_name="gsm8k" \
  batch_size=$batch_size \
  num_batches=$num_batches \
  num_test_batches=100 \
  num_train_epochs=$num_train_epochs \
  rl_training_config.actor_optimizer_config.opt_type="adamw" \
  rl_training_config.actor_optimizer_config.peak_value=3e-6 \
  rl_training_config.actor_optimizer_config.schedule_type="warmup_cosine_decay_schedule" \
  rl_training_config.actor_optimizer_config.init_value=0.0 \
  rl_training_config.actor_optimizer_config.end_value=0.0 \
  rl_training_config.actor_optimizer_config.warmup_ratio=$warmup_ratio \
  rl_training_config.actor_optimizer_config.warmup_steps=$warmup_steps \
  rl_training_config.actor_optimizer_config.decay_steps=$max_steps \
  rl_training_config.actor_optimizer_config.b1=0.9 \
  rl_training_config.actor_optimizer_config.b2=0.99 \
  rl_training_config.actor_optimizer_config.weight_decay=0.1 \
  rl_training_config.actor_optimizer_config.max_grad_norm=0.1 \
  rl_training_config.eval_every_n_steps=10 \
  rl_training_config.max_steps=$max_steps \
  rl_training_config.metrics_logging_options.log_dir="/tmp/tensorboard/grpo" \
  rl_training_config.metrics_logging_options.flush_every_n_steps=20 \
  rl_training_config.checkpointing_options.save_interval_steps=500 \
  rl_training_config.checkpointing_options.max_to_keep=4 \
  rl_training_config.profiler_options={} \
  rollout_config.total_generation_steps=768 \
  rollout_config.max_prompt_length=256 \
  rollout_config.temperature=0.9 \
  rollout_config.top_p=1.0 \
  rollout_config.top_k=50 \
  rollout_engine="vanilla" \
  offload_to_cpu=false \
  grpo_config.num_generations=4 \
  grpo_config.num_iterations=1 \
  grpo_config.beta=0.08 \
  grpo_config.epsilon=0.2 \
  reward_functions="['tunix/cli/reward_fn/gsm8k.py']"