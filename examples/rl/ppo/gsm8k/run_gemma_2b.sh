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
train_micro_batch_size=${train_micro_batch_size:-8}
total_generation_steps=${total_generation_steps:-768}
max_prompt_length=${max_prompt_length:-2048}
num_batches=${num_batches:-3738}
num_train_epochs=${num_train_epochs:-1}
warmup_ratio=${warmup_ratio:-0.1}
train_fraction=${train_fraction:-1.0}

echo "Using parameters:"
echo "  Batch Size: $batch_size"
echo "  Train Micro Batch Size: $train_micro_batch_size"
echo "  Total Generation Steps: $total_generation_steps"
echo "  Max Prompt Length: $max_prompt_length"
echo "  Num Batches: $num_batches"
echo "  Num Epochs: $num_train_epochs"
echo "  Warmup Ratio: $warmup_ratio"
echo "  Train Fraction: $train_fraction"

max_steps_float=$(awk "BEGIN {print $num_batches * $num_train_epochs * $train_fraction}")

max_steps=$(printf "%.0f" "$max_steps_float")


warmup_steps=$(awk "BEGIN {printf \"%.0f\", $warmup_ratio * $max_steps}")

echo "Max steps: $max_steps"
echo "Rounded warmup steps: $warmup_steps"

intermediate_ckpt_dir=${intermediate_ckpt_dir:-"/tmp/intermediate_ckpt/gemma_2b_ppo"}


python3 -m tunix.cli.ppo_main \
  tunix/cli/base_config.yaml \
  override_config_file=examples/rl/ppo/gsm8k/configs/gemma_2b.yaml \
  model_config.model_download_path="/tmp/models/gemma-2b" \
  model_config.intermediate_ckpt_dir="$intermediate_ckpt_dir" \
  batch_size=$batch_size \
  num_batches=$num_batches \
  num_train_epochs=$num_train_epochs \
  train_fraction=$train_fraction \
  rl_training_config.actor_optimizer_config.warmup_ratio=$warmup_ratio \
  rl_training_config.actor_optimizer_config.warmup_steps=$warmup_steps \
  rl_training_config.actor_optimizer_config.decay_steps=$max_steps \
  rl_training_config.critic_optimizer_config.warmup_ratio=$warmup_ratio \
  rl_training_config.critic_optimizer_config.warmup_steps=$warmup_steps \
  rl_training_config.critic_optimizer_config.decay_steps=$max_steps \
  rl_training_config.max_steps=$max_steps \
  rl_training_config.metrics_logging_options.log_dir="/tmp/tensorboard/ppo" \
  rl_training_config.checkpoint_root_directory="/tmp/ckpts_gemma_$RANDOM" \
  rl_training_config.mini_batch_size=$batch_size \
  rl_training_config.train_micro_batch_size=$train_micro_batch_size \
  rollout_config.total_generation_steps=$total_generation_steps \
  rollout_config.max_prompt_length=$max_prompt_length \
  "$@"
