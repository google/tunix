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

batch_size=${batch_size:-1}
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
  tunix/cli/base_config.yaml \
  override_config_file=examples/rl/grpo/gsm8k/configs/gemma2_2b.yaml \
  reference_model_config.model_download_path="/tmp/models/gemma2-2b" \
  reference_model_config.intermediate_ckpt_dir="/tmp/intermediate_ckpt/1" \
  tokenizer_config.tokenizer_path="/tmp/models/gemma2-2b/models/google/gemma-2/flax/gemma2-2b-it/1/tokenizer.model" \
  batch_size=$batch_size \
  num_batches=$num_batches \
  num_train_epochs=$num_train_epochs \
  rl_training_config.actor_optimizer_config.warmup_ratio=$warmup_ratio \
  rl_training_config.actor_optimizer_config.warmup_steps=$warmup_steps \
  rl_training_config.actor_optimizer_config.decay_steps=$max_steps \
  rl_training_config.max_steps=$max_steps \
  rl_training_config.metrics_logging_options.log_dir="/tmp/tensorboard/grpo"

