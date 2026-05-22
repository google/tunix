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

# specify at cmd line to override defaults, e.g.
model_name=${model_name:-"qwen3-1.7b-base"}
batch_size=${batch_size:-8}
num_train_epochs=${num_train_epochs:-1}
warmup_ratio=${warmup_ratio:-0.1}
train_fraction=${train_fraction:-0.8}
checkpoint_dir=${checkpoint_dir:-"/tmp/grpo_checkpoints/${model_name}"}

echo "Using parameters:"
echo "  Batch Size: $batch_size"
echo "  Num Epochs: $num_train_epochs"
echo "  Warmup Ratio: $warmup_ratio"
echo "  Train Fraction: $train_fraction"
echo "  Checkpoint Directory: $checkpoint_dir"

python3 -m tunix.cli.grpo_main \
  tunix/cli/base_config.yaml \
  override_config_file=examples/rl/grpo/gsm8k/configs/qwen3.yaml \
  model_config.model_name="${model_name}" \
  model_config.model_id="Qwen/${model_name}" \
  model_config.intermediate_ckpt_dir="/tmp/intermediate_ckpt/${model_name}" \
  model_config.model_download_path="/tmp/models/${model_name}" \
  tokenizer_config.tokenizer_path="Qwen/${model_name}" \
  batch_size=$batch_size \
  num_train_epochs=$num_train_epochs \
  train_fraction=$train_fraction \
  rl_training_config.actor_optimizer_config.warmup_ratio=$warmup_ratio \
  rl_training_config.metrics_logging_options.log_dir="/tmp/tensorboard/${model_name}" \
  rl_training_config.checkpoint_root_directory="$checkpoint_dir"
