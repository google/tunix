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

batch_size=${batch_size:-16}
max_steps=${max_steps:-10}

echo "Using parameters:"
echo "  Batch Size: $batch_size"
echo "  Max Steps: $max_steps"

python3 -m tunix.cli.peft_main \
  tunix/cli/base_config.yaml \
  override_config_file=examples/sft/mtnt/configs/gemma2_2b.yaml \
  model_config.model_download_path="/tmp/models/gemma-2b" \
  model_config.intermediate_ckpt_dir="/tmp/intermediate_ckpt/" \
  tokenizer_config.tokenizer_path="/tmp/models/gemma-2b/models/google/gemma-2/flax/gemma2-2b-it/1/tokenizer.model" \
  batch_size=$batch_size \
  training_config.max_steps=$max_steps \
  training_config.metrics_logging_options.log_dir="/tmp/tensorboard/full"
