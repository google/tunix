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

python3 -m tunix.cli.peft_main \
  base_config.yaml \
  model_config.model_name="qwen2.5-0.5b" \
  model_config.model_id="Qwen/Qwen2.5-0.5B" \
  model_config.model_source="huggingface" \
  model_config.lora_config={} \
  model_config.mesh.shape="(2,2)" \
  model_config.mesh.axis_names="('fsdp','tp')" \
  model_config.rng_seed=0 \
  model_config.model_download_path="/tmp/models" \
  tokenizer_config.tokenizer_path="Qwen/Qwen2.5-0.5B"\
  tokenizer_config.tokenizer_type="huggingface" \
  dataset_name="mtnt/en-fr" \
  optimizer_config.opt_type="adamw" \
  optimizer_config.learning_rate=1e-5 \
  max_target_length=256 \
  training_config.eval_every_n_steps=1 \
  training_config.max_steps=1 \
  training_config.metrics_logging_options.log_dir="/tmp/tensorboard/full" \
  training_config.metrics_logging_options.flush_every_n_steps=1


