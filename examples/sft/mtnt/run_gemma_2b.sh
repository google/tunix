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
CONFIG="configs/sft.yaml"

python3 -m tunix.cli.peft_main \
  $CONFIG \
  model_config.model_name="gemma-2b" \
  model_config.model_id="google/gemma/flax/2b" \
  model_config.model_source="kaggle" \
  model_config.model_download_path="/tmp/models/" \
  model_config.intermediate_ckpt_dir="/tmp/intermediate_ckpt/gemma-2b" \
  model_config.mesh.shape="(2,2)" \
  model_config.mesh.axis_names="('fsdp','tp')" \
  model_config.lora_config.rank=16 \
  model_config.lora_config.alpha=2.0 \
  model_config.lora_config.weight_qtype="nf4" \
  model_config.lora_config.tile_size=256 \
  model_config.lora_config.module_path=".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj" \
  tokenizer_config.tokenizer_path="/tmp/models/models/google/gemma/flax/2b/1/tokenizer.model" \
  tokenizer_config.tokenizer_type="sentencepiece"
 


