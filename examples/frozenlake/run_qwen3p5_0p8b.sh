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

# Install dependencies if running outside an already-configured virtualenv:
#   pip install -e '.[frozenlake]'
#
# Default topology requires >=5 TPU chips (`(4,1)` actor + `(1,1)` TP=1 vLLM
# rollout). On a 4-chip host, pass `actor_model_config.mesh.shape="(2,1)"`.
#
# To run Qwen3.5-2B on the same TP=1 rollout recipe, override the model fields:
#   ./examples/frozenlake/run_qwen3p5_0p8b.sh \
#     model_config.model_name=Qwen3.5-2B \
#     model_config.model_id=Qwen/Qwen3.5-2B \
#     tokenizer_config.tokenizer_path=Qwen/Qwen3.5-2B \
#     vllm_config.model_version=Qwen/Qwen3.5-2B

set -Eeuo pipefail
set -x

batch_size=${batch_size:-64}
num_batches=${num_batches:-5}

echo "Using parameters:"
echo "  Batch Size: $batch_size"
echo "  Num Batches: $num_batches"

python3 -m tunix.cli.grpo_main \
  tunix/cli/base_agentic_config.yaml \
  override_config_file=examples/frozenlake/configs/qwen3p5_0p8b.yaml \
  batch_size="$batch_size" \
  num_batches="$num_batches" \
  "$@"
