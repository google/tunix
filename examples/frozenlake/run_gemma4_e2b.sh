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

# Install dependencies:
pip install -e .[frozenlake]

set -x # Enable xtrace

batch_size=${batch_size:-64}
num_batches=${num_batches:-5}

echo "Using parameters:"
echo "  Batch Size: $batch_size"
echo "  Num Batches: $num_batches"

python3 -m tunix.cli.grpo_main \
  tunix/cli/base_agentic_config.yaml \
  override_config_file=examples/frozenlake/configs/gemma4_e2b.yaml \
  batch_size=$batch_size \
  num_batches=$num_batches \
  "$@"
