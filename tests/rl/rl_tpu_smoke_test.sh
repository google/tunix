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

set -ex

echo "=== Running Non-Agentic RL Smoke Test (Vanilla Rollout) ==="
model_name="Qwen2.5-0.5B" \
num_batches=1 \
bash examples/rl/grpo/gsm8k/run_qwen3_simplereward.sh \
  model_config.mesh.shape="(4,2)" \
  actor_model_config.mesh.shape="(4,2)" \
  rollout_config.total_generation_steps=64

echo "=== Running Agentic RL Smoke Test (vLLM Rollout) ==="
model_name="Qwen2.5-0.5B" \
model_id="Qwen/Qwen2.5-0.5B" \
num_batches=1 \
total_tpus=8 \
train_mesh="(4,1)" \
rollout_mesh="(1,4)" \
checkpoint_dir="/tmp/rl_smoke_agentic_ckpts" \
bash examples/rl/grpo/gsm8k/run_qwen3_8b_disagg.sh \
  rollout_config.total_generation_steps=64
