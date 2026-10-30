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

# Two-step sampled-token OPD on the existing native Tunix GSM8K smoke layout.
# The fixed SGLang teacher must already be running on separate resources.
set -Ee

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${OPD_TEACHER_URL:?Set OPD_TEACHER_URL to the running SGLang teacher}"
: "${OPD_TEACHER_TOKENIZER_PATH:?Set the exact teacher tokenizer path or HF ID}"
export OPD_TEACHER_URL OPD_TEACHER_TOKENIZER_PATH
export OPD_COEFFICIENT="${OPD_COEFFICIENT:-1.0}"
export OPD_TASK_REWARD_COEFFICIENT="${OPD_TASK_REWARD_COEFFICIENT:-0.0}"
export TEMPERATURE="${TEMPERATURE:-1.0}"
export TOP_P="${TOP_P:-1.0}"
export TOP_K="${TOP_K:--1}"
export USE_ROLLOUT_LOGPS="${USE_ROLLOUT_LOGPS:-true}"

# Reuse the model, chip layout, optimizer, data subset, sync and checkpoint
# defaults rather than maintaining a second copy of the smoke recipe.
exec bash "${DIR}/ci_smoke_gsm8k_qwen3_0p6b.sh" "$@"
