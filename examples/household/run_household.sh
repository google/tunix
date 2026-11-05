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

set -euo pipefail

MODEL_ID=${1:-"google/gemma-4-E2B-it"}
BATCH_SIZE=${2:-16}
NUM_BATCHES=${3:-100}
shift $(( $# >= 3 ? 3 : $# ))

echo "=== Running Household Task Exploration GRPO Training ==="
echo "Model: ${MODEL_ID}"
echo "Batch size: ${BATCH_SIZE}"
echo "Batches: ${NUM_BATCHES}"

PYTHONPATH="${PYTHONPATH:-.}" python3 -m examples.household.train_household \
  --model_id="${MODEL_ID}" \
  --batch_size="${BATCH_SIZE}" \
  --num_batches="${NUM_BATCHES}" \
  "$@"
