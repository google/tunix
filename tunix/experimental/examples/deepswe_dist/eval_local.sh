#!/usr/bin/env bash
# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Four-chip Qwen3-1.7B smoke eval. Extra CLI arguments override these defaults.
set -euo pipefail
DEEPSWE_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEEPSWE_REPO_ROOT="$(cd -- "$DEEPSWE_SCRIPT_DIR/../../../.." && pwd)"
: "${MODEL_PATH:?Set MODEL_PATH to the Qwen3-1.7B HF/tokenizer directory}"
: "${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to the matching MaxText Orbax 0/items directory}"
export PYTHONPATH="$DEEPSWE_REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export JAX_PLATFORMS=tpu,cpu
unset EVAL_JOBSET_NAME JAX_BACKEND_TARGET VLLM_TPU_USING_PATHWAYS

DEEPSWE_DATASET_ARGS=()
if [[ -n "${DATASET_PATH:-}" ]]; then
  DEEPSWE_DATASET_ARGS+=(--dataset_path "$DATASET_PATH")
fi
exec "${PYTHON:-python}" -u "$DEEPSWE_SCRIPT_DIR/eval_launcher.py" \
  --model_id Qwen/Qwen3-1.7B --tokenizer_path "$MODEL_PATH" \
  --model_absolute_path "$CHECKPOINT_PATH" \
  --maxtext_model_name qwen3-1.7b --scan_layers false \
  --checkpoint_storage_use_ocdbt true --checkpoint_storage_use_zarr3 true \
  --checkpoint_storage_concurrent_gb 4 --use_ocdbt_with_pathways false \
  --mesh_fsdp 1 --mesh_tp "${TPU_CHIPS:-4}" \
  --use_agent_sandbox false --scaffold r2egym \
  --dataset_name R2E-Gym/R2E-Gym-Subset --dataset_split train \
  --tasks_limit 1 --num_rollouts_per_instance 1 --max_concurrent 1 \
  --vllm_max_num_seqs 1 --vllm_max_num_batched_tokens 4096 \
  --vllm_utilization 0.5 --max_model_len 8192 \
  --max_response_length 1024 --max_steps 2 --enable_thinking false \
  --output_dir "${OUTPUT_DIR:-$DEEPSWE_REPO_ROOT/artifacts/deepswe_eval_qwen3_1p7b}" \
  "${DEEPSWE_DATASET_ARGS[@]}" "$@"
