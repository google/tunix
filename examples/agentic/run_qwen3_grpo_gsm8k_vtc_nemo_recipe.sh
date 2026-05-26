#!/usr/bin/env bash
# Reproduce the NeMo VTC-style GSM8K GRPO recipe with the Tunix agentic demo.
#
# This launcher keeps the Tunix-side defaults aligned with the reference setup:
#   - Qwen3-1.7B
#   - GSM8K train/test
#   - VTC prompt + 0/0.1/0.5/1.0 reward
#   - 4 prompts/step x 8 generations
#   - effective 16-sequence optimizer batch
#   - beta=0.04, lr=2e-7, wd=0.01, max_grad_norm=1.0
#   - train-only by default; pass --enable_eval to turn validation back on

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_STEPS="${MAX_STEPS:-200}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-2}"
TRAIN_MICRO_BATCH_SIZE="${TRAIN_MICRO_BATCH_SIZE:-1}"
COMPUTE_LOGPS_MICRO_BATCH_SIZE="${COMPUTE_LOGPS_MICRO_BATCH_SIZE:-1}"
ROLLOUT_SPLIT_FRACTION="${ROLLOUT_SPLIT_FRACTION:-1.0}"
ROLLOUT_VLLM_HBM_UTILIZATION="${ROLLOUT_VLLM_HBM_UTILIZATION:-0.6}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "WARNING: HF_TOKEN is not set. This is fine if Qwen3-1.7B is already cached locally." >&2
fi

cmd=(
  python3
  "${REPO_ROOT}/examples/agentic/qwen3_grpo_gsm8k_vtc_demo.py"
  --batch_size "${BATCH_SIZE}"
  --max_steps "${MAX_STEPS}"
  --max_response_length "${MAX_RESPONSE_LENGTH}"
  --mini_batch_size "${MINI_BATCH_SIZE}"
  --train_micro_batch_size "${TRAIN_MICRO_BATCH_SIZE}"
  --compute_logps_micro_batch_size "${COMPUTE_LOGPS_MICRO_BATCH_SIZE}"
  --rollout_split_fraction "${ROLLOUT_SPLIT_FRACTION}"
  --rollout_vllm_hbm_utilization "${ROLLOUT_VLLM_HBM_UTILIZATION}"
)

for opt in rollout_mesh_fsdp rollout_mesh_tp train_mesh_fsdp train_mesh_tp train_mesh_sp \
  rollout_vllm_max_num_seqs rollout_vllm_max_num_batched_tokens; do
  val="${!opt:-}"
  if [[ -n "${val}" ]]; then
    cmd+=("--${opt}" "${val}")
  fi
done

cd "${REPO_ROOT}"
"${cmd[@]}" "$@"
