#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUNNER="${SCRIPT_DIR}/run_qwen3_grpo_gsm8k_vtc_ablations.py"

echo "Running VTC screening ablations (7 runs, 200 steps each)..."
echo "W&B URLs will be recorded under:"
echo "  ${REPO_ROOT}/artifacts/qwen3_grpo_gsm8k_vtc/ablations/wandb_urls.tsv"

python "${RUNNER}" --stage screening --execute "$@"
