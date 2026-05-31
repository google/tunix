#!/usr/bin/env bash
# Ablation: checkout each commit from 2c347e05 onwards and run training.
# Usage: bash ablation_run.sh
# Each run logs to its own wandb run (commit hash in the run name).

set -euo pipefail

REPO="/home/haoyugao_google_com/tunix"
SCRIPT="${REPO}/examples/agentic/qwen3_grpo_gsm8k_vtc_demo.py"
TRAIN_ARGS="--shared_mesh_fsdp=1 --shared_mesh_tp=4"

COMMITS=(
  "2c347e05"
  "ae7707ea"
  "b1cfd605"
  "086d9620"
  "90b87821"
  "bf48560f"
  "41185adf"
  "ac1f4984"
  "62235e58"
  "349d2bcb"
)

LOG_DIR="${REPO}/ablation_logs"
mkdir -p "${LOG_DIR}"

cd "${REPO}"

for COMMIT in "${COMMITS[@]}"; do
  echo ""
  echo "========================================"
  echo "Checking out: ${COMMIT}"
  echo "========================================"

  git checkout -f "${COMMIT}"

  LOG_FILE="${LOG_DIR}/${COMMIT}.log"
  echo "Running training for ${COMMIT}, logging to ${LOG_FILE}"

  WANDB_RUN_ID="ablation-${COMMIT}" \
  python3 "${SCRIPT}" ${TRAIN_ARGS} 2>&1 | tee "${LOG_FILE}"

  EXIT_CODE=${PIPESTATUS[0]}
  if [[ ${EXIT_CODE} -ne 0 ]]; then
    echo "Training FAILED for ${COMMIT} (exit code ${EXIT_CODE})" | tee -a "${LOG_FILE}"
  else
    echo "Training DONE for ${COMMIT}" | tee -a "${LOG_FILE}"
  fi
done

echo ""
echo "All commits done."
