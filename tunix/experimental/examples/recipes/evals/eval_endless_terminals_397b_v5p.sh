#!/bin/bash
set -e

# ==============================================================================
# Endless Terminals (arXiv:2601.16443) Evaluation Launcher: Qwen3.5-397B-A17B on TPU v5p
# ==============================================================================
# Evaluates Qwen3.5-397B-A17B on the Endless Terminals dataset
# (obiwan96/endless-terminals, 2,492 Harbor Linux terminal tasks) on TPU v5p
# (bodaborg-v5p-nap in europe-west4) using in-place Harbor verification
# (/tests/test.sh -> /logs/verifier/reward.txt + score_tests: partial counts).
#
# Usage:
#   # Dry-run (inspect K8s JobSet YAML without launching):
#   DRY_RUN=true bash tunix/experimental/examples/recipes/evals/eval_endless_terminals_397b_v5p.sh
#
#   # Evaluate the official 100-task held-out dev/val split (k=8 rollouts):
#   DATASET_PATH=gs://wuhao-storage-europe-west4/datasets/endless_terminals/endless_terminals_val_100.jsonl \
#   TASKS_LIMIT=100 \
#   EVAL_OUTPUT_DIR=gs://wuhao-storage-europe-west4/eval_results/endless_terminals_qwen35_397b_val100_k8_max80 \
#   EVAL_RUN_ID=run_endless_val100_k8_max80 \
#     bash tunix/experimental/examples/recipes/evals/eval_endless_terminals_397b_v5p.sh
#
#   # Full evaluation on all 2,492 tasks (k=8 rollouts):
#   bash tunix/experimental/examples/recipes/evals/eval_endless_terminals_397b_v5p.sh
#
#   # Stop a running evaluation job:
#   bash tunix/experimental/examples/recipes/evals/eval_endless_terminals_397b_v5p.sh stop_eval
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export JOB_PREFIX="${JOB_PREFIX:-${USER}-endless}"
export EVAL_JOBSET_NAME="${EVAL_JOBSET_NAME:-${JOB_PREFIX}-eval}"
export EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-gs://wuhao-storage-europe-west4/eval_results/endless_terminals_qwen35_397b_all2492_k8_max80}"
export EVAL_RUN_ID="${EVAL_RUN_ID:-run_endless_all2492_k8_max80}"

export DATASET_PATH="${DATASET_PATH:-gs://wuhao-storage-europe-west4/datasets/endless_terminals/endless_terminals_all_2492.jsonl}"
export TASKS_LIMIT="${TASKS_LIMIT:-2492}"
export OPENHANDS_WORKING_DIR="${OPENHANDS_WORKING_DIR:-/home/user}"

exec bash "${SCRIPT_DIR}/eval_swe_rebench_397b_v5p.sh" "$@"
