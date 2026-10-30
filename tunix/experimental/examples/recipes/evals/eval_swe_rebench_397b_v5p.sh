#!/bin/bash
set -e

# ==============================================================================
# SWE-rebench-V2 / Harbor Dataset Evaluation Launcher: Qwen3.5-397B-A17B on TPU v5p
# ==============================================================================
# Evaluates how many instances of a dataset (default: swe-rebench-v2) pass the
# 397B model on TPU v5p (bodaborg-v5p-nap in europe-west4) using late-bound
# Harbor verification (/tests/test.sh -> /logs/verifier/reward.txt).
#
# Usage:
#   # Dry-run (inspect K8s JobSet YAML without launching):
#   DRY_RUN=true bash tunix/experimental/examples/recipes/evals/eval_swe_rebench_397b_v5p.sh
#
#   # Smoke test on 4 tasks (1 rollout per task):
#   TASKS_LIMIT=4 NUM_GENERATIONS=1 \
#     bash tunix/experimental/examples/recipes/evals/eval_swe_rebench_397b_v5p.sh
#
#   # Full evaluation on swe_rebench_v2_python_codeA.jsonl (4 rollouts per task):
#   NUM_GENERATIONS=4 \
#     bash tunix/experimental/examples/recipes/evals/eval_swe_rebench_397b_v5p.sh
#
#   # Stop a running evaluation job:
#   bash tunix/experimental/examples/recipes/evals/eval_swe_rebench_397b_v5p.sh stop_eval
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPES_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${RECIPES_DIR}/../../../.." && pwd)"

export JOB_PREFIX="${JOB_PREFIX:-${USER}-rebench}"
export EVAL_JOBSET_NAME="${EVAL_JOBSET_NAME:-${JOB_PREFIX}-eval}"
export REGION="${REGION:-europe-west4}"
export CLUSTER="${CLUSTER:-bodaborg-v5p-nap}"
export K8S_NAMESPACE="${K8S_NAMESPACE:-trellis}"
export BUCKET="${BUCKET:-gs://atwigg-trellis-europe-west4-dev}"
export EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-gs://wuhao-storage-europe-west4/eval_results/swe_rebench_v2_qwen35_397b_remaining5243_k8_max80}"
export EVAL_MIRROR_OUTPUT_DIR="${EVAL_MIRROR_OUTPUT_DIR:-${BUCKET}/eval_results/${JOB_PREFIX}}"
export EVAL_RUN_ID="${EVAL_RUN_ID:-run_remaining5243_k8_max80}"

# Default 397B checkpoint (europe-west4) & SWE-rebench-v2 remaining 5,243-problem dataset (europe-west4)
export MAXTEXT_CKPT="${MAXTEXT_CKPT:-gs://sanbao-europe/qwen35_397b/scanned_reshard_fsdp32_tp2/0/items}"
export DATASET_PATH="${DATASET_PATH:-gs://wuhao-storage-europe-west4/datasets/swe_rebench_v2/swe_rebench_v2_python_codeA_remaining5243.jsonl}"
export DATASET_SPLIT="${DATASET_SPLIT:-validation}"

# SWE-rebench-v2 Docker images live on Docker Hub (docker.io/swerebenchv2/...);
# disable GAR prefix rewriting and pre-pulling unless explicitly overridden.
export IMAGE_REWRITE_PREFIX="${IMAGE_REWRITE_PREFIX-}"
export PREPULL_SANDBOX_IMAGES="${PREPULL_SANDBOX_IMAGES:-false}"
export PREPULL_IMAGES_FILE="${PREPULL_IMAGES_FILE-}"
export PREWARM_IMAGES_FILE="${PREWARM_IMAGES_FILE-}"
export SANDBOX_IMAGE_PULL_POLICY="${SANDBOX_IMAGE_PULL_POLICY:-IfNotPresent}"
export SANDBOX_TOLERATIONS="${SANDBOX_TOLERATIONS-}"

# Evaluation & Sampling Parameters
export NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
export ROLLOUT_REPLICAS="${ROLLOUT_REPLICAS:-16}"
export BATCH_SIZE="${BATCH_SIZE:-32}"
export TASKS_LIMIT="${TASKS_LIMIT:-5243}"
export TEMPERATURE="${TEMPERATURE:-1.0}"
export TOP_P="${TOP_P:-0.95}"
export MAX_TURNS="${MAX_TURNS:-80}"
export STEP_TIMEOUT_SECS="${STEP_TIMEOUT_SECS:-120}"
export REWARD_TIMEOUT_SECS="${REWARD_TIMEOUT_SECS:-300}"
export EPISODE_TIMEOUT_SECS="${EPISODE_TIMEOUT_SECS:-2400}"
export MAX_WARMPOOL_REPLICAS="${MAX_WARMPOOL_REPLICAS:-16}"
export MAX_CONCURRENCY="${MAX_CONCURRENCY:-$(( 16 * ROLLOUT_REPLICAS ))}"
export ENABLE_THINKING="${ENABLE_THINKING:-false}"
export OPENHANDS_MULTI_TOOL_CALLS="${OPENHANDS_MULTI_TOOL_CALLS:-true}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-16384}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-49152}"

# Sync local Harbor/SWE-rebench-v2 evaluation modules into /app on the K8s pod
# so new dataset/verifier support works without rebuilding TUNIX_IMAGE.
export SYNC_LOCAL_DEEPSWE_CODE="${SYNC_LOCAL_DEEPSWE_CODE:-true}"
if [[ "${SYNC_LOCAL_DEEPSWE_CODE}" == "true" && -z "${PRE_EVAL_CMD:-}" && "${1:-eval}" == "eval" ]]; then
  _BASE_IMAGE_COMMIT="${BASE_IMAGE_COMMIT:-62f48fcf547796e51b45973d24cc6b72b47b7ee8}"
  mapfile -t _OVERLAY_FILES < <(
    cd "${REPO_ROOT}" && {
      git diff --name-only --diff-filter=ACMRT "${_BASE_IMAGE_COMMIT}" -- examples/ tunix/ 2>/dev/null || true
      git ls-files --others --exclude-standard -- examples/ tunix/ 2>/dev/null || true
      printf '%s\n' \
        examples/deepswe/deepswe_data.py \
        examples/deepswe/harbor_utils.py \
        examples/deepswe/opencode_fuzzy.py \
        examples/deepswe/openhands_utils.py \
        examples/deepswe/sandbox_utils.py \
        examples/deepswe/score_tests.py \
        examples/deepswe/swe_agent.py \
        examples/deepswe/swe_env.py \
        examples/deepswe/template.py \
        tunix/generate/utils.py \
        tunix/generate/vllm_sampler.py \
        tunix/utils/maxtext_utils.py \
        tunix/experimental/examples/deepswe_dist/deepswe.py \
        tunix/experimental/examples/deepswe_dist/eval_deepswe.py \
        tunix/experimental/examples/deepswe_dist/eval_worker.py
    } | sort -u
  )
  unset _BASE_IMAGE_COMMIT
  if [[ "${DRY_RUN:-false}" == "true" ]]; then
    export PRE_EVAL_CMD="export EVAL_RUN_ID='${EVAL_RUN_ID}' EVAL_MIRROR_OUTPUT_DIR='${EVAL_MIRROR_OUTPUT_DIR}'; echo '[dry-run] sync local deepswe code to /app'"
  else
    _OVERLAY_HASH="$(cd "${REPO_ROOT}" && sha256sum "${_OVERLAY_FILES[@]}" | sha256sum | cut -c1-12)"
    _OVERLAY_GCS="${BUCKET}/code_overlays/${JOB_PREFIX}/deepswe_overlay_${_OVERLAY_HASH}.tar.gz"
    _TMP_TAR="$(mktemp /tmp/deepswe_overlay_XXXXXX.tar.gz)"
    tar -czf "${_TMP_TAR}" -C "${REPO_ROOT}" "${_OVERLAY_FILES[@]}"
    if ! gcloud storage cp --quiet "${_TMP_TAR}" "${_OVERLAY_GCS}" 2>/dev/null; then
      _POD_FOR_UPLOAD="$(kubectl get pods -n "${K8S_NAMESPACE}" --field-selector=status.phase=Running -o name 2>/dev/null | grep "proc-0-0" | head -n 1 | sed 's|pod/||')"
      if [[ -n "${_POD_FOR_UPLOAD}" ]]; then
        kubectl exec -i -n "${K8S_NAMESPACE}" "${_POD_FOR_UPLOAD}" -c main -- python3 -c "import sys, fsspec; fsspec.open('${_OVERLAY_GCS}', 'wb').open().write(sys.stdin.buffer.read())" < "${_TMP_TAR}"
      else
        echo "ERROR: Failed to upload overlay to ${_OVERLAY_GCS}" >&2
        rm -f "${_TMP_TAR}"
        exit 1
      fi
      unset _POD_FOR_UPLOAD
    fi
    rm -f "${_TMP_TAR}"
    export PRE_EVAL_CMD="export EVAL_RUN_ID='${EVAL_RUN_ID}' EVAL_MIRROR_OUTPUT_DIR='${EVAL_MIRROR_OUTPUT_DIR}'; python3 -c \"import fsspec, tarfile, io; data = fsspec.open('${_OVERLAY_GCS}', 'rb').open().read(); tarfile.open(fileobj=io.BytesIO(data), mode='r:gz').extractall('/app')\""
  fi
  unset _OVERLAY_FILES _OVERLAY_HASH _OVERLAY_GCS _TMP_TAR
fi

COMMAND="${1:-eval}"
WAIT_FOR_COMPLETION="${WAIT_FOR_COMPLETION:-false}"

# Dispatch to mlperf_397b_v5p_eval.sh
if [[ "${WAIT_FOR_COMPLETION}" != "true" || "${DRY_RUN:-false}" == "true" || "${COMMAND}" != "eval" ]]; then
  exec bash "${SCRIPT_DIR}/mlperf_397b_v5p_eval.sh" "$@"
fi

bash "${SCRIPT_DIR}/mlperf_397b_v5p_eval.sh" "$@"

HEAD_JOBSET="${EVAL_JOBSET_NAME}"
if [[ "${ROLLOUT_REPLICAS:-1}" -gt 1 ]]; then
  HEAD_JOBSET="${EVAL_JOBSET_NAME}-0"
fi

echo "Waiting for evaluation JobSet ${HEAD_JOBSET} in namespace ${K8S_NAMESPACE}..."
while true; do
  if ! kubectl get jobset "${HEAD_JOBSET}" -n "${K8S_NAMESPACE}" &>/dev/null; then
    echo "JobSet ${HEAD_JOBSET} no longer exists."
    break
  fi
  COMPLETED="$(kubectl get jobset "${HEAD_JOBSET}" -n "${K8S_NAMESPACE}" -o jsonpath='{.status.conditions[?(@.type=="Completed")].status}' 2>/dev/null || true)"
  FAILED="$(kubectl get jobset "${HEAD_JOBSET}" -n "${K8S_NAMESPACE}" -o jsonpath='{.status.conditions[?(@.type=="Failed")].status}' 2>/dev/null || true)"
  if [[ "${COMPLETED}" == "True" ]]; then
    echo "Evaluation JobSet ${HEAD_JOBSET} completed!"
    break
  fi
  if [[ "${FAILED}" == "True" ]]; then
    echo "Evaluation JobSet ${HEAD_JOBSET} failed!" >&2
    exit 1
  fi
  sleep 30
done

echo "=============================================================================="
echo "Evaluation Summary (${EVAL_OUTPUT_DIR}/summary.json):"
echo "=============================================================================="
gcloud storage cat "${EVAL_OUTPUT_DIR}/summary.json" || true
echo ""
