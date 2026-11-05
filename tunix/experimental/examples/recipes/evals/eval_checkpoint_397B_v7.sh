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

set -e

# ==============================================================================
# Single-checkpoint MLPerf DeepSWE evaluation on TPU v7x (397B)
# ==============================================================================
# Usage:
#   ./eval_checkpoint_397B_v7.sh [gs://path_to_checkpoint] [--dry-run] [--require-target]
#   JOB_PREFIX=<run_prefix> [STEP=<step>] ./eval_checkpoint_397B_v7.sh [--dry-run]
#   ./eval_checkpoint_397B_v7.sh stop_eval
#
# Behavior:
# - Inherits rollout worker, model, cluster, and sandbox settings directly from
#   mlperf_gbs1024_submission.sh via mlperf_397b_v7x_eval.sh (128 replicas x
#   8 chips tpu7x:2x2x1 on jobset.tpu.yaml, DP=1, TP=4, EP=8, FP8 rollout;
#   no Pathways, no cpu-np nodes).
# - Auto-discovers <run>/mllog/eval_checkpoints.jsonl when present so
#   eval_start, eval_accuracy, eval_stop, and run_stop are appended to the
#   training run's MLLOG file (seed_<seed>.out).
# - Blocks until the evaluation JobSet completes, tears down the JobSets and
#   sandboxes, and prints the evaluation metrics to stdout.
# ==============================================================================

usage() {
  cat >&2 <<EOF
Usage: $(basename "$0") [gs://path_to_checkpoint] [--dry-run] [--require-target]
       JOB_PREFIX=<run_prefix> [STEP=<step>] $(basename "$0") [--dry-run] [--require-target]
       $(basename "$0") stop_eval

Arguments:
  <gs://path_to_checkpoint>  GCS path to checkpoint (e.g. gs://.../checkpoints/7/model_params
                             or gs://.../checkpoints/7). Optional if JOB_PREFIX (and STEP /
                             CHECKPOINT_STEP or eval_checkpoints.jsonl) is provided.
  --dry-run, --render        Render K8s JobSet YAML without submitting to the cluster
  --require-target           Exit with code 2 if evaluation completes with target_reached=false
  stop, stop_eval            Tear down running evaluation JobSets and sandboxes
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${1:-}" == "stop" || "${1:-}" == "stop_eval" ]]; then
  exec bash "${SCRIPT_DIR}/mlperf_397b_v7x_eval.sh" stop_eval "${@:2}"
fi

CKPT_ARG=""
REQUIRE_TARGET="false"
DRY_RUN_MODE="false"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --require-target|--require_target)
      REQUIRE_TARGET="true"
      shift
      ;;
    --dry-run|--dry_run|--render)
      DRY_RUN_MODE="true"
      EXTRA_ARGS+=("$1")
      shift
      ;;
    --*)
      EXTRA_ARGS+=("$1")
      shift
      ;;
    *)
      if [[ -z "${CKPT_ARG}" ]]; then
        CKPT_ARG="$1"
      else
        EXTRA_ARGS+=("$1")
      fi
      shift
      ;;
  esac
done

CKPT_PATH="${CKPT_ARG:-${MAXTEXT_CKPT:-}}"
if [[ -n "${CKPT_PATH}" ]]; then
  # Strip trailing slash; if given a step directory (.../<step>), append /model_params.
  CKPT_PATH="${CKPT_PATH%/}"
  if [[ "${CKPT_PATH}" =~ /[0-9]+$ ]]; then
    CKPT_PATH="${CKPT_PATH}/model_params"
  fi
  export MAXTEXT_CKPT="${CKPT_PATH}"
fi

# If checkpoint URI is in us-east1 and POD/REGION were not set, select pod2;
# otherwise let mlperf_gbs1024_submission.sh set its default pod/cluster.
if [[ -z "${POD:-}" && -z "${REGION:-}" && "${MAXTEXT_CKPT:-}" == *us-east1* ]]; then
  export POD="pod2"
fi

# Infer JOB_PREFIX and CHECKPOINT_STEP from the checkpoint path when not explicitly set.
CKPT_RUN_TAG="single_ckpt"
if [[ "${MAXTEXT_CKPT:-}" =~ ^(.*)/([0-9]+)/model_params$ ]]; then
  CKPT_ROOT="${BASH_REMATCH[1]}"
  DETECTED_STEP="${BASH_REMATCH[2]}"
  _run_dir="${CKPT_ROOT%/checkpoints}"
  CKPT_RUN_TAG="$(basename "${_run_dir}")"
  CKPT_RUN_TAG="${CKPT_RUN_TAG%-checkpoints}"
  CKPT_RUN_TAG="${CKPT_RUN_TAG%-train}"
  export JOB_PREFIX="${JOB_PREFIX:-${CKPT_RUN_TAG}}"
  export CHECKPOINT_STEP="${CHECKPOINT_STEP:-${STEP:-${DETECTED_STEP}}}"
elif [[ -n "${STEP:-}" && -z "${CHECKPOINT_STEP:-}" ]]; then
  export CHECKPOINT_STEP="${STEP}"
fi

USER_EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-}"
USER_MANIFEST_FILE="${CHECKPOINT_MANIFEST_FILE:-}"

# Unset CHECKPOINT_MANIFEST_FILE before sourcing mlperf_397b_v7x_eval.sh so
# mlperf_base.sh does not enter the multi-checkpoint manifest loop.
unset CHECKPOINT_MANIFEST_FILE

# Inherit all rollout worker, model, cluster, and eval settings from
# mlperf_397b_v7x_eval.sh -> mlperf_gbs1024_submission.sh without launching yet.
MLPERF_NO_LAUNCH=1 source "${SCRIPT_DIR}/mlperf_397b_v7x_eval.sh" >&2

# If the caller did not explicitly pass a checkpoint path (positional arg or
# MAXTEXT_CKPT env var), ignore the base-model default set by mlperf_397b_256_v7x.sh
# and resolve the trained checkpoint from the manifest (or JOB_PREFIX / BUCKET).
if [[ -z "${CKPT_PATH}" ]]; then
  export MAXTEXT_CKPT=""
  CKPT_RUN_TAG="${JOB_PREFIX}"
  CKPT_ROOT="${BUCKET%/}/${JOB_PREFIX}-train/checkpoints"
fi

# ==============================================================================
# Auto-discover Manifest Metadata (<run>/mllog/eval_checkpoints.jsonl)
# ==============================================================================
if [[ -n "${CKPT_ROOT:-}" || -n "${MAXTEXT_OUTPUT_DIR:-}" ]]; then
  if MANIFEST_META="$(python3 - "${USER_MANIFEST_FILE}" "${CKPT_ROOT:-}" "${MAXTEXT_OUTPUT_DIR:-}" "${BUCKET:-}" "${JOB_PREFIX:-}" "${CHECKPOINT_STEP:--1}" "${MAXTEXT_CKPT:-}" <<'PY'
import json
import os
import subprocess
import sys

(
    explicit_manifest,
    ckpt_root,
    maxtext_output_dir,
    bucket,
    job_prefix,
    target_step_str,
    ckpt_path,
) = sys.argv[1:8]
target_step = int(target_step_str)
ckpt_root = ckpt_root.rstrip("/")
maxtext_output_dir = maxtext_output_dir.rstrip("/")
bucket = bucket.rstrip("/")

candidates = []
if explicit_manifest:
  candidates.append(explicit_manifest)
else:
  if ckpt_root:
    if ckpt_root.endswith("-checkpoints"):
      candidates.append(f"{ckpt_root[:-len('-checkpoints')]}/mllog/eval_checkpoints.jsonl")
    if ckpt_root.endswith("/checkpoints"):
      train_dir = ckpt_root[:-len("/checkpoints")]
      parent_dir = os.path.dirname(train_dir)
      if parent_dir:
        candidates.append(f"{parent_dir}/mllog/eval_checkpoints.jsonl")
      candidates.append(f"{train_dir}/mllog/eval_checkpoints.jsonl")
    candidates.append(f"{ckpt_root}/mllog/eval_checkpoints.jsonl")
  if maxtext_output_dir:
    candidates.append(f"{maxtext_output_dir}/mllog/eval_checkpoints.jsonl")
  if bucket and job_prefix:
    candidates.append(f"{bucket}/{job_prefix}-train/mllog/eval_checkpoints.jsonl")
    candidates.append(f"{bucket}/maxtext/{job_prefix}/mllog/eval_checkpoints.jsonl")

def read_gcs_or_local(path: str) -> str:
  if path.startswith("gs://"):
    for cmd in (["gcloud", "storage", "cat", path], ["gsutil", "cat", path]):
      res = subprocess.run(cmd, capture_output=True, text=True, check=False)
      if res.returncode == 0 and res.stdout.strip():
        return res.stdout
    return ""
  try:
    with open(path, encoding="utf-8") as f:
      return f.read()
  except OSError:
    return ""

text = ""
used_manifest = ""
for path in dict.fromkeys(candidates):
  content = read_gcs_or_local(path)
  if content.strip():
    text = content
    used_manifest = path
    break

if not text:
  sys.exit(1)

records = [json.loads(line) for line in text.splitlines() if line.strip()]
if not records:
  sys.exit(1)

matched = None
if target_step < 0 and not ckpt_path:
  matched = records[-1]
else:
  if ckpt_path:
    for r in reversed(records):
      if str(r.get("checkpoint_path", "")).rstrip("/") == ckpt_path.rstrip("/"):
        matched = r
        break
  if matched is None and target_step >= 0:
    for r in reversed(records):
      if int(r["step"]) == target_step:
        matched = r
        break

if matched is None:
  sys.exit(1)

matched_root = os.path.dirname(
    os.path.dirname(str(matched.get("checkpoint_path", "")).rstrip("/"))
)
same_run_records = [
    r
    for r in records
    if os.path.dirname(
        os.path.dirname(str(r.get("checkpoint_path", "")).rstrip("/"))
    )
    == matched_root
] or records
max_step = max(int(r["step"]) for r in same_run_records)

mllog_file = str(matched["mllog_file"]) if "mllog_file" in matched else ""
canonical_ckpt = str(matched["checkpoint_path"])
is_last = "true" if int(matched["step"]) == max_step else "false"
print("|".join([
    used_manifest,
    str(int(matched["step"])),
    str(int(matched["samples_count"])),
    str(int(matched["timestamp_ms"])),
    is_last,
    mllog_file,
    canonical_ckpt,
]))
PY
  )"; then
    IFS='|' read -r _M_PATH _M_STEP _M_SAMPLES _M_TS_MS _M_IS_LAST _M_MLLOG _M_CKPT <<< "${MANIFEST_META}"
    if [[ -n "${_M_CKPT}" && "${_M_CKPT}" != "${MAXTEXT_CKPT:-}" ]]; then
      echo "[eval_checkpoint_397B_v7] Resolved canonical checkpoint path from manifest: ${_M_CKPT}" >&2
      export MAXTEXT_CKPT="${_M_CKPT}"
    fi
    export CHECKPOINT_STEP="${_M_STEP}"
    export SAMPLES_COUNT="${SAMPLES_COUNT:-${_M_SAMPLES}}"
    export CHECKPOINT_TIMESTAMP_MS="${_M_TS_MS}"
    export IS_LAST_CHECKPOINT="${IS_LAST_CHECKPOINT:-${_M_IS_LAST}}"
    if [[ "${RCP_LOGGING:-auto}" != "false" ]]; then
      if [[ -n "${_M_MLLOG}" ]]; then
        export METRIC_LOGGER_DIR="${METRIC_LOGGER_DIR:-${_M_MLLOG}}"
      fi
      export RCP_LOGGING="${RCP_LOGGING:-true}"
    fi
    echo "[eval_checkpoint_397B_v7] Loaded manifest metadata from ${_M_PATH}: step=${CHECKPOINT_STEP} samples_count=${SAMPLES_COUNT} timestamp_ms=${CHECKPOINT_TIMESTAMP_MS} is_last=${IS_LAST_CHECKPOINT} mllog=${METRIC_LOGGER_DIR:-none}" >&2
  fi
fi

if [[ -z "${MAXTEXT_CKPT:-}" && -n "${CKPT_ROOT:-}" && -n "${CHECKPOINT_STEP:-}" ]]; then
  export MAXTEXT_CKPT="${CKPT_ROOT%/}/${CHECKPOINT_STEP}/model_params"
fi

if [[ -z "${MAXTEXT_CKPT:-}" ]]; then
  echo "Error: could not resolve MAXTEXT_CKPT. Pass <gs://path_to_checkpoint> or set STEP/CHECKPOINT_STEP." >&2
  usage
  exit 1
fi

if [[ "${RCP_LOGGING:-auto}" == "auto" ]]; then
  export RCP_LOGGING="false"
fi

# Default EVAL_OUTPUT_DIR to a checkpoint-specific subpath so summary.json files
# from different runs/steps are cleanly isolated.
if [[ -z "${USER_EVAL_OUTPUT_DIR}" ]]; then
  export EVAL_OUTPUT_DIR="${BUCKET}/eval_results/${JOB_PREFIX}/${CKPT_RUN_TAG}/step_${CHECKPOINT_STEP:-0}"
fi

echo "[eval_checkpoint_397B_v7] Checkpoint:      ${MAXTEXT_CKPT}" >&2
echo "[eval_checkpoint_397B_v7] Cluster / Pod:   ${CLUSTER} (${REGION}, ${POD})" >&2
echo "[eval_checkpoint_397B_v7] Topology:        ${ROLLOUT_REPLICAS}x ${ROLLOUT_TPU_SLICE} (${ROLLOUT_JOBSET_YAML}), batch_size=${BATCH_SIZE}, max_concurrency=${MAX_CONCURRENCY}" >&2
echo "[eval_checkpoint_397B_v7] Output Dir:      ${EVAL_OUTPUT_DIR}" >&2

if [[ "${DRY_RUN_MODE}" == "true" || "${DRY_RUN:-false}" == "true" ]]; then
  exec "${LAUNCHER}" --command eval --image "${TUNIX_IMAGE}" "${EXTRA_ARGS[@]}"
fi

list_summaries() {
  local out_dir="${EVAL_OUTPUT_DIR%/}"
  if [[ "${out_dir}" == gs://* ]]; then
    (gcloud storage ls "${out_dir}/*/summary.json" 2>/dev/null || gsutil ls "${out_dir}/*/summary.json" 2>/dev/null) | sort || true
  else
    compgen -G "${out_dir}/*/summary.json" | sort || true
  fi
}

BEFORE_SUMMARIES="$(list_summaries)"

cleanup_eval() {
  "${LAUNCHER}" --command stop_eval --image "${TUNIX_IMAGE}" >&2 || true
}

# Clear any leftover JobSet from a prior run with the same EVAL_JOBSET_NAME.
"${LAUNCHER}" --command stop_eval --image "${TUNIX_IMAGE}" &>/dev/null || true

# Ensure JobSets and sandboxes are torn down on any exit or interrupt.
trap cleanup_eval EXIT

echo "[eval_checkpoint_397B_v7] Launching evaluation JobSet ${EVAL_JOBSET_NAME}..." >&2
"${LAUNCHER}" --command eval --image "${TUNIX_IMAGE}" "${EXTRA_ARGS[@]}" >&2

HEAD_JOBSET="${EVAL_JOBSET_NAME}"
if [[ "${ROLLOUT_REPLICAS:-1}" -gt 1 ]]; then
  HEAD_JOBSET="${EVAL_JOBSET_NAME}-0"
fi
PROC_SELECTOR="jobset.sigs.k8s.io/jobset-name=${HEAD_JOBSET},jobset.sigs.k8s.io/replicatedjob-name=proc,batch.kubernetes.io/job-completion-index=0"

echo "[eval_checkpoint_397B_v7] Waiting for ${HEAD_JOBSET} (pod 0 main container) in namespace ${K8S_NAMESPACE:-default}..." >&2
while true; do
  if ! JS_COND="$(kubectl get jobset "${HEAD_JOBSET}" ${K8S_NAMESPACE:+-n "${K8S_NAMESPACE}"} -o jsonpath='{.status.conditions[?(@.status=="True")].type}' 2>&1)"; then
    if [[ "${JS_COND}" == *NotFound* ]]; then
      echo "[eval_checkpoint_397B_v7] JobSet ${HEAD_JOBSET} no longer exists." >&2
      break
    fi
    sleep 10
    continue
  fi
  if [[ "${JS_COND}" =~ (Failed|Completed) ]]; then
    echo "[eval_checkpoint_397B_v7] JobSet ${HEAD_JOBSET} reached terminal condition: ${JS_COND}." >&2
    break
  fi
  POD_STATUS="$(kubectl get pods ${K8S_NAMESPACE:+-n "${K8S_NAMESPACE}"} -l "${PROC_SELECTOR}" \
    -o jsonpath='{.items[0].status.containerStatuses[?(@.name=="main")].state.terminated.exitCode}{"|"}{.items[0].status.containerStatuses[?(@.name=="main")].state.waiting.reason}{"|"}{.items[0].status.containerStatuses[?(@.name=="main")].restartCount}{"|"}{.items[0].status.containerStatuses[?(@.name=="main")].lastState.terminated.exitCode}' 2>/dev/null || true)"
  IFS='|' read -r MAIN_EXIT WAIT_REASON RESTARTS LAST_EXIT <<< "${POD_STATUS}"
  if [[ -n "${MAIN_EXIT}" ]]; then
    echo "[eval_checkpoint_397B_v7] Main evaluation container finished with exit code ${MAIN_EXIT}." >&2
    break
  fi
  if [[ "${WAIT_REASON}" == "CrashLoopBackOff" ]] || [[ "${RESTARTS:-0}" -ge 2 ]]; then
    echo "[eval_checkpoint_397B_v7] Main evaluation container is crashing (reason=${WAIT_REASON}, restarts=${RESTARTS}, lastExit=${LAST_EXIT})." >&2
    break
  fi
  sleep 10
done

# Let the EXIT trap handle cleanup on exit.

AFTER_SUMMARIES="$(list_summaries)"

python3 - "${BEFORE_SUMMARIES}" "${AFTER_SUMMARIES}" "${MAXTEXT_CKPT}" "${REQUIRE_TARGET}" <<'PY'
import json
import subprocess
import sys

before_raw, after_raw, ckpt_path, require_target_str = sys.argv[1:5]
require_target = require_target_str.lower() == "true"
before = {line.strip() for line in before_raw.splitlines() if line.strip()}
after = [line.strip() for line in after_raw.splitlines() if line.strip()]
new_summaries = [p for p in after if p not in before]

if not new_summaries:
  sys.stderr.write(
      f"[eval_checkpoint_397B_v7] ERROR: No new summary.json found for {ckpt_path}.\n"
  )
  sys.exit(1)

summary_uri = new_summaries[-1]
if summary_uri.startswith("gs://"):
  res = subprocess.run(["gcloud", "storage", "cat", summary_uri], capture_output=True, text=True, check=False)
  if res.returncode == 0 and res.stdout.strip():
    text = res.stdout
  else:
    text = subprocess.check_output(["gsutil", "cat", summary_uri], text=True)
else:
  with open(summary_uri, encoding="utf-8") as f:
    text = f.read()

summary = json.loads(text)
pass_at_k = summary["pass_at_k"]
pass_at_1 = pass_at_k.get("1")
if pass_at_1 is None:
  pass_at_1 = summary["avg_at_k"]
pass_at_1 = float(pass_at_1)

pass_at_4 = pass_at_k.get("4")
if pass_at_4 is None:
  pass_at_4 = pass_at_1
pass_at_4 = float(pass_at_4)

instances = int(summary["instances"])
resolved_instances = int(round(pass_at_4 * instances))
resolved_attempts = int(summary["resolved_attempts"])
completed_att = int(summary["completed_attempts"])
expected_att = int(summary["expected_attempts"])
error_att = int(summary["error_attempts"])
complete = bool(summary["complete"])
target_acc = float(summary["target_accuracy"])
target_reached = bool(summary["target_reached"])
step = int(summary["checkpoint_step"])
samples_count = int(summary["samples_count"])

output_metrics = {
    "checkpoint": ckpt_path,
    "summary_uri": summary_uri,
    "resolved_instances": resolved_instances,
    **summary,
}

print(
    f"pass@4={pass_at_4:.4f} ({resolved_instances}/{instances}) "
    f"pass@1={pass_at_1:.4f} ({resolved_attempts}/{expected_att}) "
    f"attempts={completed_att}/{expected_att} "
    f"errors={error_att} "
    f"target_reached={str(target_reached).lower()} "
    f"step={step} samples_count={samples_count}"
)
print(json.dumps(output_metrics, indent=2))

if not complete or error_att > 0:
  sys.stderr.write(
      f"[eval_checkpoint_397B_v7] ERROR: Evaluation incomplete or had errors ({summary_uri}).\n"
  )
  sys.exit(1)

if require_target and not target_reached:
  sys.stderr.write(
      f"[eval_checkpoint_397B_v7] Target accuracy {target_acc} not reached (pass@4={pass_at_4:.4f}).\n"
  )
  sys.exit(2)
PY
