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
#   ./eval_checkpoint_397B_v7.sh gs://path_to_checkpoint [--dry-run] [--require-target]
#   ./eval_checkpoint_397B_v7.sh stop_eval
#
# Behavior:
# - Evaluates Qwen3.5-397B-A17B on 256 TPU v7x chips by default
#   (16 replicas x 16 chips tpu7x:2x2x4, BATCH_SIZE=64, MAX_CONCURRENCY=256,
#   MAX_WARMPOOL_REPLICAS=4, HEAD_NODEPOOL=sandbox-np so 0 cpu-np nodes are used).
# - Auto-detects cluster/pod (us-east1 -> pod2, us-central1 -> pod1).
# - Auto-discovers <run>/mllog/eval_checkpoints.jsonl when present so
#   eval_start, eval_accuracy, eval_stop, and run_stop are appended to the
#   training run's MLLOG file (seed_<seed>.out).
# - Blocks until the evaluation JobSet completes, tears down the JobSets and
#   sandboxes, and prints the evaluation metrics to stdout.
# ==============================================================================

usage() {
  cat >&2 <<EOF
Usage: $(basename "$0") <gs://path_to_checkpoint> [--dry-run] [--require-target]
       $(basename "$0") stop_eval

Arguments:
  <gs://path_to_checkpoint>  GCS path to checkpoint (e.g. gs://.../checkpoints/7/model_params
                             or gs://.../checkpoints/7)
  --dry-run, --render        Render K8s JobSet YAML without submitting to the cluster
  --require-target           Exit with code 2 if evaluation completes with target_reached=false
  stop, stop_eval            Tear down running evaluation JobSets and sandboxes
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${1:-}" == "stop" || "${1:-}" == "stop_eval" ]]; then
  if [[ -z "${POD:-}" && -z "${REGION:-}" ]]; then
    _ctx="$(kubectl config current-context 2>/dev/null || true)"
    if [[ "${_ctx}" == *us-central1* ]]; then
      export POD="pod1"
    else
      export POD="pod2"
    fi
  fi
  export ROLLOUT_REPLICAS="${ROLLOUT_REPLICAS:-32}"
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
if [[ -z "${CKPT_PATH}" ]]; then
  echo "Error: checkpoint path is required." >&2
  usage
  exit 1
fi

# Strip trailing slash; if given a step directory (.../<step>), append /model_params.
CKPT_PATH="${CKPT_PATH%/}"
if [[ "${CKPT_PATH}" =~ /[0-9]+$ ]]; then
  CKPT_PATH="${CKPT_PATH}/model_params"
fi
export MAXTEXT_CKPT="${CKPT_PATH}"

# ==============================================================================
# Auto-detect Pod / Region / Cluster from Checkpoint URI or Kubectl Context
# ==============================================================================
if [[ -z "${POD:-}" && -z "${REGION:-}" ]]; then
  if [[ "${MAXTEXT_CKPT}" == *us-east1* ]]; then
    export POD="pod2"
  elif [[ "${MAXTEXT_CKPT}" == *us-central1* ]]; then
    export POD="pod1"
  else
    _ctx="$(kubectl config current-context 2>/dev/null || true)"
    if [[ "${_ctx}" == *us-central1* ]]; then
      export POD="pod1"
    else
      export POD="pod2"
    fi
  fi
fi

if [[ "${POD:-pod2}" == "pod2" || "${POD:-}" == "2" || "${POD:-}" == "elm" || "${REGION:-}" == us-east1* ]]; then
  export IMAGE_REWRITE_PREFIX="${IMAGE_REWRITE_PREFIX:-us-east1-docker.pkg.dev/cloud-tpu-multipod-dev/tunix/}"
fi

# ==============================================================================
# 256 TPU v7x Chip Defaults (16 replicas x 16 chips tpu7x:2x2x4 = 256 chips)
# ==============================================================================
export ROLLOUT_TPU_SLICE="${ROLLOUT_TPU_SLICE:-tpu7x:2x2x4}"
export ROLLOUT_REPLICAS="${ROLLOUT_REPLICAS:-16}"
export NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
export BATCH_SIZE="${BATCH_SIZE:-64}"
export MAX_CONCURRENCY="${MAX_CONCURRENCY:-$(( ROLLOUT_REPLICAS * 16 ))}"
export MAX_WARMPOOL_REPLICAS="${MAX_WARMPOOL_REPLICAS:-${NUM_GENERATIONS}}"
export HEAD_NODEPOOL="${HEAD_NODEPOOL:-sandbox-np}"

# ==============================================================================
# Auto-discover Manifest Metadata (<run>/mllog/eval_checkpoints.jsonl)
# ==============================================================================
# Both base resharded checkpoints (.../0/items) and 397B MaxText RL checkpoints
# (.../checkpoints/<step>/model_params) use scanned layers (SCAN_LAYERS=true).
export SCAN_LAYERS="${SCAN_LAYERS:-true}"

CKPT_RUN_TAG="single_ckpt"
if [[ "${MAXTEXT_CKPT}" =~ ^(.*)/([0-9]+)/model_params$ ]]; then
  CKPT_ROOT="${BASH_REMATCH[1]}"
  DETECTED_STEP="${BASH_REMATCH[2]}"
  _run_dir="${CKPT_ROOT%/checkpoints}"
  CKPT_RUN_TAG="$(basename "${_run_dir}")"
  CKPT_RUN_TAG="${CKPT_RUN_TAG%-checkpoints}"
  CKPT_RUN_TAG="${CKPT_RUN_TAG%-train}"
  export CHECKPOINT_STEP="${CHECKPOINT_STEP:-${DETECTED_STEP}}"

  if MANIFEST_META="$(python3 - "${CHECKPOINT_MANIFEST_FILE:-}" "${CKPT_ROOT}" "${CHECKPOINT_STEP}" "${MAXTEXT_CKPT}" <<'PY'
import json
import os
import subprocess
import sys

explicit_manifest, ckpt_root, target_step_str, ckpt_path = sys.argv[1:5]
target_step = int(target_step_str)
ckpt_root = ckpt_root.rstrip("/")

candidates = []
if explicit_manifest:
  candidates.append(explicit_manifest)
else:
  if ckpt_root.endswith("-checkpoints"):
    candidates.append(f"{ckpt_root[:-len('-checkpoints')]}/mllog/eval_checkpoints.jsonl")
  if ckpt_root.endswith("/checkpoints"):
    train_dir = ckpt_root[:-len("/checkpoints")]
    parent_dir = os.path.dirname(train_dir)
    if parent_dir:
      candidates.append(f"{parent_dir}/mllog/eval_checkpoints.jsonl")
    candidates.append(f"{train_dir}/mllog/eval_checkpoints.jsonl")
  candidates.append(f"{ckpt_root}/mllog/eval_checkpoints.jsonl")

text = ""
used_manifest = ""
for path in dict.fromkeys(candidates):
  if path.startswith("gs://"):
    res = subprocess.run(
        ["gsutil", "cat", path],
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode == 0 and res.stdout.strip():
      text = res.stdout
      used_manifest = path
      break
  else:
    try:
      with open(path, encoding="utf-8") as f:
        content = f.read()
      if content.strip():
        text = content
        used_manifest = path
        break
    except OSError:
      continue

if not text:
  sys.exit(1)

records = [json.loads(line) for line in text.splitlines() if line.strip()]
if not records:
  sys.exit(1)

max_step = max(int(r["step"]) for r in records)
matched = None
for r in records:
  if int(r["step"]) == target_step or str(r["checkpoint_path"]).rstrip("/") == ckpt_path.rstrip("/"):
    matched = r
    break

if matched is None:
  sys.exit(1)

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
    if [[ -n "${_M_CKPT}" && "${_M_CKPT}" != "${MAXTEXT_CKPT}" ]]; then
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

if [[ "${RCP_LOGGING:-auto}" == "auto" ]]; then
  export RCP_LOGGING="false"
fi

USER_EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-}"
USER_ROLLOUT_JOBSET_YAML="${ROLLOUT_JOBSET_YAML:-}"

# Unset CHECKPOINT_MANIFEST_FILE before sourcing mlperf_397b_v7x_eval.sh so
# mlperf_base.sh does not enter the multi-checkpoint manifest loop.
unset CHECKPOINT_MANIFEST_FILE

# Load shared 397B v7x and MLPerf base configuration without launching yet.
# Redirect stdout to stderr so `kubectl config use-context` messages do not
# pollute stdout metrics/YAML output.
MLPERF_NO_LAUNCH=1 source "${SCRIPT_DIR}/mlperf_397b_v7x_eval.sh" >&2

# eval_worker.py uses Pathways (JAX_PLATFORMS=proxy, VLLM_TPU_USING_PATHWAYS=1)
# to expose all 32 devices across the 4-host tpu7x:2x2x4 slice; override
# mlperf_397b_v7x_eval.sh's unconditional jobset.mcjax.ray.yaml unless the caller
# explicitly set ROLLOUT_JOBSET_YAML.
export ROLLOUT_JOBSET_YAML="${USER_ROLLOUT_JOBSET_YAML:-jobset.pathways.yaml}"

# Default EVAL_OUTPUT_DIR to a checkpoint-specific subpath so summary.json files
# from different runs/steps are cleanly isolated.
if [[ -z "${USER_EVAL_OUTPUT_DIR}" ]]; then
  export EVAL_OUTPUT_DIR="${BUCKET}/eval_results/${JOB_PREFIX}/${CKPT_RUN_TAG}/step_${CHECKPOINT_STEP:-0}"
fi

_slice_dims="${ROLLOUT_TPU_SLICE#*:}"
_chips_per_replica=$(( ${_slice_dims//x/*} ))
_total_chips=$(( ROLLOUT_REPLICAS * _chips_per_replica ))

echo "[eval_checkpoint_397B_v7] Checkpoint:      ${MAXTEXT_CKPT}" >&2
echo "[eval_checkpoint_397B_v7] Cluster / Pod:   ${CLUSTER} (${REGION}, ${POD})" >&2
echo "[eval_checkpoint_397B_v7] Topology:        ${ROLLOUT_REPLICAS}x ${ROLLOUT_TPU_SLICE} (${_total_chips} v7x chips, head_nodepool=${HEAD_NODEPOOL}), batch_size=${BATCH_SIZE}, max_concurrency=${MAX_CONCURRENCY}, max_warmpool_replicas=${MAX_WARMPOOL_REPLICAS}" >&2
echo "[eval_checkpoint_397B_v7] Output Dir:      ${EVAL_OUTPUT_DIR}" >&2

if [[ "${DRY_RUN_MODE}" == "true" || "${DRY_RUN:-false}" == "true" ]]; then
  exec "${LAUNCHER}" --command eval --image "${TUNIX_IMAGE}" "${EXTRA_ARGS[@]}"
fi

list_summaries() {
  local out_dir="${EVAL_OUTPUT_DIR%/}"
  if [[ "${out_dir}" == gs://* ]]; then
    gsutil ls "${out_dir}/*/summary.json" 2>/dev/null | sort || true
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
