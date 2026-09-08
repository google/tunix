#!/usr/bin/env bash
set -euo pipefail

repo="${V2_FL_REPO:?}"
pkg="$repo/canon-zero-tim"
root="${V2_FL_ROOT:?}"
workload="${V2_FL_WORKLOAD:?}"
arm="${V2_FL_ARM:?}"
geometry="${V2_FL_GEOMETRY:-dp2-tp2}"
dp_size="${V2_FL_DP_SIZE:-2}"
tp_size="${V2_FL_TP_SIZE:-2}"
model_dir="${V2_FL_MODEL_DIR:-qwen8b_tp2}"
profile_rel="${V2_FL_PROFILE_REL:-cluster/profiles/qwen3-8b-dp2-tp2-frozenlake-onehost.env}"
case "$geometry:$dp_size:$tp_size:$model_dir:$profile_rel" in
  "dp4-tp1:4:1:qwen8b_tp1:cluster/profiles/qwen3-8b-dp4-tp1-frozenlake-onehost.env") ;;
  "dp2-tp2:2:2:qwen8b_tp2:cluster/profiles/qwen3-8b-dp2-tp2-frozenlake-onehost.env") ;;
  "dp1-tp4:1:4:qwen8b:cluster/profiles/qwen3-8b-dp1-tp4-frozenlake-onehost.env") ;;
  *) echo "[V2.FL.ONEHOST] unregistered geometry delivery tuple" >&2; exit 2 ;;
esac

case "$workload" in
  p45)
    workload_name="frozenlake-p45-onehost-dp${dp_size}-tp${tp_size}"
    export CANON_P57_WORKLOAD_CANDIDATE=
    export CANON_P57_DATA_SPLIT=
    ;;
  m15)
    workload_name="frozenlake-m15-onehost-dp${dp_size}-tp${tp_size}"
    export CANON_P57_WORKLOAD_CANDIDATE=m15
    export CANON_P57_DATA_SPLIT=main
    ;;
  *) echo "[V2.FL.ONEHOST] unknown workload: $workload" >&2; exit 2 ;;
esac
case "$arm" in
  r0) keep_tape=0; reduce_once=0; length_sort=0; report_buckets=0 ;;
  r0b) keep_tape=0; reduce_once=0; length_sort=0; report_buckets=1; chunk_ticket=0; chunk_backpressure=0 ;;
  r0c) keep_tape=0; reduce_once=0; length_sort=0; report_buckets=1; chunk_ticket=1; chunk_backpressure=0 ;;
  r0d) keep_tape=0; reduce_once=0; length_sort=0; report_buckets=1; chunk_ticket=0; chunk_backpressure=1 ;;
  r1) keep_tape=stream; reduce_once=0; length_sort=0; report_buckets=0 ;;
  r2)
    if [ "$dp_size" = 1 ]; then echo "[V2.FL.ONEHOST] DP1 has no reduce-once arm" >&2; exit 2; fi
    keep_tape=stream; reduce_once=1; length_sort=0; report_buckets=0
    ;;
  r3)
    keep_tape=stream
    if [ "$dp_size" = 1 ]; then reduce_once=0; else reduce_once=1; fi
    length_sort=1; report_buckets=0
    ;;
  *) echo "[V2.FL.ONEHOST] unknown arm: $arm" >&2; exit 2 ;;
esac
chunk_ticket="${chunk_ticket:-0}"
chunk_backpressure="${chunk_backpressure:-0}"
if { [ "$arm" = r0b ] || [ "$arm" = r0c ] || [ "$arm" = r0d ]; } && \
   { [ "$workload" != p45 ] || [ "$geometry" != dp2-tp2 ]; }; then
  echo "[V2.FL.ONEHOST] $arm P75/P76/P77 capacity arm admits only P45 DP2xTP2" >&2
  exit 2
fi
export CANON_PROFILE_FILE="$profile_rel"
export CANON_P32_KEEP_TAPE="$keep_tape"
export CANON_DP_REDUCE_ONCE="$reduce_once"
export CANON_P32_LENGTH_SORT="$length_sort"
if [ "${CANON_P75_REPORT_ADJOINT_BUCKETS:-}" != "$report_buckets" ]; then
  echo "[V2.FL.ONEHOST] outer/inner report-bucket selector mismatch" >&2
  exit 2
fi
export CANON_P75_REPORT_ADJOINT_BUCKETS="$report_buckets"
if [ "${CANON_P76_CHUNK_DEPENDENCY_TICKET:-}" != "$chunk_ticket" ]; then
  echo "[V2.FL.ONEHOST] outer/inner chunk-ticket selector mismatch" >&2
  exit 2
fi
export CANON_P76_CHUNK_DEPENDENCY_TICKET="$chunk_ticket"
if [ "${CANON_P77_CHUNK_BACKPRESSURE:-}" != "$chunk_backpressure" ]; then
  echo "[V2.FL.ONEHOST] outer/inner chunk-backpressure selector mismatch" >&2
  exit 2
fi
export CANON_P77_CHUNK_BACKPRESSURE="$chunk_backpressure"
export CANON_WANDB_RUN_NAME="v2-fl-${workload}-${arm}-${V2_FL_LABEL:?}"
export CANON_P57_RUN_KIND=
export CANON_P57_TIM_ARM=
# shellcheck disable=SC1090
source "$pkg/$profile_rel"
if [ "${CANON_P75_REPORT_ADJOINT_BUCKETS:-}" != "$report_buckets" ]; then
  echo "[V2.FL.ONEHOST] profile changed report-bucket selector" >&2
  exit 2
fi
if [ "${CANON_P76_CHUNK_DEPENDENCY_TICKET:-}" != "$chunk_ticket" ]; then
  echo "[V2.FL.ONEHOST] profile changed chunk-ticket selector" >&2
  exit 2
fi
if [ "${CANON_P77_CHUNK_BACKPRESSURE:-}" != "$chunk_backpressure" ]; then
  echo "[V2.FL.ONEHOST] profile changed chunk-backpressure selector" >&2
  exit 2
fi

python3 "$pkg/tasks/p41-optimizer-residency/scripts/admit_frozenlake_runtime.py" \
  --gymnasium-wheel "${V2_FL_DEPS:?}/gymnasium-1.3.0-py3-none-any.whl" \
  --farama-wheel "${V2_FL_DEPS:?}/farama_notifications-0.0.6-py3-none-any.whl" \
  --report "$root/runtime.json"

python3 - <<'PY'
import jax
devices = jax.devices()
print(f"[V2.FL.ONEHOST] devices={len(devices)} ids={[d.id for d in devices]} backend={jax.default_backend()}", flush=True)
if len(devices) != 4 or jax.default_backend() != "tpu":
  raise SystemExit("FrozenLake one-host carrier requires exactly four TPU devices")
PY

python3 - <<'PY'
import json
import os
from pathlib import Path
from tunix.rl import dp_workloads

short = os.environ["V2_FL_WORKLOAD"]
name = {
    "p45": f"frozenlake-p45-onehost-dp{os.environ['V2_FL_DP_SIZE']}-tp{os.environ['V2_FL_TP_SIZE']}",
    "m15": f"frozenlake-m15-onehost-dp{os.environ['V2_FL_DP_SIZE']}-tp{os.environ['V2_FL_TP_SIZE']}",
}[short]
workload = dp_workloads.get_workload(name)
manifest = {
    "schema": "canon.v2-frozenlake-onehost.run.v1",
    "source_commit": os.environ["V2_FL_SOURCE_SHA"],
    "source_diff_sha256": os.environ["V2_FL_DIFF_SHA"],
    "image_id_sha256": os.environ["V2_FL_IMAGE_SHA"],
    "model_snapshot_sha256": os.environ["V2_FL_MODEL_SHA256"],
    "dataset_train_sha256": os.environ["V2_FL_TRAIN_SHA256"],
    "dataset_test_sha256": os.environ["V2_FL_TEST_SHA256"],
    "runner_sha256": os.environ["V2_FL_RUNNER_SHA256"],
    "label": os.environ["V2_FL_LABEL"],
    "workload": short,
    "workload_name": workload.name,
    "arm": os.environ["V2_FL_ARM"],
    "stage": "backward-no-commit",
    "model_id": workload.model_id,
    "model_dir_name": workload.model_dir_name,
    "topology": {"dp": workload.dp_size, "tp": workload.tp_size, "devices": 4},
    "global_prompts": workload.global_prompts,
    "num_generations": workload.num_generations,
    "global_trajectories": workload.global_trajectories,
    "gradient_groups": workload.gradient_groups,
    "local_m": workload.local_m,
    "global_m": workload.global_m,
    "max_prompt_length": workload.max_prompt_length,
    "max_response_length": workload.max_response_length,
    "max_turns": workload.frozenlake_max_turns,
    "data_shuffle_seed": 42,
    "vllm_global_seed": 0,
    "vllm_hbm_utilization": float(os.environ["FL_VLLM_HBM_UTIL"]),
    "selectors": {
        "keep_tape": os.environ["CANON_P32_KEEP_TAPE"],
        "reduce_once": os.environ["CANON_DP_REDUCE_ONCE"],
        "length_sort": os.environ["CANON_P32_LENGTH_SORT"],
        "report_adjoint_buckets": os.environ[
            "CANON_P75_REPORT_ADJOINT_BUCKETS"
        ],
        "chunk_dependency_ticket": os.environ[
            "CANON_P76_CHUNK_DEPENDENCY_TICKET"
        ],
        "chunk_backpressure": os.environ["CANON_P77_CHUNK_BACKPRESSURE"],
    },
    "reducer_schedule": {
        "kind": "fixed-local-byte-buckets",
        "max_local_bytes": 2 * 1024**3,
    },
    "checked_vma": os.environ["CANON_P66_P59_CHECK_VMA"] == "1",
    "wandb_mode": os.environ["WANDB_MODE"],
    "classification_mode": os.environ["V2_FL_MODE"],
    "xprof": (
        {
            "phase": os.environ["CANON_XPROF_PHASE"],
            "skip_steps": int(os.environ["CANON_XPROF_SKIP_STEPS"]),
            "steps": int(os.environ["CANON_XPROF_STEPS"]),
            "host_tracer": int(os.environ["CANON_XPROF_HOST_TRACER"]),
            "python_tracer": int(os.environ["CANON_XPROF_PYTHON_TRACER"]),
            "tpu_trace_mode": os.environ["CANON_XPROF_TPU_TRACE_MODE"],
            "labels": int(os.environ["CANON_XPROF_LABELS"]),
        }
        if os.environ["V2_FL_MODE"] == "profile"
        else None
    ),
    "training_capsule": {
        "mode": os.environ["V2_FL_CAPSULE_MODE"],
        "capture_run": os.environ.get("V2_FL_CAPSULE_CAPTURE_RUN") or None,
        "sha256": (
            os.environ.get("CANON_V2_TRAINING_CAPSULE_SHA256") or None
        ),
        "model_binding_sha256": (
            os.environ.get("CANON_V2_MODEL_BINDING_SHA256") or None
        ),
    },
    "hbm_stage_diagnostic": (
        os.environ["V2_FL_MODE"] == "measure"
        and os.environ["V2_FL_ARM"] in ("r0", "r0b")
    ),
}
path = Path(os.environ["V2_FL_ROOT"]) / "run_manifest.json"
with path.open("x", encoding="utf-8") as output:
  json.dump(manifest, output, indent=2, sort_keys=True)
  output.write("\n")
print("[V2.FL.ONEHOST] manifest=" + json.dumps(manifest, sort_keys=True), flush=True)
PY

exec python3 - <<'PY'
import os
import shlex
from tunix.rl import dp_workloads

name = {
    "p45": f"frozenlake-p45-onehost-dp{os.environ['V2_FL_DP_SIZE']}-tp{os.environ['V2_FL_TP_SIZE']}",
    "m15": f"frozenlake-m15-onehost-dp{os.environ['V2_FL_DP_SIZE']}-tp{os.environ['V2_FL_TP_SIZE']}",
}[os.environ["V2_FL_WORKLOAD"]]
workload = dp_workloads.get_workload(name)
dp_workloads.validate_environment(
    workload, os.environ, require_reduction_admission=True
)
command = workload.command(run_stage="backward-no-commit")
print("[V2.FL.ONEHOST] exec=" + shlex.join(command), flush=True)
os.execvp(command[0], command)
PY
