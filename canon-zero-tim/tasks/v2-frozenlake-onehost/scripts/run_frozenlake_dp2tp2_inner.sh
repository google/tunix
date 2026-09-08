#!/usr/bin/env bash
set -euo pipefail

repo="${V2_FL_REPO:?}"
pkg="$repo/canon-zero-tim"
root="${V2_FL_ROOT:?}"
workload="${V2_FL_WORKLOAD:?}"
arm="${V2_FL_ARM:?}"
profile_rel=cluster/profiles/qwen3-8b-dp2-tp2-frozenlake-onehost.env

case "$workload" in
  p45)
    workload_name=frozenlake-p45-onehost-dp2-tp2
    export CANON_P57_WORKLOAD_CANDIDATE=
    export CANON_P57_DATA_SPLIT=
    ;;
  m15)
    workload_name=frozenlake-m15-onehost-dp2-tp2
    export CANON_P57_WORKLOAD_CANDIDATE=m15
    export CANON_P57_DATA_SPLIT=main
    ;;
  *) echo "[V2.FL.ONEHOST] unknown workload: $workload" >&2; exit 2 ;;
esac
case "$arm" in
  r0) keep_tape=0; reduce_once=0; length_sort=0 ;;
  r1) keep_tape=stream; reduce_once=0; length_sort=0 ;;
  r2) keep_tape=stream; reduce_once=1; length_sort=0 ;;
  r3) keep_tape=stream; reduce_once=1; length_sort=1 ;;
  *) echo "[V2.FL.ONEHOST] unknown arm: $arm" >&2; exit 2 ;;
esac
export CANON_PROFILE_FILE="$profile_rel"
export CANON_P32_KEEP_TAPE="$keep_tape"
export CANON_DP_REDUCE_ONCE="$reduce_once"
export CANON_P32_LENGTH_SORT="$length_sort"
export CANON_WANDB_RUN_NAME="v2-fl-${workload}-${arm}-${V2_FL_LABEL:?}"
export CANON_P57_RUN_KIND=
export CANON_P57_TIM_ARM=
# shellcheck disable=SC1090
source "$pkg/$profile_rel"

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
    "p45": "frozenlake-p45-onehost-dp2-tp2",
    "m15": "frozenlake-m15-onehost-dp2-tp2",
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
    "topology": {"dp": 2, "tp": 2, "devices": 4},
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
    },
    "checked_vma": os.environ["CANON_P66_P59_CHECK_VMA"] == "1",
    "wandb_mode": os.environ["WANDB_MODE"],
    "classification_mode": os.environ["V2_FL_MODE"],
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
    "p45": "frozenlake-p45-onehost-dp2-tp2",
    "m15": "frozenlake-m15-onehost-dp2-tp2",
}[os.environ["V2_FL_WORKLOAD"]]
workload = dp_workloads.get_workload(name)
dp_workloads.validate_environment(
    workload, os.environ, require_reduction_admission=True
)
command = workload.command(run_stage="backward-no-commit")
print("[V2.FL.ONEHOST] exec=" + shlex.join(command), flush=True)
os.execvp(command[0], command)
PY
