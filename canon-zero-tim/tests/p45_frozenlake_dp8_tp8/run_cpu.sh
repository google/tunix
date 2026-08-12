#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKTREE="$(cd "$ROOT/.." && pwd)"
cd "$WORKTREE"

python3 -c "import ast,pathlib; files=(\
'tunix/rl/dp_workloads.py',\
'tunix/rl/agentic/agentic_rl_learner.py',\
'tunix/rl/canonical_qwen3_adapter.py',\
'examples/frozenlake/train_frozenlake_qwen3.py',\
'canon-zero-tim/cluster/render_p33_jobsets.py',\
'canon-zero-tim/cluster/render_p45_frozenlake.py',\
'canon-zero-tim/src/engine_shims/models/qwen8b_tp8/p22xf_contract.py',\
'canon-zero-tim/tests/p33_workloads/classify_run.py',\
'canon-zero-tim/tests/p45_frozenlake_dp8_tp8/test_renderer.py',\
'canon-zero-tim/tests/p45_frozenlake_dp8_tp8/test_qwen8b_tp8.py',\
'canon-zero-tim/tests/p45_frozenlake_dp8_tp8/probe_overlay_import.py',\
'canon-zero-tim/tests/p45_frozenlake_dp8_tp8/probe_qwen8b_tp8.py'); \
[ast.parse(pathlib.Path(path).read_text(), filename=path) for path in files]"

bash -n \
  canon-zero-tim/cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-resident.env \
  canon-zero-tim/tests/p45_frozenlake_dp8_tp8/run_cpu.sh \
  canon-zero-tim/tests/p45_frozenlake_dp8_tp8/run_exact_image.sh

JAX_PLATFORMS=cpu python3 -m unittest \
  canon-zero-tim/tests/p33_workloads/test_dp_workloads.py \
  canon-zero-tim/tests/p33_workloads/test_classify_run.py \
  canon-zero-tim/tests/p33_workloads/test_render_p33_jobsets.py \
  canon-zero-tim/tests/p45_frozenlake_dp8_tp8/test_renderer.py \
  canon-zero-tim/tests/p45_frozenlake_dp8_tp8/test_qwen8b_tp8.py
JAX_PLATFORMS=cpu python3 -m unittest discover \
  -s tests/rl \
  -p alignment_test.py

validate_p45_profile() (
  set -euo pipefail
  local state
  state="$(mktemp -d)"
  trap 'rm -r "$state"' EXIT
  export CANON_PKG="$ROOT"
  export CANON_STATE="$state"
  export CANON_MODE=run
  export CANON_PROFILE_FILE=cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-resident.env
  export CANON_P32_TRAIN_ADMITTED=1
  export CANON_P32_DP_REDUCTION_ADMITTED=1
  export CANON_P33_WORKLOAD_LAUNCH_ADMITTED=1
  export CANON_P33_SHARED_MESH=8,8
  export CANON_P33_RUN_STAGE=full
  export CANON_P33_NO_COMMIT=0
  export CANON_P33_ENABLE_EVAL=1
  export CANON_OPT_STATE_RESIDENT=1
  export CANON_P30_OPT_STATE_OFFLOAD=0
  export CANON_FROZENLAKE_ALIGNMENT_WARN_ONLY=1
  export CANON_RUN_CMD="printf p45-preflight"
  export CANON_RUN_LOG="$state/run.log"
  export CANON_PRE_ALIGN_REPORT="$state/pre_alignment.jsonl"
  export CANON_ALIGN_REPORT="$state/alignment.jsonl"
  export CANON_UPDATE_REPORT="$state/updates.jsonl"
  export INJECTED_WANDB_API_KEY=test-key-not-a-credential
  bash "$ROOT/cluster/steps/00_env.sh" >/dev/null
  # shellcheck disable=SC1090
  source "$state/env.sh"
  [ "$CANON_P32_WORKLOAD" = frozenlake-dp8-tp8 ]
  [ "$CANON_MODEL_DIR_NAME" = qwen8b_tp8 ]
  [ "$CANON_DP_SIZE:$CANON_TP_SIZE" = 8:8 ]
  [ "$CANON_LOCAL_TRAJECTORIES:$MIN_TOKEN_BUCKET" = 32:2048 ]
  [ "$CANON_OPT_STATE_RESIDENT:$CANON_P30_OPT_STATE_OFFLOAD" = 1:0 ]
  [ "$CANON_P33_ENABLE_EVAL:$CANON_P31_ENABLE_EVAL" = 1:1 ]
  [ "$CANON_FROZENLAKE_ALIGNMENT_WARN_ONLY" = 1 ]
  echo "[P45.PROFILE] ADMITTED_PREFLIGHT_PASS topology=DP8xTP8 model_dir=qwen8b_tp8 local_trajectories=32 global_m=2048 optimizer=device-resident eval=on warning_only=on"
)

validate_p45_profile
echo "[P45.FROZENLAKE] CPU_GATE PASS topology=DP8xTP8 optimizer=device-resident"
