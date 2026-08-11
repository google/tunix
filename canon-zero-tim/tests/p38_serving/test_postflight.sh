#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKTREE="$(cd "$ROOT/.." && pwd)"

run_case() (
  set -euo pipefail
  local mode="$1" state command rc
  state="$(mktemp -d)"
  trap 'rm -r "$state"' EXIT
  export CANON_PKG="$ROOT"
  export CANON_STATE="$state"
  export CANON_RUN_LOG="$state/run.log"
  export CANON_P33_WORKLOAD_LAUNCH_ADMITTED=0
  export CANON_P32_TRAIN_ADMITTED=0
  export CANON_P38_PRECHECK_ONLY=1
  export CANON_P38_SERVING_CAPTURE_DIR="$state/capture"
  export CANON_P38_SERVING_CAPTURE_EXPECTED_RECORDS=1
  export CANON_P38_SERVING_CAPTURE_CLASSIFICATION="$state/capture.json"
  export CANON_P38_SERVING_CAPTURE_ARCHIVE="$state/capture.tar"
  command="python3 $ROOT/tests/p38_serving/make_fixture.py --directory $state/capture"
  command+="; printf '%s\\n' 'CANON_FIXED_AR=1 fixed-order tree'"
  command+="; printf '%s\\n' 'CANON_FIXED_AR_EMBED=1 fixed-order embed gather'"
  if [ "$mode" = exact ]; then
    command+="; printf '%s\\n' '[CANON_P38] PRECHECK_COMPLETE STOP_BEFORE_BACKWARD step=0 N_action=1'"
  fi
  command+="; exit 1"
  export CANON_RUN_CMD="$command"
  export -p > "$state/env.sh"
  rc=0
  bash "$ROOT/cluster/steps/90_run.sh" > "$state/driver.log" 2>&1 || rc=$?
  if [ "$mode" = exact ]; then
    [ "$rc" -eq 0 ]
    grep -q 'P38 serving expected precheck exit=1 accepted' "$state/driver.log"
    grep -q '"verdict": "PASS"' "$CANON_P38_SERVING_CAPTURE_CLASSIFICATION"
    test -s "$CANON_P38_SERVING_CAPTURE_ARCHIVE"
  else
    [ "$rc" -ne 0 ]
    if grep -q 'P38 serving expected precheck exit=1 accepted' "$state/driver.log"; then
      echo "[P38.SERVING] postflight accepted capture without exact precheck" >&2
      exit 1
    fi
  fi
)

run_case exact
run_case red
echo "[P38.SERVING] POSTFLIGHT_PASS exact_stop=accepted red_stop=rejected"
