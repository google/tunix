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
  export CANON_P38_SERVING_CAPTURE_EXPECTED_RECORDS=4
  export CANON_P38_SERVING_CAPTURE_EXPECTED_PATH=standard
  export CANON_P38_SERVING_CAPTURE_PREFIX_BOUNDS=1536,1792,2048,2304,2560
  export CANON_P38_SERVING_CAPTURE_FREE_SPACE_MULTIPLIER=5
  export CANON_P38_SERVING_CAPTURE_CLASSIFICATION="$state/capture.json"
  export CANON_P38_SERVING_CAPTURE_ARCHIVE="$state/capture.tar"
  export CANON_P38_MISMATCH_CAPSULE="$state/mismatch.npz"
  export CANON_KV_UNIFIED=0
  command="python3 $ROOT/tests/p38_serving/make_fixture.py --directory $state/capture --mismatch-capsule $CANON_P38_MISMATCH_CAPSULE"
  command+="; printf '%s\\n' '[CANON_P38_SERVING_CAPTURE_INIT] enabled=1 max_calls=4 expected_path=standard'"
  command+="; printf '%s\\n' '[CANON_P38_SERVING_CAPTURE_OBSERVE] {\"call\":1,\"program_path\":\"standard\",\"one_token_requests\":1}'"
  command+="; printf '%s\\n' 'CANON_FIXED_AR=1 fixed-order tree'"
  command+="; printf '%s\\n' 'CANON_FIXED_AR_EMBED=1 fixed-order embed gather'"
  if [ "$mode" = exact ]; then
    command+="; printf '%s\\n' '[CANON_P38] PRECHECK_COMPLETE STOP_BEFORE_BACKWARD step=0 N_action=1'"
  elif [ "$mode" = stock-hit ]; then
    command+="; printf '%s\\n' '[PATHTRACE] KV_UNIFIED_two_pass'"
  elif [ "$mode" = unified-missing ]; then
    export CANON_KV_UNIFIED=1
  elif [ "$mode" = unified-exact ]; then
    export CANON_KV_UNIFIED=1
    command+="; printf '%s\\n' '[PATHTRACE] KV_UNIFIED_two_pass'"
    command+="; printf '%s\\n' '[CANON_P38] PRECHECK_COMPLETE STOP_BEFORE_BACKWARD step=0 N_action=1'"
  fi
  command+="; exit 1"
  export CANON_RUN_CMD="$command"
  export -p > "$state/env.sh"
  rc=0
  bash "$ROOT/cluster/steps/90_run.sh" > "$state/driver.log" 2>&1 || rc=$?
  if [ "$mode" = exact ] || [ "$mode" = unified-exact ]; then
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
    if [ "$mode" = stock-hit ]; then
      grep -q 'P38 stock arm executed KV_UNIFIED_two_pass' "$state/driver.log"
    elif [ "$mode" = unified-missing ]; then
      grep -q 'P38 U arm did not execute KV_UNIFIED_two_pass' "$state/driver.log"
    fi
  fi
)

run_case exact
run_case red
run_case stock-hit
run_case unified-missing
run_case unified-exact
echo "[P38.SERVING] POSTFLIGHT_PASS exact_stop=accepted red_stop=rejected stock_hit=rejected unified_missing=rejected unified_exact=accepted"
