#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKTREE="$(cd "$ROOT/.." && pwd)"

run_case() (
  set -euo pipefail
  local mode="$1" state command rc
  echo "[P38.SERVING] CASE mode=$mode"
  state="$(mktemp -d)"
  if [ "${P38_KEEP_TEST_STATE:-0}" = "1" ]; then
    echo "[P38.SERVING] KEEP_STATE path=$state"
  else
    trap 'rm -r "$state"' EXIT
  fi
  mkdir -p "$state/bin" "$state/fake-gcs"
  cp "$ROOT/tests/p38_serving/fake_gcloud.sh" "$state/bin/gcloud"
  chmod +x "$state/bin/gcloud"
  export PATH="$state/bin:$PATH"
  export FAKE_GCS_ROOT="$state/fake-gcs"
  export CANON_PKG="$ROOT"
  export CANON_STATE="$state"
  export CANON_RUN_LOG="$state/run.log"
  export CANON_PRE_ALIGN_REPORT="$state/pre-alignment.jsonl"
  export CANON_P33_WORKLOAD_LAUNCH_ADMITTED=0
  export CANON_P32_TRAIN_ADMITTED=0
  export CANON_P38_PRECHECK_ONLY=1
  export CANON_P38_CONTROLLED_EXIT=1
  export CANON_P38_DIAGNOSTIC_ROUNDS=1
  export CANON_P38_DURABILITY_PROFILE=full-v1
  export CANON_P38_DIAGNOSTIC_ROUND_FILE="$state/p38_diagnostic_round"
  export CANON_P38_ROUND_SEAL_REQUEST_DIR="$state/p38_round_seal_requests"
  export CANON_P38_ROUND_SEAL_ACK_DIR="$state/p38_round_seal_acks"
  export CANON_P38_MIN_ACTION_KV=1686
  export CANON_P38_SERVING_CAPTURE_DIR="$state/capture"
  export CANON_P38_REQUEST_JOURNAL="$state/capture/p38_request_journal.jsonl"
  export CANON_P38_INCIDENT_LEDGER="$state/capture/p38_incident_ledger.jsonl"
  export CANON_P38_INCIDENT_MIN_PREFIX=1400
  export CANON_P38_INCIDENT_MAX_PREFIX=3072
  export CANON_P38_INCIDENT_MAX_BYTES=134217728
  export CANON_P38_LIVE_SNAPSHOT_INTERVAL_SECONDS=1
  export CANON_P38_LIVE_SNAPSHOT_STOP_FILE="$state/p38_live.stop"
  export CANON_P38_LIVE_SNAPSHOT_WORKER_LOG="$state/p38_live_worker.log"
  export CANON_P38_LIVE_COLLECT_REQUEST_FILE="$state/p38_collect.request"
  export CANON_P38_LIVE_COLLECT_ACK_FILE="$state/p38_collect.ack"
  export CANON_P38_LIVE_COMPLETE_REQUEST_FILE="$state/p38_complete.request"
  export CANON_P38_LIVE_COMPLETE_ACK_FILE="$state/p38_complete.ack"
  export CANON_P38_SERVING_CAPTURE_EXPECTED_RECORDS=4
  export CANON_P38_SERVING_CAPTURE_EXPECTED_PATH=standard
  export CANON_P38_SERVING_CAPTURE_PREFIX_BOUNDS=1536,1664,1792,1920,2048
  export CANON_P38_SERVING_CAPTURE_FREE_SPACE_MULTIPLIER=5
  export CANON_P38_SERVING_CAPTURE_CLASSIFICATION="$state/capture.json"
  export CANON_P38_SERVING_CAPTURE_ARCHIVE="$state/capture.tar"
  export CANON_P38_GCS_PREFIX="gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/postflight-$mode/attempt-0"
  export CANON_P38_MISMATCH_CAPSULE="$state/mismatch.npz"
  export CANON_KV_UNIFIED=0
  export CANON_P38_KV_OBSERVER_DIR="$state/capture"
  export CANON_P38_KV_OBSERVER_MAX_CANDIDATES=3
  export CANON_P38_KV_OBSERVER_MAX_PAGES=16
  export CANON_P38_KV_OBSERVER_MAX_BYTES=134217728
  export CANON_P38_KV_OBSERVER_MAX_READ_BYTES=671088640
  export CANON_P38_KV_OBSERVER_CLASSIFICATION="$state/kv-observer.json"
  command="python3 $ROOT/tests/p38_serving/make_fixture.py --directory $state/capture --mismatch-capsule $CANON_P38_MISMATCH_CAPSULE"
  if [ "$mode" = seam-layer ] || [ "$mode" = tail-layer ] || \
     [ "$mode" = tail-missing ]; then
    unset CANON_P38_KV_OBSERVER_DIR \
      CANON_P38_KV_OBSERVER_MAX_CANDIDATES \
      CANON_P38_KV_OBSERVER_MAX_PAGES \
      CANON_P38_KV_OBSERVER_MAX_BYTES \
      CANON_P38_KV_OBSERVER_MAX_READ_BYTES \
      CANON_P38_KV_OBSERVER_CLASSIFICATION
    export CANON_P38_SEAM_OBSERVER=layer
    export CANON_P38_SEAM_OBSERVER_DIR="$state/capture"
    export CANON_P38_SEAM_MIN_POSITION=1400
    export CANON_P38_SEAM_MAX_POSITION=3072
    export CANON_P38_SEAM_MAX_BYTES=4294967296
    export CANON_P38_SEAM_CLASSIFICATION="$state/seam-classification.json"
    command+=" --seam"
    if [ "$mode" = tail-layer ] || [ "$mode" = tail-missing ]; then
      export CANON_P38_TAIL_OBSERVER=1
      export CANON_P38_TAIL_MAX_BYTES=268435456
      if [ "$mode" = tail-layer ]; then
        command+=" --tail"
      fi
    fi
  fi
  if [ "$mode" = missing-journal ]; then
    command+=" --omit-request-journal"
  elif [ "$mode" = missing-incident ]; then
    command+=" --omit-incident-ledger"
  fi
  if [ "$mode" != exact-stable ]; then
    command+="; cp '$CANON_P38_MISMATCH_CAPSULE' '${CANON_P38_MISMATCH_CAPSULE%.npz}.round-000000.npz'"
  fi
  depth=1700
  if [ "$mode" = shallow ]; then
    depth=1600
  fi
  command+="; printf '%s\\n' '{\"action_geometry\":{\"valid\":true,\"max_logical_kv_prefix_length\":$depth}}' > '$CANON_PRE_ALIGN_REPORT'"
  command+="; printf '%s\\n' '[CANON_P38_SERVING_CAPTURE_INIT] enabled=1 max_calls=4 expected_path=standard'"
  command+="; printf '%s\\n' '[CANON_P38_SERVING_CAPTURE_OBSERVE] {\"call\":1,\"program_path\":\"standard\",\"one_token_requests\":1}'"
  command+="; printf '%s\\n' '[CANON_P38_REQUEST_JOURNAL] record=1 request=request-0 prefix=1600 stratum=0 dp=0'"
  command+="; printf '%s\\n' '[CANON_P38_INCIDENT_LEDGER] record=1 call=1 requests=1 bytes=1'"
  if [ "$mode" = seam-layer ] || [ "$mode" = tail-layer ] || \
     [ "$mode" = tail-missing ]; then
    command+="; printf '%s\\n' '[CANON_P38_SEAM_OBSERVER_INIT] mode=layer min_position=1400 max_position=3072 max_bytes=4294967296'"
    command+="; printf '%s\\n' '[CANON_P38_SEAM_OBSERVER_RECORD] arm=A record=0 rows=1 bytes=1'"
    command+="; printf '%s\\n' '[CANON_P38_SEAM_OBSERVER_RECORD] arm=B record=1 rows=1 bytes=1'"
    if [ "$mode" = tail-layer ] || [ "$mode" = tail-missing ]; then
      command+="; printf '%s\\n' '[CANON_P38_TAIL_OBSERVER_INIT] enabled=1 max_bytes=268435456'"
      command+="; printf '%s\\n' '[CANON_P38_TAIL_OBSERVER_RECORD] record=0 arm=A call=1 rows=1 bytes=1'"
      command+="; printf '%s\\n' '[CANON_P38_TAIL_OBSERVER_RECORD] record=1 arm=B call=2 rows=1 bytes=1'"
    fi
  elif [[ "$mode" != unified-* ]]; then
    command+="; printf '%s\\n' '[CANON_P38_KV_OBSERVER_INIT] enabled=1 candidates=3 pages=16 max_output_bytes=134217728 max_read_bytes=671088640'"
    command+="; printf '%s\\n' '[CANON_P38_KV_OBSERVER_CANDIDATE] round=0 request=decode-0 call=1 prefix=1600'"
    command+="; printf '%s\\n' '[CANON_P38_KV_OBSERVER_CANDIDATE] round=1 request=decode-1 call=2 prefix=1600'"
    command+="; printf '%s\\n' '[CANON_P38_KV_OBSERVER_CANDIDATE] round=2 request=decode-2 call=3 prefix=1600'"
    for observer_record in 0 1 2; do
      command+="; printf '%s\\n' '[CANON_P38_KV_OBSERVER_RECORD] arm=A record=$((observer_record * 2)) request=decode-$observer_record tokens=1601 pages=7 bytes=1'"
      command+="; printf '%s\\n' '[CANON_P38_KV_OBSERVER_RECORD] arm=B record=$((observer_record * 2 + 1)) request=clean-$observer_record tokens=1601 pages=7 bytes=2'"
    done
  fi
  if [ "$mode" != missing-coverage ]; then
    command+="; printf '%s\\n' '[CANON_P38] DIAGNOSTIC_COVERAGE_CONTRACT prompt_groups=32 unit_prompts=4 units=8 generations=8 trajectories=256 partial_tail=reject verdict=PASS'"
  fi
  command+="; printf '%s\\n' 'CANON_FIXED_AR=1 fixed-order tree'"
  command+="; printf '%s\\n' 'CANON_FIXED_AR_EMBED=1 fixed-order embed gather'"
  if [ "$mode" = exact ] || [ "$mode" = exact-stable ] || \
     [ "$mode" = shallow ] || [ "$mode" = seam-layer ] || \
     [ "$mode" = tail-layer ] || [ "$mode" = tail-missing ]; then
    command+="; printf '%s\\n' '[CANON_P38] PRECHECK_ROUND_COMPLETE round=1/1 step=0 N_action=1 verdict=PASS a_b_differing_bytes=0 backward=0 optimizer_commits=0'"
    command+="; printf '%s\\n' '[CANON_P38] PRECHECK_COMPLETE STOP_BEFORE_BACKWARD step=0 N_action=1'"
  elif [ "$mode" = stock-hit ]; then
    command+="; printf '%s\\n' '[PATHTRACE] KV_UNIFIED_two_pass'"
  elif [ "$mode" = unified-missing ]; then
    export CANON_KV_UNIFIED=1
    unset CANON_P38_KV_OBSERVER_DIR \
      CANON_P38_KV_OBSERVER_MAX_CANDIDATES \
      CANON_P38_KV_OBSERVER_MAX_PAGES \
      CANON_P38_KV_OBSERVER_MAX_BYTES \
      CANON_P38_KV_OBSERVER_MAX_READ_BYTES \
      CANON_P38_KV_OBSERVER_CLASSIFICATION
  elif [ "$mode" = unified-exact ]; then
    export CANON_KV_UNIFIED=1
    unset CANON_P38_KV_OBSERVER_DIR \
      CANON_P38_KV_OBSERVER_MAX_CANDIDATES \
      CANON_P38_KV_OBSERVER_MAX_PAGES \
      CANON_P38_KV_OBSERVER_MAX_BYTES \
      CANON_P38_KV_OBSERVER_MAX_READ_BYTES \
      CANON_P38_KV_OBSERVER_CLASSIFICATION
    command+="; printf '%s\\n' '[PATHTRACE] KV_UNIFIED_two_pass'"
    command+="; printf '%s\\n' '[CANON_P38] PRECHECK_ROUND_COMPLETE round=1/1 step=0 N_action=1 verdict=PASS a_b_differing_bytes=0 backward=0 optimizer_commits=0'"
    command+="; printf '%s\\n' '[CANON_P38] PRECHECK_COMPLETE STOP_BEFORE_BACKWARD step=0 N_action=1'"
  elif [ "$mode" = capture-error ]; then
    command+="; printf '%s\\n' '[CANON_P38_SERVING_CAPTURE_ERROR] stage=begin error=TypeError: injected'"
    command+="; printf '%s\\n' '[CANON_P38] PRECHECK_ROUND_COMPLETE round=1/1 step=0 N_action=1 verdict=PASS a_b_differing_bytes=0 backward=0 optimizer_commits=0'"
    command+="; printf '%s\\n' '[CANON_P38] PRECHECK_COMPLETE STOP_BEFORE_BACKWARD step=0 N_action=1'"
  elif [ "$mode" = missing-coverage ]; then
    command+="; printf '%s\\n' '[CANON_P38] PRECHECK_ROUND_COMPLETE round=1/1 step=0 N_action=1 verdict=PASS a_b_differing_bytes=0 backward=0 optimizer_commits=0'"
    command+="; printf '%s\\n' '[CANON_P38] PRECHECK_COMPLETE STOP_BEFORE_BACKWARD step=0 N_action=1'"
  fi
  case "$mode" in
    exact|exact-stable|shallow|seam-layer|tail-layer|tail-missing|unified-exact|capture-error|missing-coverage)
      command+="; printf '%s\\n' '[CANON_P38] CONTROLLED_EXIT code=42 backward=0 optimizer_commits=0'; exit 42"
      ;;
    *) command+="; exit 1" ;;
  esac
  export CANON_RUN_CMD="$command"
  export -p > "$state/env.sh"
  rc=0
  bash "$ROOT/cluster/steps/90_run.sh" > "$state/driver.log" 2>&1 || rc=$?
  if [ "$mode" = exact ] || [ "$mode" = exact-stable ] || \
     [ "$mode" = seam-layer ] || [ "$mode" = tail-layer ] || \
     [ "$mode" = unified-exact ]; then
    [ "$rc" -eq 0 ]
    grep -q 'P38 serving controlled precheck accepted exit=42' "$state/driver.log"
    grep -q '\[CANON_P38\] DEPTH_SUFFICIENCY min=1686 observed=1700 verdict=PASS' "$state/driver.log"
    grep -q '"verdict": "PASS"' "$CANON_P38_SERVING_CAPTURE_CLASSIFICATION"
    if [ "$mode" = seam-layer ] || [ "$mode" = tail-layer ]; then
      if [ "$mode" = tail-layer ]; then
        grep -q '"classification": "decode_terminal_first_difference_measured"' \
          "$CANON_P38_SEAM_CLASSIFICATION"
      else
        grep -q '"classification": "decode_seam_first_difference_measured"' \
          "$CANON_P38_SEAM_CLASSIFICATION"
      fi
      grep -q '^\[CANON_P38_SEAM_INPUTS\] source=immutable-rounds capsules=1 mode=layer$' \
        "$state/driver.log"
      find "$FAKE_GCS_ROOT" -name seam-classification.json -type f | grep -q .
    elif [ "$mode" = exact ] || [ "$mode" = exact-stable ]; then
      grep -q '"classification": "live_kv_fingerprint_equal_on_red_row"' \
        "$CANON_P38_KV_OBSERVER_CLASSIFICATION"
      if [ "$mode" = exact ]; then
        grep -q '^\[CANON_P38_KV_OBSERVER_INPUTS\] source=immutable-rounds capsules=1$' \
          "$state/driver.log"
      else
        grep -q '^\[CANON_P38_KV_OBSERVER_INPUTS\] source=stable-fallback capsules=1$' \
          "$state/driver.log"
      fi
    fi
    test -s "$CANON_P38_SERVING_CAPTURE_ARCHIVE"
    grep -q '\[P38.GCS\] LIVE_WORKER_JOINED rc=0' "$state/driver.log"
    grep -q '\[P38.GCS\] LIVE_ACTION_ACKNOWLEDGED action=collect' "$state/driver.log"
    grep -q '\[P38.GCS\] LIVE_ACTION_ACKNOWLEDGED action=complete' "$state/driver.log"
    find "$FAKE_GCS_ROOT" -path '*/live/*/LIVE.json' -type f | grep -q .
    test -s "$CANON_P38_LIVE_COLLECT_ACK_FILE"
    test -s "$CANON_P38_LIVE_COMPLETE_ACK_FILE"
  else
    [ "$rc" -ne 0 ]
    if grep -q 'P38 serving controlled precheck accepted' "$state/driver.log"; then
      echo "[P38.SERVING] postflight accepted capture without exact precheck" >&2
      exit 1
    fi
    if [ "$mode" = stock-hit ]; then
      grep -q 'P38 stock arm executed KV_UNIFIED_two_pass' "$state/driver.log"
    elif [ "$mode" = unified-missing ]; then
      grep -q 'P38 U arm did not execute KV_UNIFIED_two_pass' "$state/driver.log"
    elif [ "$mode" = capture-error ]; then
      grep -q 'P38 serving capture reported internal errors: 1' "$state/driver.log"
    elif [ "$mode" = missing-coverage ]; then
      grep -q 'P38 diagnostic did not attest full 32-prompt coverage: 0' "$state/driver.log"
    elif [ "$mode" = missing-journal ]; then
      grep -q 'P38 request journal is absent: markers=1' "$state/driver.log"
    elif [ "$mode" = missing-incident ]; then
      grep -q 'P38 incident ledger is absent: markers=1' "$state/driver.log"
    elif [ "$mode" = tail-missing ]; then
      grep -q 'P38 seam observer contract failed' "$state/driver.log"
      grep -q 'P38 terminal-tail observer produced no records' "$state/driver.log"
    elif [ "$mode" = shallow ]; then
      grep -q 'P38 depth sufficiency failed: min=1686 observed=1600' "$state/driver.log"
    fi
  fi
)

run_case exact
run_case exact-stable
run_case seam-layer
run_case tail-layer
run_case tail-missing
run_case red
run_case stock-hit
run_case unified-missing
run_case unified-exact
run_case capture-error
run_case missing-coverage
run_case missing-journal
run_case missing-incident
run_case shallow
echo "[P38.SERVING] POSTFLIGHT_PASS controlled_exact=accepted seam_layer=classified tail_layer=classified tail_missing=rejected immutable_rounds=preferred stable_fallback=accepted shallow=rejected red_stop=rejected stock_hit=rejected unified_missing=rejected unified_exact=accepted capture_error=rejected missing_coverage=rejected missing_journal=rejected missing_incident=rejected"
