#!/usr/bin/env bash
# Periodically persist immutable host-only P38 evidence while the workload runs.
set -euo pipefail

: "${CANON_RUN_LOG:?CANON_RUN_LOG unset}"
: "${CANON_PRE_ALIGN_REPORT:?CANON_PRE_ALIGN_REPORT unset}"
: "${CANON_P38_MISMATCH_CAPSULE:?CANON_P38_MISMATCH_CAPSULE unset}"
: "${CANON_P38_REQUEST_JOURNAL:?CANON_P38_REQUEST_JOURNAL unset}"
: "${CANON_P38_INCIDENT_LEDGER:?CANON_P38_INCIDENT_LEDGER unset}"
: "${CANON_P38_DIAGNOSTIC_ROUND_FILE:?CANON_P38_DIAGNOSTIC_ROUND_FILE unset}"
: "${CANON_P38_LIVE_SNAPSHOT_STOP_FILE:?CANON_P38_LIVE_SNAPSHOT_STOP_FILE unset}"
: "${CANON_P38_LIVE_SNAPSHOT_INTERVAL_SECONDS:?CANON_P38_LIVE_SNAPSHOT_INTERVAL_SECONDS unset}"
: "${CANON_P38_LIVE_COLLECT_REQUEST_FILE:?CANON_P38_LIVE_COLLECT_REQUEST_FILE unset}"
: "${CANON_P38_LIVE_COLLECT_ACK_FILE:?CANON_P38_LIVE_COLLECT_ACK_FILE unset}"
: "${CANON_P38_LIVE_COMPLETE_REQUEST_FILE:?CANON_P38_LIVE_COMPLETE_REQUEST_FILE unset}"
: "${CANON_P38_LIVE_COMPLETE_ACK_FILE:?CANON_P38_LIVE_COMPLETE_ACK_FILE unset}"
: "${CANON_P38_ROUND_SEAL_REQUEST_DIR:?CANON_P38_ROUND_SEAL_REQUEST_DIR unset}"
: "${CANON_P38_ROUND_SEAL_ACK_DIR:?CANON_P38_ROUND_SEAL_ACK_DIR unset}"
: "${CANON_P38_DURABILITY_PROFILE:?CANON_P38_DURABILITY_PROFILE unset}"
: "${CANON_PKG:?CANON_PKG unset}"

case "$CANON_P38_DURABILITY_PROFILE" in
  full-v1|round-alignment-v1) ;;
  *)
    echo "[P38.GCS] REFUSING: invalid durability profile: $CANON_P38_DURABILITY_PROFILE" >&2
    exit 2
    ;;
esac

case "$CANON_P38_LIVE_SNAPSHOT_INTERVAL_SECONDS" in
  ''|*[!0-9]*)
    echo "[P38.GCS] REFUSING: live snapshot interval must be an integer" >&2
    exit 2
    ;;
esac
if [ "$CANON_P38_LIVE_SNAPSHOT_INTERVAL_SECONDS" -lt 1 ]; then
  echo "[P38.GCS] REFUSING: live snapshot interval must be positive" >&2
  exit 2
fi
for control_path in \
    "$CANON_P38_LIVE_SNAPSHOT_STOP_FILE" \
    "$CANON_P38_LIVE_COLLECT_REQUEST_FILE" \
    "$CANON_P38_LIVE_COLLECT_ACK_FILE" \
    "$CANON_P38_LIVE_COMPLETE_REQUEST_FILE" \
    "$CANON_P38_LIVE_COMPLETE_ACK_FILE"; do
  if [ -e "$control_path" ]; then
    echo "[P38.GCS] REFUSING: live worker control path already exists: $control_path" >&2
    exit 2
  fi
done

persist="$CANON_PKG/tasks/p38-pathways-decode-prefill-carrier/scripts/persist_p38_gcs.sh"
sequence=0
last_signature=""
last_observer_signature=""
next_round_to_seal=0

observer_signature_for_round() {
  local observer_dir="$1" round_value="$2"
  python3 - "$observer_dir" "$round_value" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
round_index = int(sys.argv[2])
paths = []
for pattern in (
    "p38_kv_observer_*.json",
    "p38_seam_*.json",
    "p38_tail_*.json",
    "p38_terminal_*.json",
):
  for json_path in root.glob(pattern):
    record = json.loads(json_path.read_text(encoding="utf-8"))
    if int(record.get("diagnostic_round", -1)) != round_index:
      continue
    npz_path = json_path.with_suffix(".npz")
    if not npz_path.is_file():
      raise SystemExit(f"paired observer NPZ is absent: {npz_path}")
    paths.extend((json_path, npz_path))
for path in sorted(paths, key=lambda item: item.name):
  print(f"{path.name}:{path.stat().st_size}")
PY
}

snapshot_if_changed() {
  local run_size=0 journal_size=0 incident_size=0 report_size=0 capsule_signature=""
  local signature sequence_text rc capsule_path observer_path observer_dir
  local observer_signature="" observer_changed=0 round_value=missing
  # The fixed-lm-head causal arm seals a compact, independently verified
  # alignment bundle at every round.  Periodic snapshots use the same worker
  # and cannot be preempted once a GCS transfer starts, so running them here
  # can starve a later round request past the learner's 900-second deadline.
  if [ "$CANON_P38_DURABILITY_PROFILE" = round-alignment-v1 ]; then
    return 0
  fi
  [ ! -e "$CANON_RUN_LOG" ] || run_size="$(wc -c < "$CANON_RUN_LOG" | tr -d '[:space:]')"
  [ ! -e "$CANON_P38_REQUEST_JOURNAL" ] || journal_size="$(wc -c < "$CANON_P38_REQUEST_JOURNAL" | tr -d '[:space:]')"
  [ ! -e "$CANON_P38_INCIDENT_LEDGER" ] || incident_size="$(wc -c < "$CANON_P38_INCIDENT_LEDGER" | tr -d '[:space:]')"
  [ ! -e "$CANON_PRE_ALIGN_REPORT" ] || report_size="$(wc -c < "$CANON_PRE_ALIGN_REPORT" | tr -d '[:space:]')"
  [ ! -e "$CANON_P38_DIAGNOSTIC_ROUND_FILE" ] || round_value="$(tr -d '[:space:]' < "$CANON_P38_DIAGNOSTIC_ROUND_FILE")"
  shopt -s nullglob
  for capsule_path in \
      "${CANON_P38_MISMATCH_CAPSULE%.npz}".round-*.npz \
      "$CANON_P38_MISMATCH_CAPSULE"; do
    [ -e "$capsule_path" ] || continue
    capsule_signature+="$(basename "$capsule_path"):$(wc -c < "$capsule_path" | tr -d '[:space:]'),"
  done
  shopt -u nullglob
  observer_dir="${CANON_P38_SEAM_OBSERVER_DIR:-${CANON_P38_KV_OBSERVER_DIR:-}}"
  if [ -n "$observer_dir" ] && [ -d "$observer_dir" ] && \
     [ "$round_value" != missing ]; then
    observer_signature="$(observer_signature_for_round \
      "$observer_dir" "$round_value" | tr '\n' ',')"
  fi
  if [ -n "$observer_signature" ] && \
     [ "$observer_signature" != "$last_observer_signature" ]; then
    observer_changed=1
  fi
  signature="$run_size:$journal_size:$incident_size:$report_size:$round_value:$capsule_signature:$observer_signature"
  if [ "$signature" = "$last_signature" ] || \
     [ "$signature" = "0:0:0:0:missing::" ]; then
    return 0
  fi
  printf -v sequence_text '%06d' "$sequence"
  rc=0
  CANON_P38_LIVE_INCLUDE_OBSERVER="$observer_changed" \
    CANON_P38_LIVE_OBSERVER_ROUND="$round_value" \
    bash "$persist" snapshot "$sequence_text" || rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "[P38.GCS] FATAL: live snapshot failed sequence=$sequence_text rc=$rc" >&2
    return "$rc"
  fi
  last_signature="$signature"
  last_observer_signature="$observer_signature"
  sequence=$((sequence + 1))
}

write_ack() {
  local ack_path="$1" action="$2" partial="${1}.partial"
  (umask 077; printf 'action=%s status=PASS\n' "$action" > "$partial")
  mv -- "$partial" "$ack_path"
  echo "[P38.GCS] LIVE_${action^^}_PASS ack=$ack_path"
}

handle_terminal_requests() {
  if [ -e "$CANON_P38_LIVE_COLLECT_REQUEST_FILE" ] && \
     [ ! -e "$CANON_P38_LIVE_COLLECT_ACK_FILE" ]; then
    snapshot_if_changed
    bash "$persist" collect
    write_ack "$CANON_P38_LIVE_COLLECT_ACK_FILE" collect
  fi
  if [ -e "$CANON_P38_LIVE_COMPLETE_REQUEST_FILE" ] && \
     [ ! -e "$CANON_P38_LIVE_COMPLETE_ACK_FILE" ]; then
    if [ ! -e "$CANON_P38_LIVE_COLLECT_ACK_FILE" ]; then
      echo "[P38.GCS] FATAL: completion requested before collection acknowledgement" >&2
      return 2
    fi
    bash "$persist" complete
    write_ack "$CANON_P38_LIVE_COMPLETE_ACK_FILE" complete
  fi
}

handle_round_requests() {
  local request_path request_name round_text expected_name ack_path partial
  shopt -s nullglob
  local requests=("$CANON_P38_ROUND_SEAL_REQUEST_DIR"/round-*.request)
  shopt -u nullglob
  for request_path in "${requests[@]}"; do
    request_name="$(basename "$request_path")"
    round_text="${request_name#round-}"
    round_text="${round_text%.request}"
    if [ "$((10#$round_text))" -lt "$next_round_to_seal" ] && \
       [ -s "$CANON_P38_ROUND_SEAL_ACK_DIR/round-$round_text.ack" ]; then
      continue
    fi
    printf -v expected_name 'round-%06d.request' "$next_round_to_seal"
    if [ "$request_name" != "$expected_name" ]; then
      echo "[P38.GCS] FATAL: round-seal request order drifted: expected=$expected_name observed=$request_name" >&2
      return 2
    fi
    python3 - "$request_path" "$next_round_to_seal" <<'PY'
import json
import pathlib
import sys

record = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = {
    "action": "seal-round",
    "diagnostic_round": int(sys.argv[2]),
    "schema": "canon-p38-round-seal-request-v1",
}
if record != expected:
  raise SystemExit(f"round-seal request drifted: {record!r} != {expected!r}")
PY
    bash "$persist" round "$round_text"
    # The learner is still blocked here, so every record for this round is
    # already inside the verified round archive.  Mark that observer state as
    # durable before publishing the ACK; the next periodic snapshot then does
    # not redundantly archive the just-sealed round.
    observer_dir="${CANON_P38_SEAM_OBSERVER_DIR:-${CANON_P38_KV_OBSERVER_DIR:-}}"
    if [ -n "$observer_dir" ] && [ -d "$observer_dir" ]; then
      last_observer_signature="$(observer_signature_for_round \
        "$observer_dir" "$next_round_to_seal" | tr '\n' ',')"
    fi
    ack_path="$CANON_P38_ROUND_SEAL_ACK_DIR/round-$round_text.ack"
    partial="$ack_path.partial"
    python3 - "$partial" "$next_round_to_seal" <<'PY'
import json
import pathlib
import sys

pathlib.Path(sys.argv[1]).write_text(json.dumps({
    "action": "seal-round",
    "diagnostic_round": int(sys.argv[2]),
    "schema": "canon-p38-round-seal-ack-v1",
    "status": "PASS",
}, sort_keys=True) + "\n", encoding="utf-8")
PY
    mv -- "$partial" "$ack_path"
    echo "[P38.GCS] LIVE_ROUND_PASS round=$next_round_to_seal ack=$ack_path"
    next_round_to_seal=$((next_round_to_seal + 1))
  done
}

echo "[P38.GCS] LIVE_WORKER_START interval=$CANON_P38_LIVE_SNAPSHOT_INTERVAL_SECONDS profile=$CANON_P38_DURABILITY_PROFILE"
while [ ! -e "$CANON_P38_LIVE_SNAPSHOT_STOP_FILE" ]; do
  # Round durability is on the learner's critical path.  Never put a periodic
  # live snapshot ahead of an already-published seal or terminal request.
  handle_round_requests
  handle_terminal_requests
  snapshot_if_changed
  sleep "$CANON_P38_LIVE_SNAPSHOT_INTERVAL_SECONDS" &
  wait "$!" || true
done
handle_round_requests
handle_terminal_requests
snapshot_if_changed
if [ "$CANON_P38_DURABILITY_PROFILE" = full-v1 ]; then
  if [ "$sequence" -eq 0 ]; then
    echo "[P38.GCS] FATAL: live worker observed no host evidence" >&2
    exit 1
  fi
elif [ "$next_round_to_seal" -eq 0 ]; then
  echo "[P38.GCS] FATAL: round-alignment worker sealed no rounds" >&2
  exit 1
fi
echo "[P38.GCS] LIVE_WORKER_COMPLETE snapshots=$sequence rounds=$next_round_to_seal profile=$CANON_P38_DURABILITY_PROFILE"
