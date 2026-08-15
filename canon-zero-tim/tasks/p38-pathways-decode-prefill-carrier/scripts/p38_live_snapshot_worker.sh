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
: "${CANON_PKG:?CANON_PKG unset}"

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

snapshot_if_changed() {
  local run_size=0 journal_size=0 incident_size=0 report_size=0 capsule_signature=""
  local signature sequence_text rc capsule_path observer_path observer_dir
  local observer_signature="" observer_changed=0 round_value=missing
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
  if [ -n "$observer_dir" ] && [ -d "$observer_dir" ]; then
    shopt -s nullglob
    for observer_path in \
        "$observer_dir"/p38_kv_observer_*.json \
        "$observer_dir"/p38_kv_observer_*.npz \
        "$observer_dir"/p38_seam_*.json \
        "$observer_dir"/p38_seam_*.npz; do
      observer_signature+="$(basename "$observer_path"):$(wc -c < "$observer_path" | tr -d '[:space:]'),"
    done
    shopt -u nullglob
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

echo "[P38.GCS] LIVE_WORKER_START interval=$CANON_P38_LIVE_SNAPSHOT_INTERVAL_SECONDS"
while [ ! -e "$CANON_P38_LIVE_SNAPSHOT_STOP_FILE" ]; do
  snapshot_if_changed
  handle_terminal_requests
  sleep "$CANON_P38_LIVE_SNAPSHOT_INTERVAL_SECONDS" &
  wait "$!" || true
done
snapshot_if_changed
handle_terminal_requests
if [ "$sequence" -eq 0 ]; then
  echo "[P38.GCS] FATAL: live worker observed no host evidence" >&2
  exit 1
fi
echo "[P38.GCS] LIVE_WORKER_COMPLETE snapshots=$sequence"
