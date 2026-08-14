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
if [ -e "$CANON_P38_LIVE_SNAPSHOT_STOP_FILE" ]; then
  echo "[P38.GCS] REFUSING: live snapshot stop file already exists" >&2
  exit 2
fi

persist="$CANON_PKG/tasks/p38-pathways-decode-prefill-carrier/scripts/persist_p38_gcs.sh"
sequence=0
last_signature=""

snapshot_if_changed() {
  local run_size=0 journal_size=0 incident_size=0 report_size=0 capsule_signature=""
  local signature sequence_text rc capsule_path round_value=missing
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
  signature="$run_size:$journal_size:$incident_size:$report_size:$round_value:$capsule_signature"
  if [ "$signature" = "$last_signature" ] || \
     [ "$signature" = "0:0:0:0:missing:" ]; then
    return 0
  fi
  printf -v sequence_text '%06d' "$sequence"
  rc=0
  bash "$persist" snapshot "$sequence_text" || rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "[P38.GCS] FATAL: live snapshot failed sequence=$sequence_text rc=$rc" >&2
    return "$rc"
  fi
  last_signature="$signature"
  sequence=$((sequence + 1))
}

echo "[P38.GCS] LIVE_WORKER_START interval=$CANON_P38_LIVE_SNAPSHOT_INTERVAL_SECONDS"
while [ ! -e "$CANON_P38_LIVE_SNAPSHOT_STOP_FILE" ]; do
  snapshot_if_changed
  sleep "$CANON_P38_LIVE_SNAPSHOT_INTERVAL_SECONDS" &
  wait "$!" || true
done
snapshot_if_changed
if [ "$sequence" -eq 0 ]; then
  echo "[P38.GCS] FATAL: live worker observed no host evidence" >&2
  exit 1
fi
echo "[P38.GCS] LIVE_WORKER_COMPLETE snapshots=$sequence"
