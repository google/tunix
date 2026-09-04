#!/usr/bin/env bash
# Periodically mirror immutable P57 TiTO evidence and finalize on request.
set -euo pipefail

: "${CANON_PKG:?CANON_PKG unset}"
: "${CANON_STATE:?CANON_STATE unset}"
: "${CANON_P57_TITO_GCS_PREFIX:?CANON_P57_TITO_GCS_PREFIX unset}"
: "${CANON_P57_TITO_GCS_INTERVAL_SECONDS:?interval unset}"
: "${CANON_P57_TITO_GCS_READY:?ready file unset}"
: "${CANON_P57_TITO_GCS_STOP_FILE:?stop file unset}"
: "${CANON_P57_TITO_GCS_FINALIZE_FILE:?finalize file unset}"
: "${CANON_P57_TITO_GCS_FINAL_ACK:?final ack unset}"
: "${CANON_P57_TITO_GCS_HEARTBEAT:?heartbeat file unset}"

case "$CANON_P57_TITO_GCS_INTERVAL_SECONDS" in
  ''|*[!0-9]*)
    echo "[P57.TITO.GCS] REFUSING: interval must be an integer" >&2
    exit 2
    ;;
esac
if [ "$CANON_P57_TITO_GCS_INTERVAL_SECONDS" -lt 1 ]; then
  echo "[P57.TITO.GCS] REFUSING: interval must be positive" >&2
  exit 2
fi
for control in \
    "$CANON_P57_TITO_GCS_READY" \
    "$CANON_P57_TITO_GCS_STOP_FILE" \
    "$CANON_P57_TITO_GCS_FINALIZE_FILE" \
    "$CANON_P57_TITO_GCS_FINAL_ACK" \
    "$CANON_P57_TITO_GCS_HEARTBEAT"; do
  if [ -e "$control" ]; then
    echo "[P57.TITO.GCS] REFUSING: stale worker control file" >&2
    exit 2
  fi
done

sync_script="$CANON_PKG/tasks/multiturn-tito-cross-workload/scripts/sync_tito_evidence_to_gcs.py"
run_sync() {
  local command=(python3 "$sync_script" "$@")
  if command -v ionice >/dev/null 2>&1; then
    ionice -c 3 "${command[@]}"
  else
    "${command[@]}"
  fi
}
sync_once() {
  run_sync \
    --state "$CANON_STATE" \
    --gcs-prefix "$CANON_P57_TITO_GCS_PREFIX"
}
heartbeat() {
  local status="$1"
  local failures="$2"
  local partial="${CANON_P57_TITO_GCS_HEARTBEAT}.partial"
  (umask 077; printf 'status=%s failures=%s unix=%s\n' \
    "$status" "$failures" "$(date +%s)" > "$partial")
  mv -- "$partial" "$CANON_P57_TITO_GCS_HEARTBEAT"
  echo "[P57.TITO.GCS] HEARTBEAT status=$status failures=$failures"
}

run_sync \
  --state "$CANON_STATE" \
  --gcs-prefix "$CANON_P57_TITO_GCS_PREFIX" \
  --probe
ready_partial="${CANON_P57_TITO_GCS_READY}.partial"
(umask 077; printf 'action=ready status=PASS\n' > "$ready_partial")
mv -- "$ready_partial" "$CANON_P57_TITO_GCS_READY"
echo "[P57.TITO.GCS] READY"
echo "[P57.TITO.GCS] WORKER_START interval=$CANON_P57_TITO_GCS_INTERVAL_SECONDS"
live_failures=0
heartbeat PASS "$live_failures"
while [ ! -e "$CANON_P57_TITO_GCS_STOP_FILE" ]; do
  if [ -e "$CANON_P57_TITO_GCS_FINALIZE_FILE" ]; then
    run_sync \
      --state "$CANON_STATE" \
      --gcs-prefix "$CANON_P57_TITO_GCS_PREFIX" \
      --final
    partial="${CANON_P57_TITO_GCS_FINAL_ACK}.partial"
    (umask 077; printf 'action=finalize status=PASS\n' > "$partial")
    mv -- "$partial" "$CANON_P57_TITO_GCS_FINAL_ACK"
    heartbeat PASS "$live_failures"
    echo "[P57.TITO.GCS] FINAL_PASS"
    exit 0
  fi
  if sync_once; then
    heartbeat PASS "$live_failures"
  else
    live_failures=$((live_failures + 1))
    heartbeat DEGRADED "$live_failures"
  fi
  sleep "$CANON_P57_TITO_GCS_INTERVAL_SECONDS" &
  wait "$!" || true
done
if sync_once; then
  heartbeat PASS "$live_failures"
else
  live_failures=$((live_failures + 1))
  heartbeat DEGRADED "$live_failures"
fi
echo "[P57.TITO.GCS] WORKER_STOPPED final=0"
