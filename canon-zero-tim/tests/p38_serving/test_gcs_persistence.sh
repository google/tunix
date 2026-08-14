#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PERSIST="$ROOT/tasks/p38-pathways-decode-prefill-carrier/scripts/persist_p38_gcs.sh"
tmp="$(mktemp -d)"
trap 'rm -r "$tmp"' EXIT

install_fake_gcloud() {
  local root="$1"
  mkdir -p "$root/bin" "$root/gcs"
  cp "$ROOT/tests/p38_serving/fake_gcloud.sh" "$root/bin/gcloud"
  chmod +x "$root/bin/gcloud"
  export PATH="$root/bin:$PATH"
  export FAKE_GCS_ROOT="$root/gcs"
}

make_case() {
  local root="$1" job="$2"
  mkdir -p "$root/state/capture"
  printf 'run\n' > "$root/state/run.log"
  printf '{}\n' > "$root/state/pre.jsonl"
  printf 'capsule\n' > "$root/state/capsule.npz"
  printf '{"verdict":"PASS"}\n' > "$root/state/classification.json"
  printf '{}\n' > "$root/state/capture/p38_request_journal.jsonl"
  printf '{}\n' > "$root/state/capture/p38_incident_ledger.jsonl"
  printf '0\n' > "$root/state/p38_diagnostic_round"
  tar -C "$root/state/capture" -cf "$root/state/capture.tar" .
  export CANON_STATE="$root/state"
  export CANON_PKG="$ROOT"
  export CANON_RUN_LOG="$root/state/run.log"
  export CANON_PRE_ALIGN_REPORT="$root/state/pre.jsonl"
  export CANON_P38_MISMATCH_CAPSULE="$root/state/capsule.npz"
  export CANON_P38_REQUEST_JOURNAL="$root/state/capture/p38_request_journal.jsonl"
  export CANON_P38_INCIDENT_LEDGER="$root/state/capture/p38_incident_ledger.jsonl"
  export CANON_P38_DIAGNOSTIC_ROUND_FILE="$root/state/p38_diagnostic_round"
  export CANON_P38_SERVING_CAPTURE_CLASSIFICATION="$root/state/classification.json"
  export CANON_P38_SERVING_CAPTURE_ARCHIVE="$root/state/capture.tar"
  export CANON_P38_GCS_PREFIX="gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/$job/attempt-0"
  export CANON_EXPECT_COMMIT="$(printf 'a%.0s' {1..40})"
  export CANON_POD_NAME="$job-head"
  export JOBSET_RESTART_ATTEMPT=0
}

install_fake_gcloud "$tmp/pass"
make_case "$tmp/pass" canon-p38-test-pass
bash "$PERSIST" probe > "$tmp/pass/probe.log"
if bash "$PERSIST" probe > "$tmp/pass/reused-prefix.log" 2>&1; then
  echo "[P38.GCS] reused attempt prefix was accepted" >&2
  exit 1
fi
grep -q 'remote marker already exists: PREFLIGHT.json' \
  "$tmp/pass/reused-prefix.log"
printf 'round-0\n' > "${CANON_P38_MISMATCH_CAPSULE%.npz}.round-000000.npz"
bash "$PERSIST" snapshot 000000 > "$tmp/pass/snapshot.log"
live="$FAKE_GCS_ROOT/yuxzhang-tunix-models/canon-zero-tim/evidence/p38/canon-p38-test-pass/attempt-0/live/000000"
for name in run.log request-journal.jsonl incident-ledger.jsonl \
    diagnostic-round.txt pre-alignment.jsonl capsule.npz \
    capsule.round-000000.npz \
    SHA256SUMS LIVE.json; do
  test -s "$live/$name"
done
(cd "$live" && sha256sum -c SHA256SUMS --quiet)
grep -q '"schema": "canon-p38-gcs-live-v1"' "$live/LIVE.json"
if bash "$PERSIST" snapshot 000000 > "$tmp/pass/reused-live.log" 2>&1; then
  echo "[P38.GCS] repeated live snapshot was accepted" >&2
  exit 1
fi
grep -q 'live snapshot already exists' "$tmp/pass/reused-live.log"
bash "$PERSIST" collect > "$tmp/pass/collect.log"
remote="$FAKE_GCS_ROOT/yuxzhang-tunix-models/canon-zero-tim/evidence/p38/canon-p38-test-pass/attempt-0"
for name in PREFLIGHT.json run.log pre-alignment.jsonl mismatch-capsule.npz \
    mismatch-capsule.round-000000.npz \
    serving-classification.json serving-capture.tar SHA256SUMS COLLECTED.json; do
  test -s "$remote/$name"
done
test ! -e "$remote/COMPLETE.json"
bash "$PERSIST" complete > "$tmp/pass/complete.log"
test -s "$remote/COMPLETE.json"
grep -q '"status": "postflight-accepted"' "$remote/COMPLETE.json"
(cd "$remote" && sha256sum -c SHA256SUMS --quiet)
if bash "$PERSIST" complete > "$tmp/pass/repeat.log" 2>&1; then
  echo "[P38.GCS] repeated completion was accepted" >&2
  exit 1
fi

install_fake_gcloud "$tmp/missing"
make_case "$tmp/missing" canon-p38-test-missing
: > "$CANON_P38_SERVING_CAPTURE_ARCHIVE"
if bash "$PERSIST" collect > "$tmp/missing/run.log" 2>&1; then
  echo "[P38.GCS] collection accepted an empty archive" >&2
  exit 1
fi
grep -q 'required artifact missing or empty' "$tmp/missing/run.log"

install_fake_gcloud "$tmp/upload-fail"
make_case "$tmp/upload-fail" canon-p38-test-upload-fail
export FAKE_GCS_FAIL_CP=1
if bash "$PERSIST" probe > "$tmp/upload-fail/run.log" 2>&1; then
  echo "[P38.GCS] preflight accepted a failed upload" >&2
  exit 1
fi
unset FAKE_GCS_FAIL_CP

install_fake_gcloud "$tmp/worker"
make_case "$tmp/worker" canon-p38-test-worker
bash "$PERSIST" probe > "$tmp/worker/probe.log"
export CANON_P38_LIVE_SNAPSHOT_INTERVAL_SECONDS=1
export CANON_P38_LIVE_SNAPSHOT_STOP_FILE="$tmp/worker/state/live.stop"
worker_log="$tmp/worker/state/live-worker.log"
bash "$ROOT/tasks/p38-pathways-decode-prefill-carrier/scripts/p38_live_snapshot_worker.sh" \
  > "$worker_log" 2>&1 &
worker_pid=$!
printf 'run-more\n' >> "$CANON_RUN_LOG"
for unused in 1 2 3 4 5; do
  if find "$FAKE_GCS_ROOT" -path '*/live/*/LIVE.json' -type f | grep -q .; then
    break
  fi
  sleep 1
done
if ! kill -0 "$worker_pid" 2>/dev/null; then
  echo "[P38.GCS] live snapshot worker exited before the stop signal" >&2
  cat "$worker_log" >&2
  wait "$worker_pid" || true
  exit 1
fi
touch "$CANON_P38_LIVE_SNAPSHOT_STOP_FILE"
wait "$worker_pid"
grep -q 'LIVE_WORKER_COMPLETE snapshots=' "$worker_log"
worker_live="$(find "$FAKE_GCS_ROOT" -path '*/live/*/LIVE.json' -type f | head -n 1)"
test -s "$worker_live"
worker_live_dir="$(dirname "$worker_live")"
(cd "$worker_live_dir" && sha256sum -c SHA256SUMS --quiet)

echo "[P38.GCS] PERSISTENCE_TEST_PASS probe=verified prefix_reuse=rejected live=immutable worker=durable collected=verified complete=last missing=rejected upload_failure=rejected"
