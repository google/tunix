#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PERSIST="$ROOT/tasks/p38-pathways-decode-prefill-carrier/scripts/persist_p38_gcs.sh"
ARCHIVE_TOOL="$ROOT/tasks/p38-pathways-decode-prefill-carrier/scripts/p38_evidence_archive.py"
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
  unset CANON_P38_KV_OBSERVER_DIR CANON_P38_KV_OBSERVER_CLASSIFICATION \
    CANON_P38_SEAM_OBSERVER_DIR CANON_P38_TAIL_OBSERVER \
    CANON_P38_FIXED_LM_HEAD CANON_APC_M15_REPLAY_LEDGER || true
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
  export CANON_P38_ROUND_SEAL_REQUEST_DIR="$root/state/p38_round_seal_requests"
  export CANON_P38_ROUND_SEAL_ACK_DIR="$root/state/p38_round_seal_acks"
  mkdir -p "$CANON_P38_ROUND_SEAL_REQUEST_DIR" \
    "$CANON_P38_ROUND_SEAL_ACK_DIR"
  export CANON_P38_SERVING_CAPTURE_CLASSIFICATION="$root/state/classification.json"
  export CANON_P38_SERVING_CAPTURE_ARCHIVE="$root/state/capture.tar"
  export CANON_P38_GCS_PREFIX="gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/$job/attempt-0"
  export CANON_P38_DURABILITY_PROFILE=full-v1
  export CANON_EXPECT_COMMIT="$(git -C "$ROOT/.." rev-parse HEAD)"
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
printf '{"schema":"m15-apc-serving-envelope-v1"}\n' > \
  "$tmp/pass/state/capture/m15_replay_envelope.jsonl"
export CANON_APC_M15_REPLAY_LEDGER="$tmp/pass/state/capture/m15_replay_envelope.jsonl"
bash "$PERSIST" snapshot 000000 > "$tmp/pass/snapshot.log"
live="$FAKE_GCS_ROOT/yuxzhang-tunix-models/canon-zero-tim/evidence/p38/canon-p38-test-pass/attempt-0/live/000000"
for name in LIVE_ARCHIVE.tar SHA256SUMS LIVE.json; do
  test -s "$live/$name"
done
test "$(find "$live" -maxdepth 1 -type f | wc -l)" -eq 3
python3 "$ARCHIVE_TOOL" extract \
  --archive "$live/LIVE_ARCHIVE.tar" \
  --output "$tmp/pass/live-extracted" > "$tmp/pass/live-extract.log"
(cd "$tmp/pass/live-extracted" && sha256sum -c SHA256SUMS --quiet)
for name in run.log request-journal.jsonl incident-ledger.jsonl \
    m15-replay-envelope.jsonl \
    diagnostic-round.txt pre-alignment.jsonl capsule.npz \
    capsule.round-000000.npz; do
  test -s "$tmp/pass/live-extracted/$name"
done
grep -q '"schema": "canon-p38-gcs-live-v1"' "$live/LIVE.json"
grep -q '"transport": "single-deterministic-tar-v1"' "$live/LIVE.json"
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
export CANON_P38_KV_OBSERVER_DIR="$tmp/worker/state/capture"
printf '{"arm":"A","diagnostic_round":0}\n' > \
  "$CANON_P38_KV_OBSERVER_DIR/p38_kv_observer_0000_a.json"
printf 'observer-npz\n' > \
  "$CANON_P38_KV_OBSERVER_DIR/p38_kv_observer_0000_a.npz"
export CANON_P38_LIVE_SNAPSHOT_INTERVAL_SECONDS=1
export CANON_P38_LIVE_SNAPSHOT_STOP_FILE="$tmp/worker/state/live.stop"
export CANON_P38_LIVE_COLLECT_REQUEST_FILE="$tmp/worker/state/collect.request"
export CANON_P38_LIVE_COLLECT_ACK_FILE="$tmp/worker/state/collect.ack"
export CANON_P38_LIVE_COMPLETE_REQUEST_FILE="$tmp/worker/state/complete.request"
export CANON_P38_LIVE_COMPLETE_ACK_FILE="$tmp/worker/state/complete.ack"
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
printf 'action=collect\n' > "$CANON_P38_LIVE_COLLECT_REQUEST_FILE.partial"
mv "$CANON_P38_LIVE_COLLECT_REQUEST_FILE.partial" \
  "$CANON_P38_LIVE_COLLECT_REQUEST_FILE"
for unused in 1 2 3 4 5; do
  [ ! -s "$CANON_P38_LIVE_COLLECT_ACK_FILE" ] || break
  sleep 1
done
grep -q '^action=collect status=PASS$' "$CANON_P38_LIVE_COLLECT_ACK_FILE"
test -s "$FAKE_GCS_ROOT/yuxzhang-tunix-models/canon-zero-tim/evidence/p38/canon-p38-test-worker/attempt-0/COLLECTED.json"
test ! -e "$FAKE_GCS_ROOT/yuxzhang-tunix-models/canon-zero-tim/evidence/p38/canon-p38-test-worker/attempt-0/COMPLETE.json"
printf 'action=complete\n' > "$CANON_P38_LIVE_COMPLETE_REQUEST_FILE.partial"
mv "$CANON_P38_LIVE_COMPLETE_REQUEST_FILE.partial" \
  "$CANON_P38_LIVE_COMPLETE_REQUEST_FILE"
for unused in 1 2 3 4 5; do
  [ ! -s "$CANON_P38_LIVE_COMPLETE_ACK_FILE" ] || break
  sleep 1
done
grep -q '^action=complete status=PASS$' "$CANON_P38_LIVE_COMPLETE_ACK_FILE"
test -s "$FAKE_GCS_ROOT/yuxzhang-tunix-models/canon-zero-tim/evidence/p38/canon-p38-test-worker/attempt-0/COMPLETE.json"
touch "$CANON_P38_LIVE_SNAPSHOT_STOP_FILE"
wait "$worker_pid"
grep -q 'LIVE_COLLECT_PASS' "$worker_log"
grep -q 'LIVE_COMPLETE_PASS' "$worker_log"
grep -q 'LIVE_WORKER_COMPLETE snapshots=' "$worker_log"
worker_live="$(find "$FAKE_GCS_ROOT" -path '*/live/*/LIVE.json' -type f | head -n 1)"
test -s "$worker_live"
worker_live_dir="$(dirname "$worker_live")"
test "$(find "$worker_live_dir" -maxdepth 1 -type f | wc -l)" -eq 3
python3 "$ARCHIVE_TOOL" extract \
  --archive "$worker_live_dir/LIVE_ARCHIVE.tar" \
  --output "$tmp/worker/live-extracted" > "$tmp/worker/live-extract.log"
(cd "$tmp/worker/live-extracted" && sha256sum -c SHA256SUMS --quiet)
test -s "$tmp/worker/live-extracted/p38_kv_observer_0000_a.json"
test -s "$tmp/worker/live-extracted/p38_kv_observer_0000_a.npz"

install_fake_gcloud "$tmp/rounds"
make_case "$tmp/rounds" canon-p38-test-rounds
bash "$PERSIST" probe > "$tmp/rounds/probe.log"
unset CANON_P38_KV_OBSERVER_DIR CANON_P38_KV_OBSERVER_CLASSIFICATION || true
export CANON_P38_SEAM_OBSERVER_DIR="$tmp/rounds/state/capture"
: > "$CANON_PRE_ALIGN_REPORT"
: > "$CANON_P38_REQUEST_JOURNAL"
: > "$CANON_P38_INCIDENT_LEDGER"
for round_index in 0 1; do
  printf '{"diagnostic_round":%s,"step":0}\n' "$round_index" \
    >> "$CANON_PRE_ALIGN_REPORT"
  printf '{"schema":"p38-request-journal-v1","kind":"journal","sequence":%s}\n' "$round_index" \
    >> "$CANON_P38_REQUEST_JOURNAL"
  printf '{"diagnostic_round":%s,"schema":"p38-incident-ledger-v1","kind":"incident"}\n' "$round_index" \
    >> "$CANON_P38_INCIDENT_LEDGER"
  printf 'capsule-%s\n' "$round_index" > \
    "${CANON_P38_MISMATCH_CAPSULE%.npz}.round-$(printf '%06d' "$round_index").npz"
  python3 - "$CANON_P38_SEAM_OBSERVER_DIR" "$round_index" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
round_index = int(sys.argv[2])
stem = root / f"p38_seam_{round_index:06d}"
npz = pathlib.Path(str(stem) + ".npz")
npz.write_bytes(f"seam-{round_index}\n".encode())
pathlib.Path(str(stem) + ".json").write_text(json.dumps({
    "diagnostic_round": round_index,
    "npz_sha256": hashlib.sha256(npz.read_bytes()).hexdigest(),
    "schema": "p38-seam-fingerprint-v1",
}) + "\n", encoding="utf-8")
PY
done
export CANON_P38_LIVE_SNAPSHOT_INTERVAL_SECONDS=1
export CANON_P38_LIVE_SNAPSHOT_STOP_FILE="$tmp/rounds/state/live.stop"
export CANON_P38_LIVE_COLLECT_REQUEST_FILE="$tmp/rounds/state/collect.request"
export CANON_P38_LIVE_COLLECT_ACK_FILE="$tmp/rounds/state/collect.ack"
export CANON_P38_LIVE_COMPLETE_REQUEST_FILE="$tmp/rounds/state/complete.request"
export CANON_P38_LIVE_COMPLETE_ACK_FILE="$tmp/rounds/state/complete.ack"
make_round_request() {
  local round_index="$1" request
  request="$CANON_P38_ROUND_SEAL_REQUEST_DIR/round-$(printf '%06d' "$round_index").request"
  python3 - "$request.partial" "$round_index" <<'PY'
import json
import pathlib
import sys

pathlib.Path(sys.argv[1]).write_text(json.dumps({
    "action": "seal-round",
    "diagnostic_round": int(sys.argv[2]),
    "schema": "canon-p38-round-seal-request-v1",
}, sort_keys=True) + "\n", encoding="utf-8")
PY
  mv "$request.partial" "$request"
}
# Publish round 0 before worker startup.  The critical-path request must be
# serviced before the first periodic live snapshot.
make_round_request 0
round_worker_log="$tmp/rounds/state/live-worker.log"
bash "$ROOT/tasks/p38-pathways-decode-prefill-carrier/scripts/p38_live_snapshot_worker.sh" \
  > "$round_worker_log" 2>&1 &
round_worker_pid=$!
for round_index in 0 1; do
  if [ "$round_index" -ne 0 ]; then
    make_round_request "$round_index"
  fi
  ack="$CANON_P38_ROUND_SEAL_ACK_DIR/round-$(printf '%06d' "$round_index").ack"
  for unused in 1 2 3 4 5 6 7 8 9 10; do
    [ ! -s "$ack" ] || break
    sleep 1
  done
  grep -q '"status": "PASS"' "$ack"
done
python3 - "$round_worker_log" <<'PY'
import pathlib
import sys

lines = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
round_pass = next(i for i, line in enumerate(lines) if "LIVE_ROUND_PASS round=0" in line)
live_snapshot = next(i for i, line in enumerate(lines) if "[P38.GCS] LIVE sequence=" in line)
assert round_pass < live_snapshot, (round_pass, live_snapshot)
PY
# Simulate the pod disappearing after round 1.  Neither final collect nor
# COMPLETE is allowed to be necessary for the two sealed round bundles.
kill "$round_worker_pid"
wait "$round_worker_pid" || true
round_remote="$FAKE_GCS_ROOT/yuxzhang-tunix-models/canon-zero-tim/evidence/p38/canon-p38-test-rounds/attempt-0/rounds"
for sequence in 000000 000001; do
  test -s "$round_remote/$sequence/ROUND_COMPLETE.json"
  test -s "$round_remote/$sequence/ROUND_ARCHIVE.tar"
  test -s "$round_remote/$sequence/SHA256SUMS"
  test "$(find "$round_remote/$sequence" -maxdepth 1 -type f | wc -l)" -eq 3
  python3 "$ARCHIVE_TOOL" extract \
    --archive "$round_remote/$sequence/ROUND_ARCHIVE.tar" \
    --output "$tmp/rounds/extracted-$sequence" \
    > "$tmp/rounds/extract-$sequence.log"
  (cd "$tmp/rounds/extracted-$sequence" && sha256sum -c SHA256SUMS --quiet)
done
python3 - "$tmp/rounds" "$round_remote" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
remote = pathlib.Path(sys.argv[2])
for round_index in (0, 1):
  bundle = root / f"extracted-{round_index:06d}"
  pre = [
      json.loads(line)
      for line in (bundle / "pre-alignment.jsonl").read_text().splitlines()
  ]
  incident = [
      json.loads(line)
      for line in (bundle / "incident-ledger.jsonl").read_text().splitlines()
  ]
  journal = [
      json.loads(line)
      for line in (bundle / "request-journal.jsonl").read_text().splitlines()
  ]
  inventory = json.loads((bundle / "ROUND_INVENTORY.json").read_text())
  assert {record["diagnostic_round"] for record in pre} == {round_index}, pre
  assert {record["diagnostic_round"] for record in incident} == {round_index}, incident
  assert len(journal) == 2, journal
  assert {record["schema"] for record in journal} == {
      "p38-request-journal-v1"
  }, journal
  assert inventory["journal_scope"] == "cumulative-unscoped", inventory
  assert inventory["profile"] == "full", inventory
  completion = json.loads(
      (remote / f"{round_index:06d}" / "ROUND_COMPLETE.json").read_text()
  )
  assert completion["logical_file_count"] == len(
      (bundle / "SHA256SUMS").read_text().splitlines()
  ), completion
  assert completion["transport"] == "single-deterministic-tar-v1", completion
PY
test ! -e "$FAKE_GCS_ROOT/yuxzhang-tunix-models/canon-zero-tim/evidence/p38/canon-p38-test-rounds/attempt-0/COLLECTED.json"

# The fixed-lm-head discriminator has no use for the old KV/seam evidence.
# Its durability profile must seal one bounded alignment archive without
# starting a periodic live snapshot that could starve the round request.
install_fake_gcloud "$tmp/alignment-round"
make_case "$tmp/alignment-round" canon-p38-test-alignment-round
bash "$PERSIST" probe > "$tmp/alignment-round/probe.log"
export CANON_P38_DURABILITY_PROFILE=round-alignment-v1
export CANON_P38_FIXED_LM_HEAD=1
: > "$CANON_PRE_ALIGN_REPORT"
printf '{"diagnostic_round":0,"step":0}\n' > "$CANON_PRE_ALIGN_REPORT"
# Exact P38s23r3 rounds intentionally produce no mismatch capsule.  The
# durability path must still seal and acknowledge the alignment record.
rm "$CANON_P38_MISMATCH_CAPSULE"
# These full-forensics inputs are deliberately absent.  The alignment-only
# round must neither read nor silently recreate them.
rm "$CANON_P38_REQUEST_JOURNAL" "$CANON_P38_INCIDENT_LEDGER"
export CANON_P38_LIVE_SNAPSHOT_INTERVAL_SECONDS=1
export CANON_P38_LIVE_SNAPSHOT_STOP_FILE="$tmp/alignment-round/state/live.stop"
export CANON_P38_LIVE_COLLECT_REQUEST_FILE="$tmp/alignment-round/state/collect.request"
export CANON_P38_LIVE_COLLECT_ACK_FILE="$tmp/alignment-round/state/collect.ack"
export CANON_P38_LIVE_COMPLETE_REQUEST_FILE="$tmp/alignment-round/state/complete.request"
export CANON_P38_LIVE_COMPLETE_ACK_FILE="$tmp/alignment-round/state/complete.ack"
make_round_request 0
alignment_worker_log="$tmp/alignment-round/state/live-worker.log"
bash "$ROOT/tasks/p38-pathways-decode-prefill-carrier/scripts/p38_live_snapshot_worker.sh" \
  > "$alignment_worker_log" 2>&1 &
alignment_worker_pid=$!
alignment_ack="$CANON_P38_ROUND_SEAL_ACK_DIR/round-000000.ack"
for unused in 1 2 3 4 5 6 7 8 9 10; do
  [ ! -s "$alignment_ack" ] || break
  sleep 1
done
grep -q '"status": "PASS"' "$alignment_ack"
touch "$CANON_P38_LIVE_SNAPSHOT_STOP_FILE"
wait "$alignment_worker_pid"
grep -q 'LIVE_WORKER_START .*profile=round-alignment-v1' \
  "$alignment_worker_log"
grep -q 'LIVE_WORKER_COMPLETE snapshots=0 rounds=1 profile=round-alignment-v1' \
  "$alignment_worker_log"
alignment_remote="$FAKE_GCS_ROOT/yuxzhang-tunix-models/canon-zero-tim/evidence/p38/canon-p38-test-alignment-round/attempt-0"
test ! -e "$alignment_remote/live"
test "$(find "$alignment_remote/rounds/000000" -maxdepth 1 -type f | wc -l)" -eq 3
python3 "$ARCHIVE_TOOL" extract \
  --archive "$alignment_remote/rounds/000000/ROUND_ARCHIVE.tar" \
  --output "$tmp/alignment-round/extracted" \
  > "$tmp/alignment-round/extract.log"
(cd "$tmp/alignment-round/extracted" && sha256sum -c SHA256SUMS --quiet)
for name in pre-alignment.jsonl run.log ROUND_INVENTORY.json; do
  test -s "$tmp/alignment-round/extracted/$name"
done
for name in mismatch-capsule.npz request-journal.jsonl incident-ledger.jsonl; do
  test ! -e "$tmp/alignment-round/extracted/$name"
done
python3 - "$tmp/alignment-round/extracted/ROUND_INVENTORY.json" \
  "$alignment_remote/rounds/000000/ROUND_COMPLETE.json" <<'PY'
import json
import pathlib
import sys

inventory = json.loads(pathlib.Path(sys.argv[1]).read_text())
completion = json.loads(pathlib.Path(sys.argv[2]).read_text())
assert inventory["profile"] == "alignment-only", inventory
assert inventory["journal_scope"] == "omitted-by-alignment-only-profile", inventory
assert inventory["journal_records"] == 0, inventory
assert inventory["incident_records"] == 0, inventory
assert inventory["kv_records"] == 0, inventory
assert inventory["capsule_present"] is False, inventory
assert completion["durability_profile"] == "round-alignment-v1", completion
assert completion["logical_file_count"] == 3, completion
PY
bash "$PERSIST" collect > "$tmp/alignment-round/collect.log"
bash "$PERSIST" complete > "$tmp/alignment-round/complete.log"
test -s "$alignment_remote/COLLECTED.json"
test -s "$alignment_remote/COMPLETE.json"
test ! -e "$alignment_remote/mismatch-capsule.npz"
(cd "$alignment_remote" && sha256sum -c SHA256SUMS --quiet)

# The E0 KV carrier seals, classifies, uploads, and reads back each of exactly
# three rounds before the learner ACK.  Verify that all round evidence remains
# independently recoverable before root collection exists.
install_fake_gcloud "$tmp/m15-e0-kv"
make_case "$tmp/m15-e0-kv" canon-p38-test-m15-e0-kv
export CANON_P38_DURABILITY_PROFILE=m15-e0-kv-v1
export CANON_P38_FIXED_LM_HEAD=1
export CANON_P38_DIAGNOSTIC_ROUNDS=3
export CANON_APC_M15_TARGET_DEBUG=off
export CANON_P38_KV_OBSERVER_DIR="$tmp/m15-e0-kv/state/observer"
export CANON_P38_KV_OBSERVER_CLASSIFICATION="$tmp/m15-e0-kv/state/p38_kv_observer.classification.json"
export CANON_APC_M15_REPLAY_LEDGER="$tmp/m15-e0-kv/state/m15-replay-envelope.jsonl"
export CANON_P38_MISMATCH_CAPSULE="$tmp/m15-e0-kv/state/mismatch-capsule.npz"
python3 - "$ROOT/tasks/v1-apc-m15-target-debug/scripts/test_m15_e0_kv_three_round.py" \
    "$tmp/m15-e0-kv/state" <<'PY'
import importlib.util
import pathlib
import sys

spec = importlib.util.spec_from_file_location("e0_fixture", sys.argv[1])
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
root = pathlib.Path(sys.argv[2])
fixture = module.E0KvThreeRoundTest()._stage_fixture(root)
assert fixture["observer"] == root / "observer"
PY
export CANON_PRE_ALIGN_REPORT="$tmp/m15-e0-kv/state/pre-alignment.jsonl"
bash "$PERSIST" probe > "$tmp/m15-e0-kv/probe.log"
for round_index in 0 1 2; do
  printf -v round_text '%06d' "$round_index"
  bash "$PERSIST" m15-e0-round "$round_text" \
    > "$tmp/m15-e0-kv/round-$round_text.log"
done
m15_e0_remote="$FAKE_GCS_ROOT/yuxzhang-tunix-models/canon-zero-tim/evidence/p38/canon-p38-test-m15-e0-kv/attempt-0"
test ! -e "$m15_e0_remote/COLLECTED.json"
test ! -e "$m15_e0_remote/COMPLETE.json"
for round_index in 0 1 2; do
  printf -v round_text '%06d' "$round_index"
  round_remote="$m15_e0_remote/rounds/$round_text"
  for name in ROUND_ARCHIVE.tar SHA256SUMS ROUND_INPUT.json \
      kv-observer-classification.json ROUND_COMPLETE.json; do
    test -s "$round_remote/$name"
  done
  for name in CLASSIFIER_INPUT_ARCHIVE.tar CLASSIFIER_INPUT_SHA256SUMS \
      CLASSIFIER_INPUT_RECEIPT.json; do
    test -s "$round_remote/classifier-input/$name"
  done
  grep -q 'M15_E0_CLASSIFIER_INPUT_CHECKPOINT .*status=uploaded-readback-verified' \
    "$tmp/m15-e0-kv/round-$round_text.log"
  grep -q 'M15_E0_ROUND_COMPLETE .*input_checkpoint=PASS classifier=PASS upload=PASS readback=PASS' \
    "$tmp/m15-e0-kv/round-$round_text.log"
done
python3 "$ROOT/tasks/v1-apc-m15-target-debug/scripts/aggregate_m15_e0_kv_rounds.py" \
  --root "$CANON_STATE/p38_m15_e0_kv_rounds" \
  --arm off --rounds 3 --expected-source "$CANON_EXPECT_COMMIT" \
  --output "$CANON_P38_KV_OBSERVER_CLASSIFICATION" \
  > "$tmp/m15-e0-kv/aggregate.log"
bash "$PERSIST" collect > "$tmp/m15-e0-kv/collect.log"
test -s "$m15_e0_remote/COLLECTED.json"
grep -q '"status": "CONTROL_EXACT_3_OF_3"' \
  "$m15_e0_remote/kv-observer-classification.json"
grep -q 'profile=m15-e0-kv-v1 rounds=3' "$tmp/m15-e0-kv/collect.log"

# The M15 wide observer never waits for a terminal multi-GiB tar.  Completed
# JSON/NPZ pairs are copied into bounded shards and each shard publishes its
# completion marker only after a remote download-and-verify round trip.
install_fake_gcloud "$tmp/m15-wide"
make_case "$tmp/m15-wide" canon-p38-test-m15-wide
export CANON_P38_DURABILITY_PROFILE=m15-wide-v1
export CANON_P38_SEAM_OBSERVER_DIR="$tmp/m15-wide/state/capture"
export CANON_P38_SEAM_OBSERVER=full
bash "$PERSIST" probe > "$tmp/m15-wide/probe.log"
grep -q 'RUNTIME_SOURCE_PASS' "$tmp/m15-wide/probe.log"
for index in $(seq 0 39); do
  python3 - "$CANON_P38_SEAM_OBSERVER_DIR" "$index" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
index = int(sys.argv[2])
npz = root / f"p38_seam_{index:06d}.npz"
npz.write_bytes((f"seam-{index}\n" * 32).encode())
(root / f"p38_seam_{index:06d}.json").write_text(json.dumps({
    "diagnostic_round": 0,
    "npz_sha256": hashlib.sha256(npz.read_bytes()).hexdigest(),
    "record_index": index,
    "schema": "p38-seam-fingerprint-v1",
}, sort_keys=True) + "\n", encoding="utf-8")
PY
done
bash "$PERSIST" m15-shard 000000 > "$tmp/m15-wide/shard-0.log"
bash "$PERSIST" m15-shard 000001 > "$tmp/m15-wide/shard-1.log"
m15_empty_rc=0
bash "$PERSIST" m15-shard 000002 \
  > "$tmp/m15-wide/shard-empty.log" 2>&1 || m15_empty_rc=$?
test "$m15_empty_rc" -eq 3
# Advance the live round and prove that the same worker can seal a fresh local
# shard union without rereading round 0 or colliding with its global indices.
printf '1\n' > "$CANON_P38_DIAGNOSTIC_ROUND_FILE"
python3 - "$CANON_P38_SEAM_OBSERVER_DIR" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
index = 40
npz = root / f"p38_seam_{index:06d}.npz"
npz.write_bytes(b"seam-round-1\n")
(root / f"p38_seam_{index:06d}.json").write_text(json.dumps({
    "diagnostic_round": 1,
    "npz_sha256": hashlib.sha256(npz.read_bytes()).hexdigest(),
    "record_index": index,
    "schema": "p38-seam-fingerprint-v1",
}, sort_keys=True) + "\n", encoding="utf-8")
PY
bash "$PERSIST" m15-shard 000002 > "$tmp/m15-wide/shard-round-1.log"
m15_wide_remote="$FAKE_GCS_ROOT/yuxzhang-tunix-models/canon-zero-tim/evidence/p38/canon-p38-test-m15-wide/attempt-0"
for sequence in 000000 000001 000002; do
  shard_remote="$m15_wide_remote/wide/shards/$sequence"
  test "$(find "$shard_remote" -maxdepth 1 -type f | wc -l)" -eq 3
  for name in SHARD_ARCHIVE.tar SHA256SUMS SHARD_COMPLETE.json; do
    test -s "$shard_remote/$name"
  done
  python3 "$ARCHIVE_TOOL" extract \
    --archive "$shard_remote/SHARD_ARCHIVE.tar" \
    --output "$tmp/m15-wide/extracted-$sequence" \
    > "$tmp/m15-wide/extract-$sequence.log"
  (cd "$tmp/m15-wide/extracted-$sequence" && \
    sha256sum -c SHA256SUMS --quiet)
done
# This is the forced-death state: useful observer shards are already durable,
# while terminal publication correctly has not happened.
test ! -e "$m15_wide_remote/COLLECTED.json"
test ! -e "$m15_wide_remote/COMPLETE.json"
python3 - "$m15_wide_remote" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1]) / "wide" / "shards"
receipts = [
    json.loads((root / sequence / "SHARD_COMPLETE.json").read_text())
    for sequence in ("000000", "000001", "000002")
]
assert [row["record_pairs"] for row in receipts] == [32, 8, 1], receipts
assert [row["diagnostic_round"] for row in receipts] == [0, 0, 1], receipts
assert all(row["status"] == "sealed-uploaded-verified" for row in receipts)
assert all(row["claim_ceiling"] ==
           "INCONCLUSIVE_PARTIAL_LIVE_EVIDENCE_UNTIL_WIDE_ROUND_COMPLETE"
           for row in receipts)
assert all(row["expected_source_commit"] == row["runtime_source_commit"]
           for row in receipts)
PY

# A classifier defect must happen only after the exact host inputs needed for
# offline reclassification are uploaded and independently read back.  The fake
# seam payloads are intentionally not valid NumPy archives, so this arm reaches
# stage 15 and then fails at stage 20 without publishing a round completion.
export CANON_APC_M15_TARGET_DEBUG=off
export CANON_P38_SEAM_CLASSIFICATION="$tmp/m15-wide/state/m15.classification.json"
export CANON_APC_M15_SEAM_BUNDLE="$tmp/m15-wide/state/m15.bundle.tar"
export CANON_APC_M15_REPLAY_LEDGER="$tmp/m15-wide/state/capture/m15-replay-envelope.jsonl"
export CANON_P38_MISMATCH_CAPSULE="$tmp/m15-wide/state/absent-capsule.npz"
export CANON_P38_SEAM_LAYER=0
printf '{"diagnostic_round":0,"schema":"m15-apc-serving-envelope-v1"}\n' \
  > "$CANON_APC_M15_REPLAY_LEDGER"
python3 - "$CANON_P38_MISMATCH_CAPSULE" <<'PY'
import json
import numpy as np
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
np.savez(
    path,
    metadata_json=np.frombuffer(
        json.dumps({"diagnostic_round": 0}).encode(), dtype=np.uint8
    ),
    selected_rows=np.asarray([0], dtype=np.int32),
    prompt_ids=np.asarray([[10, 11]], dtype=np.int32),
    prompt_mask=np.asarray([[True, True]], dtype=np.bool_),
    completion_ids=np.asarray([[12]], dtype=np.int32),
    completion_valid_mask=np.asarray([[True]], dtype=np.bool_),
    action_mask=np.asarray([[True]], dtype=np.bool_),
    s_decode=np.asarray([[-2.0]], dtype=np.float32),
    s_prefill=np.asarray([[-1.0]], dtype=np.float32),
)
PY
cat > "$CANON_PRE_ALIGN_REPORT" <<'JSON'
{"N_action":1,"boundaries":{"S_decode_vs_S_prefill":{"differing_bytes":1,"differing_elements":1,"finite":true,"valid":true},"S_prefill_vs_T_old":{"differing_bytes":0,"differing_elements":0,"finite":true,"valid":true}},"diagnostic_round":0}
JSON
m15_classifier_fail_rc=0
bash "$PERSIST" m15-round 000000 \
  > "$tmp/m15-wide/classifier-fail.log" 2>&1 || m15_classifier_fail_rc=$?
test "$m15_classifier_fail_rc" -ne 0
m15_round_zero="$m15_wide_remote/wide/rounds/000000"
test -s "$m15_round_zero/stages/STAGE_10_assemble_PASS.json"
test -s "$m15_round_zero/stages/STAGE_15_checkpoint-input_PASS.json"
if [ ! -s "$m15_round_zero/stages/STAGE_20_classify_FAIL.json" ]; then
  cat "$tmp/m15-wide/classifier-fail.log" >&2
  find "$m15_round_zero" -maxdepth 3 -type f -print >&2
  exit 1
fi
classifier_input_remote="$m15_round_zero/classifier-input"
for name in ROUND_INPUT_RECEIPT.json m15-replay-envelope.jsonl \
    pre-alignment.jsonl CLASSIFIER_INPUT_SHA256SUMS \
    CLASSIFIER_INPUT_RECEIPT.json; do
  test -s "$classifier_input_remote/$name"
done
(cd "$classifier_input_remote" && \
  sha256sum -c CLASSIFIER_INPUT_SHA256SUMS --quiet)
python3 - "$classifier_input_remote/CLASSIFIER_INPUT_RECEIPT.json" <<'PY'
import json
import pathlib
import sys

receipt = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert receipt["status"] == "prepared-for-durable-upload", receipt
assert receipt["diagnostic_round"] == 0, receipt
assert receipt["a_b_differing_bytes"] == 1, receipt
assert receipt["files"] == [
    "ROUND_INPUT_RECEIPT.json",
    "m15-replay-envelope.jsonl",
    "mismatch-capsule.npz",
    "pre-alignment.jsonl",
], receipt
PY
test ! -e "$m15_round_zero/WIDE_ROUND_COMPLETE.json"

# A failed M15 round assembly must leave both a durable remote sub-stage and a
# local failure receipt that the blocked learner can observe immediately.
export CANON_APC_M15_TARGET_DEBUG=off
export CANON_P38_SEAM_CLASSIFICATION="$tmp/m15-wide/state/m15.classification.json"
export CANON_APC_M15_SEAM_BUNDLE="$tmp/m15-wide/state/m15.bundle.tar"
export CANON_APC_M15_REPLAY_LEDGER="$tmp/m15-wide/state/capture/m15-replay-envelope.jsonl"
printf '{"diagnostic_round":2,"schema":"m15-apc-serving-envelope-v1"}\n' \
  > "$CANON_APC_M15_REPLAY_LEDGER"
m15_round_fail_rc=0
bash "$PERSIST" m15-round 000002 \
  > "$tmp/m15-wide/round-fail.log" 2>&1 || m15_round_fail_rc=$?
test "$m15_round_fail_rc" -eq 2
m15_round_failure="$CANON_P38_ROUND_SEAL_ACK_DIR/round-000002.failure.json"
python3 - "$m15_round_failure" <<'PY'
import json
import pathlib
import sys

failure = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert failure == {
    "action": "seal-round",
    "diagnostic_round": 2,
    "exit_code": 2,
    "schema": "canon-p38-round-seal-failure-v1",
    "stage": "assemble",
    "status": "FAIL",
}, failure
PY
m15_round_stage_remote="$m15_wide_remote/wide/rounds/000002/stages"
test -s "$m15_round_stage_remote/STAGE_10_assemble_STARTED.json"
test -s "$m15_round_stage_remote/STAGE_10_assemble_FAIL.json"
test ! -e "$m15_wide_remote/wide/rounds/000002/WIDE_ROUND_COMPLETE.json"

export FAKE_GCS_FAIL_CP=1
m15_stage_upload_rc=0
bash "$PERSIST" m15-round 000003 \
  > "$tmp/m15-wide/stage-upload-fail.log" 2>&1 || m15_stage_upload_rc=$?
unset FAKE_GCS_FAIL_CP
test "$m15_stage_upload_rc" -ne 0
python3 - "$CANON_P38_ROUND_SEAL_ACK_DIR/round-000003.failure.json" <<'PY'
import json
import pathlib
import sys

failure = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert failure["diagnostic_round"] == 3, failure
assert failure["stage"] == "assemble-receipt", failure
assert failure["status"] == "FAIL", failure
PY
test ! -e "$CANON_P38_ROUND_SEAL_ACK_DIR/round-000003.ack"

# Isolate the live-worker coordination contract from classifier payloads.  A
# fake persistence backend proves three ordered ACKs; a forced round failure
# must instead emit one failure receipt and no ACK.
worker_contract_root="$tmp/m15-worker-contract"
fake_pkg="$worker_contract_root/pkg"
mkdir -p "$fake_pkg/tasks/p38-pathways-decode-prefill-carrier/scripts"
fake_persist="$fake_pkg/tasks/p38-pathways-decode-prefill-carrier/scripts/persist_p38_gcs.sh"
apply_worker_contract_env() {
  local case_root="$1" profile="${2:-m15-wide-v1}"
  mkdir -p "$case_root/state/capture" \
    "$case_root/state/p38_round_seal_requests" \
    "$case_root/state/p38_round_seal_acks"
  printf 'run\n' > "$case_root/state/run.log"
  printf '{}\n' > "$case_root/state/pre.jsonl"
  printf 'capsule\n' > "$case_root/state/capsule.npz"
  printf '{}\n' > "$case_root/state/request.jsonl"
  printf '{}\n' > "$case_root/state/incident.jsonl"
  printf '0\n' > "$case_root/state/p38_diagnostic_round"
  export CANON_PKG="$fake_pkg"
  export CANON_RUN_LOG="$case_root/state/run.log"
  export CANON_PRE_ALIGN_REPORT="$case_root/state/pre.jsonl"
  export CANON_P38_MISMATCH_CAPSULE="$case_root/state/capsule.npz"
  export CANON_P38_REQUEST_JOURNAL="$case_root/state/request.jsonl"
  export CANON_P38_INCIDENT_LEDGER="$case_root/state/incident.jsonl"
  export CANON_P38_DIAGNOSTIC_ROUND_FILE="$case_root/state/p38_diagnostic_round"
  export CANON_P38_ROUND_SEAL_REQUEST_DIR="$case_root/state/p38_round_seal_requests"
  export CANON_P38_ROUND_SEAL_ACK_DIR="$case_root/state/p38_round_seal_acks"
  export CANON_P38_LIVE_SNAPSHOT_INTERVAL_SECONDS=1
  export CANON_P38_LIVE_SNAPSHOT_STOP_FILE="$case_root/state/live.stop"
  export CANON_P38_LIVE_COLLECT_REQUEST_FILE="$case_root/state/collect.request"
  export CANON_P38_LIVE_COLLECT_ACK_FILE="$case_root/state/collect.ack"
  export CANON_P38_LIVE_COMPLETE_REQUEST_FILE="$case_root/state/complete.request"
  export CANON_P38_LIVE_COMPLETE_ACK_FILE="$case_root/state/complete.ack"
  export CANON_P38_DURABILITY_PROFILE="$profile"
}
cat > "$fake_persist" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
case "${1:?}" in
  m15-shard) exit 3 ;;
  m15-round|m15-e0-round)
    if [ "${FAKE_M15_FAIL_ROUND:-}" = "${2:?}" ]; then
      exit 17
    fi
    printf '%s\n' "$2" >> "${FAKE_M15_CALL_LOG:?}"
    ;;
  *) exit 2 ;;
esac
SH
chmod +x "$fake_persist"

worker_pass_root="$worker_contract_root/pass"
apply_worker_contract_env "$worker_pass_root"
export FAKE_M15_CALL_LOG="$worker_pass_root/state/calls.log"
unset FAKE_M15_FAIL_ROUND
worker_pass_log="$worker_pass_root/state/worker.log"
bash "$ROOT/tasks/p38-pathways-decode-prefill-carrier/scripts/p38_live_snapshot_worker.sh" \
  > "$worker_pass_log" 2>&1 &
worker_pass_pid=$!
for round_index in 0 1 2; do
  make_round_request "$round_index"
  ack="$CANON_P38_ROUND_SEAL_ACK_DIR/round-$(printf '%06d' "$round_index").ack"
  for unused in 1 2 3 4 5; do
    [ ! -s "$ack" ] || break
    sleep 1
  done
  grep -q '"status": "PASS"' "$ack"
done
touch "$CANON_P38_LIVE_SNAPSHOT_STOP_FILE"
wait "$worker_pass_pid"
test "$(cat "$FAKE_M15_CALL_LOG")" = $'000000\n000001\n000002'
grep -q 'LIVE_WORKER_COMPLETE snapshots=0 rounds=3 profile=m15-wide-v1' \
  "$worker_pass_log"

worker_fail_root="$worker_contract_root/fail"
apply_worker_contract_env "$worker_fail_root"
export FAKE_M15_CALL_LOG="$worker_fail_root/state/calls.log"
export FAKE_M15_FAIL_ROUND=000000
make_round_request 0
worker_fail_log="$worker_fail_root/state/worker.log"
worker_fail_rc=0
bash "$ROOT/tasks/p38-pathways-decode-prefill-carrier/scripts/p38_live_snapshot_worker.sh" \
  > "$worker_fail_log" 2>&1 || worker_fail_rc=$?
test "$worker_fail_rc" -eq 17
test ! -e "$CANON_P38_ROUND_SEAL_ACK_DIR/round-000000.ack"
python3 - "$CANON_P38_ROUND_SEAL_ACK_DIR/round-000000.failure.json" <<'PY'
import json
import pathlib
import sys

failure = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert failure["diagnostic_round"] == 0, failure
assert failure["exit_code"] == 17, failure
assert failure["stage"] == "persist-m15-round", failure
assert failure["status"] == "FAIL", failure
PY
grep -q 'LIVE_ROUND_FAILURE round=0 stage=persist-m15-round exit_code=17' \
  "$worker_fail_log"
unset FAKE_M15_FAIL_ROUND

# The E0 KV profile must preserve the first two durable ACKs when round 2
# fails.  This is the exact salvage boundary needed when the pod disappears
# before root collection: completed round evidence is not rolled back.
worker_e0_root="$worker_contract_root/e0-round-2-fail"
apply_worker_contract_env "$worker_e0_root" m15-e0-kv-v1
export FAKE_M15_CALL_LOG="$worker_e0_root/state/calls.log"
export FAKE_M15_FAIL_ROUND=000002
worker_e0_log="$worker_e0_root/state/worker.log"
bash "$ROOT/tasks/p38-pathways-decode-prefill-carrier/scripts/p38_live_snapshot_worker.sh" \
  > "$worker_e0_log" 2>&1 &
worker_e0_pid=$!
for round_index in 0 1 2; do
  make_round_request "$round_index"
  round_text="$(printf '%06d' "$round_index")"
  ack="$CANON_P38_ROUND_SEAL_ACK_DIR/round-$round_text.ack"
  failure="$CANON_P38_ROUND_SEAL_ACK_DIR/round-$round_text.failure.json"
  for unused in 1 2 3 4 5; do
    [ ! -s "$ack" ] && [ ! -s "$failure" ] || break
    sleep 1
  done
  if [ "$round_index" -lt 2 ]; then
    grep -q '"status": "PASS"' "$ack"
  else
    test ! -e "$ack"
    grep -q '"status": "FAIL"' "$failure"
  fi
done
worker_e0_rc=0
wait "$worker_e0_pid" || worker_e0_rc=$?
test "$worker_e0_rc" -eq 17
test "$(cat "$FAKE_M15_CALL_LOG")" = $'000000\n000001'
grep -q 'LIVE_ROUND_FAILURE round=2 stage=persist-m15-e0-round exit_code=17' \
  "$worker_e0_log"
unset FAKE_M15_FAIL_ROUND
export CANON_PKG="$ROOT"

install_fake_gcloud "$tmp/m15-source-mismatch"
make_case "$tmp/m15-source-mismatch" canon-p38-test-m15-source-mismatch
export CANON_P38_DURABILITY_PROFILE=m15-wide-v1
export CANON_EXPECT_COMMIT="$(printf 'a%.0s' {1..40})"
if bash "$PERSIST" probe > "$tmp/m15-source-mismatch/run.log" 2>&1; then
  echo "[P38.GCS] mismatched M15 runtime source was accepted" >&2
  exit 1
fi
grep -q 'runtime source mismatch' "$tmp/m15-source-mismatch/run.log"
test ! -e "$FAKE_GCS_ROOT/yuxzhang-tunix-models/canon-zero-tim/evidence/p38/canon-p38-test-m15-source-mismatch"

install_fake_gcloud "$tmp/out-of-order"
make_case "$tmp/out-of-order" canon-p38-test-out-of-order
bash "$PERSIST" probe > "$tmp/out-of-order/probe.log"
export CANON_P38_LIVE_SNAPSHOT_INTERVAL_SECONDS=1
export CANON_P38_LIVE_SNAPSHOT_STOP_FILE="$tmp/out-of-order/state/live.stop"
export CANON_P38_LIVE_COLLECT_REQUEST_FILE="$tmp/out-of-order/state/collect.request"
export CANON_P38_LIVE_COLLECT_ACK_FILE="$tmp/out-of-order/state/collect.ack"
export CANON_P38_LIVE_COMPLETE_REQUEST_FILE="$tmp/out-of-order/state/complete.request"
export CANON_P38_LIVE_COMPLETE_ACK_FILE="$tmp/out-of-order/state/complete.ack"
out_of_order_log="$tmp/out-of-order/state/live-worker.log"
bash "$ROOT/tasks/p38-pathways-decode-prefill-carrier/scripts/p38_live_snapshot_worker.sh" \
  > "$out_of_order_log" 2>&1 &
out_of_order_pid=$!
for unused in 1 2 3 4 5; do
  grep -q 'LIVE_WORKER_START' "$out_of_order_log" 2>/dev/null && break
  sleep 1
done
grep -q 'LIVE_WORKER_START' "$out_of_order_log"
printf 'action=complete\n' > "$CANON_P38_LIVE_COMPLETE_REQUEST_FILE"
out_of_order_rc=0
wait "$out_of_order_pid" || out_of_order_rc=$?
test "$out_of_order_rc" -ne 0
grep -q 'completion requested before collection acknowledgement' \
  "$out_of_order_log"
test ! -e "$CANON_P38_LIVE_COMPLETE_ACK_FILE"
test ! -e "$FAKE_GCS_ROOT/yuxzhang-tunix-models/canon-zero-tim/evidence/p38/canon-p38-test-out-of-order/attempt-0/COMPLETE.json"

install_fake_gcloud "$tmp/p58-seam"
make_case "$tmp/p58-seam" canon-p58-test-seam
export CANON_P38_DURABILITY_PROFILE=p58-seam-v1
export CANON_P58_SEAM_LOCALIZATION=coarse
export CANON_P38_DIAGNOSTIC_ROUNDS=3
export CANON_P38_GCS_PREFIX="gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p58/canon-p58-test-seam/attempt-0"
export CANON_P38_SEAM_CLASSIFICATION="$tmp/p58-seam/state/p58-seam.classification.json"
printf '{"schema":"canon.p58.coarse-seam-three-round-classification.v1","verdict":"PASS"}\n' \
  > "$CANON_P38_SEAM_CLASSIFICATION"
for round_index in 0 1 2; do
  printf -v round_text '%06d' "$round_index"
  round_root="$tmp/p58-seam/state/p38_gcs_rounds/$round_text"
  mkdir -p "$round_root"
  printf '{"diagnostic_round":%s,"schema":"canon-p38-round-completion-v1","status":"sealed-and-verified"}\n' \
    "$round_index" > "$round_root/ROUND_COMPLETE.json"
  printf '{"diagnostic_round":%s,"schema":"canon.p58.coarse-seam-round-classification.v1","verdict":"PASS"}\n' \
    "$round_index" > "$round_root/p58-seam.round.classification.json"
done
bash "$PERSIST" probe > "$tmp/p58-seam/probe.log"
bash "$PERSIST" collect > "$tmp/p58-seam/collect.log"
p58_remote="$FAKE_GCS_ROOT/yuxzhang-tunix-models/canon-zero-tim/evidence/p58/canon-p58-test-seam/attempt-0"
for name in PREFLIGHT.json run.log pre-alignment.jsonl \
    p58-seam.classification.json \
    ROUND_COMPLETE.000000.json ROUND_COMPLETE.000001.json \
    ROUND_COMPLETE.000002.json \
    p58-seam.round.000000.classification.json \
    p58-seam.round.000001.classification.json \
    p58-seam.round.000002.classification.json SHA256SUMS COLLECTED.json; do
  test -s "$p58_remote/$name"
done
(cd "$p58_remote" && sha256sum -c SHA256SUMS --quiet)
grep -q '"schema": "canon-p58-seam-gcs-collection-v1"' \
  "$p58_remote/COLLECTED.json"
grep -q 'profile=p58-seam-v1 rounds=3' "$tmp/p58-seam/collect.log"

install_fake_gcloud "$tmp/p58-seam-missing-selector"
make_case "$tmp/p58-seam-missing-selector" canon-p58-test-seam-missing-selector
export CANON_P38_DURABILITY_PROFILE=p58-seam-v1
unset CANON_P58_SEAM_LOCALIZATION
export CANON_P38_GCS_PREFIX="gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p58/canon-p58-test-seam-missing-selector/attempt-0"
if bash "$PERSIST" probe > "$tmp/p58-seam-missing-selector/run.log" 2>&1; then
  echo "[P38.GCS] P58 seam profile accepted a missing selector" >&2
  exit 1
fi
grep -q 'p58-seam-v1 requires the coarse selector' \
  "$tmp/p58-seam-missing-selector/run.log"

echo "[P38.GCS] PERSISTENCE_TEST_PASS probe=verified prefix_reuse=rejected live=immutable bounded_objects=3 round_request=priority alignment_round=periodic_snapshot_disabled minimal_payload=verified worker=durable round_bundles=survive_abrupt_exit m15_shards=bounded-survive-abrupt-exit m15_round_stage_failure=durable m15_stage_upload_failure=fail_closed m15_worker_rounds=3 m15_worker_failure=fail_fast m15_e0_three_round=sealed-classified-readback m15_e0_round2_failure=rounds0_1_preserved p58_three_round_collection=verified p58_missing_selector=rejected source_mismatch=rejected collected=verified complete=last out_of_order=rejected missing=rejected upload_failure=rejected"
