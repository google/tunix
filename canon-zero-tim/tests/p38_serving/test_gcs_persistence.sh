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
m15_wide_remote="$FAKE_GCS_ROOT/yuxzhang-tunix-models/canon-zero-tim/evidence/p38/canon-p38-test-m15-wide/attempt-0"
for sequence in 000000 000001; do
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
    for sequence in ("000000", "000001")
]
assert [row["record_pairs"] for row in receipts] == [32, 8], receipts
assert all(row["status"] == "sealed-uploaded-verified" for row in receipts)
assert all(row["claim_ceiling"] ==
           "INCONCLUSIVE_PARTIAL_LIVE_EVIDENCE_UNTIL_WIDE_ROUND_COMPLETE"
           for row in receipts)
assert all(row["expected_source_commit"] == row["runtime_source_commit"]
           for row in receipts)
PY

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

echo "[P38.GCS] PERSISTENCE_TEST_PASS probe=verified prefix_reuse=rejected live=immutable bounded_objects=3 round_request=priority alignment_round=periodic_snapshot_disabled minimal_payload=verified worker=durable round_bundles=survive_abrupt_exit m15_shards=bounded-survive-abrupt-exit source_mismatch=rejected collected=verified complete=last out_of_order=rejected missing=rejected upload_failure=rejected"
