#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COLLECT="$ROOT/tasks/p38-pathways-decode-prefill-carrier/scripts/collect_p38s23r3_return.sh"
ARCHIVE="$ROOT/tasks/p38-pathways-decode-prefill-carrier/scripts/p38_evidence_archive.py"
SOURCE="$(printf 'a%.0s' {1..40})"
JOBSET="canon-p38-fl-stock-p38s23r3-${SOURCE:0:8}"
tmp="$(mktemp -d)"
trap 'rm -r "$tmp"' EXIT

mkdir -p "$tmp/bin" "$tmp/gcs" "$tmp/launch"
cp "$ROOT/tests/p38_serving/fake_gcloud.sh" "$tmp/bin/gcloud"
chmod +x "$tmp/bin/gcloud"
export PATH="$tmp/bin:$PATH"
export FAKE_GCS_ROOT="$tmp/gcs"

for name in rendered-stock.yaml render.txt semantic-preflight.txt dry-run.txt \
    apply.txt; do
  printf '%s\n' "$name" > "$tmp/launch/$name"
done
printf '%s\n' "$SOURCE" > "$tmp/launch/source_commit.txt"
(
  cd "$tmp/launch"
  find . -maxdepth 1 -type f ! -name LAUNCH_SHA256SUMS -printf '%f\n' \
    | LC_ALL=C sort | xargs sha256sum > LAUNCH_SHA256SUMS
)

cat > "$tmp/head.full.log" <<EOF
[sync] HEAD=$SOURCE
CANON_P38_FIXED_LM_HEAD=1 semantic_M=8 fixed_M=256 chunks=1
CANON_P38_FIXED_LM_HEAD=1 semantic_M=16 fixed_M=256 chunks=1
CANON_P38_FIXED_LM_HEAD=1 semantic_M=32 fixed_M=256 chunks=1
CANON_P38_FIXED_LM_HEAD=1 semantic_M=64 fixed_M=256 chunks=1
CANON_P38_FIXED_LM_HEAD=1 semantic_M=128 fixed_M=256 chunks=1
CANON_P38_FIXED_LM_HEAD=1 semantic_M=256 fixed_M=256 chunks=1
CANON_P38_FIXED_LM_HEAD=1 semantic_M=4096 fixed_M=256 chunks=16
[CANON_P38] PRECHECK_ROUND_COMPLETE round=1/3
[CANON_P38] ROUND_SEAL_ACKNOWLEDGED round=0
[P38.GCS] LIVE_ROUND_PASS round=0
[CANON_P38] PRECHECK_ROUND_COMPLETE round=2/3
[CANON_P38] ROUND_SEAL_ACKNOWLEDGED round=1
[P38.GCS] LIVE_ROUND_PASS round=1
[CANON_P38] PRECHECK_ROUND_COMPLETE round=3/3
[CANON_P38] ROUND_SEAL_ACKNOWLEDGED round=2
[P38.GCS] LIVE_ROUND_PASS round=2
[CANON_P38] CONTROLLED_EXIT code=42 backward=0 optimizer_commits=0
EOF

remote="$FAKE_GCS_ROOT/yuxzhang-tunix-models/canon-zero-tim/evidence/p38/$JOBSET/attempt-0"
mkdir -p "$remote/rounds"
printf '%s  %s\n' "$(printf root | sha256sum | awk '{print $1}')" run.log \
  > "$remote/SHA256SUMS"
python3 - "$remote" "$SOURCE" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
source = sys.argv[2]
for name, schema, status in (
    ("PREFLIGHT.json", "canon-p38-gcs-preflight-v1", "writable"),
    ("COLLECTED.json", "canon-p38-gcs-collection-v1", "collected"),
    ("COMPLETE.json", "canon-p38-gcs-completion-v1", "postflight-accepted"),
):
  (root / name).write_text(json.dumps({
      **({"manifest_sha256": hashlib.sha256(
          (root / "SHA256SUMS").read_bytes()).hexdigest()}
         if name == "COMPLETE.json" else {}),
      "schema": schema,
      "source_commit": source,
      "status": status,
  }, sort_keys=True) + "\n")
PY

for round_index in 0 1 2; do
  printf -v sequence '%06d' "$round_index"
  stage="$tmp/stage-$sequence"
  destination="$remote/rounds/$sequence"
  mkdir -p "$stage" "$destination"
  printf 'round=%s\n' "$round_index" > "$stage/run.log"
  python3 - "$stage" "$round_index" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
index = int(sys.argv[2])
(root / "pre-alignment.jsonl").write_text(json.dumps({
    "N_action": 49177 + index,
    "boundaries": {
        "S_decode_vs_S_prefill": {
            "differing_bytes": 0, "differing_elements": 0,
            "finite": True, "max_abs": 0.0, "valid": True,
        },
        "S_prefill_vs_T_old": {
            "differing_bytes": 0, "differing_elements": 0,
            "finite": True, "max_abs": 0.0, "valid": True,
        },
    },
    "diagnostic_round": index,
    "verdict": "PASS",
}, sort_keys=True) + "\n")
(root / "ROUND_INVENTORY.json").write_text(json.dumps({
    "capsule_present": False,
    "diagnostic_round": index,
    "incident_records": 0,
    "journal_records": 0,
    "kv_records": 0,
    "pre_alignment_records": 1,
    "profile": "alignment-only",
    "schema": "canon-p38-round-stage-v1",
    "seam_records": 0,
    "tail_records": 0,
    "terminal_records": 0,
}, sort_keys=True) + "\n")
PY
  (
    cd "$stage"
    find . -maxdepth 1 -type f ! -name SHA256SUMS -printf '%f\n' \
      | LC_ALL=C sort \
      | xargs sha256sum > SHA256SUMS
  )
  python3 "$ARCHIVE" create --root "$stage" \
    --manifest "$stage/SHA256SUMS" \
    --output "$destination/ROUND_ARCHIVE.tar" > "$tmp/archive-$sequence.log"
  cp "$stage/SHA256SUMS" "$destination/SHA256SUMS"
  archive_sha="$(sha256sum "$destination/ROUND_ARCHIVE.tar" | awk '{print $1}')"
  manifest_sha="$(sha256sum "$destination/SHA256SUMS" | awk '{print $1}')"
  python3 - "$destination/ROUND_COMPLETE.json" "$round_index" "$SOURCE" \
      "$archive_sha" "$manifest_sha" <<'PY'
import json
import pathlib
import sys

pathlib.Path(sys.argv[1]).write_text(json.dumps({
    "archive_sha256": sys.argv[4],
    "diagnostic_round": int(sys.argv[2]),
    "durability_profile": "round-alignment-v1",
    "logical_file_count": 3,
    "manifest_sha256": sys.argv[5],
    "schema": "canon-p38-round-completion-v1",
    "source_commit": sys.argv[3],
    "status": "sealed-and-verified",
}, sort_keys=True) + "\n")
PY
done

bash "$COLLECT" --source-commit "$SOURCE" \
  --head-log "$tmp/head.full.log" --launch-dir "$tmp/launch" \
  --output-dir "$tmp/return" > "$tmp/collect.log"
grep -q 'P38S23R3_FORWARD_EXACT_PASS' "$tmp/return/verdict.json"
(cd "$tmp/return" && sha256sum -c RETURN_SHA256SUMS --quiet)
for sequence in 000000 000001 000002; do
  test -s "$tmp/return/rounds/$sequence/pre-alignment.jsonl"
  test -s "$tmp/return/rounds/$sequence/ROUND_ARCHIVE.tar"
done

# A finite A-B-red round with exact B-C is a valid rejecting experiment, not a
# packaging failure.  Rebuild round 2 and require the mechanical reject status.
python3 - "$tmp/stage-000002/pre-alignment.jsonl" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
record = json.loads(path.read_text())
boundary = record["boundaries"]["S_decode_vs_S_prefill"]
boundary.update({"differing_bytes": 5, "differing_elements": 3,
                 "max_abs": 0.125})
record["verdict"] = "FAIL"
path.write_text(json.dumps(record, sort_keys=True) + "\n")
(path.parent / "mismatch-capsule.npz").write_bytes(b"red-capsule\n")
inventory_path = path.parent / "ROUND_INVENTORY.json"
inventory = json.loads(inventory_path.read_text())
inventory["capsule_present"] = True
inventory_path.write_text(json.dumps(inventory, sort_keys=True) + "\n")
PY
(
  cd "$tmp/stage-000002"
  find . -maxdepth 1 -type f ! -name SHA256SUMS -printf '%f\n' \
    | LC_ALL=C sort | xargs sha256sum > "$tmp/round2.SHA256SUMS"
)
mv "$tmp/round2.SHA256SUMS" "$tmp/stage-000002/SHA256SUMS"
python3 "$ARCHIVE" create --root "$tmp/stage-000002" \
  --manifest "$tmp/stage-000002/SHA256SUMS" \
  --output "$tmp/ROUND_ARCHIVE.red.tar" > "$tmp/archive-red.log"
mv "$tmp/ROUND_ARCHIVE.red.tar" \
  "$remote/rounds/000002/ROUND_ARCHIVE.tar"
cp "$tmp/stage-000002/SHA256SUMS" \
  "$remote/rounds/000002/SHA256SUMS"
python3 - "$remote/rounds/000002/ROUND_COMPLETE.json" <<'PY'
import hashlib
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
record = json.loads(path.read_text())
root = path.parent
record["archive_sha256"] = hashlib.sha256(
    (root / "ROUND_ARCHIVE.tar").read_bytes()).hexdigest()
record["manifest_sha256"] = hashlib.sha256(
    (root / "SHA256SUMS").read_bytes()).hexdigest()
record["logical_file_count"] = len(
    [line for line in (root / "SHA256SUMS").read_text().splitlines()
     if line.strip()])
path.write_text(json.dumps(record, sort_keys=True) + "\n")
PY
bash "$COLLECT" --source-commit "$SOURCE" \
  --head-log "$tmp/head.full.log" --launch-dir "$tmp/launch" \
  --output-dir "$tmp/red-return" > "$tmp/collect-red.log"
grep -q 'P38S23R3_FIXED_LM_HEAD_INSUFFICIENT' \
  "$tmp/red-return/verdict.json"

# A complete object set is insufficient when the head log cannot prove every
# round was acknowledged.  This negative catches a truncated attempt-0 log.
grep -v 'ROUND_SEAL_ACKNOWLEDGED round=2' "$tmp/head.full.log" \
  > "$tmp/head.truncated.log"
if bash "$COLLECT" --source-commit "$SOURCE" \
    --head-log "$tmp/head.truncated.log" --launch-dir "$tmp/launch" \
    --output-dir "$tmp/rejected" > "$tmp/rejected.log" 2>&1; then
  echo "[P38S23R3.OPERATOR] truncated head log was accepted" >&2
  exit 1
fi
grep -q 'exactly three round acknowledgements' "$tmp/rejected.log"

echo "[P38S23R3.OPERATOR] TEST_PASS exact_rounds=3 capsule=absent return=verified red=classified_insufficient truncated_head=rejected"
