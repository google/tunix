#!/usr/bin/env bash
# Recover Attempt 13's real single-round flat shards and replay its classifier.
set -euo pipefail

output="${1:?usage: recover_m15_attempt13_d32.sh <output-dir> [scratch-parent]}"
scratch_parent="${2:-/tmp}"
test -d "$scratch_parent"
test ! -e "$output"

script_dir="$(cd "$(dirname "$0")" && pwd)"
task_dir="$(cd "$script_dir/.." && pwd)"
repo="$(cd "$task_dir/../../.." && pwd)"
receipt="$task_dir/evidence/v1_apc_m15_attempt13_paired_d32_20260828/receipt.json"
expected_source="7d30f3827480e6f9d5ae972f55ca4d16f07de6df"
expected_receipt_sha="d1941c2de85050a5652bc5c6e809987f6bf72b996aa817371b08b43870835f95"
archive_tool="$repo/canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/p38_evidence_archive.py"

test -f "$receipt"
test -f "$archive_tool"
actual_receipt_sha="$(sha256sum "$receipt" | awk '{print $1}')"
[ "$actual_receipt_sha" = "$expected_receipt_sha" ] || {
  echo "[M15.ATTEMPT13] REFUSING: checked-in receipt SHA drifted" >&2
  exit 2
}
git -C "$repo" cat-file -e "$expected_source^{commit}"

if command -v gcloud >/dev/null 2>&1; then
  gcs_cp() { gcloud storage cp "$1" "$2" >/dev/null; }
  gcs_list() { gcloud storage ls "$1" 2>/dev/null; }
elif command -v gsutil >/dev/null 2>&1; then
  gcs_cp() { gsutil -q cp "$1" "$2"; }
  gcs_list() { gsutil -q ls "$1" 2>/dev/null; }
else
  echo "[M15.ATTEMPT13] REFUSING: gcloud or gsutil is required" >&2
  exit 2
fi

scratch="$(mktemp -d -p "$scratch_parent" m15-attempt13-flat.XXXXXX)"
trap 'rm -rf -- "$scratch"' EXIT

mapfile -t arm_rows < <(python3 - "$receipt" "$expected_source" <<'PY'
import json
import pathlib
import re
import sys

receipt = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
source = sys.argv[2]
if receipt.get("attempt") != 13 or receipt.get("source_commit") != source:
  raise SystemExit("Attempt-13 receipt identity drifted")
expected = {
    "off": ("control_arm_off", "canon-v1-apc-m15-off-d32-7d30f382", 77, 2474),
    "on": ("treatment_arm_on", "canon-v1-apc-m15-on-d32-7d30f382", 70, 2087),
}
for arm, (field, jobset, shards, pairs) in expected.items():
  value = receipt.get(field)
  if not isinstance(value, dict) or value.get("jobset_name") != jobset:
    raise SystemExit(f"Attempt-13 {arm} JobSet identity drifted")
  uri = str(value.get("gcs_source_uri", ""))
  if re.fullmatch(
      rf"gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/{jobset}/attempt-0",
      uri,
  ) is None:
    raise SystemExit(f"Attempt-13 {arm} GCS root drifted")
  print(f"{arm}\t{jobset}\t{shards}\t{pairs}\t{uri}")
PY
)
[ "${#arm_rows[@]}" -eq 2 ] || {
  echo "[M15.ATTEMPT13] REFUSING: receipt did not resolve two arms" >&2
  exit 2
}

fetch_shards() {
  local arm="$1" root="$2" expected_shards="$3" destination="$4"
  local list_file="$scratch/$arm-shards.txt"
  if ! gcs_list "$root/wide/shards/*/SHARD_COMPLETE.json" > "$list_file"; then
    echo "[M15.ATTEMPT13] REFUSING: $arm flat-shard GCS listing failed" >&2
    exit 2
  fi
  mapfile -t sequences < <(python3 - "$list_file" "$root" "$expected_shards" <<'PY'
import pathlib
import re
import sys

lines = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
root = re.escape(sys.argv[2])
expected = int(sys.argv[3])
values = []
for line in lines:
  match = re.fullmatch(root + r"/wide/shards/([0-9]{6})/SHARD_COMPLETE\.json", line.strip())
  if match is None:
    raise SystemExit(f"unexpected flat-shard listing row: {line!r}")
  values.append(int(match.group(1)))
if sorted(values) != list(range(expected)) or len(values) != len(set(values)):
  raise SystemExit(
      f"flat-shard sequences are incomplete or duplicated: got={len(values)} expected={expected}"
  )
for value in sorted(values):
  print(f"{value:06d}")
PY
  )
  [ "${#sequences[@]}" -eq "$expected_shards" ] || {
    echo "[M15.ATTEMPT13] REFUSING: $arm flat-shard listing failed" >&2
    exit 2
  }
  mkdir -p "$destination/shards"
  for sequence in "${sequences[@]}"; do
    local remote="$root/wide/shards/$sequence"
    local transfer="$scratch/$arm-transfer-$sequence"
    local extracted="$destination/shards/$sequence"
    mkdir "$transfer"
    for name in SHARD_ARCHIVE.tar SHA256SUMS SHARD_COMPLETE.json; do
      gcs_cp "$remote/$name" "$transfer/$name"
    done
    local archive_sha
    archive_sha="$(python3 - "$transfer/SHARD_COMPLETE.json" "$sequence" "$expected_source" <<'PY'
import json
import pathlib
import re
import sys

value = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
sequence = int(sys.argv[2])
source = sys.argv[3]
if not (
    value.get("schema") == "m15-wide-observer-shard-completion-v1"
    and value.get("status") == "sealed-uploaded-verified"
    and int(value.get("sequence", -1)) == sequence
    and int(value.get("diagnostic_round", -1)) == 0
    and value.get("expected_source_commit") == source
    and value.get("runtime_source_commit") == source
    and re.fullmatch(r"[0-9a-f]{64}", str(value.get("archive_sha256", "")))
):
  raise SystemExit("flat-shard completion contract drifted")
print(value["archive_sha256"])
PY
    )"
    [ "$(sha256sum "$transfer/SHA256SUMS" | awk '{print $1}')" = \
      "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["manifest_sha256"])' "$transfer/SHARD_COMPLETE.json")" ]
    [ "$(sha256sum "$transfer/SHARD_ARCHIVE.tar" | awk '{print $1}')" = "$archive_sha" ]
    python3 "$archive_tool" extract \
      --archive "$transfer/SHARD_ARCHIVE.tar" --output "$extracted" \
      > "$transfer/extract.log"
    cmp -- "$transfer/SHA256SUMS" "$extracted/SHA256SUMS"
    cp -- "$transfer/SHARD_COMPLETE.json" "$extracted/SHARD_COMPLETE.json"
    (cd "$extracted" && sha256sum -c SHA256SUMS --quiet)
  done
  echo "[M15.ATTEMPT13] FLAT_SHARDS_READY arm=$arm shards=${#sequences[@]}"
}

fetch_live() {
  local arm="$1" jobset="$2" root="$3" destination="$4"
  local list_file="$scratch/$arm-live.txt"
  if ! gcs_list "$root/live/*/LIVE.json" > "$list_file"; then
    echo "[M15.ATTEMPT13] REFUSING: $arm live-snapshot GCS listing failed" >&2
    exit 2
  fi
  mapfile -t sequences < <(python3 - "$list_file" "$root" <<'PY'
import pathlib
import re
import sys

lines = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
root = re.escape(sys.argv[2])
values = set()
for line in lines:
  match = re.fullmatch(root + r"/live/([0-9]{6})/LIVE\.json", line.strip())
  if match is None:
    raise SystemExit(f"unexpected live listing row: {line!r}")
  values.add(int(match.group(1)))
for value in sorted(values, reverse=True):
  print(f"{value:06d}")
PY
  )
  [ "${#sequences[@]}" -gt 0 ] || {
    echo "[M15.ATTEMPT13] REFUSING: $arm has no live snapshot" >&2
    exit 2
  }
  for sequence in "${sequences[@]}"; do
    local remote="$root/live/$sequence"
    local candidate="$scratch/$arm-live-$sequence"
    mkdir "$candidate"
    gcs_cp "$remote/LIVE.json" "$candidate/LIVE.json"
    if ! python3 - "$candidate/LIVE.json" "$arm" "$jobset" "$expected_source" <<'PY'
import json
import pathlib
import sys

value = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
arm, jobset, source = sys.argv[2:]
files = set(value.get("files", ()))
required = {"pre-alignment.jsonl", "m15-replay-envelope.jsonl", "diagnostic-round.txt"}
capsules = [name for name in files if name.endswith(".npz") and "capsule" in name]
valid = (
    value.get("schema") == "canon-p38-gcs-live-v1"
    and value.get("status") == "live-snapshot"
    and value.get("source_commit") == source
    and value.get("jobset") == jobset
    and required <= files
    and (arm == "off" or bool(capsules))
)
raise SystemExit(0 if valid else 1)
PY
    then
      continue
    fi
    gcs_cp "$remote/SHA256SUMS" "$candidate/SHA256SUMS"
    gcs_cp "$remote/LIVE_ARCHIVE.tar" "$candidate/LIVE_ARCHIVE.tar"
    local archive_sha manifest_sha
    archive_sha="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["archive_sha256"])' "$candidate/LIVE.json")"
    manifest_sha="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["manifest_sha256"])' "$candidate/LIVE.json")"
    [ "$(sha256sum "$candidate/LIVE_ARCHIVE.tar" | awk '{print $1}')" = "$archive_sha" ]
    [ "$(sha256sum "$candidate/SHA256SUMS" | awk '{print $1}')" = "$manifest_sha" ]
    python3 "$archive_tool" extract \
      --archive "$candidate/LIVE_ARCHIVE.tar" --output "$destination/live" \
      > "$candidate/extract.log"
    cmp -- "$candidate/SHA256SUMS" "$destination/live/SHA256SUMS"
    cp -- "$candidate/LIVE.json" "$destination/live/LIVE.json"
    (cd "$destination/live" && sha256sum -c SHA256SUMS --quiet)
    echo "[M15.ATTEMPT13] LIVE_READY arm=$arm sequence=$sequence"
    return 0
  done
  echo "[M15.ATTEMPT13] REFUSING: $arm has no replay-complete live snapshot" >&2
  exit 2
}

for row in "${arm_rows[@]}"; do
  IFS=$'\t' read -r arm jobset shard_count pair_count root <<< "$row"
  arm_root="$scratch/$arm"
  mkdir "$arm_root"
  fetch_shards "$arm" "$root" "$shard_count" "$arm_root"
  fetch_live "$arm" "$jobset" "$root" "$arm_root"
done

python3 "$script_dir/replay_m15_attempt13_flat_shards.py" \
  --off-root "$scratch/off" --on-root "$scratch/on" \
  --work "$scratch/replay-work" --output "$output"
(cd "$output" && sha256sum -c SHA256SUMS --quiet)
decision="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["decision"])' \
  "$output/ATTEMPT13_FLAT_REPLAY.json")"
echo "[M15.ATTEMPT13] RETURN_READY decision=$decision rounds=1 numerical_repair_authorized=0 output=$output"
