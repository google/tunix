#!/usr/bin/env bash
# Verify the three immutable P38s22 round seals without requiring root postflight.
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
task_dir="$(cd "$script_dir/.." && pwd)"
repo_root="$(git -C "$script_dir" rev-parse --show-toplevel)"
contract="${1:-$script_dir/p38s22_round_salvage_contract.json}"
scratch_parent="${2:-/tmp}"
return_dir="${3:-$task_dir/evidence/p38s22/round-salvage-v1}"

test -s "$contract"
test -d "$scratch_parent"
if [ -e "$return_dir" ] || [ ! -d "$(dirname "$return_dir")" ]; then
  echo "[P38S22.ROUND_SALVAGE] REFUSING: return path is unsafe" >&2
  exit 2
fi

source_uri="$(python3 -c \
  'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["source_gcs_uri"])' \
  "$contract")"
case "$source_uri" in gs://*) ;; *)
  echo "[P38S22.ROUND_SALVAGE] REFUSING: contract source is not GCS" >&2
  exit 2
esac

analysis_source_commit="$(git -C "$repo_root" rev-parse HEAD)"
dirty="$(git -C "$repo_root" status --short)"
if [ -n "$dirty" ]; then
  case "$source_uri:${CANON_P38S22_SALVAGE_ALLOW_DIRTY_FOR_TEST:-0}" in
    gs://test-p38/*:1) ;;
    *) echo "[P38S22.ROUND_SALVAGE] REFUSING: checkout is dirty" >&2; exit 2 ;;
  esac
fi

if command -v gcloud >/dev/null 2>&1; then
  gcs_cp() { gcloud storage cp "$1" "$2" >/dev/null 2>&1; }
elif command -v gsutil >/dev/null 2>&1; then
  gcs_cp() { gsutil -q cp "$1" "$2" >/dev/null 2>&1; }
else
  echo "[P38S22.ROUND_SALVAGE] REFUSING: gcloud or gsutil is required" >&2
  exit 2
fi

scratch="$(mktemp -d -p "$scratch_parent" p38s22-round-salvage.XXXXXX)"
trap 'rm -rf -- "$scratch"' EXIT
source_root="$scratch/source-root"
round_root="$scratch/rounds"
acquisition="$scratch/ACQUISITION.jsonl"
mkdir -p "$source_root" "$round_root"
: > "$acquisition"

record_acquisition() {
  local label="$1"
  local required="$2"
  local status="$3"
  local path="$4"
  python3 - "$acquisition" "$label" "$required" "$status" "$path" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

ledger, label, required, status, raw_path = sys.argv[1:]
path = Path(raw_path)
record = {
    "label": label,
    "required": required == "1",
    "schema": "p38s22-round-salvage-acquisition-v1",
    "status": status,
}
if status == "downloaded":
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  record["sha256"] = digest.hexdigest()
  record["size_bytes"] = path.stat().st_size
with Path(ledger).open("a", encoding="utf-8") as stream:
  stream.write(json.dumps(record, sort_keys=True) + "\n")
PY
}

required_transport_failed=0
fetch_object() {
  local label="$1"
  local uri="$2"
  local target="$3"
  local required="$4"
  if gcs_cp "$uri" "$target"; then
    record_acquisition "$label" "$required" downloaded "$target"
  else
    record_acquisition "$label" "$required" missing_or_unreadable "$target"
    echo "[P38S22.ROUND_SALVAGE] SOURCE_UNAVAILABLE label=$label required=$required" >&2
    if [ "$required" = 1 ]; then required_transport_failed=1; fi
  fi
}

fetch_object root/PREFLIGHT.json "$source_uri/PREFLIGHT.json" \
  "$source_root/PREFLIGHT.json" 1
fetch_object root/COLLECTED.json "$source_uri/COLLECTED.json" \
  "$source_root/COLLECTED.json" 0
fetch_object root/COMPLETE.json "$source_uri/COMPLETE.json" \
  "$source_root/COMPLETE.json" 0
fetch_object root/SHA256SUMS "$source_uri/SHA256SUMS" \
  "$source_root/SHA256SUMS" 0

for round_index in 000000 000001 000002; do
  round_dir="$round_root/$round_index"
  round_uri="$source_uri/rounds/$round_index"
  mkdir "$round_dir"
  for name in ROUND_ARCHIVE.tar SHA256SUMS ROUND_COMPLETE.json; do
    fetch_object "rounds/$round_index/$name" "$round_uri/$name" \
      "$round_dir/$name" 1
  done
done

set +e
python3 "$script_dir/audit_p38s22_round_salvage.py" \
  --contract "$contract" \
  --source-root "$source_root" \
  --round-root "$round_root" \
  --acquisition "$acquisition" \
  --reference-evidence "$task_dir/evidence/p38s22" \
  --analysis-source-commit "$analysis_source_commit" \
  --output "$return_dir"
audit_rc=$?
set -e
if [ "$audit_rc" -ne 0 ] && [ "$audit_rc" -ne 4 ]; then
  echo "[P38S22.ROUND_SALVAGE] REFUSING: no sealed audit receipt rc=$audit_rc" >&2
  exit "$audit_rc"
fi
if [ "$required_transport_failed" -ne 0 ] && [ "$audit_rc" -eq 0 ]; then
  echo "[P38S22.ROUND_SALVAGE] REFUSING: audit passed after required fetch failure" >&2
  exit 2
fi
test -s "$return_dir/SHA256SUMS"
(cd "$return_dir" && sha256sum -c SHA256SUMS --quiet)
readarray -t summary < <(python3 - "$return_dir/AUDIT.json" <<'PY'
import json
import sys

value = json.load(open(sys.argv[1], encoding="utf-8"))
totals = value.get("totals", {})
root = value.get("root_postflight", {})
print(value["status"])
print(value["verdict"])
print(totals.get("n_action", "unknown"))
print(totals.get("a_b_differing_elements", "unknown"))
print(totals.get("a_b_differing_bytes", "unknown"))
print(totals.get("b_c_differing_elements", "unknown"))
print(root.get("receipts_present", "unknown"))
PY
)
echo "[P38S22.ROUND_SALVAGE] RETURN_READY status=${summary[0]} verdict=${summary[1]} n_action=${summary[2]} a_b_elements=${summary[3]} a_b_bytes=${summary[4]} b_c_elements=${summary[5]} root_receipts_present=${summary[6]} return_dir=$return_dir analysis_source_commit=$analysis_source_commit rc=$audit_rc"
exit "$audit_rc"
