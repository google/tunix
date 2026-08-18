#!/usr/bin/env bash
# Read the immutable P38s22 GCS evidence and return one small sealed audit.
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
task_dir="$(cd "$script_dir/.." && pwd)"
repo_root="$(git -C "$script_dir" rev-parse --show-toplevel)"
contract="${1:-$script_dir/p38s22_offsite_audit_contract.json}"
scratch_parent="${2:-/tmp}"
return_dir="${3:-$task_dir/evidence/p38s22/offsite-audit-v1}"

test -s "$contract"
test -d "$scratch_parent"
if [ -e "$return_dir" ] || [ ! -d "$(dirname "$return_dir")" ]; then
  echo "[P38S22.OFFSITE] REFUSING: return directory exists or parent is absent" >&2
  exit 2
fi

source_uri="$(python3 -c \
  'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["source_gcs_uri"])' \
  "$contract")"
case "$source_uri" in gs://*) ;; *)
  echo "[P38S22.OFFSITE] REFUSING: contract source is not a GCS URI" >&2
  exit 2
esac

analysis_source_commit="$(git -C "$repo_root" rev-parse HEAD)"
dirty="$(git -C "$repo_root" status --short)"
if [ -n "$dirty" ]; then
  case "$source_uri:${CANON_P38S22_AUDIT_ALLOW_DIRTY_FOR_TEST:-0}" in
    gs://test-p38/*:1) ;;
    *) echo "[P38S22.OFFSITE] REFUSING: execution checkout is dirty" >&2; exit 2 ;;
  esac
fi

if command -v gcloud >/dev/null 2>&1; then
  gcs_cp() {
    if ! gcloud storage cp "$1" "$2" >/dev/null 2>&1; then
      echo "[P38S22.OFFSITE] REFUSING: GCS copy failed for $(basename "$1")" >&2
      return 1
    fi
  }
elif command -v gsutil >/dev/null 2>&1; then
  gcs_cp() {
    if ! gsutil -q cp "$1" "$2" >/dev/null 2>&1; then
      echo "[P38S22.OFFSITE] REFUSING: GCS copy failed for $(basename "$1")" >&2
      return 1
    fi
  }
else
  echo "[P38S22.OFFSITE] REFUSING: gcloud or gsutil is required" >&2
  exit 2
fi

scratch="$(mktemp -d -p "$scratch_parent" p38s22-offsite.XXXXXX)"
trap 'rm -rf -- "$scratch"' EXIT
source_root="$scratch/source-root"
root_files="$source_root/files"
round_root="$scratch/rounds"
mkdir -p "$root_files" "$round_root"

transport_failed=0
for name in PREFLIGHT.json COLLECTED.json COMPLETE.json SHA256SUMS; do
  target="$source_root/$name"
  if [ "$name" = SHA256SUMS ]; then target="$root_files/$name"; fi
  if ! gcs_cp "$source_uri/$name" "$target"; then
    transport_failed=1
  fi
done

root_names_file="$scratch/root-names.txt"
if [ -s "$root_files/SHA256SUMS" ]; then
  if ! python3 - "$root_files/SHA256SUMS" > "$root_names_file" <<'PY'
import re
import sys
from pathlib import Path

pattern = re.compile(r"^[0-9a-f]{64}  ([^/]+)$")
names = []
for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
  match = pattern.fullmatch(line)
  if match is None:
    raise SystemExit("invalid root SHA256SUMS")
  names.append(match.group(1))
if len(names) != len(set(names)):
  raise SystemExit("root SHA256SUMS has duplicate members")
print("\n".join(names))
PY
  then
    echo "[P38S22.OFFSITE] REFUSING: root manifest cannot be parsed" >&2
    transport_failed=1
  fi
else
  : > "$root_names_file"
fi
mapfile -t root_names < "$root_names_file"
for name in "${root_names[@]}"; do
  if ! gcs_cp "$source_uri/$name" "$root_files/$name"; then
    transport_failed=1
  fi
done

for round_index in 000000 000001 000002; do
  round_dir="$round_root/$round_index"
  round_uri="$source_uri/rounds/$round_index"
  mkdir "$round_dir"
  for name in ROUND_ARCHIVE.tar SHA256SUMS ROUND_COMPLETE.json; do
    if ! gcs_cp "$round_uri/$name" "$round_dir/$name"; then
      transport_failed=1
    fi
  done
done

set +e
python3 "$script_dir/audit_p38s22_offsite.py" \
  --contract "$contract" \
  --source-root "$source_root" \
  --round-root "$round_root" \
  --reference-evidence "$task_dir/evidence/p38s22" \
  --analysis-source-commit "$analysis_source_commit" \
  --output "$return_dir"
audit_rc=$?
set -e
if [ "$audit_rc" -ne 0 ] && [ "$audit_rc" -ne 4 ]; then
  echo "[P38S22.OFFSITE] REFUSING: auditor failed without a sealed receipt rc=$audit_rc" >&2
  exit "$audit_rc"
fi
if [ "$transport_failed" -ne 0 ] && [ "$audit_rc" -eq 0 ]; then
  echo "[P38S22.OFFSITE] REFUSING: auditor passed after a transport failure" >&2
  exit 2
fi
test -s "$return_dir/SHA256SUMS"
(cd "$return_dir" && sha256sum -c SHA256SUMS --quiet)
readarray -t summary < <(python3 - "$return_dir/AUDIT.json" <<'PY'
import json
import sys

value = json.load(open(sys.argv[1], encoding="utf-8"))
totals = value.get("totals", {})
print(value["status"])
print(value["verdict"])
print(totals.get("n_action", "unknown"))
print(totals.get("a_b_differing_elements", "unknown"))
print(totals.get("a_b_differing_bytes", "unknown"))
print(totals.get("b_c_differing_elements", "unknown"))
print(totals.get("b_c_differing_bytes", "unknown"))
PY
)
echo "[P38S22.OFFSITE] RETURN_READY status=${summary[0]} verdict=${summary[1]} n_action=${summary[2]} a_b_elements=${summary[3]} a_b_bytes=${summary[4]} b_c_elements=${summary[5]} b_c_bytes=${summary[6]} return_dir=$return_dir analysis_source_commit=$analysis_source_commit rc=$audit_rc"
exit "$audit_rc"
