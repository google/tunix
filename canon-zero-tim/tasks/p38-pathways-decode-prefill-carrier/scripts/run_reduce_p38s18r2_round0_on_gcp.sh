#!/usr/bin/env bash
# Reduce the fixed immutable P38s18r2 Round 0 beside GCS and upload only an
# independently audited, byte-preserving seam-and-tail evidence bundle.
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
contract="${1:-$script_dir/p38s18r2_round0_contract.json}"
scratch_parent="${2:-/tmp}"
return_dir="${3:-}"

test -s "$contract"
test -d "$scratch_parent"
if [ -n "$return_dir" ]; then
  if [ -e "$return_dir" ] || [ ! -d "$(dirname "$return_dir")" ]; then
    echo "[P38S18R2.REDUCE.GCP] REFUSING: return directory exists or its parent is absent" >&2
    exit 2
  fi
fi

contract_value() {
  python3 -c \
    'import json,sys; value=json.load(open(sys.argv[1], encoding="utf-8"))[sys.argv[2]]; print(str(value).lower() if isinstance(value, bool) else value)' \
    "$contract" "$1"
}

schema="$(contract_value schema)"
source_uri="$(contract_value source_gcs_uri)"
destination_uri="$(contract_value destination_gcs_uri)"
expected_source_commit="$(contract_value expected_source_commit)"
expected_manifest_sha="$(contract_value expected_source_manifest_sha256)"
expected_round="$(contract_value expected_diagnostic_round)"
expected_seam="$(contract_value expected_seam_records)"
expected_tail="$(contract_value expected_tail_records)"
expected_objects="$(contract_value expected_object_count)"
expected_manifest_files="$(contract_value expected_manifest_files)"
expected_red="$(contract_value expected_red_points)"
expected_rounds="$(contract_value expected_rounds)"
mode="$(contract_value mode)"
require_tail="$(contract_value require_tail)"
max_output_bytes="$(contract_value max_output_bytes)"

if [ "$schema" != "p38s18r2-round0-reduction-contract-v1" ] \
  || [ "$mode" != "layer" ] || [ "$require_tail" != "true" ]; then
  echo "[P38S18R2.REDUCE.GCP] REFUSING: contract schema/mode/tail drifted" >&2
  exit 2
fi
bucket_root="gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/"
case "$source_uri" in
  "$bucket_root"*/attempt-0/rounds/[0-9][0-9][0-9][0-9][0-9][0-9]) ;;
  *) echo "[P38S18R2.REDUCE.GCP] REFUSING: invalid immutable round URI" >&2; exit 2 ;;
esac
attempt_root="${source_uri%/rounds/*}"
case "$destination_uri" in
  "$attempt_root"/derived/p38s18r2-round0-seam-tail-reduction-v2) ;;
  *) echo "[P38S18R2.REDUCE.GCP] REFUSING: invalid v2 destination URI" >&2; exit 2 ;;
esac

if command -v gcloud >/dev/null 2>&1; then
  gcs_sync_down() { gcloud storage rsync --recursive "$1" "$2"; }
  gcs_sync_up() { gcloud storage rsync --recursive "$1" "$2"; }
  gcs_cp() { gcloud storage cp "$1" "$2"; }
  gcs_exists() { gcloud storage ls "$1" >/dev/null 2>&1; }
  gcs_list_recursive() { gcloud storage ls --recursive "$1/**"; }
elif command -v gsutil >/dev/null 2>&1; then
  gcs_sync_down() { gsutil -m rsync -r "$1" "$2"; }
  gcs_sync_up() { gsutil -m rsync -r "$1" "$2"; }
  gcs_cp() { gsutil -q cp "$1" "$2"; }
  gcs_exists() { gsutil -q stat "$1" >/dev/null 2>&1; }
  gcs_list_recursive() { gsutil ls -r "$1/**" | sed '/:$/d;/^$/d'; }
else
  echo "[P38S18R2.REDUCE.GCP] REFUSING: gcloud or gsutil is required" >&2
  exit 2
fi

if gcs_exists "$destination_uri/files/SHA256SUMS"; then
  echo "[P38S18R2.REDUCE.GCP] REFUSING: immutable v2 destination already exists" >&2
  exit 3
fi

repo_root="$(git -C "$script_dir" rev-parse --show-toplevel)"
analysis_source_commit="$(git -C "$repo_root" rev-parse HEAD)"
if [ -n "$(git -C "$repo_root" status --short)" ]; then
  echo "[P38S18R2.REDUCE.GCP] REFUSING: execution checkout is dirty" >&2
  exit 2
fi

scratch="$(mktemp -d -p "$scratch_parent" p38s18r2-round0-reduce.XXXXXX)"
trap 'rm -rf -- "$scratch"' EXIT
source_dir="$scratch/source"
output_dir="$scratch/output"
listing="$scratch/OBJECT_LISTING.txt"
bundle_audit="$scratch/bundle-audit.json"
audit_stdout="$scratch/bundle-audit.stdout"
audit_stderr="$scratch/bundle-audit.stderr"
mkdir -p "$source_dir"

echo "[P38S18R2.REDUCE.GCP] INVENTORY source=$source_uri"
gcs_list_recursive "$source_uri" > "$listing"
gcs_sync_down "$source_uri" "$source_dir"

mapfile -t capsules < <(find "$source_dir" -maxdepth 1 -type f \
  -name 'p38_frozenlake_mismatch_capsule*.npz' | sort)
if [ "${#capsules[@]}" -ne 1 ]; then
  echo "[P38S18R2.REDUCE.GCP] REFUSING: expected exactly one Round-0 capsule" >&2
  exit 2
fi

if python3 "$script_dir/reduce_p38_seam_tail_evidence.py" \
  --source-dir "$source_dir" \
  --source-gcs-uri "$source_uri" \
  --object-listing "$listing" \
  --capsule "${capsules[0]}" \
  --output-dir "$output_dir" \
  --mode "$mode" \
  --analysis-source-commit "$analysis_source_commit" \
  --expected-source-commit "$expected_source_commit" \
  --expected-manifest-sha256 "$expected_manifest_sha" \
  --expected-diagnostic-round "$expected_round" \
  --expected-seam-records "$expected_seam" \
  --expected-tail-records "$expected_tail" \
  --expected-object-count "$expected_objects" \
  --expected-manifest-files "$expected_manifest_files" \
  --expected-red-points "$expected_red" \
  --expected-rounds "$expected_rounds" \
  --max-output-bytes "$max_output_bytes"; then
  reducer_rc=0
else
  reducer_rc=$?
fi
if [ "$reducer_rc" -ne 0 ] && [ "$reducer_rc" -ne 4 ] \
  && [ "$reducer_rc" -ne 5 ]; then
  echo "[P38S18R2.REDUCE.GCP] REFUSING: reducer failed rc=$reducer_rc" >&2
  exit "$reducer_rc"
fi
if [ ! -s "$output_dir/SHA256SUMS" ]; then
  echo "[P38S18R2.REDUCE.GCP] REFUSING: reducer produced no sealed bundle" >&2
  exit 2
fi
(cd "$output_dir" && sha256sum -c SHA256SUMS --quiet)

python3 "$script_dir/audit_p38_seam_tail_reduction.py" \
  --bundle-dir "$output_dir" \
  --output "$bundle_audit" \
  >"$audit_stdout" 2>"$audit_stderr"
cat "$audit_stdout"
cat "$audit_stderr" >&2
test -s "$bundle_audit"
audit_sha="$(sha256sum "$bundle_audit" | awk '{print $1}')"
printf '%s  %s\n' "$audit_sha" "bundle-audit.json" \
  > "$bundle_audit.sha256"

archive_base="$(basename "$destination_uri")"
if command -v zstd >/dev/null 2>&1; then
  archive="$scratch/$archive_base.tar.zst"
  tar --zstd -cf "$archive" -C "$output_dir" .
else
  archive="$scratch/$archive_base.tar.gz"
  tar -czf "$archive" -C "$output_dir" .
fi
archive_sha="$(sha256sum "$archive" | awk '{print $1}')"
printf '%s  %s\n' "$archive_sha" "$(basename "$archive")" \
  > "$archive.sha256"

echo "[P38S18R2.REDUCE.GCP] UPLOAD destination=$destination_uri"
upload_files="$scratch/upload-files"
mkdir "$upload_files"
cp -a "$output_dir/." "$upload_files/"
rm -- "$upload_files/SHA256SUMS"
gcs_sync_up "$upload_files" "$destination_uri/files"
gcs_cp "$bundle_audit" "$destination_uri/bundle-audit.json"
gcs_cp "$bundle_audit.sha256" "$destination_uri/bundle-audit.json.sha256"
gcs_cp "$archive" "$destination_uri/$(basename "$archive")"
gcs_cp "$archive.sha256" "$destination_uri/$(basename "$archive.sha256")"
gcs_cp "$output_dir/SHA256SUMS" "$destination_uri/files/SHA256SUMS"

if [ -n "$return_dir" ]; then
  mkdir "$return_dir"
  cp -a "$output_dir" "$return_dir/files"
  cp "$bundle_audit" "$return_dir/bundle-audit.json"
  cp "$bundle_audit.sha256" "$return_dir/bundle-audit.json.sha256"
fi

manifest_sha="$(sha256sum "$output_dir/REDUCTION_MANIFEST.json" | awk '{print $1}')"
readarray -t summary < <(python3 - "$output_dir" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
manifest = json.loads((root / "REDUCTION_MANIFEST.json").read_text())
verdict = json.loads((root / "verdict.json").read_text())
print(verdict["verdict"])
print(manifest["red_points"])
print(manifest["matched_seam_keys"])
print(manifest["matched_tail_keys"])
print(len(manifest["equivalent_alias_keys"]))
print(len(manifest["tail_equivalent_alias_keys"]))
print(len(manifest["ambiguous_keys"]))
print(verdict["classifier_rc"])
PY
)
echo "[P38S18R2.REDUCE.GCP] COMPLETE verdict=${summary[0]} reducer_rc=$reducer_rc red_points=${summary[1]} matched_seam_keys=${summary[2]} matched_tail_keys=${summary[3]} seam_aliases=${summary[4]} tail_aliases=${summary[5]} conflicts=${summary[6]} classifier_rc=${summary[7]} manifest_sha256=$manifest_sha bundle_audit_sha256=$audit_sha archive_sha256=$archive_sha destination=$destination_uri return_dir=${return_dir:-none} analysis_source_commit=$analysis_source_commit"
exit "$reducer_rc"
