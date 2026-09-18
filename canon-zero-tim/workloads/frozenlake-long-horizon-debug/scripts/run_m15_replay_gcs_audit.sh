#!/usr/bin/env bash
# Reduce one immutable target attempt beside GCS and upload only small receipts.
set -euo pipefail

source_uri="${1:?usage: run_m15_replay_gcs_audit.sh <attempt-0-gs-uri> [scratch-parent]}"
scratch_parent="${2:-/tmp}"
bucket_root="gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/"
case "$source_uri" in
  "$bucket_root"*/attempt-0) ;;
  *) echo "[M15.APC.GCS] REFUSING: invalid attempt URI" >&2; exit 2 ;;
esac
derived_uri="$source_uri/derived/m15-replay-audit-v1"
test -d "$scratch_parent"

if command -v gcloud >/dev/null 2>&1; then
  gcs_cp() { gcloud storage cp "$1" "$2"; }
  gcs_sync_up() { gcloud storage rsync --recursive "$1" "$2"; }
  gcs_exists() { gcloud storage ls "$1" >/dev/null 2>&1; }
elif command -v gsutil >/dev/null 2>&1; then
  gcs_cp() { gsutil -q cp "$1" "$2"; }
  gcs_sync_up() { gsutil -m rsync -r "$1" "$2"; }
  gcs_exists() { gsutil -q stat "$1" >/dev/null 2>&1; }
else
  echo "[M15.APC.GCS] REFUSING: gcloud or gsutil is required" >&2
  exit 2
fi
if gcs_exists "$derived_uri/files/SHA256SUMS"; then
  echo "[M15.APC.GCS] REFUSING: immutable derived audit already exists" >&2
  exit 3
fi

scratch="$(mktemp -d -p "$scratch_parent" m15-apc-audit.XXXXXX)"
trap 'rm -rf -- "$scratch"' EXIT
root="$scratch/root"
capture="$scratch/capture"
output="$scratch/output"
mkdir -p "$root" "$capture"

gcs_cp "$source_uri/SHA256SUMS" "$root/SHA256SUMS"
while IFS='  ' read -r digest name; do
  case "$name" in
    ''|*/*|../*|*'..'*)
      echo "[M15.APC.GCS] REFUSING: unsafe root manifest member: $name" >&2
      exit 2
      ;;
  esac
  gcs_cp "$source_uri/$name" "$root/$name"
  actual="$(sha256sum "$root/$name" | awk '{print $1}')"
  [ "$actual" = "$digest" ] || {
    echo "[M15.APC.GCS] REFUSING: root SHA drifted: $name" >&2
    exit 2
  }
done < "$root/SHA256SUMS"
for marker in PREFLIGHT.json COLLECTED.json COMPLETE.json; do
  gcs_cp "$source_uri/$marker" "$root/$marker"
done
tar -xf "$root/serving-capture.tar" -C "$capture"

script_dir="$(cd "$(dirname "$0")" && pwd)"
python3 "$script_dir/audit_m15_replay_capture.py" \
  --root-dir "$root" \
  --capture-dir "$capture" \
  --source-gcs-uri "$source_uri" \
  --output-dir "$output"
(cd "$output" && sha256sum -c SHA256SUMS --quiet)

# Upload every receipt first and the manifest last.  The manifest is the
# immutable completion marker for this derived analysis prefix.
upload_stage="$scratch/upload"
mkdir "$upload_stage"
cp -a "$output/." "$upload_stage/"
rm -- "$upload_stage/SHA256SUMS"
gcs_sync_up "$upload_stage" "$derived_uri/files"
gcs_cp "$output/SHA256SUMS" "$derived_uri/files/SHA256SUMS"
receipt_sha="$(sha256sum "$output/RETURN_RECEIPT.json" | awk '{print $1}')"
manifest_sha="$(sha256sum "$output/SHA256SUMS" | awk '{print $1}')"
status="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$output/RETURN_RECEIPT.json")"
echo "[M15.APC.GCS] COMPLETE status=$status receipt_sha256=$receipt_sha manifest_sha256=$manifest_sha destination=$derived_uri"
