#!/usr/bin/env bash
# Run beside GCS: download one immutable live snapshot, reduce it, upload only
# the byte-preserving red-point subset and its audit manifest.
set -euo pipefail

source_uri="${1:?usage: run_reduce_p38s18l_on_gcp.sh <live-snapshot-gs-uri> <derived-gs-uri> [scratch-parent]}"
derived_uri="${2:?usage: run_reduce_p38s18l_on_gcp.sh <live-snapshot-gs-uri> <derived-gs-uri> [scratch-parent]}"
scratch_parent="${3:-/tmp}"
bucket_root="gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/"

case "$source_uri" in
  "$bucket_root"*/attempt-0/live/[0-9][0-9][0-9][0-9][0-9][0-9]) ;;
  *) echo "[P38.REDUCE.GCP] REFUSING: invalid live snapshot URI: $source_uri" >&2; exit 2 ;;
esac
case "$derived_uri" in
  "$bucket_root"*/attempt-0/derived/p38s18l-seam-reduction-v1) ;;
  *) echo "[P38.REDUCE.GCP] REFUSING: invalid derived URI: $derived_uri" >&2; exit 2 ;;
esac
test -d "$scratch_parent"

if command -v gcloud >/dev/null 2>&1; then
  gcs_sync_down() { gcloud storage rsync --recursive "$1" "$2"; }
  gcs_sync_up() { gcloud storage rsync --recursive "$1" "$2"; }
  gcs_cp() { gcloud storage cp "$1" "$2"; }
  gcs_exists() { gcloud storage ls "$1" >/dev/null 2>&1; }
elif command -v gsutil >/dev/null 2>&1; then
  gcs_sync_down() { gsutil -m rsync -r "$1" "$2"; }
  gcs_sync_up() { gsutil -m rsync -r "$1" "$2"; }
  gcs_cp() { gsutil -q cp "$1" "$2"; }
  gcs_exists() { gsutil -q stat "$1" >/dev/null 2>&1; }
else
  echo "[P38.REDUCE.GCP] REFUSING: gcloud or gsutil is required" >&2
  exit 2
fi

if gcs_exists "$derived_uri/REDUCTION_MANIFEST.json"; then
  echo "[P38.REDUCE.GCP] REFUSING: derived evidence already exists: $derived_uri" >&2
  exit 3
fi

scratch="$(mktemp -d -p "$scratch_parent" p38s18l-reduce.XXXXXX)"
trap 'rm -rf -- "$scratch"' EXIT
source_dir="$scratch/source"
output_dir="$scratch/output"
mkdir -p "$source_dir"

echo "[P38.REDUCE.GCP] DOWNLOAD source=$source_uri"
gcs_sync_down "$source_uri" "$source_dir"
test -s "$source_dir/LIVE.json"
test -s "$source_dir/SHA256SUMS"

mapfile -t capsules < <(find "$source_dir" -maxdepth 1 -type f \
  -name 'p38_frozenlake_mismatch_capsule.round-*.npz' | sort)
if [ "${#capsules[@]}" -eq 0 ]; then
  echo "[P38.REDUCE.GCP] REFUSING: no immutable round capsules" >&2
  exit 2
fi
capsule_args=()
for capsule in "${capsules[@]}"; do
  capsule_args+=(--capsule "$capsule")
done

script_dir="$(cd "$(dirname "$0")" && pwd)"
set +e
python3 "$script_dir/reduce_p38_seam_evidence.py" \
  --source-dir "$source_dir" \
  --source-gcs-uri "$source_uri" \
  "${capsule_args[@]}" \
  --output-dir "$output_dir" \
  --mode layer \
  --expected-rounds 3
reduce_rc=$?
set -e

if [ ! -s "$output_dir/SHA256SUMS" ]; then
  echo "[P38.REDUCE.GCP] REFUSING: reducer produced no sealed output rc=$reduce_rc" >&2
  exit "$reduce_rc"
fi
(cd "$output_dir" && sha256sum -c SHA256SUMS --quiet)

if command -v zstd >/dev/null 2>&1; then
  archive="$scratch/p38s18l-seam-reduction-v1.tar.zst"
  tar --zstd -cf "$archive" -C "$output_dir" .
else
  archive="$scratch/p38s18l-seam-reduction-v1.tar.gz"
  tar -czf "$archive" -C "$output_dir" .
fi
(cd "$(dirname "$archive")" && \
  sha256sum "$(basename "$archive")" > "$(basename "$archive").sha256")

echo "[P38.REDUCE.GCP] UPLOAD destination=$derived_uri"
gcs_sync_up "$output_dir" "$derived_uri/files"
gcs_cp "$archive" "$derived_uri/$(basename "$archive")"
gcs_cp "$archive.sha256" "$derived_uri/$(basename "$archive.sha256")"

manifest_sha="$(sha256sum "$output_dir/REDUCTION_MANIFEST.json" | awk '{print $1}')"
archive_sha="$(sha256sum "$archive" | awk '{print $1}')"
verdict="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["verdict"])' "$output_dir/verdict.json")"
echo "[P38.REDUCE.GCP] COMPLETE verdict=$verdict reducer_rc=$reduce_rc manifest_sha256=$manifest_sha archive_sha256=$archive_sha destination=$derived_uri"
exit "$reduce_rc"
