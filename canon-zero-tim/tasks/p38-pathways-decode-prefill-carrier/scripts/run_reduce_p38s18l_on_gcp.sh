#!/usr/bin/env bash
# Inventory all immutable snapshots beside GCS, select the highest-coverage
# source, reduce it, and upload a self-contained byte-preserving audit bundle.
set -euo pipefail

source_spec="${1:?usage: run_reduce_p38s18l_on_gcp.sh <live-root-or-snapshot-gs-uri> <derived-gs-uri> [scratch-parent]}"
derived_uri="${2:?usage: run_reduce_p38s18l_on_gcp.sh <live-root-or-snapshot-gs-uri> <derived-gs-uri> [scratch-parent]}"
scratch_parent="${3:-/tmp}"
bucket_root="gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/"
min_capsule_rounds=2

case "$source_spec" in
  "$bucket_root"*/attempt-0/live)
    live_root="$source_spec"
    ;;
  "$bucket_root"*/attempt-0/live/[0-9][0-9][0-9][0-9][0-9][0-9])
    live_root="${source_spec%/*}"
    ;;
  *)
    echo "[P38.REDUCE.GCP] REFUSING: invalid live root/snapshot URI" >&2
    exit 2
    ;;
esac
case "$derived_uri" in
  "$bucket_root"*/attempt-0/derived/p38s18l-seam-reduction-v[0-9]*) ;;
  *) echo "[P38.REDUCE.GCP] REFUSING: invalid versioned derived URI" >&2; exit 2 ;;
esac
test -d "$scratch_parent"

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
  gcs_list_recursive() { gsutil ls -r "$1/**"; }
else
  echo "[P38.REDUCE.GCP] REFUSING: gcloud or gsutil is required" >&2
  exit 2
fi

if gcs_exists "$derived_uri/files/REDUCTION_MANIFEST.json"; then
  echo "[P38.REDUCE.GCP] REFUSING: derived evidence already exists" >&2
  exit 3
fi

scratch="$(mktemp -d -p "$scratch_parent" p38s18l-reduce.XXXXXX)"
trap 'rm -rf -- "$scratch"' EXIT
listing="$scratch/object-listing.txt"
snapshot_selection="$scratch/SNAPSHOT_SELECTION.json"
source_dir="$scratch/source"
output_dir="$scratch/output"
mkdir -p "$source_dir"

script_dir="$(cd "$(dirname "$0")" && pwd)"
echo "[P38.REDUCE.GCP] INVENTORY live_root=$live_root"
gcs_list_recursive "$live_root" > "$listing"
set +e
python3 "$script_dir/select_p38_live_snapshot.py" \
  --listing "$listing" \
  --live-root "$live_root" \
  --min-capsule-rounds "$min_capsule_rounds" \
  --output "$snapshot_selection"
selection_rc=$?
set -e
if [ "$selection_rc" -ne 0 ]; then
  if [ -s "$snapshot_selection" ]; then
    cat "$snapshot_selection"
  fi
  exit "$selection_rc"
fi
source_uri="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_source_gcs_uri"])' "$snapshot_selection")"
selected_snapshot="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_snapshot"])' "$snapshot_selection")"

echo "[P38.REDUCE.GCP] DOWNLOAD snapshot=$selected_snapshot"
gcs_sync_down "$source_uri" "$source_dir"
test -s "$source_dir/LIVE.json"
test -s "$source_dir/SHA256SUMS"

mapfile -t capsules < <(find "$source_dir" -maxdepth 1 -type f \
  -name 'p38_frozenlake_mismatch_capsule.round-*.npz' | sort)
if [ "${#capsules[@]}" -lt "$min_capsule_rounds" ]; then
  echo "[P38.REDUCE.GCP] REFUSING: selected snapshot lost required capsules" >&2
  exit 2
fi
capsule_args=()
for capsule in "${capsules[@]}"; do
  capsule_args+=(--capsule "$capsule")
done

set +e
python3 "$script_dir/reduce_p38_seam_evidence.py" \
  --source-dir "$source_dir" \
  --source-gcs-uri "$source_uri" \
  --snapshot-selection "$snapshot_selection" \
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

archive_base="$(basename "$derived_uri")"
if command -v zstd >/dev/null 2>&1; then
  archive="$scratch/$archive_base.tar.zst"
  tar --zstd -cf "$archive" -C "$output_dir" .
else
  archive="$scratch/$archive_base.tar.gz"
  tar -czf "$archive" -C "$output_dir" .
fi
(cd "$(dirname "$archive")" && \
  sha256sum "$(basename "$archive")" > "$(basename "$archive").sha256")

echo "[P38.REDUCE.GCP] UPLOAD version=$archive_base"
gcs_sync_up "$output_dir" "$derived_uri/files"
gcs_cp "$archive" "$derived_uri/$(basename "$archive")"
gcs_cp "$archive.sha256" "$derived_uri/$(basename "$archive.sha256")"

manifest_sha="$(sha256sum "$output_dir/REDUCTION_MANIFEST.json" | awk '{print $1}')"
audit_sha="$(sha256sum "$output_dir/AMBIGUITY_AUDIT.json" | awk '{print $1}')"
archive_sha="$(sha256sum "$archive" | awk '{print $1}')"
verdict="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["verdict"])' "$output_dir/verdict.json")"
coverage="$(python3 -c 'import json,sys; x=json.load(open(sys.argv[1])); print("red_points={} matched_arm_keys={} aliases={} conflicts={} unmatched={}".format(x["red_points"], x["matched_arm_keys"], len(x["equivalent_alias_keys"]), len(x["ambiguous_keys"]), len(x["unmatched_keys"])))' "$output_dir/REDUCTION_MANIFEST.json")"
echo "[P38.REDUCE.GCP] COMPLETE verdict=$verdict reducer_rc=$reduce_rc snapshot=$selected_snapshot $coverage manifest_sha256=$manifest_sha ambiguity_audit_sha256=$audit_sha archive_sha256=$archive_sha destination=$derived_uri"
exit "$reduce_rc"
