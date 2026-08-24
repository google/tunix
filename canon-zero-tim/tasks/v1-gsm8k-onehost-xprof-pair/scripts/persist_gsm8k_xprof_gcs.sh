#!/usr/bin/env bash
# Persist GSM8K one-host XProf / Perfetto / Pair artifacts into GCS bucket.
#
# Usage:
#   persist_gsm8k_xprof_gcs.sh <run_or_pair_root> [custom_gcs_prefix]
#
# Default destination:
#   gs://yuxzhang-tunix-models/canon-zero-tim/evidence/v1_gsm8k_xprof/<label>
set -euo pipefail

target_root="${1:?usage: persist_gsm8k_xprof_gcs.sh <run_or_pair_root> [custom_gcs_prefix]}"
target_root="${target_root%/}"
test -d "$target_root"
label="$(basename "$target_root")"

bucket_root="gs://yuxzhang-tunix-models/canon-zero-tim/evidence/v1_gsm8k_xprof/"
prefix="${2:-${bucket_root}${label}}"

echo "[GSM8K.GCS] Uploading $target_root -> $prefix ..."

if command -v gcloud >/dev/null 2>&1; then
  gcloud storage cp -r "$target_root" "$prefix"
elif command -v gsutil >/dev/null 2>&1; then
  gsutil -m cp -r "$target_root" "$prefix"
else
  python3 -c "
import os, sys
from google.cloud import storage
client = storage.Client()
b_name = '$prefix'.split('/')[2]
p_path = '/'.join('$prefix'.split('/')[3:])
bucket = client.bucket(b_name)
for root, _, files in os.walk('$target_root'):
    for f in files:
        full = os.path.join(root, f)
        rel = os.path.relpath(full, '$target_root')
        blob_path = f'{p_path}/{rel}' if p_path else rel
        bucket.blob(blob_path).upload_from_filename(full)
print('Uploaded via python storage client.')
"
fi

echo "✅ [GSM8K.GCS] Successfully uploaded $label to $prefix"
