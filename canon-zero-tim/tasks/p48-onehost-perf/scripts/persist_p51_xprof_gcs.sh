#!/usr/bin/env bash
# Persist one P51 xprof/perfetto capture into GCS. Adapted from
# tasks/p38-pathways-decode-prefill-carrier/scripts/persist_p38_gcs.sh:
# same gcloud->gsutil->python fallback chain, SHA256SUMS manifest,
# download-back verification, and refuse-on-existing completion marker.
#
# Usage:
#   persist_p51_xprof_gcs.sh <run_root>
#     run_root = /mnt/disks/tunix-data/logp_probe_1host/p51_gsm8k_xprof_<label>
#   P51_GCS_PREFIX overrides the destination (must stay under the bucket
#   root below); default: <bucket_root><label>.
set -euo pipefail

run_root="${1:?usage: persist_p51_xprof_gcs.sh <run_root>}"
run_root="${run_root%/}"
test -d "$run_root"
label="$(basename "$run_root")"
case "$label" in
  p51_gsm8k_xprof_*) ;;
  *)
    echo "[P51.GCS] REFUSING: run_root does not look like a P51 run: $label" >&2
    exit 2
    ;;
esac

bucket_root="gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p51/"
prefix="${P51_GCS_PREFIX:-${bucket_root}${label}}"
case "$prefix" in
  "$bucket_root"*) ;;
  *)
    echo "[P51.GCS] REFUSING: unexpected evidence prefix: $prefix" >&2
    exit 2
    ;;
esac

if command -v gcloud >/dev/null 2>&1; then
  gcs_cp() { gcloud storage cp "$1" "$2"; }
  gcs_exists() { gcloud storage ls "$1" >/dev/null 2>&1; }
elif command -v gsutil >/dev/null 2>&1; then
  gcs_cp() { gsutil -q cp "$1" "$2"; }
  gcs_exists() { gsutil -q stat "$1" >/dev/null 2>&1; }
else
  if ! python3 -c "import google.cloud.storage" >/dev/null 2>&1; then
    python3 -m pip install -q google-cloud-storage || true
  fi
  if python3 -c "import google.cloud.storage" >/dev/null 2>&1; then
    gcs_cp() {
      python3 - "$1" "$2" <<'PY'
import sys
from google.cloud import storage

src, dst = sys.argv[1], sys.argv[2]
client = storage.Client()
def parse_gs(url):
  assert url.startswith("gs://")
  parts = url[5:].split("/", 1)
  return parts[0], parts[1] if len(parts) > 1 else ""

if src.startswith("gs://"):
  b_name, o_name = parse_gs(src)
  client.bucket(b_name).blob(o_name).download_to_filename(dst)
elif dst.startswith("gs://"):
  b_name, o_name = parse_gs(dst)
  client.bucket(b_name).blob(o_name).upload_from_filename(src)
PY
    }
    gcs_exists() {
      python3 - "$1" <<'PY'
import sys
from google.cloud import storage

url = sys.argv[1]
client = storage.Client()
assert url.startswith("gs://")
parts = url[5:].split("/", 1)
blob = client.bucket(parts[0]).blob(parts[1] if len(parts) > 1 else "")
if not blob.exists():
  sys.exit(1)
PY
    }
  else
    echo "[P51.GCS] REFUSING: neither gcloud, gsutil, nor google-cloud-storage is installed" >&2
    exit 1
  fi
fi

if gcs_exists "$prefix/COMPLETE.json"; then
  echo "[P51.GCS] REFUSING: remote completion marker already exists: $prefix" >&2
  exit 1
fi

# One capture session per run; refuse ambiguity rather than guess.
mapfile -t sessions < <(
  find "$run_root/train/xprof/plugins/profile" -mindepth 1 -maxdepth 1 \
    -type d 2>/dev/null | LC_ALL=C sort
)
if [ "${#sessions[@]}" -ne 1 ]; then
  echo "[P51.GCS] REFUSING: expected exactly one capture session, found ${#sessions[@]}" >&2
  exit 1
fi
session="${sessions[0]}"
session_name="$(basename "$session")"

stage="$run_root/p51_gcs_stage"
if [ -e "$stage" ]; then
  echo "[P51.GCS] REFUSING: local stage already exists: $stage" >&2
  exit 1
fi
mkdir -p "$stage"

copy_required() {
  local source="$1" name="$2" partial
  partial="$stage/$name.partial"
  if [ ! -s "$source" ]; then
    echo "[P51.GCS] REFUSING: required artifact missing or empty: $source" >&2
    exit 1
  fi
  cp -- "$source" "$partial"
  mv -- "$partial" "$stage/$name"
}

staged_files=()
# The capture artifacts (xplane carries the device planes; keep original
# basenames so XProf serves the staged copy unchanged).
shopt -s nullglob
capture_sources=("$session"/*.xplane.pb "$session"/*.json.gz)
shopt -u nullglob
if [ "${#capture_sources[@]}" -eq 0 ]; then
  echo "[P51.GCS] REFUSING: capture session has no artifacts: $session" >&2
  exit 1
fi
for source in "${capture_sources[@]}"; do
  name="$(basename "$source")"
  copy_required "$source" "$name"
  staged_files+=("$name")
done
# Run context for whoever reads the capture later.
copy_required "$run_root/driver.log" driver.log
staged_files+=(driver.log)
copy_required "$run_root/train/raw.log" raw.log
staged_files+=(raw.log)

(
  cd "$stage"
  sha256sum "${staged_files[@]}" > SHA256SUMS
  sha256sum -c SHA256SUMS --quiet
)

for name in "${staged_files[@]}" SHA256SUMS; do
  gcs_cp "$stage/$name" "$prefix/$session_name/$name"
  echo "[P51.GCS] UPLOADED name=$name bytes=$(wc -c < "$stage/$name" | tr -d '[:space:]')"
done

verify="$(mktemp)"
trap 'rm -f "$verify"' EXIT
gcs_cp "$prefix/$session_name/SHA256SUMS" "$verify"
cmp -- "$stage/SHA256SUMS" "$verify"

manifest_sha="$(sha256sum "$stage/SHA256SUMS" | awk '{print $1}')"
python3 - "$stage/COMPLETE.json.partial" "$prefix" "$session_name" \
  "$manifest_sha" <<'PY'
import json
import pathlib
import sys

target = pathlib.Path(sys.argv[1])
record = {
    "manifest_sha256": sys.argv[4],
    "prefix": sys.argv[2],
    "schema": "canon-p51-xprof-gcs-v1",
    "session": sys.argv[3],
    "status": "uploaded-and-verified",
}
target.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
PY
mv -- "$stage/COMPLETE.json.partial" "$stage/COMPLETE.json"
gcs_cp "$stage/COMPLETE.json" "$prefix/COMPLETE.json"
gcs_cp "$prefix/COMPLETE.json" "$verify"
cmp -- "$stage/COMPLETE.json" "$verify"
echo "[P51.GCS] COMPLETE prefix=$prefix/$session_name manifest_sha256=$manifest_sha"
echo "[P51.GCS] view: xprof 'gs://…' 不可直读;下载后 xprof <dir> --port 8791,或 perfetto_trace.json.gz 拖 ui.perfetto.dev"
