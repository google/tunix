#!/usr/bin/env bash
# Persist the in-pod P38 evidence before a controlled diagnostic exit.
set -euo pipefail

mode="${1:?usage: persist_p38_gcs.sh probe|snapshot|collect|complete [sequence]}"
case "$mode" in
  probe|snapshot|collect|complete) ;;
  *) echo "[P38.GCS] REFUSING: invalid mode: $mode" >&2; exit 2 ;;
esac
snapshot_sequence="${2:-}"
if [ "$mode" = snapshot ]; then
  case "$snapshot_sequence" in
    [0-9][0-9][0-9][0-9][0-9][0-9]) ;;
    *)
      echo "[P38.GCS] REFUSING: snapshot sequence must be six digits" >&2
      exit 2
      ;;
  esac
elif [ -n "$snapshot_sequence" ]; then
  echo "[P38.GCS] REFUSING: unexpected sequence for mode $mode" >&2
  exit 2
fi

: "${CANON_STATE:?CANON_STATE unset}"
: "${CANON_P38_GCS_PREFIX:?CANON_P38_GCS_PREFIX unset}"

bucket_root="gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/"
case "$CANON_P38_GCS_PREFIX" in
  "$bucket_root"*/attempt-0) ;;
  *)
    echo "[P38.GCS] REFUSING: unexpected evidence prefix: $CANON_P38_GCS_PREFIX" >&2
    exit 2
    ;;
esac
job_component="${CANON_P38_GCS_PREFIX#"$bucket_root"}"
job_component="${job_component%/attempt-0}"
case "$job_component" in
  ''|*/*|*[!a-z0-9-]*)
    echo "[P38.GCS] REFUSING: invalid JobSet component: $job_component" >&2
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
b_name = parts[0]
o_name = parts[1] if len(parts) > 1 else ""
blob = client.bucket(b_name).blob(o_name)
if not blob.exists():
  sys.exit(1)
PY
    }
  else
    echo "[P38.GCS] REFUSING: neither gcloud, gsutil, nor google-cloud-storage is installed" >&2
    exit 1
  fi
fi

stage="$CANON_STATE/p38_gcs_stage"
mkdir -p "$stage"

copy_required() {
  local source="$1" name="$2" partial
  partial="$stage/$name.partial"
  if [ ! -s "$source" ]; then
    echo "[P38.GCS] REFUSING: required artifact missing or empty: $source" >&2
    exit 1
  fi
  cp -- "$source" "$partial"
  mv -- "$partial" "$stage/$name"
}

upload() {
  local source="$1" name="${2:-$(basename "$1")}" target
  target="$CANON_P38_GCS_PREFIX/$name"
  gcs_cp "$source" "$target"
  echo "[P38.GCS] UPLOADED name=$name bytes=$(wc -c < "$source" | tr -d '[:space:]')"
}

if [ "$mode" = probe ]; then
  for marker in PREFLIGHT.json COLLECTED.json COMPLETE.json; do
    if gcs_exists "$CANON_P38_GCS_PREFIX/$marker"; then
      echo "[P38.GCS] REFUSING: remote marker already exists: $marker" >&2
      exit 1
    fi
  done
  python3 - "$stage/PREFLIGHT.json.partial" <<'PY'
import json
import os
import pathlib
import sys

target = pathlib.Path(sys.argv[1])
record = {
    "attempt": os.environ.get("JOBSET_RESTART_ATTEMPT", "unknown"),
    "prefix": os.environ["CANON_P38_GCS_PREFIX"],
    "schema": "canon-p38-gcs-preflight-v1",
    "source_commit": os.environ.get("CANON_EXPECT_COMMIT", "unknown"),
    "status": "writable",
}
target.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
PY
  mv -- "$stage/PREFLIGHT.json.partial" "$stage/PREFLIGHT.json"
  upload "$stage/PREFLIGHT.json" PREFLIGHT.json
  verify="$(mktemp)"
  trap 'rm -f "$verify"' EXIT
  gcs_cp "$CANON_P38_GCS_PREFIX/PREFLIGHT.json" "$verify"
  cmp -- "$stage/PREFLIGHT.json" "$verify"
  echo "[P38.GCS] PREFLIGHT_PASS prefix=$CANON_P38_GCS_PREFIX"
  exit 0
fi

if [ "$mode" = snapshot ]; then
  live_prefix="$CANON_P38_GCS_PREFIX/live/$snapshot_sequence"
  live_stage="$CANON_STATE/p38_gcs_live/$snapshot_sequence"
  if [ -e "$live_stage" ]; then
    echo "[P38.GCS] REFUSING: local live snapshot already exists: $snapshot_sequence" >&2
    exit 1
  fi
  if gcs_exists "$live_prefix/LIVE.json"; then
    echo "[P38.GCS] REFUSING: remote live snapshot already exists: $snapshot_sequence" >&2
    exit 1
  fi
  mkdir -p "$live_stage"
  live_files=()
  if [ -s "${CANON_RUN_LOG:-}" ]; then
    cp -- "$CANON_RUN_LOG" "$live_stage/run.log"
    live_files+=(run.log)
  fi
  if [ -s "${CANON_P38_REQUEST_JOURNAL:-}" ]; then
    cp -- "$CANON_P38_REQUEST_JOURNAL" "$live_stage/request-journal.jsonl"
    live_files+=(request-journal.jsonl)
  fi
  if [ -s "${CANON_P38_INCIDENT_LEDGER:-}" ]; then
    cp -- "$CANON_P38_INCIDENT_LEDGER" "$live_stage/incident-ledger.jsonl"
    live_files+=(incident-ledger.jsonl)
  fi
  if [ -s "${CANON_P38_DIAGNOSTIC_ROUND_FILE:-}" ]; then
    cp -- "$CANON_P38_DIAGNOSTIC_ROUND_FILE" "$live_stage/diagnostic-round.txt"
    live_files+=(diagnostic-round.txt)
  fi
  if [ -s "${CANON_PRE_ALIGN_REPORT:-}" ]; then
    cp -- "$CANON_PRE_ALIGN_REPORT" "$live_stage/pre-alignment.jsonl"
    live_files+=(pre-alignment.jsonl)
  fi
  if [ -n "${CANON_P38_MISMATCH_CAPSULE:-}" ]; then
    shopt -s nullglob
    capsule_sources=(
      "${CANON_P38_MISMATCH_CAPSULE%.npz}".round-*.npz
      "$CANON_P38_MISMATCH_CAPSULE"
    )
    shopt -u nullglob
    for capsule_source in "${capsule_sources[@]}"; do
      [ -s "$capsule_source" ] || continue
      capsule_name="$(basename "$capsule_source")"
      cp -- "$capsule_source" "$live_stage/$capsule_name"
      live_files+=("$capsule_name")
    done
  fi
  if [ "${CANON_P38_LIVE_INCLUDE_OBSERVER:-0}" = "1" ]; then
    observer_dir="${CANON_P38_SEAM_OBSERVER_DIR:-${CANON_P38_KV_OBSERVER_DIR:-}}"
    if [ -z "$observer_dir" ] || [ ! -d "$observer_dir" ]; then
      echo "[P38.GCS] REFUSING: observer snapshot requested without a directory" >&2
      exit 1
    fi
    shopt -s nullglob
    observer_sources=(
      "$observer_dir"/p38_kv_observer_*.json
      "$observer_dir"/p38_kv_observer_*.npz
      "$observer_dir"/p38_seam_*.json
      "$observer_dir"/p38_seam_*.npz
    )
    shopt -u nullglob
    if [ "${#observer_sources[@]}" -eq 0 ]; then
      echo "[P38.GCS] REFUSING: observer snapshot requested without records" >&2
      exit 1
    fi
    for observer_source in "${observer_sources[@]}"; do
      observer_name="$(basename "$observer_source")"
      cp -- "$observer_source" "$live_stage/$observer_name"
      live_files+=("$observer_name")
    done
  fi
  if [ "${#live_files[@]}" -eq 0 ]; then
    echo "[P38.GCS] SNAPSHOT_SKIPPED sequence=$snapshot_sequence reason=no-host-evidence"
    rmdir "$live_stage"
    exit 3
  fi
  (
    cd "$live_stage"
    sha256sum "${live_files[@]}" > SHA256SUMS
    sha256sum -c SHA256SUMS --quiet
  )
  python3 - "$live_stage/LIVE.json.partial" "$snapshot_sequence" \
    "${live_files[@]}" <<'PY'
import json
import os
import pathlib
import sys

target = pathlib.Path(sys.argv[1])
record = {
    "attempt": os.environ.get("JOBSET_RESTART_ATTEMPT", "unknown"),
    "files": sys.argv[3:],
    "jobset": os.environ["CANON_P38_GCS_PREFIX"].split("/")[-2],
    "pod": os.environ.get("CANON_POD_NAME", "unknown"),
    "prefix": os.environ["CANON_P38_GCS_PREFIX"],
    "schema": "canon-p38-gcs-live-v1",
    "sequence": int(sys.argv[2]),
    "source_commit": os.environ.get("CANON_EXPECT_COMMIT", "unknown"),
    "status": "live-snapshot",
}
target.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
PY
  mv -- "$live_stage/LIVE.json.partial" "$live_stage/LIVE.json"
  for name in "${live_files[@]}" SHA256SUMS; do
    gcs_cp "$live_stage/$name" "$live_prefix/$name"
    echo "[P38.GCS] LIVE_UPLOADED sequence=$snapshot_sequence name=$name bytes=$(wc -c < "$live_stage/$name" | tr -d '[:space:]')"
  done
  gcs_cp "$live_stage/LIVE.json" "$live_prefix/LIVE.json"
  verify="$(mktemp)"
  trap 'rm -f "$verify"' EXIT
  gcs_cp "$live_prefix/LIVE.json" "$verify"
  cmp -- "$live_stage/LIVE.json" "$verify"
  echo "[P38.GCS] LIVE sequence=$snapshot_sequence prefix=$live_prefix files=${live_files[*]}"
  exit 0
fi

if [ "$mode" = collect ]; then
  if [ -e "$stage/COLLECTED.json" ]; then
    echo "[P38.GCS] REFUSING: local collection marker already exists" >&2
    exit 1
  fi
  if gcs_exists "$CANON_P38_GCS_PREFIX/COLLECTED.json"; then
    echo "[P38.GCS] REFUSING: remote collection marker already exists" >&2
    exit 1
  fi

  copy_required "${CANON_RUN_LOG:?}" run.log
  copy_required "${CANON_PRE_ALIGN_REPORT:?}" pre-alignment.jsonl
  copy_required "${CANON_P38_MISMATCH_CAPSULE:?}" mismatch-capsule.npz
  copy_required "${CANON_P38_SERVING_CAPTURE_CLASSIFICATION:?}" serving-classification.json
  copy_required "${CANON_P38_SERVING_CAPTURE_ARCHIVE:?}" serving-capture.tar

  collected_files=(
    run.log pre-alignment.jsonl mismatch-capsule.npz
    serving-classification.json serving-capture.tar
  )
  if [ -n "${CANON_P38_KV_OBSERVER_CLASSIFICATION:-}" ]; then
    copy_required "$CANON_P38_KV_OBSERVER_CLASSIFICATION" \
      kv-observer-classification.json
    collected_files+=(kv-observer-classification.json)
  fi
  if [ -n "${CANON_P38_SEAM_CLASSIFICATION:-}" ]; then
    copy_required "$CANON_P38_SEAM_CLASSIFICATION" \
      seam-classification.json
    collected_files+=(seam-classification.json)
  fi
  shopt -s nullglob
  round_capsules=("${CANON_P38_MISMATCH_CAPSULE%.npz}".round-*.npz)
  shopt -u nullglob
  for round_capsule in "${round_capsules[@]}"; do
    round_suffix="${round_capsule#"${CANON_P38_MISMATCH_CAPSULE%.npz}."}"
    round_name="mismatch-capsule.$round_suffix"
    copy_required "$round_capsule" "$round_name"
    collected_files+=("$round_name")
  done

  (
    cd "$stage"
    sha256sum "${collected_files[@]}" > SHA256SUMS
    sha256sum -c SHA256SUMS --quiet
  )
  python3 - "$stage/COLLECTED.json.partial" <<'PY'
import json
import os
import pathlib
import sys

target = pathlib.Path(sys.argv[1])
record = {
    "attempt": os.environ.get("JOBSET_RESTART_ATTEMPT", "unknown"),
    "jobset": os.environ["CANON_P38_GCS_PREFIX"].split("/")[-2],
    "pod": os.environ.get("CANON_POD_NAME", "unknown"),
    "prefix": os.environ["CANON_P38_GCS_PREFIX"],
    "schema": "canon-p38-gcs-collection-v1",
    "source_commit": os.environ.get("CANON_EXPECT_COMMIT", "unknown"),
    "status": "collected",
}
target.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
PY
  mv -- "$stage/COLLECTED.json.partial" "$stage/COLLECTED.json"

  for name in "${collected_files[@]}" SHA256SUMS; do
    upload "$stage/$name" "$name"
  done
  upload "$stage/COLLECTED.json" COLLECTED.json

  verify="$(mktemp)"
  trap 'rm -f "$verify"' EXIT
  gcs_cp "$CANON_P38_GCS_PREFIX/SHA256SUMS" "$verify"
  cmp -- "$stage/SHA256SUMS" "$verify"
  echo "[P38.GCS] COLLECTED prefix=$CANON_P38_GCS_PREFIX manifest_sha256=$(sha256sum "$stage/SHA256SUMS" | awk '{print $1}')"
  exit 0
fi

if [ ! -s "$stage/COLLECTED.json" ] || [ ! -s "$stage/SHA256SUMS" ]; then
  echo "[P38.GCS] REFUSING: complete requested before local collection" >&2
  exit 1
fi
if ! gcs_exists "$CANON_P38_GCS_PREFIX/COLLECTED.json"; then
  echo "[P38.GCS] REFUSING: remote collection marker is absent" >&2
  exit 1
fi
if gcs_exists "$CANON_P38_GCS_PREFIX/COMPLETE.json"; then
  echo "[P38.GCS] REFUSING: remote completion marker already exists" >&2
  exit 1
fi

manifest_sha="$(sha256sum "$stage/SHA256SUMS" | awk '{print $1}')"
python3 - "$stage/COMPLETE.json.partial" "$manifest_sha" <<'PY'
import json
import os
import pathlib
import sys

target = pathlib.Path(sys.argv[1])
record = {
    "attempt": os.environ.get("JOBSET_RESTART_ATTEMPT", "unknown"),
    "manifest_sha256": sys.argv[2],
    "prefix": os.environ["CANON_P38_GCS_PREFIX"],
    "schema": "canon-p38-gcs-completion-v1",
    "source_commit": os.environ.get("CANON_EXPECT_COMMIT", "unknown"),
    "status": "postflight-accepted",
}
target.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
PY
mv -- "$stage/COMPLETE.json.partial" "$stage/COMPLETE.json"
upload "$stage/COMPLETE.json" COMPLETE.json
verify="$(mktemp)"
trap 'rm -f "$verify"' EXIT
gcs_cp "$CANON_P38_GCS_PREFIX/COMPLETE.json" "$verify"
cmp -- "$stage/COMPLETE.json" "$verify"
echo "[P38.GCS] COMPLETE prefix=$CANON_P38_GCS_PREFIX manifest_sha256=$manifest_sha"
