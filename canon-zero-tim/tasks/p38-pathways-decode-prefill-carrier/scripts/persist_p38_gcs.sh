#!/usr/bin/env bash
# Persist the in-pod P38 evidence before a controlled diagnostic exit.
set -euo pipefail

mode="${1:?usage: persist_p38_gcs.sh probe|collect|complete}"
case "$mode" in
  probe|collect|complete) ;;
  *) echo "[P38.GCS] REFUSING: invalid mode: $mode" >&2; exit 2 ;;
esac

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
  echo "[P38.GCS] REFUSING: neither gcloud nor gsutil is installed" >&2
  exit 1
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

  (
    cd "$stage"
    sha256sum run.log pre-alignment.jsonl mismatch-capsule.npz \
      serving-classification.json serving-capture.tar > SHA256SUMS
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

  for name in run.log pre-alignment.jsonl mismatch-capsule.npz \
      serving-classification.json serving-capture.tar SHA256SUMS; do
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
