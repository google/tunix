#!/usr/bin/env bash
# Persist the in-pod P38 evidence before a controlled diagnostic exit.
set -euo pipefail

mode="${1:?usage: persist_p38_gcs.sh probe|snapshot|round|collect|complete [sequence]}"
case "$mode" in
  probe|snapshot|round|collect|complete) ;;
  *) echo "[P38.GCS] REFUSING: invalid mode: $mode" >&2; exit 2 ;;
esac
snapshot_sequence="${2:-}"
if [ "$mode" = snapshot ] || [ "$mode" = round ]; then
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
: "${CANON_P38_DURABILITY_PROFILE:?CANON_P38_DURABILITY_PROFILE unset}"
case "$CANON_P38_DURABILITY_PROFILE" in
  full-v1|round-alignment-v1) ;;
  *)
    echo "[P38.GCS] REFUSING: invalid durability profile: $CANON_P38_DURABILITY_PROFILE" >&2
    exit 2
    ;;
esac

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
archive_tool="$CANON_PKG/tasks/p38-pathways-decode-prefill-carrier/scripts/p38_evidence_archive.py"
if [ ! -f "$archive_tool" ]; then
  echo "[P38.GCS] REFUSING: evidence archive tool is absent: $archive_tool" >&2
  exit 1
fi

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
  if [ -s "${CANON_APC_M15_REPLAY_LEDGER:-}" ]; then
    cp -- "$CANON_APC_M15_REPLAY_LEDGER" \
      "$live_stage/m15-replay-envelope.jsonl"
    live_files+=(m15-replay-envelope.jsonl)
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
    observer_round="${CANON_P38_LIVE_OBSERVER_ROUND:?live observer round unset}"
    case "$observer_round" in
      ''|*[!0-9]*)
        echo "[P38.GCS] REFUSING: live observer round must be a nonnegative integer" >&2
        exit 1
        ;;
    esac
    if [ -z "$observer_dir" ] || [ ! -d "$observer_dir" ]; then
      echo "[P38.GCS] REFUSING: observer snapshot requested without a directory" >&2
      exit 1
    fi
    observer_listing="$(
      python3 - "$observer_dir" "$observer_round" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
round_index = int(sys.argv[2])
paths = []
for pattern in (
    "p38_kv_observer_*.json",
    "p38_seam_*.json",
    "p38_tail_*.json",
    "p38_terminal_*.json",
):
  for json_path in root.glob(pattern):
    record = json.loads(json_path.read_text(encoding="utf-8"))
    if int(record.get("diagnostic_round", -1)) != round_index:
      continue
    npz_path = json_path.with_suffix(".npz")
    if not npz_path.is_file():
      raise SystemExit(f"paired observer NPZ is absent: {npz_path}")
    paths.extend((json_path, npz_path))
for path in sorted(paths, key=lambda item: item.name):
  print(path)
PY
    )"
    observer_sources=()
    if [ -n "$observer_listing" ]; then
      mapfile -t observer_sources <<< "$observer_listing"
    fi
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
  mapfile -t live_files < <(
    printf '%s\n' "${live_files[@]}" | LC_ALL=C sort -u
  )
  (
    cd "$live_stage"
    sha256sum "${live_files[@]}" > SHA256SUMS
    sha256sum -c SHA256SUMS --quiet
  )
  live_archive="$CANON_STATE/p38_gcs_live/$snapshot_sequence.tar"
  python3 "$archive_tool" create \
    --root "$live_stage" \
    --manifest "$live_stage/SHA256SUMS" \
    --output "$live_archive"
  live_archive_sha="$(sha256sum "$live_archive" | awk '{print $1}')"
  live_manifest_sha="$(sha256sum "$live_stage/SHA256SUMS" | awk '{print $1}')"
  python3 - "$live_stage/LIVE.json.partial" "$snapshot_sequence" \
    "$live_archive_sha" "$live_manifest_sha" "${live_files[@]}" <<'PY'
import json
import os
import pathlib
import sys

target = pathlib.Path(sys.argv[1])
record = {
    "archive_name": "LIVE_ARCHIVE.tar",
    "archive_sha256": sys.argv[3],
    "attempt": os.environ.get("JOBSET_RESTART_ATTEMPT", "unknown"),
    "files": sys.argv[5:],
    "jobset": os.environ["CANON_P38_GCS_PREFIX"].split("/")[-2],
    "logical_file_count": len(sys.argv[5:]),
    "manifest_sha256": sys.argv[4],
    "pod": os.environ.get("CANON_POD_NAME", "unknown"),
    "prefix": os.environ["CANON_P38_GCS_PREFIX"],
    "schema": "canon-p38-gcs-live-v1",
    "sequence": int(sys.argv[2]),
    "source_commit": os.environ.get("CANON_EXPECT_COMMIT", "unknown"),
    "status": "live-snapshot",
    "transport": "single-deterministic-tar-v1",
}
target.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
PY
  mv -- "$live_stage/LIVE.json.partial" "$live_stage/LIVE.json"
  gcs_cp "$live_archive" "$live_prefix/LIVE_ARCHIVE.tar"
  gcs_cp "$live_stage/SHA256SUMS" "$live_prefix/SHA256SUMS"
  verify_dir="$(mktemp -d)"
  trap 'rm -rf "$verify_dir"' EXIT
  gcs_cp "$live_prefix/LIVE_ARCHIVE.tar" "$verify_dir/LIVE_ARCHIVE.tar"
  gcs_cp "$live_prefix/SHA256SUMS" "$verify_dir/SHA256SUMS"
  python3 "$archive_tool" verify \
    --archive "$verify_dir/LIVE_ARCHIVE.tar" \
    --expected-sha256 "$live_archive_sha"
  cmp -- "$live_stage/SHA256SUMS" "$verify_dir/SHA256SUMS"
  gcs_cp "$live_stage/LIVE.json" "$live_prefix/LIVE.json"
  gcs_cp "$live_prefix/LIVE.json" "$verify_dir/LIVE.json"
  cmp -- "$live_stage/LIVE.json" "$verify_dir/LIVE.json"
  echo "[P38.GCS] LIVE sequence=$snapshot_sequence prefix=$live_prefix logical_files=${#live_files[@]} remote_objects=3 archive_sha256=$live_archive_sha"
  exit 0
fi

if [ "$mode" = round ]; then
  round_index=$((10#$snapshot_sequence))
  round_prefix="$CANON_P38_GCS_PREFIX/rounds/$snapshot_sequence"
  round_stage="$CANON_STATE/p38_gcs_rounds/$snapshot_sequence"
  if [ -e "$round_stage" ]; then
    echo "[P38.GCS] REFUSING: local round stage already exists: $snapshot_sequence" >&2
    exit 1
  fi
  if gcs_exists "$round_prefix/ROUND_COMPLETE.json"; then
    echo "[P38.GCS] REFUSING: remote round already exists: $snapshot_sequence" >&2
    exit 1
  fi
  observer_dir="${CANON_P38_SEAM_OBSERVER_DIR:-${CANON_P38_KV_OBSERVER_DIR:-${CANON_P38_SERVING_CAPTURE_DIR:-}}}"
  if [ "$CANON_P38_DURABILITY_PROFILE" = full-v1 ] && \
     [ -z "$observer_dir" ]; then
    echo "[P38.GCS] REFUSING: round sealing requires an observer directory" >&2
    exit 1
  fi
  stage_profile=full
  round_args=()
  if [ "$CANON_P38_DURABILITY_PROFILE" = round-alignment-v1 ]; then
    [ "${CANON_P38_FIXED_LM_HEAD:-0}" = "1" ] || {
      echo "[P38.GCS] REFUSING: round-alignment-v1 requires fixed lm-head" >&2
      exit 2
    }
    stage_profile=alignment-only
  else
    if [ -n "${CANON_P38_SEAM_OBSERVER:-}" ]; then
      round_args+=(--require-seam)
    fi
    if [ -n "${CANON_P38_KV_OBSERVER_DIR:-}" ]; then
      round_args+=(--require-kv)
    fi
    if [ "${CANON_P38_TAIL_OBSERVER:-0}" = "1" ]; then
      round_args+=(--require-tail)
    fi
    if [ "${CANON_P38_TERMINAL_DISCRIMINATOR:-0}" = "1" ]; then
      round_args+=(--require-terminal)
    fi
  fi
  python3 "$CANON_PKG/tasks/p38-pathways-decode-prefill-carrier/scripts/stage_p38_round.py" \
    --round "$round_index" \
    --profile "$stage_profile" \
    --output "$round_stage" \
    --run-log "${CANON_RUN_LOG:?}" \
    --pre-alignment "${CANON_PRE_ALIGN_REPORT:?}" \
    --capsule "${CANON_P38_MISMATCH_CAPSULE:?}" \
    --request-journal "${CANON_P38_REQUEST_JOURNAL:?}" \
    --incident-ledger "${CANON_P38_INCIDENT_LEDGER:?}" \
    --observer-dir "$observer_dir" \
    "${round_args[@]}"
  mapfile -t round_files < <(
    cd "$round_stage"
    find . -maxdepth 1 -type f ! -name 'SHA256SUMS' \
      ! -name 'ROUND_COMPLETE.json' -printf '%f\n' | LC_ALL=C sort
  )
  if [ "${#round_files[@]}" -eq 0 ]; then
    echo "[P38.GCS] REFUSING: round stage is empty" >&2
    exit 1
  fi
  (
    cd "$round_stage"
    sha256sum "${round_files[@]}" > SHA256SUMS
    sha256sum -c SHA256SUMS --quiet
  )
  round_archive="$CANON_STATE/p38_gcs_rounds/$snapshot_sequence.tar"
  python3 "$archive_tool" create \
    --root "$round_stage" \
    --manifest "$round_stage/SHA256SUMS" \
    --output "$round_archive"
  round_archive_sha="$(sha256sum "$round_archive" | awk '{print $1}')"
  verify_dir="$(mktemp -d)"
  trap 'rm -rf "$verify_dir"' EXIT
  gcs_cp "$round_archive" "$round_prefix/ROUND_ARCHIVE.tar"
  gcs_cp "$round_stage/SHA256SUMS" "$round_prefix/SHA256SUMS"
  gcs_cp "$round_prefix/ROUND_ARCHIVE.tar" "$verify_dir/ROUND_ARCHIVE.tar"
  gcs_cp "$round_prefix/SHA256SUMS" "$verify_dir/SHA256SUMS"
  python3 "$archive_tool" verify \
    --archive "$verify_dir/ROUND_ARCHIVE.tar" \
    --expected-sha256 "$round_archive_sha"
  cmp -- "$round_stage/SHA256SUMS" "$verify_dir/SHA256SUMS"
  manifest_sha="$(sha256sum "$round_stage/SHA256SUMS" | awk '{print $1}')"
  python3 - "$round_stage/ROUND_COMPLETE.json.partial" "$round_index" \
    "$manifest_sha" "$round_archive_sha" "${#round_files[@]}" \
    "$CANON_P38_DURABILITY_PROFILE" <<'PY'
import json
import os
import pathlib
import sys

target = pathlib.Path(sys.argv[1])
record = {
    "archive_name": "ROUND_ARCHIVE.tar",
    "archive_sha256": sys.argv[4],
    "attempt": os.environ.get("JOBSET_RESTART_ATTEMPT", "unknown"),
    "diagnostic_round": int(sys.argv[2]),
    "durability_profile": sys.argv[6],
    "logical_file_count": int(sys.argv[5]),
    "manifest_sha256": sys.argv[3],
    "schema": "canon-p38-round-completion-v1",
    "source_commit": os.environ.get("CANON_EXPECT_COMMIT", "unknown"),
    "status": "sealed-and-verified",
    "transport": "single-deterministic-tar-v1",
}
target.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
PY
  mv -- "$round_stage/ROUND_COMPLETE.json.partial" \
    "$round_stage/ROUND_COMPLETE.json"
  gcs_cp "$round_stage/ROUND_COMPLETE.json" \
    "$round_prefix/ROUND_COMPLETE.json"
  gcs_cp "$round_prefix/ROUND_COMPLETE.json" \
    "$verify_dir/ROUND_COMPLETE.json"
  cmp -- "$round_stage/ROUND_COMPLETE.json" \
    "$verify_dir/ROUND_COMPLETE.json"
  echo "[P38.GCS] ROUND_COMPLETE round=$round_index prefix=$round_prefix logical_files=${#round_files[@]} remote_objects=3 manifest_sha256=$manifest_sha archive_sha256=$round_archive_sha"
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
  if [ "$CANON_P38_DURABILITY_PROFILE" = full-v1 ]; then
    copy_required "${CANON_P38_MISMATCH_CAPSULE:?}" mismatch-capsule.npz
  elif [ -s "${CANON_P38_MISMATCH_CAPSULE:?}" ]; then
    copy_required "$CANON_P38_MISMATCH_CAPSULE" mismatch-capsule.npz
  fi
  copy_required "${CANON_P38_SERVING_CAPTURE_CLASSIFICATION:?}" serving-classification.json
  copy_required "${CANON_P38_SERVING_CAPTURE_ARCHIVE:?}" serving-capture.tar

  collected_files=(
    run.log pre-alignment.jsonl serving-classification.json serving-capture.tar
  )
  if [ -s "$stage/mismatch-capsule.npz" ]; then
    collected_files+=(mismatch-capsule.npz)
  fi
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
  if [ -n "${CANON_P38_TERMINAL_CLASSIFICATION:-}" ]; then
    copy_required "$CANON_P38_TERMINAL_CLASSIFICATION" \
      terminal-classification.json
    collected_files+=(terminal-classification.json)
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
