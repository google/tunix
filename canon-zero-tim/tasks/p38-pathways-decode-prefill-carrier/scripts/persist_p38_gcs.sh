#!/usr/bin/env bash
# Persist the in-pod P38 evidence before a controlled diagnostic exit.
set -euo pipefail

mode="${1:?usage: persist_p38_gcs.sh probe|snapshot|round|m15-shard|m15-round|collect|complete [sequence]}"
case "$mode" in
  probe|snapshot|round|m15-shard|m15-round|collect|complete) ;;
  *) echo "[P38.GCS] REFUSING: invalid mode: $mode" >&2; exit 2 ;;
esac
snapshot_sequence="${2:-}"
if [ "$mode" = snapshot ] || [ "$mode" = round ] || \
   [ "$mode" = m15-shard ] || [ "$mode" = m15-round ]; then
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
  full-v1|round-alignment-v1|m15-wide-v1|p58-seam-v1) ;;
  *)
    echo "[P38.GCS] REFUSING: invalid durability profile: $CANON_P38_DURABILITY_PROFILE" >&2
    exit 2
    ;;
esac

: "${CANON_EXPECT_COMMIT:?CANON_EXPECT_COMMIT unset}"
: "${CANON_PKG:?CANON_PKG unset}"
case "$CANON_EXPECT_COMMIT" in
  [0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]\
[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]\
[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]\
[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]\
[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]) ;;
  *) echo "[P38.GCS] REFUSING: expected source is not a full SHA" >&2; exit 2 ;;
esac
runtime_source_commit="$(git -C "$CANON_PKG/.." rev-parse HEAD)" || {
  echo "[P38.GCS] REFUSING: executing checkout has no Git source identity" >&2
  exit 2
}
if [ "$runtime_source_commit" != "$CANON_EXPECT_COMMIT" ]; then
  echo "[P38.GCS] REFUSING: runtime source mismatch expected=$CANON_EXPECT_COMMIT observed=$runtime_source_commit" >&2
  exit 2
fi
echo "[P38.GCS] RUNTIME_SOURCE_PASS expected=$CANON_EXPECT_COMMIT observed=$runtime_source_commit"

bucket_namespace=p38
if [ "$CANON_P38_DURABILITY_PROFILE" = p58-seam-v1 ]; then
  [ "${CANON_P58_SEAM_LOCALIZATION:-}" = coarse ] || {
    echo "[P38.GCS] REFUSING: p58-seam-v1 requires the coarse selector" >&2
    exit 2
  }
  bucket_namespace=p58
fi
bucket_root="gs://yuxzhang-tunix-models/canon-zero-tim/evidence/$bucket_namespace/"
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

write_m15_round_failure() {
  local round_text="$1" stage_name="$2" exit_code="$3"
  local failure_path partial
  : "${CANON_P38_ROUND_SEAL_ACK_DIR:?}"
  failure_path="$CANON_P38_ROUND_SEAL_ACK_DIR/round-$round_text.failure.json"
  partial="$failure_path.partial"
  if [ -e "$failure_path" ]; then
    return 0
  fi
  python3 - "$partial" "$((10#$round_text))" "$stage_name" "$exit_code" <<'PY'
import json
import pathlib
import sys

pathlib.Path(sys.argv[1]).write_text(json.dumps({
    "action": "seal-round",
    "diagnostic_round": int(sys.argv[2]),
    "exit_code": int(sys.argv[4]),
    "schema": "canon-p38-round-seal-failure-v1",
    "stage": sys.argv[3],
    "status": "FAIL",
}, sort_keys=True) + "\n", encoding="utf-8")
PY
  mv -- "$partial" "$failure_path"
}

publish_m15_round_stage() {
  local ordinal="$1" stage_name="$2" status="$3" exit_code="$4"
  local stage_dir stage_file stage_path remote_path verify_path rc=0
  stage_file="STAGE_${ordinal}_${stage_name}_${status}.json"
  stage_dir="$round_root/stages-$snapshot_sequence"
  mkdir -p "$stage_dir"
  stage_path="$stage_dir/$stage_file"
  remote_path="$round_prefix/stages/$stage_file"
  if gcs_exists "$remote_path"; then
    echo "[P38.GCS] REFUSING: remote M15 round stage already exists: $stage_file" >&2
    return 2
  fi
  python3 - "$stage_path.partial" "$round_index" "$stage_name" \
      "$status" "$exit_code" "$runtime_source_commit" <<'PY'
import json
import pathlib
import sys

pathlib.Path(sys.argv[1]).write_text(json.dumps({
    "diagnostic_round": int(sys.argv[2]),
    "exit_code": int(sys.argv[5]),
    "runtime_source_commit": sys.argv[6],
    "schema": "m15-wide-round-stage-v1",
    "stage": sys.argv[3],
    "status": sys.argv[4],
}, sort_keys=True) + "\n", encoding="utf-8")
PY
  rc=$?
  [ "$rc" -eq 0 ] || return "$rc"
  mv -- "$stage_path.partial" "$stage_path" || return $?
  gcs_cp "$stage_path" "$remote_path" || return $?
  verify_path="$(mktemp)" || return $?
  gcs_cp "$remote_path" "$verify_path" || rc=$?
  if [ "$rc" -eq 0 ]; then
    cmp -- "$stage_path" "$verify_path" || rc=$?
  fi
  rm -f "$verify_path"
  [ "$rc" -eq 0 ] || return "$rc"
  echo "[P38.GCS] M15_ROUND_STAGE round=$round_index stage=$stage_name status=$status exit_code=$exit_code"
}

begin_m15_round_stage() {
  local ordinal="$1" stage_name="$2" rc=0
  publish_m15_round_stage "$ordinal" "$stage_name" STARTED 0 || rc=$?
  if [ "$rc" -ne 0 ]; then
    write_m15_round_failure \
      "$snapshot_sequence" "$stage_name-receipt" "$rc" || true
  fi
  return "$rc"
}

finish_m15_round_stage() {
  local ordinal="$1" stage_name="$2" exit_code="$3" receipt_rc=0
  if [ "$exit_code" -eq 0 ]; then
    publish_m15_round_stage "$ordinal" "$stage_name" PASS 0 || receipt_rc=$?
    if [ "$receipt_rc" -ne 0 ]; then
      write_m15_round_failure \
        "$snapshot_sequence" "$stage_name-receipt" "$receipt_rc" || true
      return "$receipt_rc"
    fi
    return 0
  fi
  write_m15_round_failure "$snapshot_sequence" "$stage_name" "$exit_code" || true
  publish_m15_round_stage "$ordinal" "$stage_name" FAIL "$exit_code" || true
  return "$exit_code"
}

if [ "$mode" = probe ]; then
  for marker in PREFLIGHT.json COLLECTED.json COMPLETE.json; do
    if gcs_exists "$CANON_P38_GCS_PREFIX/$marker"; then
      echo "[P38.GCS] REFUSING: remote marker already exists: $marker" >&2
      exit 1
    fi
  done
  python3 - "$stage/PREFLIGHT.json.partial" "$runtime_source_commit" <<'PY'
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
    "runtime_source_commit": sys.argv[2],
    "source_verified": True,
    "status": "writable-and-source-verified",
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

if [ "$mode" = m15-shard ]; then
  [ "$CANON_P38_DURABILITY_PROFILE" = m15-wide-v1 ] || {
    echo "[P38.GCS] REFUSING: m15-shard requires m15-wide-v1" >&2
    exit 2
  }
  : "${CANON_P38_SEAM_OBSERVER_DIR:?}"
  : "${CANON_P38_DIAGNOSTIC_ROUND_FILE:?}"
  round_text="$(tr -d '[:space:]' < "$CANON_P38_DIAGNOSTIC_ROUND_FILE")"
  case "$round_text" in
    ''|*[!0-9]*)
      echo "[P38.GCS] REFUSING: M15 shard round is invalid" >&2
      exit 2
      ;;
  esac
  printf -v round_sequence '%06d' "$((10#$round_text))"
  shard_prefix="$CANON_P38_GCS_PREFIX/wide/shards/$snapshot_sequence"
  shard_root="$CANON_STATE/p38_m15_wide_shards/round-$round_sequence"
  shard_stage="$shard_root/$snapshot_sequence"
  for remote_name in SHARD_ARCHIVE.tar SHA256SUMS SHARD_COMPLETE.json; do
    if gcs_exists "$shard_prefix/$remote_name"; then
      echo "[P38.GCS] REFUSING: remote M15 shard object already exists: $snapshot_sequence/$remote_name" >&2
      exit 2
    fi
  done
  mkdir -p "$shard_root"
  shard_rc=0
  python3 "$CANON_PKG/tasks/v1-apc-m15-target-debug/scripts/stage_m15_wide_shard.py" \
    --directory "$CANON_P38_SEAM_OBSERVER_DIR" \
    --shard-root "$shard_root" \
    --output "$shard_stage" \
    --round "$round_text" \
    --sequence "$((10#$snapshot_sequence))" \
    --max-records 32 \
    --max-bytes $((256 * 1024 * 1024)) \
    --expected-commit "$CANON_EXPECT_COMMIT" \
    --runtime-commit "$runtime_source_commit" || shard_rc=$?
  if [ "$shard_rc" -eq 3 ]; then
    exit 3
  elif [ "$shard_rc" -ne 0 ]; then
    exit "$shard_rc"
  fi
  shard_archive="$shard_root/$snapshot_sequence.tar"
  python3 "$archive_tool" create \
    --root "$shard_stage" \
    --manifest "$shard_stage/SHA256SUMS" \
    --output "$shard_archive"
  shard_archive_sha="$(sha256sum "$shard_archive" | awk '{print $1}')"
  shard_manifest_sha="$(sha256sum "$shard_stage/SHA256SUMS" | awk '{print $1}')"
  gcs_cp "$shard_archive" "$shard_prefix/SHARD_ARCHIVE.tar"
  gcs_cp "$shard_stage/SHA256SUMS" "$shard_prefix/SHA256SUMS"
  verify_dir="$(mktemp -d)"
  trap 'rm -rf "$verify_dir"' EXIT
  gcs_cp "$shard_prefix/SHARD_ARCHIVE.tar" "$verify_dir/SHARD_ARCHIVE.tar"
  gcs_cp "$shard_prefix/SHA256SUMS" "$verify_dir/SHA256SUMS"
  python3 "$archive_tool" verify \
    --archive "$verify_dir/SHARD_ARCHIVE.tar" \
    --expected-sha256 "$shard_archive_sha"
  cmp -- "$shard_stage/SHA256SUMS" "$verify_dir/SHA256SUMS"
  python3 - "$shard_stage/SHARD_COMPLETE.json.partial" \
    "$snapshot_sequence" "$round_text" "$shard_archive_sha" \
    "$shard_manifest_sha" "$runtime_source_commit" <<'PY'
import json
import os
import pathlib
import sys

inventory = json.loads(
    pathlib.Path(sys.argv[1]).with_name("SHARD_INVENTORY.json").read_text(
        encoding="utf-8"
    )
)
record = {
    "schema": "m15-wide-observer-shard-completion-v1",
    "status": "sealed-uploaded-verified",
    "claim_ceiling": "INCONCLUSIVE_PARTIAL_LIVE_EVIDENCE_UNTIL_WIDE_ROUND_COMPLETE",
    "sequence": int(sys.argv[2]),
    "diagnostic_round": int(sys.argv[3]),
    "archive_sha256": sys.argv[4],
    "manifest_sha256": sys.argv[5],
    "record_pairs": int(inventory["record_pairs"]),
    "payload_bytes": int(inventory["payload_bytes"]),
    "expected_source_commit": os.environ["CANON_EXPECT_COMMIT"],
    "runtime_source_commit": sys.argv[6],
}
pathlib.Path(sys.argv[1]).write_text(
    json.dumps(record, sort_keys=True) + "\n", encoding="utf-8"
)
PY
  mv -- "$shard_stage/SHARD_COMPLETE.json.partial" \
    "$shard_stage/SHARD_COMPLETE.json"
  gcs_cp "$shard_stage/SHARD_COMPLETE.json" \
    "$shard_prefix/SHARD_COMPLETE.json"
  gcs_cp "$shard_prefix/SHARD_COMPLETE.json" \
    "$verify_dir/SHARD_COMPLETE.json"
  cmp -- "$shard_stage/SHARD_COMPLETE.json" \
    "$verify_dir/SHARD_COMPLETE.json"
  shard_pairs="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["record_pairs"])' "$shard_stage/SHARD_INVENTORY.json")"
  echo "[P38.GCS] M15_SHARD_COMPLETE sequence=$snapshot_sequence round=$round_text pairs=$shard_pairs archive_sha256=$shard_archive_sha manifest_sha256=$shard_manifest_sha"
  exit 0
fi

if [ "$mode" = m15-round ]; then
  [ "$CANON_P38_DURABILITY_PROFILE" = m15-wide-v1 ] || {
    echo "[P38.GCS] REFUSING: m15-round requires m15-wide-v1" >&2
    exit 2
  }
  : "${CANON_APC_M15_TARGET_DEBUG:?}"
  : "${CANON_P38_SEAM_OBSERVER:?}"
  : "${CANON_P38_SEAM_OBSERVER_DIR:?}"
  : "${CANON_P38_SEAM_CLASSIFICATION:?}"
  : "${CANON_APC_M15_SEAM_BUNDLE:?}"
  : "${CANON_PRE_ALIGN_REPORT:?}"
  : "${CANON_P38_MISMATCH_CAPSULE:?}"
  : "${CANON_APC_M15_REPLAY_LEDGER:?}"
  round_index=$((10#$snapshot_sequence))
  round_prefix="$CANON_P38_GCS_PREFIX/wide/rounds/$snapshot_sequence"
  round_root="$CANON_STATE/p38_m15_wide_rounds"
  round_stage="$round_root/$snapshot_sequence"
  printf -v round_sequence '%06d' "$round_index"
  shard_root="$CANON_STATE/p38_m15_wide_shards/round-$round_sequence"
  if [ -e "$round_stage" ]; then
    echo "[P38.GCS] REFUSING: M15 wide round already exists: $snapshot_sequence" >&2
    exit 2
  fi
  for remote_name in ROUND_INPUT_RECEIPT.json p38_seam.classification.json \
      m15_wide_seam_bundle.tar WIDE_SHA256SUMS WIDE_ROUND_COMPLETE.json \
      classifier-input/CLASSIFIER_INPUT_RECEIPT.json; do
    if gcs_exists "$round_prefix/$remote_name"; then
      echo "[P38.GCS] REFUSING: remote M15 wide-round object already exists: $snapshot_sequence/$remote_name" >&2
      exit 2
    fi
  done
  mkdir -p "$round_root"
  begin_m15_round_stage 10 assemble || exit $?
  stage_rc=0
  python3 "$CANON_PKG/tasks/v1-apc-m15-target-debug/scripts/assemble_m15_wide_round.py" \
    --live-directory "$CANON_P38_SEAM_OBSERVER_DIR" \
    --shard-root "$shard_root" \
    --output "$round_stage" \
    --round "$round_index" \
    --pre-alignment "$CANON_PRE_ALIGN_REPORT" \
    --capsule "$CANON_P38_MISMATCH_CAPSULE" \
    --replay-ledger "$CANON_APC_M15_REPLAY_LEDGER" \
    --observer-mode "$CANON_P38_SEAM_OBSERVER" \
    --expected-commit "$CANON_EXPECT_COMMIT" \
    --runtime-commit "$runtime_source_commit" || stage_rc=$?
  finish_m15_round_stage 10 assemble "$stage_rc" || exit $?
  # The immutable observer values already live in verified wide shards.  Seal
  # the remaining host-only classifier inputs before running analysis code so
  # a classifier defect cannot force another 64-TPU rollout.
  begin_m15_round_stage 15 checkpoint-input || exit $?
  stage_rc=0
  python3 "$CANON_PKG/tasks/v1-apc-m15-target-debug/scripts/checkpoint_m15_classifier_input.py" \
    --directory "$round_stage" \
    --arm "$CANON_APC_M15_TARGET_DEBUG" \
    --output "$round_stage/CLASSIFIER_INPUT_RECEIPT.json" || stage_rc=$?
  classifier_input_prefix="$round_prefix/classifier-input"
  classifier_input_files=(
    ROUND_INPUT_RECEIPT.json
    m15-replay-envelope.jsonl
    pre-alignment.jsonl
  )
  if [ -s "$round_stage/mismatch-capsule.npz" ]; then
    classifier_input_files+=(mismatch-capsule.npz)
  fi
  if [ "$stage_rc" -eq 0 ]; then
    for name in "${classifier_input_files[@]}"; do
      gcs_cp "$round_stage/$name" "$classifier_input_prefix/$name" || stage_rc=$?
      [ "$stage_rc" -eq 0 ] || break
    done
  fi
  if [ "$stage_rc" -eq 0 ]; then
    for name in CLASSIFIER_INPUT_SHA256SUMS CLASSIFIER_INPUT_RECEIPT.json; do
      gcs_cp "$round_stage/$name" "$classifier_input_prefix/$name" || stage_rc=$?
      [ "$stage_rc" -eq 0 ] || break
    done
  fi
  classifier_verify_dir=""
  if [ "$stage_rc" -eq 0 ]; then
    classifier_verify_dir="$(mktemp -d)" || stage_rc=$?
  fi
  if [ "$stage_rc" -eq 0 ]; then
    for name in "${classifier_input_files[@]}" \
        CLASSIFIER_INPUT_SHA256SUMS CLASSIFIER_INPUT_RECEIPT.json; do
      gcs_cp "$classifier_input_prefix/$name" \
        "$classifier_verify_dir/$name" || stage_rc=$?
      [ "$stage_rc" -eq 0 ] || break
    done
  fi
  if [ "$stage_rc" -eq 0 ]; then
    cmp -- "$round_stage/CLASSIFIER_INPUT_SHA256SUMS" \
      "$classifier_verify_dir/CLASSIFIER_INPUT_SHA256SUMS" || stage_rc=$?
  fi
  if [ "$stage_rc" -eq 0 ]; then
    cmp -- "$round_stage/CLASSIFIER_INPUT_RECEIPT.json" \
      "$classifier_verify_dir/CLASSIFIER_INPUT_RECEIPT.json" || stage_rc=$?
  fi
  if [ "$stage_rc" -eq 0 ]; then
    (cd "$classifier_verify_dir" && \
      sha256sum -c CLASSIFIER_INPUT_SHA256SUMS --quiet) || stage_rc=$?
  fi
  if [ -n "$classifier_verify_dir" ]; then
    rm -rf -- "$classifier_verify_dir"
  fi
  finish_m15_round_stage 15 checkpoint-input "$stage_rc" || exit $?
  m15_round_args=(
    --directory "$round_stage"
    --alignment-report "$round_stage/pre-alignment.jsonl"
    --mode "$CANON_P38_SEAM_OBSERVER"
    --arm "$CANON_APC_M15_TARGET_DEBUG"
    --replay-ledger "$round_stage/m15-replay-envelope.jsonl"
  )
  if [ "$CANON_APC_M15_TARGET_DEBUG" = on ]; then
    m15_round_args+=(--require-first-action)
  fi
  if [ -s "$round_stage/mismatch-capsule.npz" ]; then
    m15_round_args+=(--capsule "$round_stage/mismatch-capsule.npz")
  fi
  if [ "$CANON_P38_SEAM_OBSERVER" = full ]; then
    : "${CANON_P38_SEAM_LAYER:?}"
    m15_round_args+=(--expected-layer "$CANON_P38_SEAM_LAYER")
  fi
  begin_m15_round_stage 20 classify || exit $?
  stage_rc=0
  python3 "$CANON_PKG/tasks/v1-apc-m15-target-debug/scripts/classify_m15_apc_wide_seam.py" \
    "${m15_round_args[@]}" \
    --output "$round_stage/p38_seam.classification.json" || stage_rc=$?
  finish_m15_round_stage 20 classify "$stage_rc" || exit $?
  package_args=()
  if [ -s "$round_stage/mismatch-capsule.npz" ]; then
    package_args+=(--capsule "$round_stage/mismatch-capsule.npz")
  fi
  begin_m15_round_stage 30 package || exit $?
  stage_rc=0
  python3 "$CANON_PKG/tasks/v1-apc-m15-target-debug/scripts/package_m15_apc_wide_seam.py" \
    --directory "$round_stage" \
    --classification "$round_stage/p38_seam.classification.json" \
    --alignment-report "$round_stage/pre-alignment.jsonl" \
    "${package_args[@]}" \
    --replay-ledger "$round_stage/m15-replay-envelope.jsonl" \
    --output "$round_stage/m15_wide_seam_bundle.tar" || stage_rc=$?
  finish_m15_round_stage 30 package "$stage_rc" || exit $?
  begin_m15_round_stage 35 local-export || exit $?
  stage_rc=0
  cp -- "$round_stage/p38_seam.classification.json" \
    "$CANON_P38_SEAM_CLASSIFICATION" || stage_rc=$?
  if [ "$stage_rc" -eq 0 ]; then
    cp -- "$round_stage/m15_wide_seam_bundle.tar" \
      "$CANON_APC_M15_SEAM_BUNDLE" || stage_rc=$?
  fi
  finish_m15_round_stage 35 local-export "$stage_rc" || exit $?
  round_files=(
    ROUND_INPUT_RECEIPT.json
    p38_seam.classification.json
    m15_wide_seam_bundle.tar
  )
  begin_m15_round_stage 40 manifest || exit $?
  stage_rc=0
  (
    cd "$round_stage"
    sha256sum "${round_files[@]}" > WIDE_SHA256SUMS
    sha256sum -c WIDE_SHA256SUMS --quiet
  ) || stage_rc=$?
  finish_m15_round_stage 40 manifest "$stage_rc" || exit $?
  begin_m15_round_stage 50 upload || exit $?
  stage_rc=0
  for name in "${round_files[@]}" WIDE_SHA256SUMS; do
    if [ "$stage_rc" -eq 0 ]; then
      gcs_cp "$round_stage/$name" "$round_prefix/$name" || stage_rc=$?
    fi
  done
  finish_m15_round_stage 50 upload "$stage_rc" || exit $?
  begin_m15_round_stage 60 remote-verify || exit $?
  stage_rc=0
  verify_dir="$(mktemp -d)"
  trap 'rm -rf "$verify_dir"' EXIT
  gcs_cp "$round_prefix/WIDE_SHA256SUMS" "$verify_dir/WIDE_SHA256SUMS" || stage_rc=$?
  if [ "$stage_rc" -eq 0 ]; then
    cmp -- "$round_stage/WIDE_SHA256SUMS" "$verify_dir/WIDE_SHA256SUMS" || stage_rc=$?
  fi
  for name in "${round_files[@]}"; do
    if [ "$stage_rc" -eq 0 ]; then
      gcs_cp "$round_prefix/$name" "$verify_dir/$name" || stage_rc=$?
    fi
  done
  if [ "$stage_rc" -eq 0 ]; then
    (cd "$verify_dir" && sha256sum -c WIDE_SHA256SUMS --quiet) || stage_rc=$?
  fi
  finish_m15_round_stage 60 remote-verify "$stage_rc" || exit $?
  round_manifest_sha="$(sha256sum "$round_stage/WIDE_SHA256SUMS" | awk '{print $1}')"
  begin_m15_round_stage 70 completion || exit $?
  stage_rc=0
  python3 - "$round_stage/WIDE_ROUND_COMPLETE.json.partial" \
    "$round_index" "$round_manifest_sha" "$runtime_source_commit" <<'PY' || stage_rc=$?
import json
import os
import pathlib
import sys

classification = json.loads(
    pathlib.Path(sys.argv[1]).with_name("p38_seam.classification.json").read_text(
        encoding="utf-8"
    )
)
receipt = json.loads(
    pathlib.Path(sys.argv[1]).with_name("ROUND_INPUT_RECEIPT.json").read_text(
        encoding="utf-8"
    )
)
record = {
    "schema": "m15-wide-round-completion-v1",
    "status": "classified-and-uploaded",
    "diagnostic_round": int(sys.argv[2]),
    "manifest_sha256": sys.argv[3],
    "classification": classification["classification"],
    "record_pairs": int(receipt["record_pairs"]),
    "shards": receipt["shards"],
    "expected_source_commit": os.environ["CANON_EXPECT_COMMIT"],
    "runtime_source_commit": sys.argv[4],
}
pathlib.Path(sys.argv[1]).write_text(
    json.dumps(record, sort_keys=True) + "\n", encoding="utf-8"
)
PY
  if [ "$stage_rc" -eq 0 ]; then
    mv -- "$round_stage/WIDE_ROUND_COMPLETE.json.partial" \
      "$round_stage/WIDE_ROUND_COMPLETE.json" || stage_rc=$?
  fi
  if [ "$stage_rc" -eq 0 ]; then
    gcs_cp "$round_stage/WIDE_ROUND_COMPLETE.json" \
      "$round_prefix/WIDE_ROUND_COMPLETE.json" || stage_rc=$?
  fi
  if [ "$stage_rc" -eq 0 ]; then
    gcs_cp "$round_prefix/WIDE_ROUND_COMPLETE.json" \
      "$verify_dir/WIDE_ROUND_COMPLETE.json" || stage_rc=$?
  fi
  if [ "$stage_rc" -eq 0 ]; then
    cmp -- "$round_stage/WIDE_ROUND_COMPLETE.json" \
      "$verify_dir/WIDE_ROUND_COMPLETE.json" || stage_rc=$?
  fi
  finish_m15_round_stage 70 completion "$stage_rc" || exit $?
  echo "[P38.GCS] M15_WIDE_ROUND_COMPLETE round=$round_index prefix=$round_prefix manifest_sha256=$round_manifest_sha"
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
  if [ "$CANON_P38_DURABILITY_PROFILE" = p58-seam-v1 ]; then
    p58_round_args=()
    if [ -s "$round_stage/mismatch-capsule.npz" ]; then
      p58_round_args+=(--capsule "$round_stage/mismatch-capsule.npz")
    fi
    JAX_PLATFORMS=cpu PYTHONPATH="$CANON_PKG/..:${PYTHONPATH:-}" \
      python3 "$CANON_PKG/tasks/p58-deepswe-native-zero-comparison/scripts/classify_p58_coarse_seam_round.py" \
        --directory "$round_stage" \
        --alignment-report "$round_stage/pre-alignment.jsonl" \
        "${p58_round_args[@]}" \
        --output "$round_stage/p58-seam.round.classification.json"
  fi
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
classification = target.with_name("p58-seam.round.classification.json")
if classification.is_file():
  payload = json.loads(classification.read_text(encoding="utf-8"))
  record["classification"] = payload["outcome"]
  record["classification_sha256"] = __import__("hashlib").sha256(
      classification.read_bytes()
  ).hexdigest()
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

  if [ "$CANON_P38_DURABILITY_PROFILE" = m15-wide-v1 ]; then
    case "${CANON_P38_DIAGNOSTIC_ROUNDS:-}" in
      ''|*[!0-9]*)
        echo "[P38.GCS] REFUSING: M15 diagnostic round count is invalid" >&2
        exit 2
        ;;
    esac
    [ "$CANON_P38_DIAGNOSTIC_ROUNDS" -ge 1 ] && \
      [ "$CANON_P38_DIAGNOSTIC_ROUNDS" -le 8 ] || {
      echo "[P38.GCS] REFUSING: M15 diagnostic round count must be in [1,8]" >&2
      exit 2
    }
    printf -v wide_round_text '%06d' "$((CANON_P38_DIAGNOSTIC_ROUNDS - 1))"
    wide_round="$CANON_STATE/p38_m15_wide_rounds/$wide_round_text"
    copy_required "${CANON_RUN_LOG:?}" run.log
    copy_required "${CANON_PRE_ALIGN_REPORT:?}" pre-alignment.jsonl
    copy_required "${CANON_P38_SEAM_CLASSIFICATION:?}" \
      p38_seam.classification.json
    copy_required "${CANON_APC_M15_SEAM_BUNDLE:?}" \
      m15_wide_seam_bundle.tar
    copy_required "$wide_round/WIDE_ROUND_COMPLETE.json" \
      WIDE_ROUND_COMPLETE.json
    collected_files=(
      run.log pre-alignment.jsonl p38_seam.classification.json
      m15_wide_seam_bundle.tar WIDE_ROUND_COMPLETE.json
    )
    if [ -s "${CANON_P38_MISMATCH_CAPSULE:?}" ]; then
      copy_required "$CANON_P38_MISMATCH_CAPSULE" mismatch-capsule.npz
      collected_files+=(mismatch-capsule.npz)
    fi
    for remote_name in "${collected_files[@]}" SHA256SUMS COLLECTED.json; do
      if gcs_exists "$CANON_P38_GCS_PREFIX/$remote_name"; then
        echo "[P38.GCS] REFUSING: remote M15 collection object already exists: $remote_name" >&2
        exit 2
      fi
    done
    (
      cd "$stage"
      sha256sum "${collected_files[@]}" > SHA256SUMS
      sha256sum -c SHA256SUMS --quiet
    )
    python3 - "$stage/COLLECTED.json.partial" "$runtime_source_commit" <<'PY'
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
    "schema": "m15-wide-gcs-collection-v1",
    "source_commit": os.environ["CANON_EXPECT_COMMIT"],
    "runtime_source_commit": sys.argv[2],
    "status": "collected-from-sealed-shards",
}
target.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
PY
    mv -- "$stage/COLLECTED.json.partial" "$stage/COLLECTED.json"
    for name in "${collected_files[@]}" SHA256SUMS; do
      upload "$stage/$name" "$name"
    done
    upload "$stage/COLLECTED.json" COLLECTED.json
    verify_dir="$(mktemp -d)"
    trap 'rm -rf "$verify_dir"' EXIT
    gcs_cp "$CANON_P38_GCS_PREFIX/SHA256SUMS" \
      "$verify_dir/SHA256SUMS"
    cmp -- "$stage/SHA256SUMS" "$verify_dir/SHA256SUMS"
    for name in "${collected_files[@]}"; do
      gcs_cp "$CANON_P38_GCS_PREFIX/$name" "$verify_dir/$name"
    done
    (cd "$verify_dir" && sha256sum -c SHA256SUMS --quiet)
    gcs_cp "$CANON_P38_GCS_PREFIX/COLLECTED.json" \
      "$verify_dir/COLLECTED.json"
    cmp -- "$stage/COLLECTED.json" "$verify_dir/COLLECTED.json"
    echo "[P38.GCS] COLLECTED prefix=$CANON_P38_GCS_PREFIX profile=m15-wide-v1 manifest_sha256=$(sha256sum "$stage/SHA256SUMS" | awk '{print $1}')"
    exit 0
  fi

  if [ "$CANON_P38_DURABILITY_PROFILE" = p58-seam-v1 ]; then
    [ "${CANON_P38_DIAGNOSTIC_ROUNDS:-}" = 3 ] || {
      echo "[P38.GCS] REFUSING: P58 seam collection requires three rounds" >&2
      exit 2
    }
    copy_required "${CANON_RUN_LOG:?}" run.log
    copy_required "${CANON_PRE_ALIGN_REPORT:?}" pre-alignment.jsonl
    copy_required "${CANON_P38_SEAM_CLASSIFICATION:?}" \
      p58-seam.classification.json
    collected_files=(run.log pre-alignment.jsonl p58-seam.classification.json)
    for round_index in 0 1 2; do
      printf -v round_text '%06d' "$round_index"
      round_root="$CANON_STATE/p38_gcs_rounds/$round_text"
      copy_required "$round_root/ROUND_COMPLETE.json" \
        "ROUND_COMPLETE.$round_text.json"
      copy_required "$round_root/p58-seam.round.classification.json" \
        "p58-seam.round.$round_text.classification.json"
      collected_files+=(
        "ROUND_COMPLETE.$round_text.json"
        "p58-seam.round.$round_text.classification.json"
      )
    done
    for remote_name in "${collected_files[@]}" SHA256SUMS COLLECTED.json; do
      if gcs_exists "$CANON_P38_GCS_PREFIX/$remote_name"; then
        echo "[P38.GCS] REFUSING: remote P58 seam object already exists: $remote_name" >&2
        exit 2
      fi
    done
    (
      cd "$stage"
      sha256sum "${collected_files[@]}" > SHA256SUMS
      sha256sum -c SHA256SUMS --quiet
    )
    python3 - "$stage/COLLECTED.json.partial" "$runtime_source_commit" <<'PY'
import json
import os
import pathlib
import sys

target = pathlib.Path(sys.argv[1])
record = {
    "attempt": os.environ.get("JOBSET_RESTART_ATTEMPT", "unknown"),
    "diagnostic_rounds": 3,
    "jobset": os.environ["CANON_P38_GCS_PREFIX"].split("/")[-2],
    "prefix": os.environ["CANON_P38_GCS_PREFIX"],
    "schema": "canon-p58-seam-gcs-collection-v1",
    "source_commit": os.environ["CANON_EXPECT_COMMIT"],
    "runtime_source_commit": sys.argv[2],
    "status": "collected-from-three-sealed-rounds",
}
target.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
PY
    mv -- "$stage/COLLECTED.json.partial" "$stage/COLLECTED.json"
    for name in "${collected_files[@]}" SHA256SUMS; do
      upload "$stage/$name" "$name"
    done
    upload "$stage/COLLECTED.json" COLLECTED.json
    verify_dir="$(mktemp -d)"
    trap 'rm -rf "$verify_dir"' EXIT
    gcs_cp "$CANON_P38_GCS_PREFIX/SHA256SUMS" "$verify_dir/SHA256SUMS"
    cmp -- "$stage/SHA256SUMS" "$verify_dir/SHA256SUMS"
    for name in "${collected_files[@]}"; do
      gcs_cp "$CANON_P38_GCS_PREFIX/$name" "$verify_dir/$name"
    done
    (cd "$verify_dir" && sha256sum -c SHA256SUMS --quiet)
    gcs_cp "$CANON_P38_GCS_PREFIX/COLLECTED.json" \
      "$verify_dir/COLLECTED.json"
    cmp -- "$stage/COLLECTED.json" "$verify_dir/COLLECTED.json"
    echo "[P38.GCS] COLLECTED prefix=$CANON_P38_GCS_PREFIX profile=p58-seam-v1 rounds=3"
    exit 0
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
  python3 - "$stage/COLLECTED.json.partial" "$runtime_source_commit" <<'PY'
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
    "runtime_source_commit": sys.argv[2],
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
python3 - "$stage/COMPLETE.json.partial" "$manifest_sha" \
  "$runtime_source_commit" <<'PY'
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
    "runtime_source_commit": sys.argv[3],
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
