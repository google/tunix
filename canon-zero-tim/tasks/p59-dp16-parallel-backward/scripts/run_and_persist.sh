#!/usr/bin/env bash
# Run one P59 workload and persist its irreplaceable reports before Pod exit.
set -uo pipefail

: "${CANON_STATE:?CANON_STATE unset}"
: "${CANON_PKG:?CANON_PKG unset}"
: "${CANON_P59_INNER_RUN_CMD:?CANON_P59_INNER_RUN_CMD unset}"
: "${CANON_P59_GCS_PREFIX:?CANON_P59_GCS_PREFIX unset}"
: "${CANON_P59_REQUIRE_XPROF:?CANON_P59_REQUIRE_XPROF unset}"
: "${CANON_PRE_ALIGN_REPORT:?CANON_PRE_ALIGN_REPORT unset}"
: "${CANON_ALIGN_REPORT:?CANON_ALIGN_REPORT unset}"
: "${CANON_UPDATE_REPORT:?CANON_UPDATE_REPORT unset}"

if [ "${JOBSET_RESTART_ATTEMPT:-unknown}" != "0" ]; then
  echo "[P59.PERSIST] REFUSING: evidence requires JOBSET_ATTEMPT 0" >&2
  exit 2
fi
case "$CANON_P59_REQUIRE_XPROF" in
  0|1) ;;
  *) echo "[P59.PERSIST] REFUSING: CANON_P59_REQUIRE_XPROF must be 0/1" >&2; exit 2 ;;
esac
bucket_root="gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p59/"
case "$CANON_P59_GCS_PREFIX" in
  "$bucket_root"*/attempt-0) ;;
  *) echo "[P59.PERSIST] REFUSING: unexpected GCS prefix" >&2; exit 2 ;;
esac

if command -v gcloud >/dev/null 2>&1; then
  gcs_cp() { gcloud storage cp "$1" "$2"; }
  gcs_exists() { gcloud storage ls "$1" >/dev/null 2>&1; }
elif command -v gsutil >/dev/null 2>&1; then
  gcs_cp() { gsutil -q cp "$1" "$2"; }
  gcs_exists() { gsutil -q stat "$1" >/dev/null 2>&1; }
else
  echo "[P59.PERSIST] REFUSING: gcloud or gsutil is required" >&2
  exit 2
fi

stage="$CANON_STATE/p59_persist"
if [ -e "$stage" ]; then
  echo "[P59.PERSIST] REFUSING: local persistence stage already exists" >&2
  exit 2
fi
if gcs_exists "$CANON_P59_GCS_PREFIX/PREFLIGHT.json"; then
  echo "[P59.PERSIST] REFUSING: remote label has already been used" >&2
  exit 2
fi
mkdir -p "$stage"
python3 - "$stage/PREFLIGHT.json" <<'PY'
import json
import os
import pathlib
import sys

pathlib.Path(sys.argv[1]).write_text(json.dumps({
    "attempt": os.environ.get("JOBSET_RESTART_ATTEMPT", "unknown"),
    "jobset": os.environ.get("CANON_WANDB_RUN_NAME", "unknown"),
    "schema": "canon-p59-persistence-preflight-v1",
    "source_commit": os.environ.get("CANON_EXPECT_COMMIT", "unknown"),
    "status": "armed",
}, sort_keys=True) + "\n", encoding="utf-8")
PY
gcs_cp "$stage/PREFLIGHT.json" "$CANON_P59_GCS_PREFIX/PREFLIGHT.json" || exit 3
echo "[P59.PERSIST] PREFLIGHT_PASS prefix=$CANON_P59_GCS_PREFIX"

inner_log="$stage/inner.log"
bash -c "$CANON_P59_INNER_RUN_CMD" 2>&1 | tee "$inner_log"
pipe_status=("${PIPESTATUS[@]}")
inner_rc="${pipe_status[0]}"
tee_rc="${pipe_status[1]}"
if [ "$tee_rc" -ne 0 ] && [ "$inner_rc" -eq 0 ]; then
  inner_rc=3
fi

kind=candidate
if [ "${CANON_P59_RANK_PARALLEL_BACKWARD:-0}" = "0" ]; then
  kind=control
elif [ "$CANON_P59_REQUIRE_XPROF" = "1" ]; then
  kind=profile
fi
classification="$stage/classification.json"
if [ "$inner_rc" -eq 0 ]; then
  JAX_PLATFORMS=cpu PYTHONPATH="$CANON_PKG/..:${PYTHONPATH:-}" \
    python3 "$CANON_PKG/tests/p59_backward/classify_and_analyze.py" \
      --kind "$kind" \
      --run-log "$inner_log" \
      --pre-alignment-report "$CANON_PRE_ALIGN_REPORT" \
      --update-report "$CANON_UPDATE_REPORT" \
      --alignment-report "$CANON_ALIGN_REPORT" \
      --output "$classification" || inner_rc=8
fi

files=("inner.log")
for pair in \
  "pre-alignment.jsonl:$CANON_PRE_ALIGN_REPORT" \
  "alignment.jsonl:$CANON_ALIGN_REPORT" \
  "updates.jsonl:$CANON_UPDATE_REPORT"; do
  name="${pair%%:*}"
  source="${pair#*:}"
  if [ -s "$source" ]; then
    cp -- "$source" "$stage/$name"
    files+=("$name")
  elif [ "$inner_rc" -eq 0 ]; then
    echo "[P59.PERSIST] FATAL: successful run lacks $name" >&2
    inner_rc=4
  fi
done
if [ -s "$classification" ]; then
  files+=("classification.json")
fi

if [ "${#files[@]}" -gt 0 ]; then
  (
    cd "$stage" || exit 1
    sha256sum "${files[@]}" > SHA256SUMS
    tar --sort=name --mtime=@0 --owner=0 --group=0 \
      -cf EVIDENCE.tar "${files[@]}" SHA256SUMS
  ) || exit 5
  gcs_cp "$stage/EVIDENCE.tar" "$CANON_P59_GCS_PREFIX/EVIDENCE.tar" || exit 5
  gcs_cp "$stage/SHA256SUMS" "$CANON_P59_GCS_PREFIX/SHA256SUMS" || exit 5
fi

xprof_sha=""
if [ "$CANON_P59_REQUIRE_XPROF" = "1" ]; then
  if [ ! -d "${CANON_XPROF_DIR:-}" ] || \
     ! find "$CANON_XPROF_DIR" -type f -print -quit | grep -q .; then
    echo "[P59.PERSIST] FATAL: required XProf directory is absent or empty" >&2
    inner_rc=6
  else
    tar --sort=name --mtime=@0 --owner=0 --group=0 \
      -C "$CANON_XPROF_DIR" -cf "$stage/XPROF.tar" . || exit 6
    xprof_sha="$(sha256sum "$stage/XPROF.tar" | awk '{print $1}')"
    printf '%s  %s\n' "$xprof_sha" XPROF.tar > "$stage/XPROF.sha256"
    gcs_cp "$stage/XPROF.tar" "$CANON_P59_GCS_PREFIX/XPROF.tar" || exit 6
    gcs_cp "$stage/XPROF.sha256" "$CANON_P59_GCS_PREFIX/XPROF.sha256" || exit 6
  fi
fi

python3 - "$stage/RESULT.json" "$inner_rc" "${#files[@]}" "$xprof_sha" <<'PY'
import json
import os
import pathlib
import sys

pathlib.Path(sys.argv[1]).write_text(json.dumps({
    "attempt": os.environ.get("JOBSET_RESTART_ATTEMPT", "unknown"),
    "evidence_files": int(sys.argv[3]),
    "inner_exit": int(sys.argv[2]),
    "schema": "canon-p59-persistence-result-v1",
    "source_commit": os.environ.get("CANON_EXPECT_COMMIT", "unknown"),
    "status": "complete" if int(sys.argv[2]) == 0 else "failed",
    "xprof_sha256": sys.argv[4] or None,
}, sort_keys=True) + "\n", encoding="utf-8")
PY
result_name=COMPLETE.json
if [ "$inner_rc" -ne 0 ]; then
  result_name=FAILED.json
fi
gcs_cp "$stage/RESULT.json" "$CANON_P59_GCS_PREFIX/$result_name" || exit 7
echo "[P59.PERSIST] RESULT status=$result_name inner_rc=$inner_rc files=${#files[@]} xprof_sha256=${xprof_sha:-none}"
exit "$inner_rc"
