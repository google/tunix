#!/usr/bin/env bash
# Fail-closed transport for immutable P64 training capsules. This file is
# sourced by step 90; keep it free of shell-option changes.

canon_p64_capsule_sync() {
  local phase="$1"
  local capsule="${CANON_P64_TRAINING_CAPSULE:-}"
  local binding="${capsule}.model.json"
  local remote="${CANON_P64_TRAINING_CAPSULE_GCS_URI:-}"
  local remote_binding="${remote}.model.json"
  local state="${CANON_STATE:-}"
  local receipt="${state%/}/p64_capsule_${phase}.receipt"
  local tool="none"
  local rc=0
  local capsule_sha=""
  local binding_sha=""
  local marker

  case "$phase" in
    capture|replay) ;;
    *)
      echo "[P64.CAPSULE] FATAL invalid transport phase: $phase" >&2
      return 2
      ;;
  esac
  if [ "$phase" != "${CANON_P64_TRAINING_CAPSULE_MODE:-}" ] || \
     [ "$capsule" != "${state%/}/p64_training_capsule.npz" ] || \
     [[ ! "$remote" =~ ^gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p64/[a-z0-9][a-z0-9-]*/training-capsule\.npz$ ]] || \
     [ -e "$receipt" ]; then
    echo "[P64.CAPSULE] FATAL transport identity or receipt path drifted" >&2
    return 2
  fi

  if command -v gcloud >/dev/null 2>&1; then
    tool="gcloud"
    if [ "$phase" = "capture" ]; then
      [ -s "$capsule" ] && [ -s "$binding" ] || rc=3
      if [ "$rc" -eq 0 ]; then
        gcloud storage cp --no-clobber "$capsule" "$remote" || rc=$?
      fi
      if [ "$rc" -eq 0 ]; then
        gcloud storage cp --no-clobber "$binding" "$remote_binding" || rc=$?
      fi
    else
      [ ! -e "$capsule" ] && [ ! -e "$binding" ] || rc=4
      if [ "$rc" -eq 0 ]; then
        mkdir -p "$(dirname -- "$capsule")"
        gcloud storage cp "$remote" "$capsule" || rc=$?
      fi
      if [ "$rc" -eq 0 ]; then
        gcloud storage cp "$remote_binding" "$binding" || rc=$?
      fi
    fi
  elif command -v gsutil >/dev/null 2>&1; then
    tool="gsutil"
    if [ "$phase" = "capture" ]; then
      [ -s "$capsule" ] && [ -s "$binding" ] || rc=3
      if [ "$rc" -eq 0 ]; then
        gsutil -n cp "$capsule" "$remote" || rc=$?
      fi
      if [ "$rc" -eq 0 ]; then
        gsutil -n cp "$binding" "$remote_binding" || rc=$?
      fi
    else
      [ ! -e "$capsule" ] && [ ! -e "$binding" ] || rc=4
      if [ "$rc" -eq 0 ]; then
        mkdir -p "$(dirname -- "$capsule")"
        gsutil cp "$remote" "$capsule" || rc=$?
      fi
      if [ "$rc" -eq 0 ]; then
        gsutil cp "$remote_binding" "$binding" || rc=$?
      fi
    fi
  elif python3 -c "from google.cloud import storage" >/dev/null 2>&1; then
    tool="python"
    if [ "$phase" = "capture" ]; then
      [ -s "$capsule" ] && [ -s "$binding" ] || rc=3
      if [ "$rc" -eq 0 ]; then
        python3 -c '
import sys
from google.cloud import storage

c_local, b_local, c_uri, b_uri = sys.argv[1:5]
client = storage.Client()
def parse(uri):
    return uri[5:].split("/", 1)

cb, ckey = parse(c_uri)
bb, bkey = parse(b_uri)
b_capsule = client.bucket(cb)
b_binding = client.bucket(bb)

try:
    b_capsule.blob(ckey).upload_from_filename(c_local, if_generation_match=0)
    b_binding.blob(bkey).upload_from_filename(b_local, if_generation_match=0)
except Exception as e:
    # If PreconditionFailed or already uploaded, verify existence
    if not (b_capsule.blob(ckey).exists() and b_binding.blob(bkey).exists()):
        raise
' "$capsule" "$binding" "$remote" "$remote_binding" || rc=$?
      fi
    else
      [ ! -e "$capsule" ] && [ ! -e "$binding" ] || rc=4
      if [ "$rc" -eq 0 ]; then
        mkdir -p "$(dirname -- "$capsule")"
        python3 -c '
import sys
from google.cloud import storage

c_local, b_local, c_uri, b_uri = sys.argv[1:5]
client = storage.Client()
def parse(uri):
    return uri[5:].split("/", 1)

cb, ckey = parse(c_uri)
bb, bkey = parse(b_uri)
client.bucket(cb).blob(ckey).download_to_filename(c_local)
client.bucket(bb).blob(bkey).download_to_filename(b_local)
' "$capsule" "$binding" "$remote" "$remote_binding" || rc=$?
      fi
    fi
  else
    rc=127
  fi

  if [ -s "$capsule" ]; then
    capsule_sha="$(sha256sum "$capsule" | awk '{print $1}')"
  fi
  if [ -s "$binding" ]; then
    binding_sha="$(sha256sum "$binding" | awk '{print $1}')"
  fi
  if [ "$phase" = "replay" ] && \
     { [ "$capsule_sha" != "${CANON_P64_TRAINING_CAPSULE_SHA256:-}" ] || \
       [ "$binding_sha" != "${CANON_P64_MODEL_BINDING_SHA256:-}" ]; }; then
    rc=5
  fi
  if [ "$rc" -ne 0 ]; then
    marker="[P64.CAPSULE] transport_failed mode=$phase tool=$tool rc=$rc remote=$remote"
  else
    marker="[P64.CAPSULE] transport_ready mode=$phase tool=$tool capsule_sha256=$capsule_sha binding_sha256=$binding_sha remote=$remote"
  fi
  printf '%s\n' "$marker"
  printf '%s\n' "$marker" > "$receipt"
  [ "$rc" -eq 0 ]
}
