#!/usr/bin/env bash
# Fail-closed GCS restore for Pathways XProf captures. This file is sourced by
# step 90; keep it free of shell-option changes.

canon_xprof_nonempty_count() {
  local directory="$1"
  local pattern="$2"
  if [ ! -d "$directory" ]; then
    printf '0\n'
    return 0
  fi
  find "$directory" -type f -name "$pattern" -size +0c 2>/dev/null \
    | wc -l | tr -d '[:space:]'
}

canon_xprof_gcs_receipt() {
  local receipt_path="$1"
  local status="$2"
  local tool="$3"
  local rc="$4"
  local xplanes="$5"
  local traces="$6"
  local remote="$7"
  local local_dir="$8"
  local receipt
  receipt="[P51.XPROF.GCS] phase=restore status=$status tool=$tool rc=$rc xplanes=$xplanes traces=$traces remote=$remote local=$local_dir"
  printf '%s\n' "$receipt"
  mkdir -p "$(dirname -- "$receipt_path")"
  if [ -e "$receipt_path" ]; then
    echo "[P51.XPROF.GCS] FATAL receipt already exists: $receipt_path" >&2
    return 2
  fi
  printf '%s\n' "$receipt" > "$receipt_path"
}

canon_xprof_gcs_restore() {
  local local_dir="$1"
  local receipt_path="$2"
  local remote="${CANON_XPROF_DIR:-}"
  local state="${CANON_STATE:-}"
  local state_name="${state##*/}"
  local attempt
  local expected_local
  local expected_receipt
  local tool="none"
  local rc=0
  local status
  local xplanes=0
  local traces=0

  if [[ ! "$remote" =~ ^gs://yuxzhang-tunix-models/tmp/canon-zero-tim/p33/${state_name}/attempt-(direct|[0-9]+)/xprof-update$ ]]; then
    canon_xprof_gcs_receipt \
      "$receipt_path" INVALID_CONTRACT "$tool" 2 0 0 \
      "${remote:-unset}" "$local_dir"
    return 2
  fi
  attempt="${BASH_REMATCH[1]}"
  if [ "$attempt" = direct ]; then
    expected_local="${state%/}/xprof-update"
  else
    expected_local="${state%/}/attempt-$attempt/xprof-update"
  fi
  expected_receipt="$(dirname -- "$expected_local")/xprof_gcs_restore.receipt"
  if [ "$local_dir" != "$expected_local" ] || \
     [ "$receipt_path" != "$expected_receipt" ] || \
     [ -e "$local_dir" ]; then
    canon_xprof_gcs_receipt \
      "$receipt_path" INVALID_CONTRACT "$tool" 2 0 0 "$remote" "$local_dir"
    return 2
  fi

  if command -v gcloud >/dev/null 2>&1; then
    tool="gcloud"
    gcloud storage rsync -r "$remote" "$local_dir" || rc=$?
  elif command -v gsutil >/dev/null 2>&1; then
    tool="gsutil"
    gsutil -m rsync -r "$remote" "$local_dir" || rc=$?
  else
    rc=127
  fi
  xplanes="$(canon_xprof_nonempty_count "$local_dir" '*.xplane.pb')"
  traces="$(canon_xprof_nonempty_count "$local_dir" '*.trace.json.gz')"
  if [ "$rc" -eq 0 ] && [ "$xplanes" -gt 0 ] && [ "$traces" -gt 0 ]; then
    status="PASS"
  elif [ "$rc" -eq 127 ]; then
    status="NO_TOOL"
  elif [ "$rc" -ne 0 ]; then
    status="TRANSPORT_ERROR"
  else
    status="MISSING_ARTIFACTS"
    rc=3
  fi
  canon_xprof_gcs_receipt \
    "$receipt_path" "$status" "$tool" "$rc" "$xplanes" "$traces" \
    "$remote" "$local_dir" || return $?
  [ "$status" = PASS ]
}
