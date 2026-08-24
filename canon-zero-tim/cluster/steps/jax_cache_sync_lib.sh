#!/usr/bin/env bash
# Shared best-effort JAX persistent-cache synchronization with auditable receipts.
# This file is sourced by step 28 and step 90; keep it free of shell options.

canon_jax_cache_file_count() {
  local cache_dir="$1"
  if [ ! -d "$cache_dir" ]; then
    printf '0\n'
    return 0
  fi
  find "$cache_dir" -type f 2>/dev/null | wc -l | tr -d '[:space:]'
}

canon_jax_cache_receipt() {
  local phase="$1"
  local status="$2"
  local tool="$3"
  local rc="$4"
  local entries="$5"
  local profile="$6"
  local bucket="$7"
  local cache_dir="$8"
  local receipt
  receipt="[JAX_CACHE_SYNC] phase=$phase status=$status tool=$tool rc=$rc entries=$entries profile=$profile bucket=$bucket local=$cache_dir"
  printf '%s\n' "$receipt"
  if [ -n "${CANON_STATE:-}" ] && [ -d "$CANON_STATE" ]; then
    printf '%s\n' "$receipt" > "$CANON_STATE/jax_cache_${phase}.receipt"
  fi
}

canon_jax_cache_sync() {
  local phase="$1"
  local bucket="${CANON_GCS_CACHE_BUCKET:-}"
  local cache_dir="${JAX_COMPILATION_CACHE_DIR:-}"
  local profile
  local gcs_path
  local tool="none"
  local rc=0
  local status
  local entries

  profile="$(basename -- "${CANON_PROFILE_FILE:-default}" .env)"
  if [ -z "$bucket" ] || [ -z "$cache_dir" ]; then
    entries="$(canon_jax_cache_file_count "$cache_dir")"
    canon_jax_cache_receipt \
      "$phase" disabled "$tool" 0 "$entries" "$profile" \
      "${bucket:-unset}" "${cache_dir:-unset}"
    return 0
  fi
  case "$phase" in
    restore|save) ;;
    *)
      canon_jax_cache_receipt \
        "$phase" invalid-phase "$tool" 2 0 "$profile" "$bucket" "$cache_dir"
      return 0
      ;;
  esac
  case "$bucket:$cache_dir" in
    gs://*:/?*) ;;
    *)
      entries="$(canon_jax_cache_file_count "$cache_dir")"
      canon_jax_cache_receipt \
        "$phase" invalid-contract "$tool" 2 "$entries" "$profile" \
        "$bucket" "$cache_dir"
      return 0
      ;;
  esac

  mkdir -p "$cache_dir"
  gcs_path="${bucket%/}/$profile"
  if command -v gcloud >/dev/null 2>&1; then
    tool="gcloud"
    if [ "$phase" = restore ]; then
      gcloud storage rsync -r "$gcs_path" "$cache_dir" || rc=$?
    else
      gcloud storage rsync -r "$cache_dir" "$gcs_path" || rc=$?
    fi
  elif command -v gsutil >/dev/null 2>&1; then
    tool="gsutil"
    if [ "$phase" = restore ]; then
      gsutil -m rsync -r "$gcs_path" "$cache_dir" || rc=$?
    else
      gsutil -m rsync -r "$cache_dir" "$gcs_path" || rc=$?
    fi
  else
    rc=127
  fi

  entries="$(canon_jax_cache_file_count "$cache_dir")"
  if [ "$rc" -ne 0 ]; then
    if [ "$rc" -eq 127 ]; then
      status="no-tool"
    else
      status="error"
    fi
  elif [ "$entries" -eq 0 ]; then
    status="empty"
  elif [ "$phase" = restore ]; then
    status="hit"
  else
    status="saved"
  fi
  canon_jax_cache_receipt \
    "$phase" "$status" "$tool" "$rc" "$entries" "$profile" \
    "$gcs_path" "$cache_dir"
  # Cache synchronization is a performance aid, never a Zero-TIM numerical
  # gate. The explicit receipt preserves misses/errors for post-run judgment.
  return 0
}
