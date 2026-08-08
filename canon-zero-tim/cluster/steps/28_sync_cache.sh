#!/usr/bin/env bash
# 28_sync_cache.sh: Pull precompiled JAX/XLA cache from GCS before training
set -uo pipefail
source "$CANON_STATE/env.sh"

if [ -n "${CANON_GCS_CACHE_BUCKET:-}" ]; then
  PROFILE_NAME="$(basename "${CANON_PROFILE_FILE:-default}" .env)"
  GCS_PATH="${CANON_GCS_CACHE_BUCKET}/${PROFILE_NAME}"
  LOCAL_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/tmp/jax_compilation_cache}"
  
  echo "[cache] Target GCS cache: $GCS_PATH"
  echo "[cache] Local cache dir: $LOCAL_CACHE_DIR"
  mkdir -p "$LOCAL_CACHE_DIR"
  
  if command -v gcloud >/dev/null 2>&1; then
    echo "[cache] Pulling precompiled XLA cache from $GCS_PATH..."
    gcloud storage rsync -r "$GCS_PATH" "$LOCAL_CACHE_DIR" 2>/dev/null || true
  elif command -v gsutil >/dev/null 2>&1; then
    echo "[cache] Pulling precompiled XLA cache using gsutil from $GCS_PATH..."
    gsutil -m rsync -r "$GCS_PATH" "$LOCAL_CACHE_DIR" 2>/dev/null || true
  fi
  
  CACHE_COUNT=$(find "$LOCAL_CACHE_DIR" -type f 2>/dev/null | wc -l || echo 0)
  echo "[cache] Pre-warmed cache entries: $CACHE_COUNT"
fi
