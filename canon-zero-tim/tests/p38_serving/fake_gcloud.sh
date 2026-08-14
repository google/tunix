#!/usr/bin/env bash
set -euo pipefail

: "${FAKE_GCS_ROOT:?FAKE_GCS_ROOT unset}"
[ "${1:-}" = storage ] || exit 2
operation="${2:-}"
shift 2

object_path() {
  local uri="$1"
  case "$uri" in
    gs://*) printf '%s/%s\n' "$FAKE_GCS_ROOT" "${uri#gs://}" ;;
    *) printf '%s\n' "$uri" ;;
  esac
}

case "$operation" in
  cp)
    [ "$#" -eq 2 ] || exit 2
    [ "${FAKE_GCS_FAIL_CP:-0}" = 0 ] || exit 9
    source="$(object_path "$1")"
    target="$(object_path "$2")"
    mkdir -p "$(dirname "$target")"
    cp -- "$source" "$target"
    ;;
  ls)
    [ "$#" -eq 1 ] || exit 2
    test -e "$(object_path "$1")"
    ;;
  *) exit 2 ;;
esac
