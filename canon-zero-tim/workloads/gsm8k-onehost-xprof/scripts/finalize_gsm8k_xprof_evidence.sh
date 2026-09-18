#!/usr/bin/env bash
# Finalize one-host GSM8K XProf evidence without mutating a hashed file after
# the manifest is written. This file is sourced by the common runner.

gsm8k_xprof_choose_terminal() {
  local arm="${1:?arm is required}"
  local root="${2:?root is required}"
  local docker_rc="${3:?docker rc is required}"
  local classifier_rc="${4:?classifier rc is required}"

  case "$arm" in
    native|zero-hp) ;;
    *) return 2 ;;
  esac
  case "$docker_rc" in
    *[!0-9]*) return 2 ;;
  esac
  case "$classifier_rc" in
    *[!0-9]*) return 2 ;;
  esac

  if [ "$docker_rc" -ne 0 ] || [ "$classifier_rc" -ne 0 ]; then
    GSM8K_XPROF_TERMINAL_RC=1
    GSM8K_XPROF_TERMINAL_MARKER="[V1.GSM8K.XPROF] RED arm=$arm docker_rc=$docker_rc classifier_rc=$classifier_rc root=$root"
  else
    GSM8K_XPROF_TERMINAL_RC=0
    GSM8K_XPROF_TERMINAL_MARKER="[V1.GSM8K.XPROF] GREEN arm=$arm backward_xprof=1 root=$root"
  fi
}

gsm8k_xprof_write_terminal_manifest() {
  local marker="${1:?terminal marker is required}"
  local driver="${2:?driver log is required}"
  local manifest="${3:?manifest path is required}"
  shift 3

  if [ "$#" -eq 0 ] || [ ! -f "$driver" ] || [ ! -r "$driver" ]; then
    return 2
  fi
  case "$marker" in
    *$'\n'*) return 2 ;;
  esac
  if grep -Eq '^\[V1\.GSM8K\.XPROF\] (GREEN|RED) ' "$driver"; then
    return 2
  fi

  local path
  declare -A seen_paths=()
  for path in "$@"; do
    if [ ! -f "$path" ] || [ ! -r "$path" ] || [ -n "${seen_paths[$path]:-}" ]; then
      return 2
    fi
    seen_paths["$path"]=1
  done

  # This is the last permitted write to any file included in the manifest.
  printf '%s\n' "$marker" >>"$driver" || return 2

  local manifest_tmp
  manifest_tmp="$(mktemp -- "${manifest}.tmp.XXXXXX")" || return 2
  if ! sha256sum "$@" >"$manifest_tmp"; then
    rm -f -- "$manifest_tmp"
    return 2
  fi
  if ! mv -- "$manifest_tmp" "$manifest"; then
    rm -f -- "$manifest_tmp"
    return 2
  fi
}

gsm8k_xprof_verify_manifest() {
  local manifest="${1:?manifest path is required}"
  [ -s "$manifest" ] || return 2
  sha256sum -c "$manifest"
}
