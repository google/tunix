#!/usr/bin/env bash
# Shared, side-effect-free predicates for the P57 runtime envelope.
#
# This file is sourced by entrypoint.sh and 90_run.sh.  Keep it free of `set`
# commands and top-level validation: callers own their shell policy and error
# reporting.

_P57_PROFILE="cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-tim.env"

p57_is_stock_fast_calibration() {
  [ "${CANON_PROFILE_FILE:-}" = "$_P57_PROFILE" ] && \
    [ "${CANON_P57_RUN_KIND:-}" = "calibration" ] && \
    [ "${CANON_P57_INFERENCE_REGIME:-}" = "stock-fast" ]
}

p57_is_isolated_eval() {
  [ "${CANON_PROFILE_FILE:-}" = "$_P57_PROFILE" ] && \
    [ "${CANON_P57_RUN_KIND:-}" = "eval" ]
}

p57_is_nontraining_runtime() {
  p57_is_stock_fast_calibration || p57_is_isolated_eval
}

p57_validate_stock_fast_runtime_markers() {
  if [ "$#" -ne 6 ]; then
    echo "[P57.STOCK_FAST] FATAL: expected six runtime marker counts" >&2
    return 2
  fi
  local label value
  local labels=(fixed_ar fixed_embed logprob_m fixed_lm_head fixed_lm_head_vjp kv_unified)
  local index=0
  for value in "$@"; do
    label="${labels[$index]}"
    case "$value" in
      ''|*[!0-9]*)
        echo "[P57.STOCK_FAST] FATAL: invalid $label marker count: $value" >&2
        return 2
        ;;
    esac
    if [ "$value" -ne 0 ]; then
      echo "[P57.STOCK_FAST] FATAL: canonical runtime marker leaked: $label=$value" >&2
      return 1
    fi
    index=$((index + 1))
  done
  echo "[P57.STOCK_FAST] RUNTIME_PATH_PASS canonical_markers=0 overlay=skipped"
}
