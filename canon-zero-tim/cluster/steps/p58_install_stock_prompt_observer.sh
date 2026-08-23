#!/usr/bin/env bash
# Install the independent P58-native B observer without exposing the canonical
# engine. Stock verification must run first; only the runner and one helper are
# then changed under an exact output manifest.
set -euo pipefail
source "$CANON_STATE/env.sh"

production_native="$([ "${CANON_PROFILE_FILE:-}" = \
     "cluster/profiles/qwen3-4b-dp8-tp8-deepswe-tim.env" ] && \
   [ "${CANON_P34_DEEPSWE:-}" = "1" ] && \
   [ "${CANON_P58_DEEPSWE_TIM:-}" = "1" ] && \
   [ "${CANON_P58_TIM_ADMITTED:-}" = "1" ] && \
   [ "${CANON_P58_TIM_ARM:-}" = "native" ] && echo 1 || echo 0)"
onehost_native="$([ "${CANON_P58_ONEHOST_XPROF_ARM:-}" = "native" ] && \
   [ "${CANON_DEEPSWE_ONEHOST_SMOKE:-0}" = "1" ] && \
   [ "${CANON_DEEPSWE_ONEHOST_STAGE:-}" = "backward-no-commit" ] && \
   [ "${CANON_DEEPSWE_ONEHOST_NO_COMMIT:-0}" = "1" ] && \
   [ "${CANON_P58_DEEPSWE_TIM:-0}" = "0" ] && echo 1 || echo 0)"
if [ "$production_native" != "1" ] && [ "$onehost_native" != "1" ]; then
  echo "[P58.STOCK_OBSERVER] FATAL: overlay used outside signed native arm" >&2
  exit 2
fi
if [ "${CANON_P58_NATIVE_STOCK_PROMPT_OBSERVER:-0}" != "1" ]; then
  echo "[P58.STOCK_OBSERVER] FATAL: native observer flag must equal 1" >&2
  exit 2
fi
if [ "${CANON_PROMPT_PROCESSED_LOGPROBS:-}" != "0" ] || \
   [ "${CANON_ENGINE_MODULE_C:-}" != "0" ]; then
  echo "[P58.STOCK_OBSERVER] FATAL: canonical processed engine must remain off" >&2
  exit 2
fi
for forbidden in CANON_FIXED_AR CANON_FIXED_AR_EMBED CANON_LOGPROB_M; do
  if [[ -v "$forbidden" ]]; then
    echo "[P58.STOCK_OBSERVER] FATAL: canonical flag present: $forbidden" >&2
    exit 2
  fi
done

SP="$(cat "$CANON_STATE/tpu_inference_path")"
RUNNER="$SP/runner/tpu_runner.py"
HELPER="$SP/runner/p58_stock_prompt_observer.py"
PATCH="$CANON_PKG/patches/p58_stock_observer/01-tpu-runner.patch"
MANIFEST="$CANON_PKG/P58_STOCK_OBSERVER_MANIFEST.sha256"
STOCK_MANIFEST="$CANON_PKG/STOCK_MANIFEST.sha256"
for required in "$RUNNER" "$PATCH" "$MANIFEST" "$STOCK_MANIFEST" \
  "$CANON_PKG/src/p58_stock_prompt_observer.py"; do
  [ -f "$required" ] || {
    echo "[P58.STOCK_OBSERVER] FATAL: missing $required" >&2
    exit 1
  }
done

stock_expected="$(awk '$2 == "runner/tpu_runner.py" {print $1}' "$STOCK_MANIFEST")"
stock_actual="$(sha256sum "$RUNNER" | cut -d' ' -f1)"
[ -n "$stock_expected" ] && [ "$stock_actual" = "$stock_expected" ] || {
  echo "[P58.STOCK_OBSERVER] FATAL: runner was not stock before install" >&2
  exit 1
}

stage="$(mktemp -d /tmp/p58-stock-observer.XXXXXX)"
trap 'rm -rf "$stage"' EXIT
cp "$RUNNER" "$stage/tpu_runner.py"
patch -s --fuzz=0 --no-backup-if-mismatch \
  "$stage/tpu_runner.py" "$PATCH" || {
  echo "[P58.STOCK_OBSERVER] FATAL: observer patch did not apply exactly" >&2
  exit 1
}
cp "$CANON_PKG/src/p58_stock_prompt_observer.py" \
  "$stage/p58_stock_prompt_observer.py"
python3 -m py_compile \
  "$stage/tpu_runner.py" "$stage/p58_stock_prompt_observer.py"

checked=0
while read -r expected relative; do
  case "$relative" in
    runner/tpu_runner.py) candidate="$stage/tpu_runner.py" ;;
    runner/p58_stock_prompt_observer.py)
      candidate="$stage/p58_stock_prompt_observer.py" ;;
    *)
      echo "[P58.STOCK_OBSERVER] FATAL: unexpected manifest path $relative" >&2
      exit 1 ;;
  esac
  actual="$(sha256sum "$candidate" | cut -d' ' -f1)"
  [ "$actual" = "$expected" ] || {
    echo "[P58.STOCK_OBSERVER] FATAL: staged hash mismatch for $relative" >&2
    exit 1
  }
  checked=$((checked + 1))
done < "$MANIFEST"
[ "$checked" -eq 2 ] || {
  echo "[P58.STOCK_OBSERVER] FATAL: manifest must contain exactly two files" >&2
  exit 1
}

install -m 0644 "$stage/p58_stock_prompt_observer.py" "$HELPER"
install -m 0644 "$stage/tpu_runner.py" "$RUNNER"
( cd "$SP" && sha256sum -c "$MANIFEST" --quiet ) || {
  echo "[P58.STOCK_OBSERVER] FATAL: installed manifest mismatch" >&2
  exit 1
}
echo "[P58.STOCK_OBSERVER] OVERLAY_PASS files=2 stock_runner_verified=1 canonical_bundle=off treatment=observer-only onehost=$onehost_native"
