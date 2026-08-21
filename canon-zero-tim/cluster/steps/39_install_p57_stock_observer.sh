#!/usr/bin/env bash
# Install the P57 B-observer correction without exposing the canonical engine.
#
# Step 38 first proves all six signed engine files are stock.  This step then
# changes only runner/tpu_runner.py and adds one helper.  The modified branch is
# reachable solely when the explicit processed-prompt observer flag is on;
# rollout sampling and model/trainer numerics retain the pinned stock program.
set -euo pipefail
source "$CANON_STATE/env.sh"
# shellcheck disable=SC1091
source "$CANON_PKG/cluster/steps/p57_runtime_contract.sh"

if ! p57_is_stock_fast_training; then
  echo "[P57.STOCK_OBSERVER] FATAL: observer overlay used outside stock training" >&2
  exit 2
fi
if [ "${CANON_PROMPT_PROCESSED_LOGPROBS:-0}" != "1" ]; then
  echo "[P57.STOCK_OBSERVER] FATAL: processed prompt observer flag must equal 1" >&2
  exit 2
fi

SP="$(cat "$CANON_STATE/tpu_inference_path")"
RUNNER="$SP/runner/tpu_runner.py"
HELPER="$SP/runner/p57_stock_prompt_observer.py"
PATCH="$CANON_PKG/patches/p57_stock_observer/01-tpu-runner.patch"
MANIFEST="$CANON_PKG/P57_STOCK_OBSERVER_MANIFEST.sha256"
STOCK_MANIFEST="$CANON_PKG/STOCK_MANIFEST.sha256"
for required in "$RUNNER" "$PATCH" "$MANIFEST" "$STOCK_MANIFEST" \
  "$CANON_PKG/src/p57_stock_prompt_observer.py"; do
  [ -f "$required" ] || {
    echo "[P57.STOCK_OBSERVER] FATAL: missing $required" >&2
    exit 1
  }
done

stock_expected="$(awk '$2 == "runner/tpu_runner.py" {print $1}' "$STOCK_MANIFEST")"
stock_actual="$(sha256sum "$RUNNER" | cut -d' ' -f1)"
[ -n "$stock_expected" ] && [ "$stock_actual" = "$stock_expected" ] || {
  echo "[P57.STOCK_OBSERVER] FATAL: runner was not stock before observer install" >&2
  exit 1
}

stage="$(mktemp -d /tmp/p57-stock-observer.XXXXXX)"
trap 'rm -rf "$stage"' EXIT
cp "$RUNNER" "$stage/tpu_runner.py"
patch -s --fuzz=0 --no-backup-if-mismatch \
  "$stage/tpu_runner.py" "$PATCH" || {
  echo "[P57.STOCK_OBSERVER] FATAL: observer patch did not apply exactly" >&2
  exit 1
}
cp "$CANON_PKG/src/p57_stock_prompt_observer.py" \
  "$stage/p57_stock_prompt_observer.py"
python3 -m py_compile \
  "$stage/tpu_runner.py" "$stage/p57_stock_prompt_observer.py"

while read -r expected relative; do
  case "$relative" in
    runner/tpu_runner.py) candidate="$stage/tpu_runner.py" ;;
    runner/p57_stock_prompt_observer.py)
      candidate="$stage/p57_stock_prompt_observer.py" ;;
    *)
      echo "[P57.STOCK_OBSERVER] FATAL: unexpected manifest path $relative" >&2
      exit 1 ;;
  esac
  actual="$(sha256sum "$candidate" | cut -d' ' -f1)"
  [ "$actual" = "$expected" ] || {
    echo "[P57.STOCK_OBSERVER] FATAL: staged hash mismatch for $relative" >&2
    exit 1
  }
done < "$MANIFEST"

install -m 0644 "$stage/p57_stock_prompt_observer.py" "$HELPER"
install -m 0644 "$stage/tpu_runner.py" "$RUNNER"
( cd "$SP" && sha256sum -c "$MANIFEST" --quiet ) || {
  echo "[P57.STOCK_OBSERVER] FATAL: installed observer manifest mismatch" >&2
  exit 1
}
echo "[P57.STOCK_OBSERVER] OVERLAY_PASS files=2 stock_runner_verified=1 treatment=observer-only"
