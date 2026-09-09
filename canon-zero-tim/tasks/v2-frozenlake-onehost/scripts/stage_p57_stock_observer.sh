#!/usr/bin/env bash
# Stage the P57 stock-observer runner for a one-host stock-engine arm.
# usage: stage_p57_stock_observer.sh <out-dir> --from-image <image>
# Mirrors cluster/steps/39_install_p57_stock_observer.sh on a development
# host: pull the stock runner/tpu_runner.py out of the pinned image, prove it
# is stock (STOCK_MANIFEST.sha256), apply patches/p57_stock_observer/
# 01-tpu-runner.patch exactly, add the helper, and prove both staged files
# against P57_STOCK_OBSERVER_MANIFEST.sha256.  The launcher bind-mounts the
# two files over the stock runner; the observer branch is reachable only
# with CANON_PROMPT_PROCESSED_LOGPROBS=1 (tasks/zero_tim_perf phase3).
set -euo pipefail
OUT="${1:?usage: stage_p57_stock_observer.sh <out-dir> --from-image <image>}"
[ "${2:-}" = "--from-image" ] || { echo "usage: stage_p57_stock_observer.sh <out-dir> --from-image <image>" >&2; exit 2; }
IMAGE="${3:?image}"
DOCKER="${DOCKER:-sudo docker}"
PKG="$(cd "$(dirname "$0")/../../.." && pwd)"
SP=/usr/local/lib/python3.12/site-packages/tpu_inference
PATCH="$PKG/patches/p57_stock_observer/01-tpu-runner.patch"
MANIFEST="$PKG/P57_STOCK_OBSERVER_MANIFEST.sha256"
STOCK_MANIFEST="$PKG/STOCK_MANIFEST.sha256"
for required in "$PATCH" "$MANIFEST" "$STOCK_MANIFEST" "$PKG/src/p57_stock_prompt_observer.py"; do
  [ -f "$required" ] || { echo "[P57.STOCK_OBSERVER] FATAL: missing $required" >&2; exit 1; }
done
mkdir -p "$OUT"
stage="$(mktemp -d)"
trap 'rm -rf "$stage"' EXIT
CID="$($DOCKER create "$IMAGE" /bin/true)"
$DOCKER cp "$CID:$SP/runner/tpu_runner.py" "$stage/tpu_runner.py" >/dev/null
$DOCKER rm "$CID" >/dev/null
sudo chown "$(id -u):$(id -g)" "$stage/tpu_runner.py" 2>/dev/null || true
stock_expected="$(awk '$2 == "runner/tpu_runner.py" {print $1}' "$STOCK_MANIFEST")"
stock_actual="$(sha256sum "$stage/tpu_runner.py" | cut -d' ' -f1)"
[ -n "$stock_expected" ] && [ "$stock_actual" = "$stock_expected" ] || {
  echo "[P57.STOCK_OBSERVER] FATAL: image runner is not the pinned stock file ($stock_actual)" >&2
  exit 1
}
patch -s --fuzz=0 --no-backup-if-mismatch "$stage/tpu_runner.py" "$PATCH" || {
  echo "[P57.STOCK_OBSERVER] FATAL: observer patch did not apply exactly" >&2
  exit 1
}
cp "$PKG/src/p57_stock_prompt_observer.py" "$stage/p57_stock_prompt_observer.py"
python3 -m py_compile "$stage/tpu_runner.py" "$stage/p57_stock_prompt_observer.py"
while read -r expected relative; do
  case "$relative" in
    runner/tpu_runner.py) candidate="$stage/tpu_runner.py" ;;
    runner/p57_stock_prompt_observer.py) candidate="$stage/p57_stock_prompt_observer.py" ;;
    *) echo "[P57.STOCK_OBSERVER] FATAL: unexpected manifest path $relative" >&2; exit 1 ;;
  esac
  actual="$(sha256sum "$candidate" | cut -d' ' -f1)"
  [ "$actual" = "$expected" ] || { echo "[P57.STOCK_OBSERVER] FATAL: staged hash mismatch for $relative" >&2; exit 1; }
done < "$MANIFEST"
install -m 0644 "$stage/tpu_runner.py" "$OUT/tpu_runner.py"
install -m 0644 "$stage/p57_stock_prompt_observer.py" "$OUT/p57_stock_prompt_observer.py"
rm -rf "$OUT/__pycache__"
echo "[P57.STOCK_OBSERVER] STAGED out=$OUT files=2 stock_runner_verified=1 treatment=observer-only"
