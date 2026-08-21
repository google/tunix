#!/usr/bin/env bash
# Prove that P57 stock-fast still uses the pinned image's untouched engine.
#
# Step 20 proves the image starts from the signed stock files. This step runs
# after the P57 installation branch and proves that no later step replaced any
# of the six engine targets. The import probe uses CPU in a separate process,
# so it catches import-time shim dependencies without compiling a TPU program.
set -euo pipefail
source "$CANON_STATE/env.sh"
# shellcheck disable=SC1091
source "$CANON_PKG/cluster/steps/p57_runtime_contract.sh"

if ! p57_is_stock_fast_calibration; then
  echo "[P57.STOCK_FAST] FATAL: stock verification used outside exact calibration tuple" >&2
  exit 2
fi

SP="$(cat "$CANON_STATE/tpu_inference_path")"
MANIFEST="$CANON_PKG/STOCK_MANIFEST.sha256"
[ -d "$SP" ] || { echo "[P57.STOCK_FAST] FATAL: missing stock package: $SP" >&2; exit 1; }
[ -f "$MANIFEST" ] || { echo "[P57.STOCK_FAST] FATAL: missing manifest: $MANIFEST" >&2; exit 1; }

checked=0
while read -r expected relative; do
  target="$SP/$relative"
  [ -f "$target" ] || {
    echo "[P57.STOCK_FAST] FATAL: stock target missing: $relative" >&2
    exit 1
  }
  actual="$(sha256sum "$target" | cut -d' ' -f1)"
  [ "$actual" = "$expected" ] || {
    echo "[P57.STOCK_FAST] FATAL: stock target changed: $relative" >&2
    exit 1
  }
  checked=$((checked + 1))
done < "$MANIFEST"
[ "$checked" -eq 6 ] || {
  echo "[P57.STOCK_FAST] FATAL: expected six stock manifest entries; got $checked" >&2
  exit 1
}

env -u CANON_SHIM_ROOT -u CANON_FIXED_AR -u CANON_FIXED_AR_EMBED \
  -u CANON_PALLAS_MPAD -u CANON_PALLAS_ALL_PROJ \
  PATHWAYS_HEAD="" JAX_BACKEND_TARGET="" JAX_PLATFORMS=cpu \
  PYTHONPATH="$CANON_PKG/..:${PYTHONPATH:-}" \
  python3 "$CANON_PKG/cluster/steps/p57_probe_stock_engine.py"

(
  cd "$CANON_PKG/.."
  # argparse's --help exits only after all module imports succeed, but before
  # the P57 top-level run-kind/CLI agreement checks.  Abseil's --helpshort is
  # not recognized by this argparse entrypoint and must not be used here.
  env -u PYTHONPATH PATHWAYS_HEAD="" JAX_BACKEND_TARGET="" JAX_PLATFORMS=cpu \
    python3 -u -m examples.frozenlake.train_frozenlake_qwen3 --help \
    > /dev/null
)
echo "[P57.STOCK_FAST] WORKLOAD_IMPORT_PASS entrypoint=module"

echo "[P57.STOCK_FAST] PREFLIGHT_PASS files=$checked import=pass overlay=absent"
