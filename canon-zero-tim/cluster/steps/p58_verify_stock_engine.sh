#!/usr/bin/env bash
# Verify that the P58 native arm retained the digest-pinned image engine before
# its independent observer-only B overlay is installed.
set -euo pipefail
source "$CANON_STATE/env.sh"

if [ "${CANON_PROFILE_FILE:-}" != \
     "cluster/profiles/qwen3-4b-dp8-tp8-deepswe-tim.env" ] || \
   [ "${CANON_P58_DEEPSWE_TIM:-}" != "1" ] || \
   [ "${CANON_P58_TIM_ARM:-}" != "native" ]; then
  echo "[P58.NATIVE] FATAL: stock verification used outside the native arm" >&2
  exit 2
fi

SP="$(cat "$CANON_STATE/tpu_inference_path")"
MANIFEST="$CANON_PKG/STOCK_MANIFEST.sha256"
[ -d "$SP" ] || { echo "[P58.NATIVE] FATAL: missing stock package: $SP" >&2; exit 1; }
[ -f "$MANIFEST" ] || { echo "[P58.NATIVE] FATAL: missing stock manifest" >&2; exit 1; }

checked=0
while read -r expected relative; do
  target="$SP/$relative"
  [ -f "$target" ] || {
    echo "[P58.NATIVE] FATAL: stock target missing: $relative" >&2
    exit 1
  }
  actual="$(sha256sum "$target" | cut -d' ' -f1)"
  [ "$actual" = "$expected" ] || {
    echo "[P58.NATIVE] FATAL: stock target changed: $relative" >&2
    exit 1
  }
  checked=$((checked + 1))
done < "$MANIFEST"
[ "$checked" -eq 6 ] || {
  echo "[P58.NATIVE] FATAL: expected six stock engine files; got $checked" >&2
  exit 1
}

env -u CANON_SHIM_ROOT -u CANON_FIXED_AR -u CANON_FIXED_AR_EMBED \
  -u CANON_PALLAS_MPAD -u CANON_PALLAS_ALL_PROJ \
  PATHWAYS_HEAD="" JAX_BACKEND_TARGET="" JAX_PLATFORMS=cpu \
  PYTHONPATH="$CANON_PKG/..:${PYTHONPATH:-}" \
  python3 "$CANON_PKG/cluster/steps/p57_probe_stock_engine.py"

(
  env -u PYTHONPATH PATHWAYS_HEAD="" JAX_BACKEND_TARGET="" \
    JAX_PLATFORMS=cpu \
    python3 -u "$CANON_PKG/../examples/deepswe/canonical_entrypoint.py" \
      --help > /dev/null
)
echo "[P58.NATIVE] STOCK_PREFLIGHT_PASS files=$checked direct_entrypoint=pass overlay=absent"
