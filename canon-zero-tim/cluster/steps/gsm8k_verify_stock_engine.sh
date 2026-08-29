#!/usr/bin/env bash
# Prove the GSM8K Native arm retained the pinned image's untreated engine.
set -euo pipefail
source "$CANON_STATE/env.sh"

if [ "${CANON_PROFILE_FILE:-}" != \
     "cluster/profiles/qwen3-1p7b-dp16-tp4-gsm8k-native.env" ] || \
   [ "${CANON_GSM8K_TRAIN:-}" != "1" ] || \
   [ "${CANON_GSM8K_VANILLA:-}" != "1" ] || \
   [ -n "${CANON_P32_WORKLOAD:-}" ]; then
  echo "[GSM8K.NATIVE] FATAL: stock verification used outside the Native arm" >&2
  exit 2
fi

SP="$(cat "$CANON_STATE/tpu_inference_path")"
MANIFEST="$CANON_PKG/STOCK_MANIFEST.sha256"
[ -d "$SP" ] || {
  echo "[GSM8K.NATIVE] FATAL: missing stock package: $SP" >&2
  exit 1
}
[ -f "$MANIFEST" ] || {
  echo "[GSM8K.NATIVE] FATAL: missing stock manifest" >&2
  exit 1
}

checked=0
while read -r expected relative; do
  target="$SP/$relative"
  [ -f "$target" ] || {
    echo "[GSM8K.NATIVE] FATAL: stock target missing: $relative" >&2
    exit 1
  }
  actual="$(sha256sum "$target" | cut -d' ' -f1)"
  [ "$actual" = "$expected" ] || {
    echo "[GSM8K.NATIVE] FATAL: stock target changed: $relative" >&2
    exit 1
  }
  checked=$((checked + 1))
done < "$MANIFEST"
[ "$checked" -eq 6 ] || {
  echo "[GSM8K.NATIVE] FATAL: expected six stock engine files; got $checked" >&2
  exit 1
}

(
  cd "$CANON_PKG/.."
  env -u PYTHONPATH -u CANON_SHIM_ROOT \
    PATHWAYS_HEAD="" JAX_BACKEND_TARGET="" JAX_PLATFORMS=cpu \
    python3 -u examples/math_gsm8k/qwen3_grpo_demo.py --help > /dev/null
)
echo "[GSM8K.NATIVE] STOCK_PREFLIGHT_PASS files=$checked driver_import=pass canonical_overlay=absent alignment=off"
