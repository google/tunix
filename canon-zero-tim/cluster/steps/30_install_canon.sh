#!/usr/bin/env bash
# Build the canonical chain from patches + shims, in-container.
#
# --from-path reads the six stock files straight off this filesystem: no docker, no image on
# disk, no network.  That is what makes the package work in a pod, and on whatever machine you
# move to next.
set -euo pipefail
source "$CANON_STATE/env.sh"
SP="$(cat "$CANON_STATE/tpu_inference_path")"
OUT="${CANON_INSTALL_DIR:-$CANON_STATE/canon}"
rm -rf "$OUT"
PATHWAYS_HEAD="" JAX_BACKEND_TARGET="" JAX_PLATFORMS=cpu bash "$CANON_PKG/install.sh" "$OUT" --from-path "$SP" --model "$CANON_MODEL_DIR_NAME"
echo "$OUT" > "$CANON_STATE/install_dir"
if ! python3 -c "import gymnasium, sentencepiece, tiktoken" 2>/dev/null; then
  python3 -m pip install --break-system-packages --no-deps -q \
    'gymnasium==1.3.0' 'sentencepiece==0.2.2' 'tiktoken==0.13.0'
fi
python3 -c "import gymnasium, numba, numpy, sentencepiece, tiktoken; print(f'[install] runtime deps OK numpy={numpy.__version__} numba={numba.__version__}')"
for runtime_package in numpy numba gymnasium sentencepiece tiktoken; do
  [ ! -e "$OUT/$runtime_package" ] || {
    echo "[install] runtime package leaked into canonical overlay: $runtime_package" >&2
    exit 1
  }
done
echo "[install] installed to $OUT ($(find "$OUT" -name '*.py' | wc -l) files)"
