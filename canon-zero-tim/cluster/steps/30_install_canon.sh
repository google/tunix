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
python3 -c "import gymnasium" 2>/dev/null || pip install -q gymnasium
python3 -c "import sentencepiece, tiktoken" 2>/dev/null || \
  python3 -m pip install -q 'sentencepiece==0.2.2' 'tiktoken==0.13.0'
python3 -c "import gymnasium, sentencepiece, tiktoken"
echo "[install] installed to $OUT ($(find "$OUT" -name '*.py' | wc -l) files)"
