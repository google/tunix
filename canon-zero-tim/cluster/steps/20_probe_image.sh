#!/usr/bin/env bash
# Is this image's tpu_inference the one the patches were cut against?
#
# The ordered patch set in patches/tpu_inference is anchored to one specific build. Against a
# different build they either fail to apply -- loud, fine -- or apply with fuzz and produce a
# file nobody has ever gated.  This step answers the question up front, per file, before the
# install runs, so a version difference is a decision instead of a mystery.
#
# Read-only.  Locates the package, hashes six files, compares with STOCK_MANIFEST.sha256.
set -euo pipefail
source "$CANON_STATE/env.sh"

find_tpu_inference() {
  if [ -n "${CANON_TPU_INFERENCE_PATH:-}" ]; then echo "$CANON_TPU_INFERENCE_PATH"; return; fi
  PATHWAYS_HEAD="" JAX_BACKEND_TARGET="" JAX_PLATFORMS=cpu python3 - <<'PY' 2>/dev/null || true
import importlib.util, os
spec = importlib.util.find_spec("tpu_inference")
if spec and spec.submodule_search_locations:
    print(os.path.dirname(list(spec.submodule_search_locations)[0] + "/x"))
PY
}

SP="$(find_tpu_inference)"
[ -n "$SP" ] && [ -d "$SP" ] || {
  echo "[probe] cannot locate the tpu_inference package." >&2
  echo "[probe] Set CANON_TPU_INFERENCE_PATH explicitly.  Known layouts:" >&2
  echo "[probe]   /usr/local/lib/python3.12/site-packages/tpu_inference          (probe host image)" >&2
  echo "[probe]   /app/vllm_tpu_inference/tpu_inference/tpu_inference            (cluster base image)" >&2
  exit 1; }
echo "[probe] tpu_inference=$SP"
echo "$SP" > "$CANON_STATE/tpu_inference_path"

MAN="$CANON_PKG/STOCK_MANIFEST.sha256"
[ -f "$MAN" ] || { echo "[probe] missing $MAN" >&2; exit 1; }

same=0; drift=0; missing=0
while read -r want rel; do
  f="$SP/$rel"
  if [ ! -f "$f" ]; then
    printf '[probe] %-9s %s\n' "MISSING" "$rel"; missing=$((missing+1)); continue
  fi
  got="$(sha256sum "$f" | cut -d' ' -f1)"
  if [ "$got" = "$want" ]; then
    printf '[probe] %-9s %s\n' "SAME" "$rel"; same=$((same+1))
  else
    printf '[probe] %-9s %s  (anchor %s.. / here %s..)\n' "DRIFT" "$rel" \
      "$(echo "$want" | cut -c1-12)" "$(echo "$got" | cut -c1-12)"
    drift=$((drift+1))
  fi
done < "$MAN"

echo "[probe] SUMMARY same=$same drift=$drift missing=$missing"
{ echo "same=$same"; echo "drift=$drift"; echo "missing=$missing"; } > "$CANON_STATE/image_probe"

if [ "$missing" -gt 0 ]; then
  echo "[probe] REFUSING: a stock file is absent -- this is not the expected package layout." >&2
  exit 1
fi
if [ "$drift" -gt 0 ]; then
  echo "[probe] VERSION DRIFT: $drift of 6 stock files differ from the anchor."
  echo "[probe] The install will attempt the patches anyway and MANIFEST verification will"
  echo "[probe] catch a wrong result, but understand what a pass would mean here: the produced"
  echo "[probe] files can no longer be byte-identical to the ones that carry the signed"
  echo "[probe] evidence.  Treat any bitwise claim on this image as UNVERIFIED until it is"
  echo "[probe] re-measured here."
  if [ "${CANON_ALLOW_IMAGE_DRIFT:-0}" != "1" ]; then
    echo "[probe] REFUSING (set CANON_ALLOW_IMAGE_DRIFT=1 to proceed deliberately, and record" >&2
    echo "[probe]           that override in the report)." >&2
    exit 1
  fi
  echo "[probe] CANON_ALLOW_IMAGE_DRIFT=1 -- proceeding with drift recorded"
else
  echo "[probe] image matches the patch anchor exactly"
fi
