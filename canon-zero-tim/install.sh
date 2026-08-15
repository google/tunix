#!/usr/bin/env bash
# Assemble the canonical engine chain into a self-contained, mountable directory.
#
#   ./install.sh <output-dir> [--from-image <image> | --from-path <tpu_inference-dir>] [--model M]
#
#   --from-image   development host: pull the six stock files out of a pinned image via docker
#   --from-path    inside a container: read them straight off the filesystem (no docker, no
#                  network, no image on disk).  This is the mode a GKE pod uses.
#   --model        qwen1p7b (default) | qwen4b | qwen8b | qwen8b_tp8 | qwen32b
#                  -- selects model modules
#
# Steps: extract stock -> apply patches/tpu_inference/*.patch -> lay down the shim chain ->
# verify every produced file against MANIFEST.sha256.
#
# The result is one flat directory.  Every shim resolves its next chain layer as a sibling of
# itself, so the directory can live anywhere; no deployment path is baked in.
#
# A SHA mismatch is fatal on purpose.  The chain is loaded by absolute-path import, and a
# missing or stale member does not raise -- the engine quietly falls back to the stock module
# and the run goes green having never exercised the intervention.
set -euo pipefail

PKG="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT=""
MODE=""
SRC_IMAGE="${CANON_IMAGE:-tunix_frozenlake_image:vllm-tpu0.25.0}"
SRC_PATH=""
MODEL="qwen1p7b"
DOCKER="${DOCKER:-sudo docker}"
SP_DEFAULT=/usr/local/lib/python3.12/site-packages/tpu_inference

usage() {
  sed -n '2,18p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' >&2
  exit "${1:-1}"
}

while [ $# -gt 0 ]; do
  case "$1" in
    --from-image) MODE=image; SRC_IMAGE="$2"; shift 2;;
    --from-path)  MODE=path;  SRC_PATH="$2";  shift 2;;
    --model)      MODEL="$2"; shift 2;;
    -h|--help)    usage 0;;
    -*)           echo "unknown option: $1" >&2; usage;;
    *)            [ -z "$OUT" ] && OUT="$1" || { echo "unexpected argument: $1" >&2; usage; }; shift;;
  esac
done
[ -n "$OUT" ] || usage
[ -n "$MODE" ] || MODE=image     # backwards compatible default

[ -d "$PKG/src/engine_shims/models/$MODEL" ] || {
  echo "unknown model '$MODEL'; available: $(ls "$PKG/src/engine_shims/models")" >&2; exit 1; }

FILES="layers/common/attention_interface.py layers/jax/embed.py layers/jax/linear.py
       runner/tpu_runner.py models/jax/qwen3.py models/jax/qwen2.py"

mkdir -p "$OUT"
STOCK="$(mktemp -d)"
trap 'rm -rf "$STOCK"' EXIT

echo "[1/4] extracting stock tpu_inference (mode=$MODE)"
case "$MODE" in
  image)
    echo "      image=$SRC_IMAGE"
    CID="$($DOCKER create "$SRC_IMAGE" /bin/true)"
    for f in $FILES; do
      mkdir -p "$STOCK/$(dirname "$f")"
      $DOCKER cp "$CID:$SP_DEFAULT/$f" "$STOCK/$f" >/dev/null
    done
    $DOCKER rm "$CID" >/dev/null
    sudo chown -R "$(id -u):$(id -g)" "$STOCK" 2>/dev/null || true
    ;;
  path)
    [ -d "$SRC_PATH" ] || { echo "--from-path: not a directory: $SRC_PATH" >&2; exit 1; }
    echo "      path=$SRC_PATH"
    for f in $FILES; do
      [ -f "$SRC_PATH/$f" ] || { echo "      missing stock file: $SRC_PATH/$f" >&2; exit 1; }
      mkdir -p "$STOCK/$(dirname "$f")"
      cp "$SRC_PATH/$f" "$STOCK/$f"
    done
    ;;
esac

echo "[2/4] applying patches"
apply() {  # <patch> <stock-relative> <output-basename>
  cp "$STOCK/$2" "$OUT/$3"
  if ! patch -s --no-backup-if-mismatch "$OUT/$3" "$PKG/patches/tpu_inference/$1"; then
    echo "      PATCH FAILED: $1 does not apply to this stock $2." >&2
    echo "      The patches are anchored to a specific tpu_inference version; this source is a" >&2
    echo "      different one.  Regenerate the patches against it (and re-run every gate), or" >&2
    echo "      pin the source to the anchored image.  Do not force it." >&2
    exit 1
  fi
}
apply 01-attention-interface.patch layers/common/attention_interface.py attn_iface_patched.py
apply 02-embed.patch               layers/jax/embed.py                  embed_patched.py
apply 03-linear.patch              layers/jax/linear.py                 linear_patched.py
apply 04-qwen3.patch               models/jax/qwen3.py                  qwen3.py
apply 05-qwen2.patch               models/jax/qwen2.py                  qwen2_patched.py
apply 06-tpu-runner.patch          runner/tpu_runner.py                 tpu_runner_p21_l30.py
patch -s --no-backup-if-mismatch "$OUT/tpu_runner_p21_l30.py" \
  "$PKG/patches/tpu_inference/07-tpu-runner-p35-metadata.patch" || {
  echo "      PATCH FAILED: 07-tpu-runner-p35-metadata.patch" >&2
  exit 1
}
patch -s --no-backup-if-mismatch "$OUT/attn_iface_patched.py" \
  "$PKG/patches/tpu_inference/08-attention-kv-unified.patch" || {
  echo "      PATCH FAILED: 08-attention-kv-unified.patch" >&2
  exit 1
}
patch -s --no-backup-if-mismatch "$OUT/tpu_runner_p21_l30.py" \
  "$PKG/patches/tpu_inference/09-tpu-runner-p38-serving-capture.patch" || {
  echo "      PATCH FAILED: 09-tpu-runner-p38-serving-capture.patch" >&2
  exit 1
}
patch -s --no-backup-if-mismatch "$OUT/tpu_runner_p21_l30.py" \
  "$PKG/patches/tpu_inference/10-tpu-runner-p38-standard-capture.patch" || {
  echo "      PATCH FAILED: 10-tpu-runner-p38-standard-capture.patch" >&2
  exit 1
}
patch -s --no-backup-if-mismatch "$OUT/tpu_runner_p21_l30.py" \
  "$PKG/patches/tpu_inference/11-tpu-runner-p38-capture-hardening.patch" || {
  echo "      PATCH FAILED: 11-tpu-runner-p38-capture-hardening.patch" >&2
  exit 1
}
patch -s --no-backup-if-mismatch "$OUT/tpu_runner_p21_l30.py" \
  "$PKG/patches/tpu_inference/12-tpu-runner-p38-prng-key-capture.patch" || {
  echo "      PATCH FAILED: 12-tpu-runner-p38-prng-key-capture.patch" >&2
  exit 1
}
patch -s --no-backup-if-mismatch "$OUT/tpu_runner_p21_l30.py" \
  "$PKG/patches/tpu_inference/13-tpu-runner-p38-request-journal.patch" || {
  echo "      PATCH FAILED: 13-tpu-runner-p38-request-journal.patch" >&2
  exit 1
}
patch -s --no-backup-if-mismatch "$OUT/tpu_runner_p21_l30.py" \
  "$PKG/patches/tpu_inference/14-tpu-runner-p38-incident-ledger.patch" || {
  echo "      PATCH FAILED: 14-tpu-runner-p38-incident-ledger.patch" >&2
  exit 1
}
patch -s --no-backup-if-mismatch "$OUT/tpu_runner_p21_l30.py" \
  "$PKG/patches/tpu_inference/15-tpu-runner-p38-fixed-m-incident.patch" || {
  echo "      PATCH FAILED: 15-tpu-runner-p38-fixed-m-incident.patch" >&2
  exit 1
}
patch -s --no-backup-if-mismatch "$OUT/tpu_runner_p21_l30.py" \
  "$PKG/patches/tpu_inference/16-tpu-runner-p38-kv-observer.patch" || {
  echo "      PATCH FAILED: 16-tpu-runner-p38-kv-observer.patch" >&2
  exit 1
}

echo "[3/4] laying down the shim chain (model=$MODEL)"
cp "$PKG"/src/engine_shims/*.py "$OUT/"
cp "$PKG/src/engine_shims/models/$MODEL"/*.py "$OUT/"

echo "[4/4] verifying against MANIFEST.sha256"
# Two manifests, because two of the installed files are model-specific.  A single combined
# manifest would pin one model's hashes and reject every other model as a mismatch -- a
# false red that looks exactly like the real failure it is meant to catch.
MODEL_MAN="$PKG/src/engine_shims/models/$MODEL/MANIFEST.sha256"
if [ -f "$PKG/MANIFEST.sha256" ] && [ -f "$MODEL_MAN" ]; then
  if ( cd "$OUT" && sha256sum -c "$PKG/MANIFEST.sha256" --quiet \
       && sha256sum -c "$MODEL_MAN" --quiet ); then
    echo "      all $(( $(wc -l < "$PKG/MANIFEST.sha256") + $(wc -l < "$MODEL_MAN") )) files match ($MODEL)"
  else
    echo "      MANIFEST MISMATCH -- refusing to report success." >&2
    echo "      A chain member that differs from the recorded one is not a warning: the chain" >&2
    echo "      is loaded by path, a stale member does not raise, and the run would go green" >&2
    echo "      without the intervention ever executing." >&2
    exit 1
  fi
else
  echo "      manifest missing (need MANIFEST.sha256 and $MODEL_MAN) -- cannot verify" >&2
  exit 1
fi

cat <<EOF

Installed to: $OUT   (model=$MODEL)

Container mounts -- target paths are under the engine's tpu_inference package
(\$SP, ${SP_DEFAULT} on the anchored image):

  \$OUT/attn_iface_patched.py   -> \$SP/layers/common/attention_interface.py
  \$OUT/linear_p22xk.py         -> \$SP/layers/jax/linear.py
  \$OUT/embed_patched.py        -> \$SP/layers/jax/embed.py
  \$OUT/tpu_runner_p21_l30.py   -> \$SP/runner/tpu_runner.py
  \$OUT/qwen3_p22xk.py          -> \$SP/models/jax/qwen3.py
  \$OUT/qwen2_p22xk.py          -> \$SP/models/jax/qwen2.py

  PYTHONPATH=$OUT:\$PYTHONPATH
  CANON_SHIM_ROOT=$OUT          # optional; defaults to each shim's own directory

Under Kubernetes there are no bind mounts: copy the six files over the target paths instead
(cluster/steps/40_overlay_engine.sh does exactly that).

The canonical switch set is in README.md.  A launch missing any of it is not a canonical run,
and the only proof that the intervention took is the [PATHTRACE] lines -- never the exit code.
EOF
