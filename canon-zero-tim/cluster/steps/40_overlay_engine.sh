#!/usr/bin/env bash
# Put the canonical files at the paths the engine imports.
#
# Kubernetes has no bind mounts, so the docker `-v file:target:ro` form used on the probe host
# becomes a copy.  Originals are saved next to the targets first, so the overlay is reversible
# without reinstalling the image.
set -euo pipefail
source "$CANON_STATE/env.sh"
SP="$(cat "$CANON_STATE/tpu_inference_path")"
OUT="$(cat "$CANON_STATE/install_dir")"
BACKUP="${CANON_OVERLAY_BACKUP:-$CANON_STATE/stock_backup}"
mkdir -p "$BACKUP"

overlay() {  # <source-basename> <target-relative>
  local src="$OUT/$1" dst="$SP/$2"
  [ -f "$src" ] || { echo "[overlay] missing installed file: $src" >&2; exit 1; }
  [ -f "$dst" ] || { echo "[overlay] missing engine target: $dst" >&2; exit 1; }
  mkdir -p "$BACKUP/$(dirname "$2")"
  [ -f "$BACKUP/$2" ] || cp "$dst" "$BACKUP/$2"
  cp "$src" "$dst"
  printf '[overlay] %-28s -> %s\n' "$1" "$2"
}

overlay attn_iface_patched.py  layers/common/attention_interface.py
overlay linear_p22xk.py        layers/jax/linear.py
overlay embed_patched.py       layers/jax/embed.py
overlay tpu_runner_p21_l30.py  runner/tpu_runner.py
overlay qwen3_p22xk.py         models/jax/qwen3.py
overlay qwen2_p22xk.py         models/jax/qwen2.py

# The chain members are imported by module name, so the install dir must be importable, and
# by absolute sibling path, so CANON_SHIM_ROOT must point at it.
export PYTHONPATH="$OUT:${PYTHONPATH:-}"
export CANON_SHIM_ROOT="$OUT"
printf 'export PYTHONPATH=%q\nexport CANON_SHIM_ROOT=%q\n' "$PYTHONPATH" "$OUT" >> "$CANON_STATE/env.sh"
echo "[overlay] PYTHONPATH and CANON_SHIM_ROOT recorded; originals in $BACKUP"
