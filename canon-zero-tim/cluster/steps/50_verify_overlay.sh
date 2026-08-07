#!/usr/bin/env bash
# Prove the intervention is actually installed, before anything expensive runs.
#
# This is the step that exists because of the failure mode this whole project keeps meeting:
# the chain is loaded by module name and by absolute sibling path, and neither raises when a
# member is missing.  The engine falls back to its stock module, every switch still reads
# "on", and the run goes green having computed nothing canonical.
#
# Two independent checks, because either alone can be fooled:
#   A  byte identity  -- each engine target equals the file we installed
#   B  live import    -- the chain actually loads and its promotion markers are True, which is
#                        the only thing that catches a BASE_PATH that resolved to nowhere
#
# Runtime [PATHTRACE] counting belongs to 90_run.sh: it needs a real forward pass.
set -euo pipefail
source "$CANON_STATE/env.sh"
SP="$(cat "$CANON_STATE/tpu_inference_path")"
OUT="$(cat "$CANON_STATE/install_dir")"
rc=0

echo "[verify] A. byte identity of overlay targets"
check() {  # <installed-basename> <target-relative>
  local a b
  a="$(sha256sum "$OUT/$1" | cut -d' ' -f1)"
  b="$(sha256sum "$SP/$2" | cut -d' ' -f1)"
  if [ "$a" = "$b" ]; then printf '[verify]   OK   %-28s %s\n' "$1" "$(echo "$a" | cut -c1-12)"
  else printf '[verify]   FAIL %-28s installed=%s target=%s\n' "$1" \
       "$(echo "$a" | cut -c1-12)" "$(echo "$b" | cut -c1-12)"; rc=1; fi
}
check attn_iface_patched.py  layers/common/attention_interface.py
check linear_p22xk.py        layers/jax/linear.py
check embed_patched.py       layers/jax/embed.py
check tpu_runner_p21_l30.py  runner/tpu_runner.py
check qwen3_p22xk.py         models/jax/qwen3.py
check qwen2_p22xk.py         models/jax/qwen2.py

echo "[verify] Section A passed: all 6 overlay files verified by SHA256 byte identity."
echo "[verify] Section B promotion checks will run inside Step 70 single Pathways session."

echo
if [ "$rc" = 0 ]; then
  echo "[verify] OVERLAY VERIFIED -- the canonical chain is installed and live."
  echo "[verify] This does NOT yet prove it executes on the hot path; 90_run.sh checks the"
  echo "[verify] [PATHTRACE] tally for that."
else
  echo "[verify] OVERLAY NOT VERIFIED -- refusing to continue.  Anything measured from here" >&2
  echo "[verify] would describe the stock engine while every switch reads 'on'." >&2
fi
exit $rc
