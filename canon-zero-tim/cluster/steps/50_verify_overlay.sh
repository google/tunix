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

echo "[verify] B. live import of the promoted chain"
PATHWAYS_HEAD="" JAX_BACKEND_TARGET="" JAX_PLATFORMS=cpu python3 - <<'PY' || rc=1
import importlib, sys

# (module, attribute, expectation)  -- expectation None means "attribute must exist"
CHECKS = [
    ("tpu_inference.layers.jax.linear", "P22XK_MATMUL_ACTIVE", True),
    ("tpu_inference.layers.jax.linear", "P22XK_LINEAR_BASE",   None),
    ("tpu_inference.layers.jax.embed",  "_CANON_F4E_ANNOUNCED", None),
    ("tpu_inference.models.jax.qwen3",  "P22XK_RMSNORM_ACTIVE", True),
    ("tpu_inference.models.jax.qwen2",  "P22XK_SWIGLU_ACTIVE",  True),
]
bad = 0
for mod, attr, want in CHECKS:
    try:
        m = importlib.import_module(mod)
    except Exception as exc:
        print(f"[verify]   FAIL import {mod}: {exc!r}")
        bad += 1
        continue
    if not hasattr(m, attr):
        print(f"[verify]   FAIL {mod}.{attr} absent -- the stock module is in place, not ours")
        bad += 1
        continue
    got = getattr(m, attr)
    if want is not None and got is not want and got != want:
        print(f"[verify]   FAIL {mod}.{attr}={got!r}, expected {want!r} -- the chain loaded but "
              f"its promotion did not take effect")
        bad += 1
        continue
    print(f"[verify]   OK   {mod}.{attr}"
          + (f"={got!r}" if want is not None else ""))
sys.exit(1 if bad else 0)
PY

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
