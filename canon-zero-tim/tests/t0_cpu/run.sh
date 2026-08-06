#!/usr/bin/env bash
# T0 -- pure-CPU mathematical gates for the chunked-cache differentiable RPA contract.
#
#   ./run.sh
#
# No TPU, no model, no container, no network.  Runs in well under a minute on a laptop.
#
# What it proves: the pure-JAX contract that VJP2's backward differentiates computes the
# SAME function as a full-prefill oracle, in fp64 -- value identical, gradients agreeing to
# rounding, and a finite-difference cross-check.  That is what makes VJP2 a legitimate VJP
# of the real kernel rather than a surrogate.  It does NOT touch the Mosaic kernel; kernel
# equality is a T1/T2 concern.
#
# Fail-closed: every expected measurement line must be present.  A gate that prints nothing
# is a gate that did not run -- it is never a pass.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHIMS="$HERE/../../src/engine_shims"
export PYTHONPATH="$SHIMS:$HERE:${PYTHONPATH:-}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"

# Pre-registered thresholds.  Reference values measured 2026-08-05 on CPU are quoted
# alongside.  Widening these to make a run pass is cheating; record the failure instead.
MAX_GRAD_REL=1e-12      # reference 5.039e-16
MAX_FD_REL=1e-6         # reference 1.106e-08
MAX_MSQ_REL=1e-12       # reference dq 4.492e-17 / dk 6.828e-17 / dv 5.082e-17
# value residuals must be exactly 0.000e+00 -- no tolerance

fail() { echo "  FAIL: $*" >&2; RC=1; }
RC=0

echo "== T0.0  canon_shim_root path resolution (unit) =="
OUT0="$(python3 "$HERE/test_shim_root.py" 2>&1)" || fail "shim-root unit test exited nonzero"
echo "$OUT0" | sed 's/^/  /'
echo "$OUT0" | grep -aq '^===== SHIM ROOT PASS =====$' || fail "shim-root unit test did not pass"

echo
echo "== T0.1  chunked-vs-full-prefill fp64 oracle (rpa_diff_chunked selftest) =="
OUT1="$(python3 "$SHIMS/rpa_diff_chunked.py" 2>&1)" || fail "selftest exited nonzero"
echo "$OUT1" | grep -a "^\[selftest\]" | sed 's/^/  /'

need() { echo "$2" | grep -aqE "$1" || fail "missing measurement line: $1"; }
need '^\[selftest\] value \|chain-oracle\| = ' "$OUT1"
need '^\[selftest\] grad rel = '               "$OUT1"
need '^\[selftest\] FD best rel = '            "$OUT1"
need '^\[selftest\] VERDICT: '                 "$OUT1"

VAL=$(echo "$OUT1" | sed -n 's/^\[selftest\] value |chain-oracle| = \(.*\)$/\1/p')
GRAD=$(echo "$OUT1" | sed -n 's/^\[selftest\] grad rel = \(.*\)$/\1/p')
FD=$(echo "$OUT1" | sed -n 's/^\[selftest\] FD best rel = \(.*\)$/\1/p')
[ "$VAL" = "0.000e+00" ] || fail "value residual must be exactly 0.000e+00, got $VAL"
python3 -c "import sys; sys.exit(0 if float('$GRAD') <= $MAX_GRAD_REL else 1)" || fail "grad rel $GRAD > $MAX_GRAD_REL"
python3 -c "import sys; sys.exit(0 if float('$FD')   <= $MAX_FD_REL   else 1)" || fail "FD rel $FD > $MAX_FD_REL"
echo "$OUT1" | grep -aq '^\[selftest\] VERDICT: PASS' || fail "selftest verdict is not PASS"

echo
echo "== T0.2  ragged multi-sequence VJP2 vs per-sequence autodiff =="
OUT2="$(python3 "$HERE/p19_vjp2_multiseq_gate.py" 2>&1)" || fail "multiseq gate exited nonzero"
echo "$OUT2" | grep -a "^\[msq\]" | sed 's/^/  /'

need '^\[msq\] value: '    "$OUT2"
need '^\[msq\] grad rel: ' "$OUT2"
need '^\[msq\] VERDICT: '  "$OUT2"

echo "$OUT2" | grep -aq '|Δ|=0.000e+00' || fail "multiseq value residual is not exactly 0.000e+00"
for k in dq dk dv; do
  v=$(echo "$OUT2" | sed -n "s/.*$k=\([0-9.e+-]*\).*/\1/p" | head -1)
  [ -n "$v" ] || { fail "missing $k in grad rel line"; continue; }
  python3 -c "import sys; sys.exit(0 if float('$v') <= $MAX_MSQ_REL else 1)" || fail "$k rel $v > $MAX_MSQ_REL"
done
echo "$OUT2" | grep -aq '^\[msq\] VERDICT: PASS' || fail "multiseq verdict is not PASS"

echo
if [ "$RC" = 0 ]; then echo "===== T0 PASS (3 gates, 7 numeric measurements + shim-root unit) ====="
else echo "===== T0 FAIL ====="; fi
exit $RC
