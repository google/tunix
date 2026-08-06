#!/usr/bin/env bash
# Negative control for tests/t0_cpu/run.sh.
#
# A gate you have not tried to break is not a gate.  This substitutes stub modules for the
# real ones and asserts that run.sh REJECTS each way a run can be wrong:
#
#   N1  a gate that prints nothing            (missing line = did not run, never a pass)
#   N2  a gate whose value residual is nonzero
#   N3  a gate whose gradient error exceeds the pre-registered threshold
#   N4  a gate that prints good numbers but VERDICT: FAIL
#
# Expected result: all four arms exit nonzero.  If any arm passes, run.sh is not
# fail-closed and its green results mean nothing.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REAL_SHIMS="$HERE/../../src/engine_shims"
RC=0

arm() {  # <name> <python-body-for-stub-selftest>
  local name="$1" body="$2"
  local tmp; tmp="$(mktemp -d)"
  cp "$REAL_SHIMS"/*.py "$tmp/" 2>/dev/null
  printf '%s\n' "$body" > "$tmp/rpa_diff_chunked.py"
  # run.sh derives SHIMS from its own location, so run a copy that points at the stub dir
  sed "s|SHIMS=\"\$HERE/../../src/engine_shims\"|SHIMS=\"$tmp\"|" "$HERE/run.sh" > "$tmp/run.sh"
  chmod +x "$tmp/run.sh"
  bash "$tmp/run.sh" >/dev/null 2>&1
  local rc=$?
  if [ "$rc" -ne 0 ]; then echo "  REJECTED (exit $rc)   $name"
  else echo "  *** ACCEPTED ***      $name   <-- run.sh is NOT fail-closed"; RC=1; fi
  rm -rf "$tmp"
}

echo "== negative control for T0 run.sh =="

arm "N1 silent gate (prints nothing)" \
'import sys
sys.exit(0)'

arm "N2 nonzero value residual" \
'print("[selftest] value |chain-oracle| = 1.000e-09")
print("[selftest] grad rel = 5.039e-16")
print("[selftest] FD best rel = 1.106e-08")
print("[selftest] VERDICT: PASS")'

arm "N3 gradient error above threshold" \
'print("[selftest] value |chain-oracle| = 0.000e+00")
print("[selftest] grad rel = 1.000e-06")
print("[selftest] FD best rel = 1.106e-08")
print("[selftest] VERDICT: PASS")'

arm "N4 good numbers but VERDICT: FAIL" \
'print("[selftest] value |chain-oracle| = 0.000e+00")
print("[selftest] grad rel = 5.039e-16")
print("[selftest] FD best rel = 1.106e-08")
print("[selftest] VERDICT: FAIL")'

echo
if [ "$RC" = 0 ]; then echo "===== NEGATIVE CONTROL PASS -- run.sh rejects all 4 arms ====="
else echo "===== NEGATIVE CONTROL FAIL -- run.sh accepted a bad run ====="; fi
exit $RC
