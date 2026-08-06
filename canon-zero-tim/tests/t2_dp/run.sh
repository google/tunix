#!/usr/bin/env bash
# P32 DP update-consistency probe. No model, checkpoint, optimizer state or
# training data is needed. On CPU, set CANON_DP_PROBE_CPU=1; on the target
# Pathways deployment use the real dp=16,tp=4 mesh.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DP="${CANON_DP_SIZE:-4}"
TP="${CANON_TP_SIZE:-1}"
LOCAL="${CANON_DP_PROBE_LOCAL_SAMPLES:-16}"

if [ "${CANON_DP_PROBE_CPU:-0}" = "1" ]; then
  export JAX_PLATFORMS=cpu
  case " ${XLA_FLAGS:-} " in
    *" --xla_force_host_platform_device_count="*) ;;
    *) export XLA_FLAGS="${XLA_FLAGS:-} --xla_force_host_platform_device_count=$((DP * TP))";;
  esac
fi

LOG="${CANON_DP_PROBE_LOG:-$(mktemp)}"
python3 "$HERE/probe_dp_update.py" \
  --dp "$DP" --tp "$TP" --local-samples "$LOCAL" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
if [ "$rc" -ne 0 ]; then
  echo "===== T2-DP FAIL -- probe exited $rc =====" >&2
  exit "$rc"
fi

for marker in CONFIG MESH CHECKS OBSERVATIONS UPDATE DECISION VERDICT; do
  count="$(grep -ac "^\[P32.DP\] $marker" "$LOG" || true)"
  if [ "$count" -ne 1 ]; then
    echo "T2-DP FAIL: expected one $marker line, got $count" >&2
    exit 1
  fi
done
grep -aq '^\[P32.DP\] VERDICT PASS$' "$LOG" || {
  echo "T2-DP FAIL: verdict was not PASS" >&2
  exit 1
}

echo "===== T2-DP PASS -- fixed-topology update contract measured ====="
echo "Read the DECISION/OBSERVATIONS lines before promoting training: PASS does"
echo "not claim arbitrary sample-to-rank placement invariance."
