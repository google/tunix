#!/usr/bin/env bash
# P32 DP16xTP4 update admission.  This is a small synthetic gradient/update probe: no model,
# checkpoint, optimizer state allocation or training data.
set -euo pipefail
source "$CANON_STATE/env.sh"
export CANON_IN_CONTAINER=1
export CANON_DP_PROBE_LOG="${CANON_DP_PROBE_LOG:-$CANON_STATE/t2_dp.log}"

echo "[dp-gate] dp=$CANON_DP_SIZE tp=$CANON_TP_SIZE local_samples=$CANON_DP_PROBE_LOCAL_SAMPLES"
bash "$CANON_PKG/tests/t2_dp/run.sh"
echo "[dp-gate] artifact=$CANON_DP_PROBE_LOG"
