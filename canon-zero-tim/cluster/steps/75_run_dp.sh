#!/usr/bin/env bash
# Validate the T2 result emitted by the same Pathways client as T1.  Starting another Python
# process here created a second IFRT proxy client and stalled on the 64-chip deployment.
set -euo pipefail
source "$CANON_STATE/env.sh"
LOG="${CANON_T1_LOG:-$CANON_STATE/t1_t2.log}"

echo "[dp-gate] dp=$CANON_DP_SIZE tp=$CANON_TP_SIZE local_samples=$CANON_DP_PROBE_LOCAL_SAMPLES"
python3 "$CANON_PKG/tests/t2_dp/validate_same_session.py" "$LOG"
