#!/usr/bin/env bash
# Serialize one workload contract without connecting to Pathways or launching training.
set -euo pipefail
source "$CANON_STATE/env.sh"

OUT="$CANON_STATE/p33_${CANON_P32_WORKLOAD}_contract.classification.json"
test ! -e "$OUT"
JAX_PLATFORMS=cpu PYTHONPATH="$CANON_PKG/..:${PYTHONPATH:-}" \
  python3 "$CANON_PKG/tests/p33_workloads/validate_workload.py" \
    --name "$CANON_P32_WORKLOAD" --output "$OUT"
echo "[workload-contract] classification=$OUT"
