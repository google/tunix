#!/usr/bin/env bash
# Give TPU worker pods time to pull images and register with the resource manager.
#
# Kept as its own step so the delay is visible and tunable, instead of a bare `sleep 60`
# buried in a YAML block.  Skipped entirely when there is no Pathways head.
set -euo pipefail
source "$CANON_STATE/env.sh"
if [ -z "${PATHWAYS_HEAD:-}" ]; then
  echo "[wait] no PATHWAYS_HEAD -- nothing to wait for"
  exit 0
fi
S="${CANON_WORKER_WAIT_SECONDS:-120}"
echo "[wait] waiting up to ${S}s for TPU workers (64 devices) to register with Pathways..."

WAITED=0
while [ "$WAITED" -lt "$S" ]; do
  if python3 -c "import sys, os; os.environ['FLAGS_pathways_enforce_subset_devices_form_subslice']='false'; import pathwaysutils; pathwaysutils.initialize(); import jax; d=jax.devices(); sys.exit(0 if len(d)>=64 else 1)" 2>/dev/null; then
    echo "[wait] successfully connected to Pathways! Visible TPU devices: $(python3 -c "import pathwaysutils; pathwaysutils.initialize(); import jax; print(len(jax.devices()))" 2>/dev/null)"
    break
  fi
  sleep 5
  WAITED=$((WAITED + 5))
  if [ $((WAITED % 20)) -eq 0 ]; then
    echo "[wait] still waiting for workers to register (${WAITED}s/${S}s)..."
  fi
done
echo "[wait] done"
