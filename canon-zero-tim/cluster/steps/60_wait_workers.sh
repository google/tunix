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
S="${CANON_WORKER_WAIT_SECONDS:-60}"
echo "[wait] sleeping ${S}s for TPU workers to initialise"
sleep "$S"
echo "[wait] done"
