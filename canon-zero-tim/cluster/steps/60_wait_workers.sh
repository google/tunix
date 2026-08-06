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
S="${CANON_WORKER_WAIT_SECONDS:-600}"
EXPECT="${CANON_TOTAL_DEVICES:-64}"
[[ "$S" =~ ^[1-9][0-9]*$ ]] || {
  echo "[wait] REFUSING: CANON_WORKER_WAIT_SECONDS must be a positive integer" >&2
  exit 2
}
[[ "$EXPECT" =~ ^[1-9][0-9]*$ ]] || {
  echo "[wait] REFUSING: expected device count must be a positive integer" >&2
  exit 2
}

echo "[wait] waiting up to ${S}s for exactly ${EXPECT} devices to register with Pathways"
WAITED=0
while [ "$WAITED" -lt "$S" ]; do
  if CANON_EXPECT_VISIBLE_DEVICES="$EXPECT" \
      python3 "$CANON_PKG/tests/t1_tpu/probe_devices.py" >/tmp/probe_devices.out 2>&1; then
    cat /tmp/probe_devices.out
    echo "[wait] Pathways readiness PASS after ${WAITED}s"
    exit 0
  fi
  sleep 10
  WAITED=$((WAITED + 10))
  if [ $((WAITED % 30)) -eq 0 ]; then
    echo "[wait] still waiting (${WAITED}s/${S}s)..."
  fi
done

cat /tmp/probe_devices.out 2>/dev/null | grep -aE '^\[T1\.PATHWAYS\]|^\[t1\.devices\]' || true
echo "[wait] REFUSING: Pathways did not expose exactly ${EXPECT} devices within ${S}s" >&2
exit 1
