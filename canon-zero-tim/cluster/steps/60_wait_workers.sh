#!/usr/bin/env bash
# Wait until Pathways exposes exactly the expected device count, and keeps exposing it.
#
# Kept as its own step so the delay is visible and tunable, instead of a bare `sleep` buried in
# a YAML block.  Skipped entirely when there is no Pathways head.
#
# Why it requires several consecutive passes rather than one: during node autoscaling the
# device count is transiently correct while pods are still dying, so a single probe can catch a
# fleeting state and hand the next step a topology that is about to change underneath it.
# CANON_WORKER_STABLE_PROBES sets how many consecutive passes are needed.  Setting it to 1
# restores single-shot behaviour -- that is allowed, but it is then a recorded choice rather
# than an accident, and the value is printed with the verdict.
set -euo pipefail
source "$CANON_STATE/env.sh"
if [ -z "${PATHWAYS_HEAD:-}" ]; then
  echo "[wait] no PATHWAYS_HEAD -- nothing to wait for"
  exit 0
fi

S="${CANON_WORKER_WAIT_SECONDS:-600}"
EXPECT="${CANON_TOTAL_DEVICES:-64}"
STABLE="${CANON_WORKER_STABLE_PROBES:-3}"
INTERVAL="${CANON_WORKER_PROBE_INTERVAL:-10}"
for v in S EXPECT STABLE INTERVAL; do
  [[ "${!v}" =~ ^[1-9][0-9]*$ ]] || {
    echo "[wait] REFUSING: $v must be a positive integer, got ${!v@Q}" >&2
    exit 2
  }
done

OUT="$(mktemp -t canon_probe_devices.XXXXXX)"
trap 'rm -f "$OUT"' EXIT

INITIAL_SYNC="${CANON_WORKER_INITIAL_SYNC_SECONDS:-60}"
echo "[wait] giving all ${EXPECT} devices on 16 TPU worker nodes ${INITIAL_SYNC}s quiet period to boot and register with Pathways RM..."
sleep "$INITIAL_SYNC"

echo "[wait] waiting up to ${S}s for exactly ${EXPECT} devices, stable across ${STABLE} consecutive probe(s) ${INTERVAL}s apart"
WAITED=0
CONSECUTIVE=0
while [ "$WAITED" -lt "$S" ]; do
  if CANON_EXPECT_VISIBLE_DEVICES="$EXPECT" \
      python3 "$CANON_PKG/tests/t1_tpu/probe_devices.py" >"$OUT" 2>&1; then
    CONSECUTIVE=$((CONSECUTIVE + 1))
    echo "[wait] probe passed (${CONSECUTIVE}/${STABLE} consecutive, ${WAITED}s/${S}s)"
    if [ "$CONSECUTIVE" -ge "$STABLE" ]; then
      cat "$OUT"
      echo "[wait] Pathways readiness PASS after ${WAITED}s (stable across ${STABLE} consecutive probe(s))"
      exit 0
    fi
  else
    # A single failure resets the run.  Counting non-consecutive passes would defeat the point:
    # the question is whether the topology is settled, not whether it was ever briefly right.
    if [ "$CONSECUTIVE" -gt 0 ]; then
      echo "[wait] probe failed after ${CONSECUTIVE} consecutive pass(es) -- resetting (${WAITED}s/${S}s)"
    fi
    CONSECUTIVE=0
  fi
  sleep "$INTERVAL"
  WAITED=$((WAITED + INTERVAL))
  if [ $(( WAITED % (INTERVAL * 3) )) -eq 0 ]; then
    echo "[wait] still waiting (${WAITED}s/${S}s)..."
  fi
done

grep -aE '^\[T1\.PATHWAYS\]|^\[t1\.devices\]' "$OUT" 2>/dev/null || true
echo "[wait] REFUSING: Pathways did not expose exactly ${EXPECT} devices, stable across ${STABLE} probe(s), within ${S}s" >&2
exit 1
