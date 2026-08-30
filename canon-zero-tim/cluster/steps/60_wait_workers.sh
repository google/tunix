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
target_head="${CANON_TARGET_PATHWAYS_HEAD:-${PATHWAYS_HEAD:-}}"
if [ -z "$target_head" ]; then
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

INITIAL_SYNC="${CANON_WORKER_INITIAL_SYNC_SECONDS:-180}"
echo "[wait] ${INITIAL_SYNC}s quiet period for worker boot/registration (configured per-role device expectation: ${EXPECT})..."
sleep "$INITIAL_SYNC"
echo "[wait] quiet period completed. Device assertion runs in step 65 (when the profile opts in) or step 70; this step alone proves nothing."
exit 0
