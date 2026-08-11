#!/usr/bin/env bash
# Fail-fast Pathways session admission probe.
#
# p39d7: the client connected while worker registration was still incomplete, the
# proxy cancelled the session one second later, and the whole Stage 1 workload fell
# into a single-CPU fallback before dying on the role split.  60_wait_workers is a
# blind quiet period and the P33 device assertion lives in step 70, which the
# mode=run path never executes -- so nothing between the sleep and the workload
# proved the session was real.
#
# This step retries a FRESH-PROCESS device count until the configured whole-slice
# expectation is visible.  Worker latecomers self-heal inside the loop; structural
# rejections (a stale client holding the cluster, a broken proxy) surface as a
# bounded, legible failure with the exact remedy, instead of a wasted launch.
#
# Opt-in: profiles that set CANON_EXPECTED_SLICE_DEVICES run the probe; every other
# profile skips it and behaves exactly as before.  The expectation is the WHOLE
# visible slice (256 on 4x8x8), not the per-role CANON_TOTAL_DEVICES contract value.
set -euo pipefail
source "$CANON_STATE/env.sh"

EXPECTED="${CANON_EXPECTED_SLICE_DEVICES:-}"
if [ -z "$EXPECTED" ]; then
  echo "[probe] CANON_EXPECTED_SLICE_DEVICES unset -- skipping the device admission probe"
  exit 0
fi
if ! printf '%s' "$EXPECTED" | grep -Eq '^[1-9][0-9]*$'; then
  echo "[probe] FATAL: CANON_EXPECTED_SLICE_DEVICES must be a positive integer, got '$EXPECTED'" >&2
  exit 1
fi
TIMEOUT="${CANON_DEVICE_PROBE_TIMEOUT_SECS:-900}"
INTERVAL="${CANON_DEVICE_PROBE_INTERVAL_SECS:-30}"
if ! printf '%s%s' "$TIMEOUT" "$INTERVAL" | grep -Eq '^[0-9]+$'; then
  echo "[probe] FATAL: probe timeout/interval must be positive integers" >&2
  exit 1
fi

DEADLINE=$(( $(date +%s) + TIMEOUT ))
ATTEMPT=0
while :; do
  ATTEMPT=$((ATTEMPT + 1))
  # One fresh interpreter per attempt: a cancelled proxy session poisons the JAX
  # backend state, so retrying inside a single process would report the stale
  # CPU fallback forever.
  set +e
  OUT="$(python3 - <<'PY' 2>&1
import pathwaysutils
pathwaysutils.initialize()
import jax
devices = jax.devices()
print(
    "[probe] platform=%s device_count=%d process_count=%d"
    % (devices[0].platform, jax.device_count(), jax.process_count())
)
PY
)"
  RC=$?
  set -e
  printf '%s\n' "$OUT" | tail -3
  # An attempt that dies before printing (import error, cancelled session) must
  # feed the retry loop, not kill the script through pipefail on a no-match grep.
  COUNT="$(printf '%s\n' "$OUT" | { grep -oE 'device_count=[0-9]+' || true; } | tail -1 | cut -d= -f2)"
  if [ "$RC" -eq 0 ] && [ "${COUNT:-0}" = "$EXPECTED" ]; then
    echo "[probe] PASS attempt=$ATTEMPT devices=$COUNT expected=$EXPECTED"
    exit 0
  fi
  NOW="$(date +%s)"
  if [ "$NOW" -ge "$DEADLINE" ]; then
    echo "[probe] FATAL: expected $EXPECTED visible devices, last saw '${COUNT:-none}' (rc=$RC) after ${TIMEOUT}s" >&2
    echo "[probe] archive the pathways-proxy and pathways-rm container logs BEFORE deleting this JobSet;" >&2
    echo "[probe] then check for stale JobSets/clients holding the slice (kubectl get jobsets,pods)." >&2
    exit 1
  fi
  echo "[probe] attempt=$ATTEMPT saw '${COUNT:-none}' (rc=$RC), expected $EXPECTED; retrying in ${INTERVAL}s"
  sleep "$INTERVAL"
done
