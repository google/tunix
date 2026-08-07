#!/usr/bin/env bash
# Run one bounded real-checkpoint DP16xTP4 release-candidate stage.
set -euo pipefail
source "$CANON_STATE/env.sh"

[ "${CANON_MODE:-}" = "dp16-rc" ] || {
  echo "[dp16-rc] REFUSING: CANON_MODE must be dp16-rc" >&2
  exit 2
}
[ "${CANON_P32_RC:-0}" = "1" ] || {
  echo "[dp16-rc] REFUSING: release-candidate profile is not active" >&2
  exit 2
}
[ "${CANON_P32_TRAIN_ADMITTED:-0}" = "0" ] || {
  echo "[dp16-rc] REFUSING: production training must remain disabled" >&2
  exit 2
}

# Secrets never enter env.sh. Reconstruct only the process environment and never print values.
if [ -n "${INJECTED_HF_TOKEN:-}" ]; then
  export HF_TOKEN
  HF_TOKEN="$(printf '%s' "$INJECTED_HF_TOKEN" | tr -d '[:space:]')"
fi

export JAX_PLATFORMS="proxy,cpu"
export JAX_BACKEND_TARGET="grpc://localhost:29000"
export PATHWAYS_HEAD="localhost"
export CANON_IN_CONTAINER=1
LOG="${CANON_P32_RC_LOG:-$CANON_STATE/p32_rc_${CANON_P32_RC_STAGE}.log}"
REPORT="${CANON_P32_RC_REPORT:-$CANON_STATE/p32_rc_${CANON_P32_RC_STAGE}.classification.json}"

echo "[dp16-rc] starting stage=$CANON_P32_RC_STAGE; production training remains refused"
cd "$CANON_PKG/.."
set +e
python3 "$CANON_PKG/tests/p32_release_candidate/probe_qwen8b_rc.py" \
  --stage "$CANON_P32_RC_STAGE" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e
if [ "$rc" -ne 0 ]; then
  echo "[dp16-rc] probe exited $rc" >&2
  exit "$rc"
fi
python3 "$CANON_PKG/tests/p32_release_candidate/classify_rc.py" \
  "$LOG" --stage "$CANON_P32_RC_STAGE" --output "$REPORT"
echo "[dp16-rc] artifact=$LOG"
echo "[dp16-rc] classification=$REPORT"
