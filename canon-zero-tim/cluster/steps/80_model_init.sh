#!/usr/bin/env bash
# Materialize the exact DP16xTP4 Qwen3-8B training state and stop.
set -euo pipefail
source "$CANON_STATE/env.sh"

[ "${CANON_MODE:-}" = "model-init-only" ] || {
  echo "[model-init] REFUSING: CANON_MODE must be model-init-only" >&2
  exit 2
}
[ "${CANON_P32_MODEL_INIT_ONLY:-0}" = "1" ] || {
  echo "[model-init] REFUSING: dedicated model-init profile is not active" >&2
  exit 2
}
[ "${CANON_P32_TRAIN_ADMITTED:-0}" = "0" ] || {
  echo "[model-init] REFUSING: model-init-only must not admit training" >&2
  exit 2
}

export JAX_PLATFORMS="proxy,cpu"
export JAX_BACKEND_TARGET="grpc://localhost:29000"
export PATHWAYS_HEAD="localhost"
export CANON_IN_CONTAINER=1
LOG="${CANON_MODEL_INIT_LOG:-$CANON_STATE/model_init.log}"
REPORT="${CANON_MODEL_INIT_REPORT:-$CANON_STATE/model_init.classification.json}"

echo "[model-init] starting structural materialization; no checkpoint, forward, backward, update, or training"
cd "$CANON_PKG/.."
set +e
python3 "$CANON_PKG/tests/p32_model_init/probe_qwen8b_init.py" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e
if [ "$rc" -ne 0 ]; then
  echo "[model-init] probe exited $rc" >&2
  exit "$rc"
fi
python3 "$CANON_PKG/tests/p32_model_init/classify_model_init.py" \
  "$LOG" --output "$REPORT"
echo "[model-init] artifact=$LOG"
echo "[model-init] classification=$REPORT"
