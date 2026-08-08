#!/usr/bin/env bash
# The actual workload.
#
# CANON_RUN_CMD carries the command so the same entry point serves gates, canaries and
# training without editing YAML.  The exit code is propagated, and the PATHTRACE tally is
# checked afterwards: an intervention that never fired produces a perfectly green run.
set -uo pipefail
source "$CANON_STATE/env.sh"
export JAX_PLATFORMS="proxy,cpu"
export JAX_BACKEND_TARGET="grpc://localhost:29000"
export PATHWAYS_HEAD="localhost"
if [ "${CANON_P32_DP_ADMISSION:-0}" = "1" ] && \
   [ "${CANON_P32_TRAIN_ADMITTED:-0}" != "1" ]; then
  echo "[run] REFUSING: P32 profile is admission-only." >&2
  echo "[run] A real (dp,tp) replicated-parameter segmented VJP has not passed remote gates;" >&2
  echo "[run] set all three P33 admissions only after the rank-local reducer gate passes." >&2
  exit 2
fi
: "${CANON_RUN_CMD:?CANON_RUN_CMD unset -- nothing to run}"
LOG="${CANON_RUN_LOG:-$CANON_STATE/run.log}"
echo "[run] cmd: $CANON_RUN_CMD"
echo "[run] log: $LOG"
cd "${CANON_RUN_CWD:-$CANON_PKG/..}"
set -o pipefail
bash -c "$CANON_RUN_CMD" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
echo "[run] exit=$rc"
# grep -a: progress-bar control characters make grep treat the log as binary and drop every
# match silently, which reads exactly like "the intervention never fired".
n_ar=$(grep -ac 'CANON_FIXED_AR=1 fixed-order tree' "$LOG" || true)
n_emb=$(grep -ac 'CANON_FIXED_AR_EMBED=1 fixed-order embed gather' "$LOG" || true)
n_lp=$(grep -ac 'CANON_LOGPROB_M on' "$LOG" || true)
n_wandb=$(grep -ac '\[CANON_P33_WANDB\] ONLINE_RUN_PASS' "$LOG" || true)
n_eval_off=$(grep -ac '\[CANON_P33_EVAL\] DISABLED workload=frozenlake' "$LOG" || true)
echo "[run] PATHTRACE fixed_ar=$n_ar embed=$n_emb logprob_m=$n_lp wandb_online=$n_wandb eval_off=$n_eval_off"
if [ "$n_ar" -eq 0 ] || [ "$n_emb" -eq 0 ]; then
  echo "[run] FATAL: no PATHTRACE for the fixed-order reductions -- the intervention did not" >&2
  echo "[run]        execute.  Any result from this run is void regardless of its exit code." >&2
  exit 1
fi
if [ "${CANON_P32_TRAIN_ADMITTED:-0}" = "1" ] && [ "$n_wandb" -ne 1 ]; then
  echo "[run] FATAL: admitted P33 workload did not attest exactly one online W&B run" >&2
  exit 1
fi
if [ "${CANON_P32_TRAIN_ADMITTED:-0}" = "1" ] && \
   [ "${CANON_P32_WORKLOAD:-}" = "frozenlake" ] && \
   [ "$n_eval_off" -ne 1 ]; then
  echo "[run] FATAL: admitted P33 FrozenLake did not attest evaluation disabled exactly once" >&2
  exit 1
fi
exit "$rc"
