#!/usr/bin/env bash
# Container entry point for a canonical run.  The Kubernetes manifest should contain nothing
# but a repo sync and `exec bash canon-zero-tim/cluster/entrypoint.sh` -- changing behaviour
# is then a git change, reviewable and diffable, instead of an edit to an indented heredoc
# inside YAML.
#
#   CANON_PROFILE_FILE=cluster/profiles/qwen3-1p7b.env \
#   CANON_MODE=gate-only \
#     bash cluster/entrypoint.sh
#
# Modes:
#   install-only 00..50 -- probe-only plus install, overlay and overlay verification.  Proves
#                the chain is live without starting a TPU program (set JAX_PLATFORMS=cpu).
#   probe-only   00..25 -- report the image version, apply the ROPE fix if this build needs
#                it, and stop.  Costs no TPU.  Run this first on a new cluster.
#   gate-only    00..50 + T1 -- install the chain, prove [PATHTRACE] fired, run the topology
#                admission probes.  No training, no optimizer, no checkpoint.
#   dp-gate-only 00..50 + T1 + T2-DP -- additionally measure DP reduction, placement
#                sensitivity and one small AdamW update.  Still no model or training.
#   model-init-only 00..60 + P32 model/optimizer/accumulator materialization.  No
#                checkpoint load, forward, backward, update or training.
#   dp16-rc      00..60 + P32 real-checkpoint release-candidate stage. The stage is
#                selected by CANON_P32_RC_STAGE and remains production-default-off.
#   workload-contract-only
#                00..50 + P33 workload serialization. No Pathways connection, model,
#                rollout, backward, optimizer update, W&B initialization or training.
#   run          00..90 -- everything, then the command in CANON_RUN_CMD.
#
# Every step is fail-closed and ordered.  A step that produces no output did not run, and a
# run whose PATHTRACE lines are missing never exercised the intervention no matter how green
# its exit code is.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG="$(cd "$HERE/.." && pwd)"
export CANON_PKG="$PKG"
export CANON_CLUSTER="$HERE"
# Steps run as separate processes, so exported variables do not flow between them.  00_env.sh
# writes one env file here and every later step sources it -- the resolved configuration is
# then a file you can cat, not state you have to reconstruct.
export CANON_STATE="${CANON_STATE:-/tmp/canon-state}"
mkdir -p "$CANON_STATE"
MODE="${CANON_MODE:-gate-only}"
export CANON_MODE="$MODE"

log() { echo "[entrypoint] $*"; }
die() { echo "[entrypoint] FATAL: $*" >&2; exit 1; }

case "$MODE" in
  probe-only|install-only|gate-only|dp-gate-only|model-init-only|dp16-rc|workload-contract-only|run) ;;
  *) die "unknown CANON_MODE: $MODE" ;;
esac

step() {
  local s="$HERE/steps/$1"
  [ -f "$s" ] || die "missing step script: $1"
  log "--> $1"
  # shellcheck disable=SC1090
  bash "$s" || die "$1 exited $?"
  log "<-- $1 ok"
}

log "start $(date -u +%Y-%m-%dT%H:%M:%SZ)  mode=$MODE  pkg=$PKG"

# Retry visibility.  The JobSet restarts on failure, and a red gate IS a failure -- so a log
# can be from attempt N while `kubectl logs` makes it look like the only run.  Print the
# attempt so a report can state it.  Unknown is reported as unknown: assuming a first attempt
# is exactly the mistake this line exists to prevent.
ATTEMPT="${JOBSET_RESTART_ATTEMPT:-}"
if [ -z "$ATTEMPT" ]; then
  log "JOBSET_ATTEMPT unknown  pod=${CANON_POD_NAME:-?}  -- the restart-attempt annotation was"
  log "JOBSET_ATTEMPT   not exposed here.  A verdict below cannot be shown to be a first attempt."
elif [ "$ATTEMPT" = "0" ]; then
  log "JOBSET_ATTEMPT 0 (first attempt)  pod=${CANON_POD_NAME:-?}"
else
  log "JOBSET_ATTEMPT $ATTEMPT  pod=${CANON_POD_NAME:-?}  -- THIS IS A RETRY.  A gate verdict from"
  log "JOBSET_ATTEMPT   a retried run is not evidence of determinism; report the attempt number."
fi

step 00_env.sh
# Resolve branching decisions from the signed profile output, not from an
# optional duplicate value in the raw JobSet environment. Secrets are excluded
# from this file by 00_env.sh.
# shellcheck disable=SC1090
source "$CANON_STATE/env.sh"
# shellcheck disable=SC1091
source "$HERE/steps/p57_runtime_contract.sh"
step 10_sync_repo.sh
step 20_probe_image.sh
step 25_rope_fix.sh
step 28_sync_cache.sh

if [ "$MODE" = "probe-only" ]; then
  log "mode=probe-only -- stopping before install.  No TPU program was started."
  exit 0
fi

if [ "${CANON_P46_EVALUATION:-0}" = "1" ]; then
  # P46 clean evaluation and its observer-only parity canary both use the stock
  # sampler. Keep the RoPE decision and pinned R2E install, but do not overlay
  # the differentiable canonical chain used for training/alignment.
  step 35_install_r2egym.sh
  log "P46_EVALUATION_STOCK_PATH mode=$CANON_P46_EVALUATION_MODE source=$CANON_EXPECT_COMMIT canonical_overlay=skipped"
elif [ "${CANON_P58_DEEPSWE_TIM:-0}" = "1" ] && \
     [ "${CANON_P58_TIM_ARM:-}" = "native" ]; then
  # P58 native is the untreated numerical baseline.  Keep the shared Tunix
  # trainer code from the pinned source. Verify every signed tpu_inference
  # target before installing the separately signed, observer-only B overlay.
  # The zero-TIM canonical chain remains absent.
  step 35_install_r2egym.sh
  step p58_verify_stock_engine.sh
  step p58_install_stock_prompt_observer.sh
  log "P58_NATIVE_STOCK_PATH source=$CANON_EXPECT_COMMIT canonical_overlay=skipped stock_observer=installed"
elif p57_is_stock_fast_runtime; then
  # P57.1 measures and trains the stock arm with the untreated pinned-image
  # serving program. Installing the canonical chain and merely unsetting its
  # flags is not stock: several shims enforce dependencies at import time.
  # Keep the independent R2E gym install, but leave all six tpu_inference
  # targets byte-identical to the pinned image established by Step 20.
  step 35_install_r2egym.sh
  step 37_install_stock_runtime.sh
  step 38_verify_stock_engine.sh
  p57_observer_overlay=absent
  if p57_is_stock_fast_training; then
    step 39_install_p57_stock_observer.sh
    p57_observer_overlay=installed
  fi
  log "P57_STOCK_FAST_PATH run_kind=$CANON_P57_RUN_KIND regime=stock-fast source=$CANON_EXPECT_COMMIT canonical_overlay=skipped observer_overlay=$p57_observer_overlay"
  unset p57_observer_overlay
else
  step 30_install_canon.sh
  step 35_install_r2egym.sh
  step 40_overlay_engine.sh
  step 50_verify_overlay.sh
fi

if [ "$MODE" = "install-only" ]; then
  log "mode=install-only -- chain installed and verified.  No TPU program was started."
  exit 0
fi

if [ "$MODE" = "workload-contract-only" ]; then
  step 86_validate_workload.sh
  log "mode=workload-contract-only -- contract serialized; no TPU program or training was started."
  exit 0
fi

if [ "$MODE" = "gate-only" ] || [ "$MODE" = "dp-gate-only" ]; then
  step 60_wait_workers.sh
  if [ "$MODE" = "dp-gate-only" ]; then
    export CANON_RUN_T2_DP=1
  else
    export CANON_RUN_T2_DP=0
  fi
  step 70_run_t1.sh
  if [ "$MODE" = "dp-gate-only" ]; then
    step 75_run_dp.sh
  fi
  log "mode=gate-only -- topology admission probes complete.  No training was run."
  [ "$MODE" = "dp-gate-only" ] && \
    log "T2-DP complete -- read DECISION/OBSERVATIONS; PASS is fixed-placement only."
  log "Read the numbers against CLUSTER_ADMISSION.md; a zero exit code is not an admission."
  exit 0
fi

if [ "$MODE" = "model-init-only" ]; then
  step 60_wait_workers.sh
  step 80_model_init.sh
  log "mode=model-init-only -- structural state materialized; no checkpoint, forward, backward, update or training was run."
  exit 0
fi

if [ "$MODE" = "dp16-rc" ]; then
  step 60_wait_workers.sh
  step 85_run_dp16_rc.sh
  log "mode=dp16-rc -- bounded release-candidate stage complete; no production training was admitted."
  exit 0
fi

step 60_wait_workers.sh
step 65_probe_devices.sh
step 90_run.sh
log "done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
