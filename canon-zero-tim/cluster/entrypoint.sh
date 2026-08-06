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

log() { echo "[entrypoint] $*"; }
die() { echo "[entrypoint] FATAL: $*" >&2; exit 1; }

step() {
  local s="$HERE/steps/$1"
  [ -f "$s" ] || die "missing step script: $1"
  log "--> $1"
  # shellcheck disable=SC1090
  bash "$s" || die "$1 exited $?"
  log "<-- $1 ok"
}

log "start $(date -u +%Y-%m-%dT%H:%M:%SZ)  mode=$MODE  pkg=$PKG"

step 00_env.sh
step 10_sync_repo.sh
step 20_probe_image.sh
step 25_rope_fix.sh

if [ "$MODE" = "probe-only" ]; then
  log "mode=probe-only -- stopping before install.  No TPU program was started."
  exit 0
fi

step 30_install_canon.sh
step 40_overlay_engine.sh
step 50_verify_overlay.sh

if [ "$MODE" = "install-only" ]; then
  log "mode=install-only -- chain installed and verified.  No TPU program was started."
  exit 0
fi

if [ "$MODE" = "gate-only" ]; then
  step 60_wait_workers.sh
  step 70_run_t1.sh
  log "mode=gate-only -- topology admission probes complete.  No training was run."
  log "Read the numbers against CLUSTER_ADMISSION.md; a zero exit code is not an admission."
  exit 0
fi

step 60_wait_workers.sh
step 90_run.sh
log "done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
