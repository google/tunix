#!/usr/bin/env bash
# Thin wrapper: rollout XProf for the native arm (CANON_XPROF_PHASE=step).
# rollout=rollout captures the STEP window (device trace holds the first ~25s of
# decode; host tracer + engine [PERF] spans cover the full phase). The census/
# classifier expectations are written for the update phase, so rollout mode is
# EXPECTED to exit 1 with census reds — it is a diagnostic capture, not a
# certification gate; the xprof/trace artifacts are the deliverable.
set -euo pipefail
label="${1:?usage: run_onehost_xprof_rollout_native.sh <unique-label>}"
script_dir="$(cd "$(dirname "$0")" && pwd)"
CANON_XPROF_PHASE=step CANON_XPROF_TPU_TRACE_MODE= exec bash "$script_dir/run_onehost_gsm8k_xprof_common.sh" native "ro_${label}"
