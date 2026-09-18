#!/usr/bin/env bash
# Thin wrapper: backward XProf for the native arm (CANON_XPROF_PHASE=update).
# backward=rollout captures the STEP window (device trace holds the first ~25s of
# decode; host tracer + engine [PERF] spans cover the full phase). The census/
# classifier expectations are written for the update phase, so rollout mode is
# EXPECTED to exit 1 with census reds — it is a diagnostic capture, not a
# certification gate; the xprof/trace artifacts are the deliverable.
set -euo pipefail
label="${1:?usage: run_onehost_xprof_backward_native.sh <unique-label>}"
script_dir="$(cd "$(dirname "$0")" && pwd)"
CANON_XPROF_PHASE=update exec bash "$script_dir/run_onehost_gsm8k_xprof_common.sh" native "ba_${label}"
