#!/usr/bin/env bash
set -euo pipefail

label="${1:?usage: run_onehost_deepswe_zero_admission.sh <unique-label>}"
script_dir="$(cd "$(dirname "$0")" && pwd)"
export CANON_P58_Q4_TP4_ZERO_ADMISSION=1
export P58_ONEHOST_PROBE_PROFILE=seam
export P58_ONEHOST_TIMEOUT_SECONDS="${P58_ONEHOST_TIMEOUT_SECONDS:-7200}"
exec bash "$script_dir/run_onehost_deepswe_xprof_common.sh" zero-hp "$label"
