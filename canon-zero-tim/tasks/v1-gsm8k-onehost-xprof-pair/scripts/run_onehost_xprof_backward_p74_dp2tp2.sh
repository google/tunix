#!/usr/bin/env bash
# Signed P74 carrier: one-host GSM8K Zero-HP backward on DP2xTP2.
set -euo pipefail

label="${1:?usage: run_onehost_xprof_backward_p74_dp2tp2.sh <unique-label>}"
script_dir="$(cd "$(dirname "$0")" && pwd)"

# These are the already-certified DP2xTP2 carrier settings. P74 changes only
# the checked-VMA cotangent repartition; the wrapper does not expose a second
# chunk policy or a condition that can turn check_vma off.
export V1_GSM8K_XPROF_GEOMETRY=dp2-tp2
export CANON_DP_COMPARE_MODE=fingerprint-hybrid
export CANON_DP_DISTINCT_SCHEDULE=first-group-warmup
export CANON_DP_FINITE_FETCH=batched-commit
export CANON_P71_SCAN=fwd

exec bash "$script_dir/run_onehost_xprof_backward_zero.sh" "$label"
