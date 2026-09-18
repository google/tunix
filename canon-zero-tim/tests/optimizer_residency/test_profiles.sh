#!/usr/bin/env bash
# Validate the default-resident optimizer placement switch without launching JAX.
# Operator decision 2026-08-12: profiles default to device-resident optimizer state;
# explicit CANON_OPT_STATE_RESIDENT=0 is the offload fallback path.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

check_profile() (
  set -euo pipefail
  local profile="$1"
  unset CANON_OPT_STATE_RESIDENT CANON_P30_OPT_STATE_OFFLOAD
  # shellcheck disable=SC1090
  source "$ROOT/cluster/profiles/$profile"
  test "$CANON_OPT_STATE_RESIDENT" = "1"
  test "$CANON_P30_OPT_STATE_OFFLOAD" = "0"
)

check_resident() (
  set -euo pipefail
  local profile="$1"
  unset CANON_P30_OPT_STATE_OFFLOAD
  export CANON_OPT_STATE_RESIDENT=1
  # shellcheck disable=SC1090
  source "$ROOT/cluster/profiles/$profile"
  test "$CANON_OPT_STATE_RESIDENT" = "1"
  test "$CANON_P30_OPT_STATE_OFFLOAD" = "0"
)

check_offload_fallback() (
  set -euo pipefail
  local profile="$1"
  unset CANON_P30_OPT_STATE_OFFLOAD
  export CANON_OPT_STATE_RESIDENT=0
  # shellcheck disable=SC1090
  source "$ROOT/cluster/profiles/$profile"
  test "$CANON_OPT_STATE_RESIDENT" = "0"
  test "$CANON_P30_OPT_STATE_OFFLOAD" = "1"
)

reject_ambiguous() (
  set -euo pipefail
  local profile="$1"
  export CANON_OPT_STATE_RESIDENT=1
  export CANON_P30_OPT_STATE_OFFLOAD=1
  # shellcheck disable=SC1090
  source "$ROOT/cluster/profiles/$profile"
)

for profile in \
  qwen3-1p7b-dp16-tp4-gsm8k.env \
  qwen3-8b-dp16-tp4-frozenlake.env; do
  check_profile "$profile"
  check_resident "$profile"
  check_offload_fallback "$profile"
  if reject_ambiguous "$profile" >/dev/null 2>&1; then
    echo "[P41.OPTIMIZER] ambiguous placement accepted: $profile" >&2
    exit 1
  fi
done

echo "[P41.OPTIMIZER] PROFILE_GATE_PASS workloads=2 default=resident fallback=offload"
