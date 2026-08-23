#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "$0")" && pwd)"
exec bash "$script_dir/run_onehost_deepswe_xprof_common.sh" zero-hp \
  "${1:?usage: run_onehost_deepswe_xprof_zero_hp.sh <unique-label>}"
