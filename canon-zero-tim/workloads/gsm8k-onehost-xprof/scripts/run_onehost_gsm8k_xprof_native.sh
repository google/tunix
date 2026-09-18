#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "$0")" && pwd)"
exec bash "$script_dir/run_onehost_gsm8k_xprof_common.sh" native \
  "${1:?usage: run_onehost_gsm8k_xprof_native.sh <unique-label>}"
