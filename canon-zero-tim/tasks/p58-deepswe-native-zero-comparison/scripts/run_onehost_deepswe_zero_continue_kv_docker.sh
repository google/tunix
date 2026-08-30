#!/usr/bin/env bash
set -euo pipefail

label="${1:?usage: run_onehost_deepswe_zero_continue_kv_docker.sh <unique-label>}"
script_dir="$(cd "$(dirname "$0")" && pwd)"
export CANON_P58_Q4_TP4_ZERO_ADMISSION=1
export CANON_P58_Q4_TP4_CONTINUE_KV_DIAGNOSTIC=1
exec bash "$script_dir/run_onehost_deepswe_seam_probe_docker.sh" "$label"
