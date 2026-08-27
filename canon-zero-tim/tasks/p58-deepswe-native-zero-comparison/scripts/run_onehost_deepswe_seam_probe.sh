#!/usr/bin/env bash
# Long-context, mutation-free Zero-HP carrier for the P58 decode/prefill seam.
set -euo pipefail

label="${1:?usage: run_onehost_deepswe_seam_probe.sh <unique-label>}"
script_dir="$(cd "$(dirname "$0")" && pwd)"
repo="$(git -C "$script_dir" rev-parse --show-toplevel)"

export P58_ONEHOST_PROBE_PROFILE=seam
export DEEPSWE_ONEHOST_WHITELIST="${DEEPSWE_ONEHOST_WHITELIST:-$repo/canon-zero-tim/clean_data/p58_seam_probe/p58z07_group3_pillow.jsonl}"
export DEEPSWE_ONEHOST_TASK_IMAGE="${DEEPSWE_ONEHOST_TASK_IMAGE:-namanjain12/pillow_final:52079cb2975fda98476c7a7f172e5519e67ba612}"
export P58_ONEHOST_TIMEOUT_SECONDS="${P58_ONEHOST_TIMEOUT_SECONDS:-7200}"

exec bash "$script_dir/run_onehost_deepswe_xprof_common.sh" zero-hp "$label"
