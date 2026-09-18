#!/usr/bin/env bash
# Long-context, mutation-free Zero-HP carrier for the P58 decode/prefill seam.
set -euo pipefail

label="${1:?usage: run_onehost_deepswe_seam_probe.sh <unique-label>}"
script_dir="$(cd "$(dirname "$0")" && pwd)"
repo="$(git -C "$script_dir" rev-parse --show-toplevel)"

export P58_ONEHOST_PROBE_PROFILE=seam
if [ "${CANON_P58_Q4_TP4_TRAJECTORY_REPLAY:-0}" = 1 ]; then
  default_whitelist="$repo/canon-zero-tim/clean_data/p58_short_backward/p58_q4_scrapy_localcache.jsonl"
  default_task_image="namanjain12/scrapy_final:439a3e59b8e858441f8d97dbc32f398db392330d"
elif [ "${CANON_P58_Q4_TP4_CARRIER_SCREEN:-0}" = 1 ]; then
  default_whitelist="$repo/canon-zero-tim/clean_data/p58_short_backward/p58_q4_scrapy_localcache.jsonl"
  default_task_image="namanjain12/scrapy_final:439a3e59b8e858441f8d97dbc32f398db392330d"
elif [ "${CANON_P58_Q4_TP4_SHORT_BACKWARD:-0}" = 1 ]; then
  default_whitelist="$repo/canon-zero-tim/clean_data/p58_seam_probe/p58z07_group3_pillow.jsonl"
  default_task_image="namanjain12/pillow_final:52079cb2975fda98476c7a7f172e5519e67ba612"
else
  default_whitelist="$repo/canon-zero-tim/clean_data/p58_seam_probe/p58z07_group3_pillow.jsonl"
  default_task_image="namanjain12/pillow_final:52079cb2975fda98476c7a7f172e5519e67ba612"
fi
export DEEPSWE_ONEHOST_WHITELIST="${DEEPSWE_ONEHOST_WHITELIST:-$default_whitelist}"
export DEEPSWE_ONEHOST_TASK_IMAGE="${DEEPSWE_ONEHOST_TASK_IMAGE:-$default_task_image}"
export P58_ONEHOST_TIMEOUT_SECONDS="${P58_ONEHOST_TIMEOUT_SECONDS:-7200}"

exec bash "$script_dir/run_onehost_deepswe_xprof_common.sh" zero-hp "$label"
