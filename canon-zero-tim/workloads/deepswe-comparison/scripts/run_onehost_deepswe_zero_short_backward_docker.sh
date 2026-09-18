#!/usr/bin/env bash
set -euo pipefail

label="${1:?usage: run_onehost_deepswe_zero_short_backward_docker.sh <unique-label>}"
script_dir="$(cd "$(dirname "$0")" && pwd)"
repo="$(git -C "$script_dir" rev-parse --show-toplevel)"
export CANON_P58_Q4_TP4_ZERO_ADMISSION=1
export CANON_P58_Q4_TP4_SHORT_BACKWARD=1
export DEEPSWE_ONEHOST_WHITELIST="$repo/canon-zero-tim/clean_data/p58_seam_probe/p58z07_group3_pillow.jsonl"
export DEEPSWE_ONEHOST_TASK_IMAGE="namanjain12/pillow_final:52079cb2975fda98476c7a7f172e5519e67ba612"
export P58_ONEHOST_TIMEOUT_SECONDS="${P58_ONEHOST_TIMEOUT_SECONDS:-21600}"
export P58_ONEHOST_COMPILATION_CACHE_DIR="${P58_ONEHOST_COMPILATION_CACHE_DIR:-/mnt/disks/tunix-data/jax-compilation-cache/p58-q4-tp4-short-backward}"
exec bash "$script_dir/run_onehost_deepswe_seam_probe_docker.sh" "$label"
