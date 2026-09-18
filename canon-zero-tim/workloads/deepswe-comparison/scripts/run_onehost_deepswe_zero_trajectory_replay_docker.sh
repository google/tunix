#!/usr/bin/env bash
set -euo pipefail

label="${1:?usage: run_onehost_deepswe_zero_trajectory_replay_docker.sh <unique-label>}"
script_dir="$(cd "$(dirname "$0")" && pwd)"
repo="$(git -C "$script_dir" rev-parse --show-toplevel)"
export CANON_P58_Q4_TP4_ZERO_ADMISSION=1
export CANON_P58_Q4_TP4_SHORT_BACKWARD=1
export CANON_P58_Q4_TP4_TRAJECTORY_REPLAY=1
export DEEPSWE_ONEHOST_WHITELIST="$repo/canon-zero-tim/clean_data/p58_short_backward/p58_q4_scrapy_localcache.jsonl"
export DEEPSWE_ONEHOST_TASK_IMAGE="namanjain12/scrapy_final:439a3e59b8e858441f8d97dbc32f398db392330d"
export P58_ONEHOST_TIMEOUT_SECONDS="${P58_ONEHOST_TIMEOUT_SECONDS:-1800}"
export P58_ONEHOST_COMPILATION_CACHE_DIR="${P58_ONEHOST_COMPILATION_CACHE_DIR:-/mnt/disks/tunix-data/jax-compilation-cache/p58-q4-tp4-systemopt-b2g2-k2560}"
exec bash "$script_dir/run_onehost_deepswe_seam_probe_docker.sh" "$label"
