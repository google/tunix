#!/usr/bin/env bash
set -euo pipefail

label="${1:?usage: run_onehost_deepswe_zero_p78_actor_logps_docker.sh <unique-label>}"
script_dir="$(cd "$(dirname "$0")" && pwd)"
export CANON_P78_SEGMENTED_ACTOR_LOGPS=1
exec bash "$script_dir/run_onehost_deepswe_zero_trajectory_replay_docker.sh" "$label"
