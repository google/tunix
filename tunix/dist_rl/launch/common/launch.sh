#!/bin/bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Unified distributed RL launcher entry point.
#
# Resolves YAML + CLI key=value configuration via `resolve_config.py` and
# dispatches to `local.sh` (when `cluster_config.target=local`) or `k8s.sh`
# (when `cluster_config.target=k8s`). This is the only script that parses
# command-line arguments.
#
# Usage:
#   bash launch.sh [start|stop|orchestrator|trainer|rollout] [--dry-run] \
#     [override_config_file=<config.yaml> ...] [section.key=value ...]

set -Eeuo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

usage() {
  echo "Usage: $0 [start|stop|orchestrator|trainer|rollout] [--dry-run]" \
    "[override_config_file=<config.yaml> ...] [section.key=value ...]"
}

COMMAND="start"
CONFIG_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    start|stop|orchestrator|trainer|rollout)
      COMMAND="$1"
      shift
      ;;
    --command)
      COMMAND="$2"
      shift 2
      ;;
    --command=*)
      COMMAND="${1#*=}"
      shift
      ;;
    --dry-run|--render)
      CONFIG_ARGS+=("cluster_config.dry_run=true")
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      CONFIG_ARGS+=("$1")
      shift
      ;;
  esac
done

resolve_config_or_die "${CONFIG_ARGS[@]}"

case "${CLUSTER_TARGET}" in
  local)
    exec "${LAUNCH_DIR}/local.sh" "${CONFIG_ARGS[@]}"
    ;;
  k8s)
    exec "${LAUNCH_DIR}/k8s.sh" "${COMMAND}" "${CONFIG_ARGS[@]}"
    ;;
  *)
    echo "Error: Unsupported CLUSTER_TARGET='${CLUSTER_TARGET}' (expected 'local' or 'k8s')." >&2
    exit 1
    ;;
esac
