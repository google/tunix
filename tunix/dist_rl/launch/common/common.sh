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

# Shared setup and helpers sourced by `launch.sh`, `local.sh`, and `k8s.sh`.

LAUNCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${LAUNCH_DIR}/../../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-python3}}"
if ! command -v "$PYTHON_BIN" &>/dev/null; then
  PYTHON_BIN="python"
fi
export PYTHON_BIN

is_true() {
  [[ "$1" == "1" || "$1" == "true" || "$1" == "True" || "$1" == "yes" || "$1" == "YES" ]]
}

# Resolves YAML + `key=value` arguments via `resolve_config` and evaluates the
# rendered `export` lines. Exits with the resolver's status (its error is on
# stderr) instead of continuing with unset variables.
resolve_config_or_die() {
  local exports
  if [[ -n "${RESOLVE_CONFIG_BIN:-}" ]]; then
    exports="$("${RESOLVE_CONFIG_BIN}" "$@")" || exit $?
  else
    exports="$("$PYTHON_BIN" "${LAUNCH_DIR}/resolve_config.py" "$@")" || exit $?
  fi
  eval "${exports}"
}

# `local.sh` and `k8s.sh` consume the exports from `launch.sh`; fail loudly if
# they are invoked directly.
require_cluster_target() {
  local expected="$1"
  if [[ "${CLUSTER_TARGET:-}" != "${expected}" ]]; then
    echo "Error: $(basename "$0") must be invoked via launch.sh with" \
      "cluster_config.target=${expected} (got CLUSTER_TARGET='${CLUSTER_TARGET:-}')." >&2
    exit 1
  fi
}
