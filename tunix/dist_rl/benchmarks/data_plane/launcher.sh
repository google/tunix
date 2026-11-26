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

# Launches the mock MoE routed-experts data-plane benchmark locally on one host.
#
# Usage:
#   bash launcher.sh [--dry-run] [section.key=value ...]

set -Eeuo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${DIR}/../../../.." && pwd)"

cd "${REPO_ROOT}"
exec "${DIR}/../../launch/common/launch.sh" \
  "override_config_file=${DIR}/data_plane_benchmark.yaml" \
  "$@"
