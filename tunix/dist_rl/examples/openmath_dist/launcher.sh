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

# Launches the distributed OpenMathInstruct-2 GRPO recipe on a single TPU host
# (`openmath.yaml` on top of `launch/common/base_dist_config.yaml`).
#
# Usage:
#   bash launcher.sh [--dry-run] [section.key=value ...]

set -Eeuo pipefail

readonly DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Not `ARTIFACT_ROOT`: `launch.sh` rejects that legacy name when exported.
artifact_root="$(cd "${DIR}/../../../.." && pwd)/artifacts/openmath_qwen3_0p6b"

exec "${DIR}/../../launch/common/launch.sh" \
  "override_config_file=${DIR}/openmath.yaml" \
  "cluster_config.artifact_root=${artifact_root}" \
  "actor_model_config.model_path=${artifact_root}/maxtext_models/qwen3-0.6b/0/items" \
  "$@"
