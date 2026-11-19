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

# Launches the distributed DeepSWE GRPO recipe on a single TPU host
# (`deepswe.yaml` on top of `launch/common/base_dist_config.yaml`).
# The local target runs `start` only, which is the default command.
#
# Usage:
#   bash launcher_v2.sh [--dry-run] [section.key=value ...]

set -Eeuo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${DIR}/../../../.." && pwd)"

# Checkpoints and metrics go under `cluster_config.artifact_root`
# (`artifacts/deepswe`); keep the dataset download there too.
exec "${DIR}/../../launch/common/launch.sh" \
  "override_config_file=${DIR}/deepswe.yaml" \
  "data_config.cache_dir=${REPO_ROOT}/artifacts/deepswe/dataset_cache" \
  "$@"
