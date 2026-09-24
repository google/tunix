#!/usr/bin/env bash
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

# Installs DeepSWE evaluation, OpenHands SDK, Kubernetes Agent Sandbox, and
# R2E-Gym dependencies for the `examples/deepswe/` recipe (used when
# `INSTALL_DEEPSWE_DEPS=true` in `Dockerfile` or locally).
#
# Usage:
#   bash examples/deepswe/install_deepswe.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

command -v uv >/dev/null 2>&1 || python3 -m pip install --upgrade uv

echo "Installing DeepSWE dependencies from pyproject.toml ([deepswe] + group deepswe-git)..."
uv pip install --directory "${REPO_ROOT}" -e ".[deepswe]"
uv pip install --directory "${REPO_ROOT}" --no-deps --group deepswe-git

# TODO(tunix-dev): Upstream the deprecated `HfFolder` import removal and
# `ParsedCommit` dict/JSON handling to `r2e-gym/r2e-gym` (or pin a patched
# commit) so we can remove these post-install `sed` monkeypatches.
SITE_PACKAGES="$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
if [[ -f "${SITE_PACKAGES}/r2egym/agenthub/utils/utils.py" ]]; then
  sed -i 's/create_repo, upload_folder, HfFolder/create_repo, upload_folder/' \
    "${SITE_PACKAGES}/r2egym/agenthub/utils/utils.py"
fi
if [[ -f "${SITE_PACKAGES}/r2egym/agenthub/runtime/docker.py" ]]; then
  sed -i 's/self.commit = ParsedCommit(\*\*json.loads(self.commit_json))/self.commit = ParsedCommit(\*\*(json.loads(self.commit_json) if isinstance(self.commit_json, str) else self.commit_json))/' \
    "${SITE_PACKAGES}/r2egym/agenthub/runtime/docker.py"
fi

echo "DeepSWE dependencies installed successfully."
