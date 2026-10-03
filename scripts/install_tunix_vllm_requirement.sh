#!/bin/bash

# Copyright 2023–2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This script installs the dependencies for running GRPO with MaxText+Tunix+vLLM on TPUs

set -euo pipefail
set -x

# Install uv
if ! command -v uv &> /dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  source $HOME/.local/bin/env
fi

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYPROJECT_TOML=${PYPROJECT_TOML:-"${ROOT_DIR}/pyproject.toml"}

python3 -m ensurepip --default-pip
uv pip install --upgrade pip setuptools wheel setuptools-rust

uv pip install aiohttp==3.12.15

# Install Python packages that enable pip to authenticate with Google Artifact Registry automatically.
uv pip install keyring keyrings.google-artifactregistry-auth

# Extract the pinned `tpu-inference` git requirement from `pyproject.toml` into
# a uv overrides file so `vllm`'s transitive PyPI `tpu-inference` requirement
# resolves to our pinned commit instead of downloading a conflicting release.
OVERRIDES_FILE=$(mktemp)
trap 'rm -f "${OVERRIDES_FILE}"' EXIT
python3 - "${PYPROJECT_TOML}" > "${OVERRIDES_FILE}" <<'PY'
import pathlib, sys, tomllib
data = tomllib.loads(pathlib.Path(sys.argv[1]).read_text())
for req in data.get("dependency-groups", {}).get("tpu-inference", []):
  print(req)
for req in data.get("tool", {}).get("uv", {}).get("override-dependencies", []):
  print(req)
PY

# Clean stale CMake build directories inside cached uv git checkouts so cached
# BuildKit runs do not fail on changed temporary source paths.
rm -rf /root/.cache/uv/git-v0/checkouts/*/*/build \
       /root/.cache/uv/git-v0/checkouts/*/*/.deps 2>/dev/null || true

UV_TPU_ARGS=(
  --extra-index-url https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/
  --find-links https://storage.googleapis.com/jax-releases/libtpu_releases.html
  --index-strategy unsafe-best-match
  --prerelease=allow
  --torch-backend=cpu
  --overrides "${OVERRIDES_FILE}"
)

uv pip install "${UV_TPU_ARGS[@]}" --group "${PYPROJECT_TOML}:tpu-inference"
VLLM_TARGET_DEVICE="tpu" uv pip install "${UV_TPU_ARGS[@]}" \
  --refresh-package vllm \
  --group "${PYPROJECT_TOML}:vllm-tpu"
uv pip install --no-deps "qwix>=0.1.6"
