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

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
REQ_FILE=${REQ_FILE:-"${ROOT_DIR}/requirements/requirements.txt"}
SPECIAL_REQ_FILE=${SPECIAL_REQ_FILE:-"${ROOT_DIR}/requirements/special_requirements.txt"}

python3 -m ensurepip --default-pip
python3 -m pip install --upgrade pip setuptools wheel

pip install aiohttp==3.12.15

# Install Python packages that enable pip to authenticate with Google Artifact Registry automatically.
pip install keyring keyrings.google-artifactregistry-auth

VLLM_TARGET_DEVICE="tpu" python3 -m pip install -r "${REQ_FILE}"
python3 -m pip install -r "${SPECIAL_REQ_FILE}" --force-reinstall
