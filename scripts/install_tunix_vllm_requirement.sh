# !/bin/bash

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

set -e
set -x

python -m ensurepip --default-pip

pip uninstall -y jax jaxlib libtpu

pip install aiohttp==3.12.15

# Install Python packages that enable pip to authenticate with Google Artifact Registry automatically.
pip install keyring keyrings.google-artifactregistry-auth

# Install vLLM for Jax and TPUs from GitHub repository
git clone https://github.com/vllm-project/vllm.git /tmp/vllm
cd /tmp/vllm
VLLM_TARGET_DEVICE="tpu" pip install -e .

# Install tpu-commons from the artifact registry
pip install --no-cache-dir --pre \
    --index-url https://us-python.pkg.dev/cloud-tpu-images/maxtext-rl/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    --extra-index-url https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/ \
    --find-links https://storage.googleapis.com/jax-releases/libtpu_releases.html \
    tpu-commons==0.1.1

pip install numba==0.61.2
