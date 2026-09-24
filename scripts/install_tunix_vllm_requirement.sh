#!/usr/bin/env bash
# Copyright 2025 Google LLC
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

# Installs the pinned vLLM and TPU-Inference commits for TPU rollout, and (when
# `pyproject.toml` is present in the repository root) installs Tunix's
# declarative TPU runtime dependencies (`[tpu,cli,test,experimental]`).
#
# Environment variable overrides:
#   - `VLLM_COMMIT`: Override the pinned `vllm-project/vllm` commit.
#   - `TPU_INFERENCE_COMMIT`: Override the pinned `vllm-project/tpu-inference` commit.

set -euo pipefail

VLLM_COMMIT="${VLLM_COMMIT:-51da0ca66c8065619c79e35dff97aa99aeaf5644}"
TPU_INFERENCE_COMMIT="${TPU_INFERENCE_COMMIT:-6f913c734cb47a1125a817e37cc418c5960675d0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

command -v uv >/dev/null 2>&1 || python3 -m pip install --upgrade uv
unset PIP_NO_CACHE_DIR

export VLLM_TARGET_DEVICE="${VLLM_TARGET_DEVICE:-tpu}"
VLLM_SPEC="vllm @ git+https://github.com/vllm-project/vllm.git@${VLLM_COMMIT}"
TPU_INFERENCE_SPEC="tpu-inference @ git+https://github.com/vllm-project/tpu-inference.git@${TPU_INFERENCE_COMMIT}"
JAX_SMI_SPEC="jax-smi @ git+https://github.com/ayaka14732/jax-smi.git"

echo "Installing vLLM (${VLLM_COMMIT}) and TPU-Inference (${TPU_INFERENCE_COMMIT})..."
rm -rf /root/.cache/uv/git-v0/checkouts/*/*/build /root/.cache/uv/git-v0/checkouts/*/*/.deps 2>/dev/null || true

OVERRIDE_FILE="$(mktemp)"
trap 'rm -f "${OVERRIDE_FILE}"' EXIT
printf '%s\nflax>=0.12.5\nnumpy==2.3.5\nprotobuf>=7.35.1\n' "${TPU_INFERENCE_SPEC}" > "${OVERRIDE_FILE}"

uv pip install \
  --refresh-package vllm \
  "${VLLM_SPEC}" \
  "${TPU_INFERENCE_SPEC}" \
  "${JAX_SMI_SPEC}" \
  --overrides "${OVERRIDE_FILE}" \
  --torch-backend=cpu

if [[ -f "${REPO_ROOT}/pyproject.toml" ]]; then
  echo "Installing Tunix declarative dependencies (pyproject.toml [tpu,cli,test,experimental])..."
  uv pip install --directory "${REPO_ROOT}" -e ".[tpu,cli,test,experimental]"
fi
