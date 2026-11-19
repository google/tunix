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

# Builds the experimental DeepSWE distributed RL base Docker image with pinned
# runtime constraints and optional MaxText / tpu-inference commit overrides.
# Example command (runnable from the repository root or any directory):
#   bash tunix/experimental/examples/deepswe_dist/build_docker.sh \
#     --maxtext --raiden --deepswe --maxtext-commit=a2cf3bc8b
#
# TODO(tunix-dev): [legacy-maxtext-a2cf3bc8b] Retire this experimental script
# once main's root Dockerfile / build_docker.sh upgrades its dependency stack
# and commit overrides are no longer required for the 35B DeepSWE recipe.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "${SCRIPT_DIR}/../../../../pyproject.toml" ]; then
    REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
else
    REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null || (cd "${SCRIPT_DIR}/../../../.." && pwd))"
fi

INSTALL_MAXTEXT=false
INSTALL_RAIDEN=false
RAIDEN_WHEEL_DIR=/app/raiden_wheels
INSTALL_DEEPSWE_DEPS=false
TPU_INFERENCE_COMMIT=""
MAXTEXT_COMMIT=""

usage() {
    cat <<'MSG'
Usage: bash tunix/experimental/examples/deepswe_dist/build_docker.sh [--maxtext] [--raiden] [--raiden-wheel-dir PATH] [--deepswe] [--tpu-inference-commit SHA] [--maxtext-commit SHA]

Options:
  --maxtext                   Install MaxText-specific dependencies.
  --raiden                    Install Raiden-specific dependencies.
  --raiden-wheel-dir PATH     Use prebuilt Raiden wheels from PATH (also accepts --raiden-wheel-dir=PATH).
  --deepswe                   Install DeepSWE evaluation dependencies.
  --tpu-inference-commit SHA  Reinstall tpu-inference at a 7-40 hex commit SHA with --no-deps
                              (also accepts --tpu-inference-commit=SHA; full 40-char SHA recommended;
                              commit must be compatible with the pinned vLLM/JAX/libtpu stack).
  --maxtext-commit SHA        Reinstall maxtext and maxtext-vllm-adapter at a 7-40 hex commit SHA
                              with --no-deps, even without --maxtext (also accepts --maxtext-commit=SHA;
                              full 40-char SHA recommended; maxtext-vllm-adapter registers a vLLM plugin).
MSG
}

die_usage() {
    echo "Error: $1" >&2
    usage >&2
    exit 1
}

require_path_arg() {
    local flag="$1"
    local val="$2"
    if [[ -z "$val" || "$val" == -* ]]; then
        die_usage "${flag} requires a non-empty path"
    fi
}

require_commit_arg() {
    local flag="$1"
    local val="$2"
    if [[ ! "$val" =~ ^[0-9a-fA-F]{7,40}$ ]]; then
        die_usage "${flag} requires a 7-40 character hex commit SHA"
    fi
}

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --maxtext) INSTALL_MAXTEXT=true; shift ;;
        --raiden) INSTALL_RAIDEN=true; shift ;;
        --raiden-wheel-dir)
            [[ $# -ge 2 ]] || die_usage "--raiden-wheel-dir requires a non-empty path"
            require_path_arg "--raiden-wheel-dir" "$2"
            RAIDEN_WHEEL_DIR="$2"
            shift 2
            ;;
        --raiden-wheel-dir=*)
            RAIDEN_WHEEL_DIR="${1#*=}"
            require_path_arg "--raiden-wheel-dir" "$RAIDEN_WHEEL_DIR"
            shift
            ;;
        --deepswe) INSTALL_DEEPSWE_DEPS=true; shift ;;
        --tpu-inference-commit)
            [[ $# -ge 2 ]] || die_usage "--tpu-inference-commit requires a 7-40 character hex commit SHA"
            require_commit_arg "--tpu-inference-commit" "$2"
            TPU_INFERENCE_COMMIT="$2"
            shift 2
            ;;
        --tpu-inference-commit=*)
            TPU_INFERENCE_COMMIT="${1#*=}"
            require_commit_arg "--tpu-inference-commit" "$TPU_INFERENCE_COMMIT"
            shift
            ;;
        --maxtext-commit)
            [[ $# -ge 2 ]] || die_usage "--maxtext-commit requires a 7-40 character hex commit SHA"
            require_commit_arg "--maxtext-commit" "$2"
            MAXTEXT_COMMIT="$2"
            shift 2
            ;;
        --maxtext-commit=*)
            MAXTEXT_COMMIT="${1#*=}"
            require_commit_arg "--maxtext-commit" "$MAXTEXT_COMMIT"
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *) die_usage "Unknown parameter: $1" ;;
    esac
done

DOCKERFILE="${SCRIPT_DIR}/Dockerfile"

if [ ! -f "$DOCKERFILE" ]; then
    echo "Error: Dockerfile not found at $DOCKERFILE" >&2
    exit 1
fi

if [ ! -f "${REPO_ROOT}/pyproject.toml" ]; then
    echo "Error: Tunix repository root (pyproject.toml) not found at ${REPO_ROOT}" >&2
    exit 1
fi

export LOCAL_IMAGE_NAME="${LOCAL_IMAGE_NAME:-tunix_base_image}"
echo "Building base image: $LOCAL_IMAGE_NAME"
echo "Using Dockerfile: $DOCKERFILE"
echo "Using Docker build context: $REPO_ROOT"

# Use Docker BuildKit so we can cache pip packages.
export DOCKER_BUILDKIT=1

echo "Starting to build your docker image. This will take a few minutes but the image can be reused as you iterate."

build_ai_image() {
    COMMIT_HASH=$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo "unknown")
    echo "Building Tunix Image at commit hash ${COMMIT_HASH}..."

    DOCKER_COMMAND="docker"
    if docker info >/dev/null 2>&1; then
        DOCKER_COMMAND="docker"
    else
        # Avoid invoking sudo interactively which can prompt for a password.
        # Check whether non-interactive sudo would work (no password).
        if sudo -n docker info >/dev/null 2>&1; then
            DOCKER_COMMAND="sudo docker"
        else
            cat <<'MSG'
Docker does not appear usable from this account and the build would prompt for a password.

Run the build with sufficient privileges (will prompt): sudo bash tunix/experimental/examples/deepswe_dist/build_docker.sh
On Linux, add your user to the docker group so sudo isn't required (you must re-login):
  sudo usermod -aG docker "$USER" && newgrp docker

MSG
            exit 1
        fi
    fi

    $DOCKER_COMMAND build \
        --network=host \
        --build-arg INSTALL_MAXTEXT="${INSTALL_MAXTEXT}" \
        --build-arg INSTALL_RAIDEN="${INSTALL_RAIDEN}" \
        --build-arg RAIDEN_WHEEL_DIR="${RAIDEN_WHEEL_DIR}" \
        --build-arg INSTALL_DEEPSWE_DEPS="${INSTALL_DEEPSWE_DEPS}" \
        --build-arg TPU_INFERENCE_COMMIT="${TPU_INFERENCE_COMMIT}" \
        --build-arg MAXTEXT_COMMIT="${MAXTEXT_COMMIT}" \
        -t "${LOCAL_IMAGE_NAME}" \
        -f "${DOCKERFILE}" "${REPO_ROOT}"
}

build_ai_image

echo ""
echo "*************************
"

echo "Built your docker image and named it ${LOCAL_IMAGE_NAME}.
It installs Tunix alongside dependencies from requirements/requirements.txt and requirements/special_requirements.txt (plus any requested commit overrides)."
