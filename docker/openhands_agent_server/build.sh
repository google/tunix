#!/bin/bash
set -euo pipefail

IMAGE_TAG="${IMAGE_TAG:-0.62}"
IMAGE_NAME="openhands-agent-server:${IMAGE_TAG}"
REGISTRY="${REGISTRY:-us-east1-docker.pkg.dev/cloud-tpu-multipod-dev/tunix}"

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Building ${IMAGE_NAME}..."
docker build -t "${IMAGE_NAME}" -f "${DIR}/Dockerfile" "${DIR}"

echo "Tagging for ${REGISTRY}..."
docker tag "${IMAGE_NAME}" "${REGISTRY}/openhands-agent-server:${IMAGE_TAG}"

echo "Pushing ${REGISTRY}/openhands-agent-server:${IMAGE_TAG}..."
docker push "${REGISTRY}/openhands-agent-server:${IMAGE_TAG}"

echo "Done! Image: ${REGISTRY}/openhands-agent-server:${IMAGE_TAG}"
