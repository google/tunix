#!/bin/bash
# Builds and pushes the Qwen3.5-35B-A3B GRPO overlay image.
#
#   bash docker/maz-q35/build_push.sh [TAG]
#
# Env overrides: BASE_IMAGE, IMAGE_REPO, PUSH=false, DOCKER_CMD.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BASE_IMAGE="${BASE_IMAGE:-gcr.io/cloud-tpu-multipod-dev/yixuannwang_google_com-runner:yixuann-e2e-0912head-v8}"
IMAGE_REPO="${IMAGE_REPO:-gcr.io/cloud-tpu-multipod-dev/mazumdera-runner}"
TAG="${1:-q35-$(date +%m%d)-v1}"
TARGET_IMAGE="${IMAGE_REPO}:${TAG}"
PUSH="${PUSH:-true}"

DOCKER_CMD="${DOCKER_CMD:-docker}"
if ! ${DOCKER_CMD} info >/dev/null 2>&1; then
  if sudo -n docker info >/dev/null 2>&1; then
    DOCKER_CMD="sudo docker"
  else
    echo "ERROR: docker is not usable from this account and passwordless sudo is" >&2
    echo "       unavailable. Re-run under sudo, or add yourself to the docker group." >&2
    exit 1
  fi
fi

echo "=== Building ${TARGET_IMAGE}"
echo "    base: ${BASE_IMAGE}"
DOCKER_BUILDKIT=1 ${DOCKER_CMD} build \
  --network=host \
  --build-arg BASE_IMAGE="${BASE_IMAGE}" \
  -t "${TARGET_IMAGE}" \
  -f "${SCRIPT_DIR}/Dockerfile" \
  "${SCRIPT_DIR}"

# The Dockerfile's own grep proves the patch landed in the layer; this proves the image
# that came out of the build is the one that will be pulled, and that maxtext still
# imports -- a syntactically valid file that fails at import time would otherwise only
# surface as a crash-looping trainer pod.
echo "=== Verifying the built image"
${DOCKER_CMD} run --rm --entrypoint bash "${TARGET_IMAGE}" -c '
  set -e
  f=/app/maxtext/src/maxtext/training_engine/maxtext_engine.py
  grep -q "dataclasses.replace(payload, metadata={})" "$f"
  python3 -c "
from maxtext.training_engine import maxtext_engine
import inspect
assert \"metadata={}\" in inspect.getsource(maxtext_engine.MaxTextTrainingEngine._prepare_batch)
print(\"ok: _prepare_batch clears payload metadata\")
"'

if [ "${PUSH}" != "true" ]; then
  echo "=== PUSH=${PUSH}, stopping after build: ${TARGET_IMAGE}"
  exit 0
fi

echo "=== Pushing ${TARGET_IMAGE}"
gcloud auth configure-docker gcr.io --quiet
${DOCKER_CMD} push "${TARGET_IMAGE}"

# Pin the manifests to the digest: the tag is mutable, and the orchestrator, trainer and
# rollout pods have to be running identical code for weight sync to mean anything.
DIGEST="$(${DOCKER_CMD} inspect --format='{{if .RepoDigests}}{{index .RepoDigests 0}}{{end}}' "${TARGET_IMAGE}")"
echo "=== Pushed"
echo "    ${TARGET_IMAGE}"
echo "    ${DIGEST}"
