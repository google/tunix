#!/bin/bash
# Build and verify the FrozenLake TPU runtime image
# (experimental/Dockerfile.frozenlake), then optionally push it.
#
# The stack is pinned by a single package: vllm-tpu==X pulls tpu-inference==X,
# which pins jax, jaxlib and libtpu exactly. Known-good rows:
#
#   VLLM_TPU_VERSION   tpu-inference   jax/jaxlib   libtpu
#   0.25.0             0.25.0          0.10.2       0.0.42.1     <- default
#   0.23.0             0.23.0          0.10.1       0.0.41
#
# Verification runs in three levels, because each catches something the
# previous one cannot:
#   L1  in the build  -- versions and a single vllm. Fails the build.
#   L2  --privileged  -- jax.devices(). The ONLY check that catches an ABI
#                        mismatch, since libtpu resolves its symbols on load.
#   L3  a real run    -- experimental/train_frozenlake_v5p_1host_docker.sh with
#                        NUM_BATCHES=2. Catches whether vLLM can actually serve.
# L1 and L2 run here; L3 is a separate command (printed at the end).
#
# Usage (on the TPU VM, from the tunix repo root):
#   bash experimental/build_frozenlake_image.sh              # build + L1 + L2
#   PROBE=1 bash experimental/build_frozenlake_image.sh      # resolve only, no build
#   NO_LOCK=1 bash experimental/build_frozenlake_image.sh    # ignore the lock and re-resolve
#   VLLM_TPU_VERSION=0.23.0 NO_LOCK=1 bash experimental/build_frozenlake_image.sh
#
# By default the build installs experimental/requirements.frozenlake.lock.txt, a
# freeze of an environment that passed all three levels, so another machine gets
# the same 275 packages rather than whatever PyPI resolved to that day. Bumping
# VLLM_TPU_VERSION therefore needs NO_LOCK=1, after which the lock should be
# re-frozen from the image the new resolution produced:
#   docker run --rm <image> pip freeze --exclude-editable \
#     > experimental/requirements.frozenlake.lock.txt
#
# Nothing here needs a registry -- the recipe travels as source. PUSH=1 (with
# IMAGE_REPO set to a registry path) is available if you additionally want a
# prebuilt copy somewhere, and needs `gcloud auth login && gcloud auth
# configure-docker <host>` first.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TUNIX_DIR="${TUNIX_DIR:-$(dirname "$SCRIPT_DIR")}"

VLLM_TPU_VERSION="${VLLM_TPU_VERSION:-0.25.0}"
# A name of its own, so tunix_base_image:latest stays untouched -- the other
# five docker wrappers run on it.
# A bare local name: the recipe travels as source, not as a 12GB artifact, so
# nothing here needs a registry. Set IMAGE_REPO to a registry path only if you
# additionally want to PUSH=1 it somewhere.
IMAGE_REPO="${IMAGE_REPO:-tunix_frozenlake_image}"
TAG="${TAG:-vllm-tpu${VLLM_TPU_VERSION}}"
IMAGE="${IMAGE:-${IMAGE_REPO}:${TAG}}"
EXPECT_DEVICES="${EXPECT_DEVICES:-4}"   # v5p-8 = 4 chips
BASE_IMAGE="${BASE_IMAGE:-python:3.12-slim}"

cd "$TUNIX_DIR"

DOCKER="docker"
docker info >/dev/null 2>&1 || DOCKER="sudo docker"

# ---------------------------------------------------------------------------
# PROBE: resolve the dependency graph without building, to see what a given
# vllm-tpu would actually drag in.
# ---------------------------------------------------------------------------
if [ -n "${PROBE:-}" ]; then
  echo "===== PROBE: resolving vllm-tpu==${VLLM_TPU_VERSION} (no build) ====="
  $DOCKER run --rm "$BASE_IMAGE" bash -c "
    pip install -q --upgrade pip >/dev/null 2>&1
    pip install --dry-run --no-cache-dir 'vllm-tpu==${VLLM_TPU_VERSION}' 2>&1 \
      | grep -Ei 'would install|error|conflict' \
      | tr ' ' '\n' | grep -Ei '^(jax|jaxlib|libtpu|tpu[-_]inference|vllm[-_]tpu|torch)-' \
      || echo '(pip printed no Would-install line; see full output above)'
  "
  exit $?
fi

# ---------------------------------------------------------------------------
# Build (L1 asserts inside the final layer).
# ---------------------------------------------------------------------------
echo "===== BUILD ${IMAGE}  (vllm-tpu ${VLLM_TPU_VERSION}) ====="
$DOCKER build \
  --network=host \
  -f experimental/Dockerfile.frozenlake \
  --build-arg "VLLM_TPU_VERSION=${VLLM_TPU_VERSION}" \
  --build-arg "NO_LOCK=${NO_LOCK:-}" \
  -t "$IMAGE" \
  . || { echo "BUILD FAILED (L1 asserts inside the build; read above)"; exit 1; }

# ---------------------------------------------------------------------------
# L2: the ABI check. Needs the chips, so it cannot live in the build.
# ---------------------------------------------------------------------------
echo
echo "===== L2: TPU init (${EXPECT_DEVICES} chips expected) ====="
$DOCKER run --rm --privileged --net=host "$IMAGE" \
  python3 experimental/verify_tpu_stack.py \
    --vllm-tpu "$VLLM_TPU_VERSION" --devices "$EXPECT_DEVICES"
l2=$?
if [ $l2 -ne 0 ]; then
  echo "L2 FAILED -- the versions are right but the stack does not initialise"
  echo "the TPU. Do NOT push this image."
  exit $l2
fi

echo
echo "################ NEXT ################"
echo "L1 (versions) and L2 (TPU init) passed for:"
echo "  $IMAGE"
echo
echo "L3 -- a real short run, still required before this image is trusted:"
echo "  NUM_BATCHES=2 RUN_TAG=fl_smoke IMAGE=$IMAGE \\"
echo "    bash experimental/train_frozenlake_v5p_1host_docker.sh"
echo
if [ -n "${PUSH:-}" ]; then
  echo "===== PUSH (requested) ====="
  $DOCKER push "$IMAGE" || { echo "push failed -- see the auth note in the header"; exit 1; }
  echo "pushed $IMAGE"
else
  echo "Not pushing. Once L3 passes: PUSH=1 IMAGE=$IMAGE bash $0"
fi
