# This scripts takes a docker image that already contains the GRL dependencies, copies the local source code in and
# uploads that image into GCR. Once in GCR the docker image can be used for development.

# Each time you update the base image via a "bash docker_build_dependency_image.sh", there will be a slow upload process
# (minutes). However, if you are simply changing local code and not updating dependencies, uploading just takes a few seconds.

# Script to buid a GRL base image locally, example cmd is:
# bash build_docker.sh

set -e

DOCKERFILE=./Dockerfile

if [ ! -f "$DOCKERFILE" ]; then
    echo "Error: Dockerfile not found at $DOCKERFILE"
    exit 1
fi

export LOCAL_IMAGE_NAME=tunix_base_image
echo "Building base image: $LOCAL_IMAGE_NAME"

echo "Using Dockerfile: $DOCKERFILE"

# Use Docker BuildKit so we can cache pip packages.
export DOCKER_BUILDKIT=1

echo "Starting to build your docker image. This will take a few minutes but the image can be reused as you iterate."

build_ai_image() {
    COMMIT_HASH=$(git rev-parse --short HEAD)
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

Run the build with sufficient privileges (will prompt): sudo bash build_docker.sh
On Linux, add your user to the docker group so sudo isn't required (you must re-login):
  sudo usermod -aG docker "$USER" && newgrp docker

MSG
            exit 1
        fi
    fi

    $DOCKER_COMMAND build \
        --network=host \
        -t ${LOCAL_IMAGE_NAME} \
        -f ${DOCKERFILE} .
}

build_ai_image

echo ""
echo "*************************
"

echo "Built your docker image and named it ${LOCAL_IMAGE_NAME}.
It now installs Tunix and the pinned vLLM and tpu-inference dependencies from requirements/requirements.txt. "
