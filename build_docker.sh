# This scripts takes a docker image that already contains the GRL dependencies, copies the local source code in and
# uploads that image into GCR. Once in GCR the docker image can be used for development.

# Each time you update the base image via a "bash docker_build_dependency_image.sh", there will be a slow upload process
# (minutes). However, if you are simply changing local code and not updating dependencies, uploading just takes a few seconds.

# Script to buid a GRL base image locally, example cmd is:
# bash build_docker.sh

set -e

INSTALL_DEEPSWE_DEPS=false
TPU_SYNC_WHEEL_GCS_PATH=${TPU_SYNC_WHEEL_GCS_PATH:-}
TPU_SYNC_WHEEL_LOCAL_DIR=.docker/tpu_sync

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --deepswe) INSTALL_DEEPSWE_DEPS=true; shift ;;
        --tpu-sync-wheel) TPU_SYNC_WHEEL_GCS_PATH="$2"; shift 2 ;;
        --tpu-sync-wheel=*) TPU_SYNC_WHEEL_GCS_PATH="${1#*=}"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
done

DOCKERFILE=./Dockerfile

if [ ! -f "$DOCKERFILE" ]; then
    echo "Error: Dockerfile not found at $DOCKERFILE"
    exit 1
fi

export LOCAL_IMAGE_NAME=${LOCAL_IMAGE_NAME:-tunix_base_image}
export LOCAL_IMAGE_TAG=${LOCAL_IMAGE_TAG:-lance-demo-0830}
LOCAL_IMAGE_REF="${LOCAL_IMAGE_NAME}:${LOCAL_IMAGE_TAG}"
echo "Building base image: $LOCAL_IMAGE_REF"

echo "Using Dockerfile: $DOCKERFILE"

# Use Docker BuildKit so we can cache pip packages.
export DOCKER_BUILDKIT=1

echo "Starting to build your docker image. This will take a few minutes but the image can be reused as you iterate."

stage_tpu_sync_wheel() {
    mkdir -p "${TPU_SYNC_WHEEL_LOCAL_DIR}"
    find "${TPU_SYNC_WHEEL_LOCAL_DIR}" -maxdepth 1 -type f -name '*.whl' -delete

    if [[ -z "${TPU_SYNC_WHEEL_GCS_PATH}" ]]; then
        return
    fi

    local wheel_name
    wheel_name=$(basename "${TPU_SYNC_WHEEL_GCS_PATH}")
    local wheel_dest="${TPU_SYNC_WHEEL_LOCAL_DIR}/${wheel_name}"
    echo "Staging TPU sync wheel from ${TPU_SYNC_WHEEL_GCS_PATH} to ${wheel_dest}"

    if command -v gcloud >/dev/null 2>&1; then
        gcloud storage cp "${TPU_SYNC_WHEEL_GCS_PATH}" "${wheel_dest}"
    elif command -v gsutil >/dev/null 2>&1; then
        gsutil cp "${TPU_SYNC_WHEEL_GCS_PATH}" "${wheel_dest}"
    else
        cat <<'MSG'
No GCS CLI found to download the TPU sync wheel.

Install gcloud or gsutil, or manually place the wheel under .docker/tpu_sync/ before building.
MSG
        exit 1
    fi
}

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
        --build-arg INSTALL_DEEPSWE_DEPS=${INSTALL_DEEPSWE_DEPS} \
        -t ${LOCAL_IMAGE_REF} \
        -f ${DOCKERFILE} .
}

stage_tpu_sync_wheel
build_ai_image

echo ""
echo "*************************
"

echo "Built your docker image and named it ${LOCAL_IMAGE_REF}.
It now installs Tunix and the pinned vLLM and tpu-inference dependencies from requirements/requirements.txt. "
