#!/bin/bash

# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This scripts takes a docker image that already contains the Tunix dependencies, copies the local source code in and
# uploads that image into GCR. Once in GCR the docker image can be used for development.

# Each time you update the base image via a "bash docker_build_dependency_image.sh", there will be a slow upload process
# (minutes). However, if you are simply changing local code and not updating dependencies, uploading just takes a few seconds.

# Script to buid a Tunix base image locally, example cmd is:
# bash docker_build_dependency_image.sh
# bash docker_build_dependency_image.sh --vllm
set -e

# Set default values
VLLM_INSTALL="false"

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --vllm) VLLM_INSTALL="true"; ;; 
        *) echo "Unknown parameter passed: $1"; exit 1 ;; 
    esac
    shift
done

DOCKERFILE=./tunix_dependencies.Dockerfile
BUILD_ARGS=""

if [[ "$VLLM_INSTALL" == "true" ]]; then
    export LOCAL_IMAGE_NAME=tunix_vllm_image
    BUILD_ARGS="--build-arg INSTALL_VLLM=true"
    echo "Building image with vLLM support: $LOCAL_IMAGE_NAME"
else
    export LOCAL_IMAGE_NAME=tunix_base_image
    echo "Building base image: $LOCAL_IMAGE_NAME"
fi

echo "Using Dockerfile: $DOCKERFILE"


# Use Docker BuildKit so we can cache pip packages.
export DOCKER_BUILDKIT=1

echo "Starting to build your docker image. This will take a few minutes but the image can be reused as you iterate."

build_ai_image() {
    if [[ -z ${LOCAL_IMAGE_NAME+x} ]]; then
        echo "Error: LOCAL_IMAGE_NAME is unset, please set it!"
        exit 1
    fi
    COMMIT_HASH=$(git rev-parse --short HEAD)
    echo "Building Tunix Image at commit hash ${COMMIT_HASH}..."

    sudo docker build \
        --network=host \
        ${BUILD_ARGS} \
        -t ${LOCAL_IMAGE_NAME} \
        -f ${DOCKERFILE} .
}

build_ai_image

echo ""
echo "*************************
"

echo "Built your docker image and named it ${LOCAL_IMAGE_NAME}.
It only has the dependencies installed. Assuming you're on a TPUVM, to run the
docker image locally with full TPU access, use the following command:"
echo ""
echo 'sudo docker run -it --rm --net=host --ipc=host --ulimit memlock=-1:-1 -v "$(pwd)":/app --workdir /app --device=/dev/vfio/0 --device=/dev/vfio/1 --device=/dev/vfio/2 --device=/dev/vfio/3 --device=/dev/vfio/4 --device=/dev/vfio/5 --device=/dev/vfio/6 --device=/dev/vfio/7 --device=/dev/vfio/vfio ${LOCAL_IMAGE_NAME} bash'
echo ""
echo "You can run tunix and your development tests inside of the docker image. Changes to your workspace will automatically
be reflected inside the docker container."
echo "Once you want you upload your docker container to GCR, take a look at docker_upload_runner.sh"

