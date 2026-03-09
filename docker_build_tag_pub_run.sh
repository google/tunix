#!/bin/bash

# Exit on error
set -e
# Print commands for debugging (optional, but helpful)
set -x

# These variables will respect existing environment variables, otherwise they use the default values
ENGINE="${ENGINE:-none}"
LOCAL_IMAGE_NAME="${LOCAL_IMAGE_NAME:-tunix_base_image}"
TAG="${TAG:-$(date +%Y%m%d_%H%M%S)}"
PROJECT_ID="${PROJECT_ID:-cloud-tpu-multipod-dev}"

# Parse command line arguments
# Note: Command line flags will take the highest precedence and override environment variables.
while [[ $# -gt 0 ]]; do
  case $1 in
    --engine=*)
      ENGINE="${1#*=}"
      shift
      ;;
    --local_image_name=*)
      LOCAL_IMAGE_NAME="${1#*=}"
      shift
      ;;
    --tag=*)
      TAG="${1#*=}"
      shift
      ;;
    --project_id=*)
      PROJECT_ID="${1#*=}"
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

./build_docker.sh --engine="${ENGINE}" --tag="${TAG}" --local_image_name="${LOCAL_IMAGE_NAME}"

docker tag "${LOCAL_IMAGE_NAME}:${TAG}" "europe-west4-docker.pkg.dev/${PROJECT_ID}/linchai-repo/${LOCAL_IMAGE_NAME}:${TAG}"

docker push "europe-west4-docker.pkg.dev/${PROJECT_ID}/linchai-repo/${LOCAL_IMAGE_NAME}:${TAG}"

