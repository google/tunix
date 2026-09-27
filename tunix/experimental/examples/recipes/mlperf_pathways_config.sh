#!/bin/bash
# ==============================================================================
# Pathways Shared Configuration for MLPerf Recipes
# ==============================================================================

export PATHWAYS_SERVER_IMAGE="${PATHWAYS_SERVER_IMAGE:-us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/datenglin/unsanitized_server:raiden_988370191}"
export PATHWAYS_PROXY_IMAGE="${PATHWAYS_PROXY_IMAGE:-us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/datenglin/unsanitized_proxy_server:raiden_988370191}"
export PATHWAYS_PROXY_MEMORY_LIMIT="${PATHWAYS_PROXY_MEMORY_LIMIT:-160G}"
export USER_CONTAINER_MEMORY="${USER_CONTAINER_MEMORY:-260G}"
export USER_CONTAINER_MEMORY_LIMIT="${USER_CONTAINER_MEMORY_LIMIT:-260G}"
