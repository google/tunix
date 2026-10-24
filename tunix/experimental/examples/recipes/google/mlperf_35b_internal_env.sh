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

# Internal cluster, storage, and image defaults for the 35B DeepSWE recipe.
# This file resides under a `google/` directory so Copybara (`exclude = ["**/google/**"]`)
# automatically omits it when exporting to GitHub.

export CLUSTER_NAME="${CLUSTER_NAME:-bodaborg-v5p-nap}"
export CLUSTER="${CLUSTER:-${CLUSTER_NAME}}"
export ZONE="${ZONE:-europe-west4-b}"
export REGION="${REGION:-europe-west4}"
export LOCATION_NAME="${LOCATION_NAME:-${ZONE}}"
export PROJECT="${PROJECT:-cloud-tpu-multipod-dev}"
export NAMESPACE="${NAMESPACE:-trellis}"
export K8S_NAMESPACE="${K8S_NAMESPACE:-${NAMESPACE}}"
export SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-xpk-sa}"
export PRIORITY_CLASS="${PRIORITY_CLASS:-medium}"
export CPU_NODEPOOL="${CPU_NODEPOOL:-cpu-np}"

export NODE_SELECTOR_KEY="${NODE_SELECTOR_KEY:-cloud.google.com/gke-nodepool}"
export NODE_SELECTOR_VAL="${NODE_SELECTOR_VAL:-sandbox-cpu-pool}"
export SANDBOX_NODE_SELECTOR_KEY="${SANDBOX_NODE_SELECTOR_KEY:-${NODE_SELECTOR_KEY}}"
export SANDBOX_NODE_SELECTOR_VAL="${SANDBOX_NODE_SELECTOR_VAL:-${NODE_SELECTOR_VAL}}"
export IMAGE_REWRITE_PREFIX="${IMAGE_REWRITE_PREFIX:-europe-west4-docker.pkg.dev/cloud-tpu-multipod-dev/tunix/}"
export OPENHANDS_SERVER_IMAGE="${OPENHANDS_SERVER_IMAGE:-${IMAGE_REWRITE_PREFIX%/}/openhands-agent-server:0.62-chardet}"

export PATHWAYS_SERVER_IMAGE="${PATHWAYS_SERVER_IMAGE:-us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/datenglin/unsanitized_server:raiden_20261014183000}"
export PATHWAYS_PROXY_IMAGE="${PATHWAYS_PROXY_IMAGE:-us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/datenglin/unsanitized_proxy_server:raiden_20261014183000}"

export BUCKET="${BUCKET:-gs://atwigg-trellis-europe-west4-dev}"
export MAXTEXT_CKPT="${MAXTEXT_CKPT:-gs://hengtaoguo-maxtext-logs/checkpoints/qwen3.5-35b-a3b/scanned/2026-06-11-10-27/0/items}"
export DATASET_PATH="${DATASET_PATH:-gs://mlperf_dataset/benchmark-r2e-gym-easy}"
