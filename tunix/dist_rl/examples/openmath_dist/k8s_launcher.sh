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

# Launches the distributed OpenMathInstruct-2 GRPO recipe on GKE (JobSet),
# layering the GKE cluster and mesh overrides below on top of `openmath.yaml`.
#
# Usage:
#   bash k8s_launcher.sh [start|stop|orchestrator|trainer|rollout|otel|stop-otel] \
#     [--dry-run] \
#     cluster_config.k8s.tunix_image=<image> \
#     actor_model_config.model_path=gs://<bucket>/checkpoints/qwen3-0.6b/0/items \
#     [section.key=value ...]

set -Eeuo pipefail

readonly DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TUNIX_IMAGE="${TUNIX_IMAGE:-}"

exec "${DIR}/../../launch/common/launch.sh" \
  "override_config_file=${DIR}/openmath.yaml" \
  model_config.model_download_path="/dev/shm/models/Qwen3-0.6B" \
  actor_model_config.mesh.shape="(16,1,1)" \
  actor_model_config.mesh.axis_names="('fsdp','tp','expert')" \
  rollout_model_config.mesh.shape="(4,1)" \
  rollout_model_config.mesh.axis_names="('fsdp','tp')" \
  cluster_config.target="k8s" \
  cluster_config.k8s.tunix_image="${TUNIX_IMAGE}" \
  cluster_config.roles.orchestrator.port=20000 \
  cluster_config.roles.rollout.port=20001 \
  cluster_config.roles.trainer.port=20002 \
  cluster_config.pathways.trainer.persistence=true \
  cluster_config.k8s.trainer.jobset_yaml="jobset.pathways.yaml" \
  cluster_config.k8s.trainer.tpu_slice="tpuv5:2x2x4" \
  cluster_config.k8s.rollout.jobset_yaml="jobset.tpu.yaml" \
  cluster_config.k8s.rollout.tpu_slice="tpuv5:2x2x1" \
  cluster_config.rollout_replicas=12 \
  "$@"
