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

# Launches the distributed DeepSWE GRPO recipe on GKE (JobSet), layering the
# GKE cluster, mesh, and sandbox namespace overrides below on top of
# `deepswe.yaml`. `otel` / `stop-otel` deploy / delete the OpenTelemetry
# collector (see `launch/common/k8s.sh`).
#
# Usage:
#   bash k8s_launcher_v2.sh [start|stop|orchestrator|trainer|rollout|otel|stop-otel] \
#     [--dry-run] [section.key=value ...]

set -Eeuo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Container image for the trainer/rollout pods. Override with
# `cluster_config.k8s.tunix_image=...` on the command line.
TUNIX_IMAGE=${TUNIX_IMAGE:-}

exec "${DIR}/../../launch/common/launch.sh" \
  "override_config_file=${DIR}/deepswe.yaml" \
  actor_model_config.mesh.shape="(8,1,1)" \
  actor_model_config.mesh.axis_names="('fsdp','tp','expert')" \
  rollout_model_config.mesh.shape="(1,4)" \
  rollout_model_config.mesh.axis_names="('fsdp','tp')" \
  data_config.cache_dir="artifacts/deepswe/dataset_cache" \
  kubernetes_config.namespace=null \
  cluster_config.target="k8s" \
  cluster_config.k8s.tunix_image="${TUNIX_IMAGE}" \
  cluster_config.ports.orchestrator=20000 \
  cluster_config.ports.rollout=20001 \
  cluster_config.ports.trainer=20002 \
  cluster_config.k8s.trainer.jobset_yaml="jobset.pathways.yaml" \
  cluster_config.k8s.trainer.tpu_slice="tpuv5:2x2x2" \
  cluster_config.k8s.rollout.jobset_yaml="jobset.tpu.yaml" \
  cluster_config.k8s.rollout.tpu_slice="tpuv5:2x2x1" \
  cluster_config.rollout_replicas=1 \
  "$@"
