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

# Launches the multi-worker mock MoE routed-experts data-plane benchmark on GKE
# TPU v5p (`jobset.tpu.yaml` + single-host `tpuv5:2x2x1` slices).
#
# Usage:
#   bash k8s_launcher.sh [start|stop|orchestrator|trainer|rollout|otel|stop-otel] \
#     [--dry-run] [section.key=value ...]

set -Eeuo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Placeholder image rewritten by Copybara on OSS export. Live GKE runs MUST pass
# `cluster_config.k8s.tunix_image=<IMAGE>` built from a commit that includes
# `tunix.dist_rl.benchmarks.data_plane`.
TUNIX_IMAGE=${TUNIX_IMAGE:-}

# `actor_model_config.mesh.shape="(64,1,1)"` emulates a 64-way FSDP trainer's
# packed-row geometry (`BatchAssembler` row divisibility and micro-batch shapes)
# while `FakeTrainer` runs on CPU inside a single-host `tpuv5:2x2x1` pod.
exec "${DIR}/../../launch/common/launch.sh" \
  "override_config_file=${DIR}/data_plane_benchmark.yaml" \
  actor_model_config.mesh.shape="(64,1,1)" \
  actor_model_config.mesh.axis_names="('fsdp','tp','expert')" \
  rollout_model_config.mesh.shape="(1,4)" \
  rollout_model_config.mesh.axis_names="('fsdp','tp')" \
  cluster_config.target="k8s" \
  cluster_config.k8s.tunix_image="${TUNIX_IMAGE}" \
  cluster_config.ports.orchestrator=20000 \
  cluster_config.ports.rollout=20001 \
  cluster_config.ports.trainer=20002 \
  cluster_config.ports.inference=20003 \
  cluster_config.k8s.trainer.jobset_yaml="jobset.tpu.yaml" \
  cluster_config.k8s.trainer.tpu_slice="tpuv5:2x2x1" \
  cluster_config.k8s.rollout.jobset_yaml="jobset.tpu.yaml" \
  cluster_config.k8s.rollout.tpu_slice="tpuv5:2x2x1" \
  cluster_config.rollout_replicas=4 \
  "$@"
