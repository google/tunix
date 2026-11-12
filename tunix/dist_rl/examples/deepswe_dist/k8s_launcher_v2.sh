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
# GKE cluster, mesh, and memory-backed `/dev/shm` checkpoint cache overrides
# below on top of `deepswe.yaml`. `otel` / `stop-otel` deploy / delete the
# OpenTelemetry collector (see `launch/common/k8s.sh`).
#
# Usage:
#   bash k8s_launcher_v2.sh [start|stop|orchestrator|trainer|rollout|otel|stop-otel] \
#     [--dry-run] [section.key=value ...]

set -Eeuo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Container image for the trainer/rollout pods. Override with
# `cluster_config.k8s.tunix_image=...` on the command line.
TUNIX_IMAGE=${TUNIX_IMAGE:-}

# Trainer: one v5p 4x4x4 slice under Pathways. Qwen3-32B float32 safetensors
# (65.5 GB) exceed standard GKE root ephemeral storage (~47 GB), so both
# `HF_HOME` and `model_download_path` target the memory-backed `/dev/shm`
# volume (`os.link` hardlinks between them with 1x RAM usage). Loading Qwen3-32B
# spikes trainer RSS (~89 GB) plus `/dev/shm` weights and 16 MiB shared-memory
# IFRT proxy transfers, covered by `user_container_memory_limit="340G"`.
# Rollout: 16 single-host v5p 2x2x1 replicas, one vLLM engine (TP=4) each;
# `jobset.tpu.yaml` does not span hosts.
export TRAINER_IFRT_PROXY_LARGE_TRANSFER_THRESHOLD_BYTES="${TRAINER_IFRT_PROXY_LARGE_TRANSFER_THRESHOLD_BYTES:-16777216}"

internal_args=()
if [[ "${DEEPSWE_SKIP_INTERNAL_ARGS:-0}" != "1" && \
  -f "${DIR}/google/k8s_internal_args.sh" ]]; then
  # shellcheck source=/dev/null
  source "${DIR}/google/k8s_internal_args.sh"
fi

exec "${DIR}/../../launch/common/launch.sh" \
  "override_config_file=${DIR}/deepswe.yaml" \
  model_config.model_download_path="/dev/shm/models" \
  actor_model_config.mesh.shape="(8,8,1)" \
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
  cluster_config.k8s.cpu_memory="64G" \
  cluster_config.k8s.user_container_memory="260G" \
  cluster_config.k8s.user_container_memory_limit="340G" \
  cluster_config.k8s.trainer.jobset_yaml="jobset.pathways.yaml" \
  cluster_config.k8s.trainer.tpu_slice="tpuv5:4x4x4" \
  cluster_config.k8s.rollout.jobset_yaml="jobset.tpu.yaml" \
  cluster_config.k8s.rollout.tpu_slice="tpuv5:2x2x1" \
  cluster_config.rollout_replicas=16 \
  cluster_config.role_env.trainer.HF_HOME="/dev/shm/hf" \
  cluster_config.role_env.rollout.HF_HOME="/dev/shm/hf" \
  "${internal_args[@]}" \
  "$@"
