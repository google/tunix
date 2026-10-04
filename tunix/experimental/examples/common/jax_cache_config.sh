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
#
# Shared JAX compilation cache configuration for rollout workers.
# Resolves default GCS cache directory based on hardware, model slug,
# and rollout sharding topology if not explicitly provided.

export LOCAL_JAX_CACHE_DIR=${LOCAL_JAX_CACHE_DIR:-${JAX_CACHE_DIR:-/tmp/jax_cache}}
export JAX_CACHE_GCS_DIR=${JAX_CACHE_GCS_DIR:-}
# The cache lives under the recipe's BUCKET (already region-specific), so there
# is a single source of truth for region -> bucket. Set JAX_CACHE_BUCKET to
# share a cache from a different bucket.
export JAX_CACHE_BUCKET="${JAX_CACHE_BUCKET:-${BUCKET:-}}"

if [[ -z "${JAX_CACHE_GCS_DIR}" && -z "${ROLLOUT_JAX_CACHE_GCS_DIR:-}" ]]; then
  if [[ -z "${JAX_CACHE_BUCKET}" ]]; then
    echo "[jax_cache] Neither BUCKET nor JAX_CACHE_BUCKET is set; rollout JAX compilation cache is disabled." >&2
  else
    _hw="unknown"
    if [[ "${ROLLOUT_TPU_SLICE:-}" == tpuv5* ]]; then
      _hw="v5p"
    elif [[ "${ROLLOUT_TPU_SLICE:-}" == tpu7x* ]]; then
      _hw="v7x"
    fi

    _model_slug="$(echo "${MODEL_NAME:-${MAXTEXT_MODEL_NAME:-model}}" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9._' '-' | sed 's/-$//')"
    _rollout_topo="${ROLLOUT_TPU_SLICE#*:}"
    _rollout_ep="${ROLLOUT_MESH_EXPERT:-1}"
    if [[ "${_rollout_ep}" == "1" && "${VLLM_ADDITIONAL_CONFIG:-}" =~ \"expert_parallelism\":\ *([0-9]+) ]]; then
      _rollout_ep="${BASH_REMATCH[1]}"
    fi
    _rollout_tp="${ROLLOUT_MESH_TP:-1}"

    export ROLLOUT_JAX_CACHE_GCS_DIR="${JAX_CACHE_BUCKET}/jax_cache/${_hw}/${_model_slug}/rollout_${_rollout_topo}_ep${_rollout_ep}_tp${_rollout_tp}"
  fi
fi

export ROLLOUT_JAX_CACHE_GCS_DIR="${ROLLOUT_JAX_CACHE_GCS_DIR:-${JAX_CACHE_GCS_DIR:+${JAX_CACHE_GCS_DIR}/rollout}}"
export SAVE_JAX_CACHE="${SAVE_JAX_CACHE:-true}"
export SKIP_JAX_PRECOMPILE="${SKIP_JAX_PRECOMPILE:-1}"
