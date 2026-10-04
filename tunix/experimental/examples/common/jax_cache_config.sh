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
export JAX_CACHE_BUCKET="${JAX_CACHE_BUCKET:-${BUCKET:-}}"

if [[ -z "${JAX_CACHE_GCS_DIR}" && -z "${ROLLOUT_JAX_CACHE_GCS_DIR:-}" ]]; then
  _cache_bucket="${JAX_CACHE_BUCKET:-${BUCKET:-}}"
  if [[ -z "${_cache_bucket}" && -n "${MAXTEXT_OUTPUT_DIR:-}" ]]; then
    _cache_bucket="$(echo "${MAXTEXT_OUTPUT_DIR}" | grep -o '^gs://[^/]*' || true)"
  fi

  if [[ -n "${_cache_bucket}" ]]; then
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
    _use_batched_rpa="${USE_BATCHED_RPA_KERNEL:-${ROLLOUT_USE_BATCHED_RPA:-0}}"

    _setup_payload=$(cat <<EOF
MODEL_NAME=${MODEL_NAME:-${MAXTEXT_MODEL_NAME:-model}}
ROLLOUT_FP8=${ROLLOUT_FP8:-false}
ROLLOUT_QUANTIZATION=${ROLLOUT_QUANTIZATION:-none}
FLOAT32_LOGITS=${FLOAT32_LOGITS:-true}
ROLLOUT_TPU_SLICE=${ROLLOUT_TPU_SLICE:-}
ROLLOUT_MESH_EXPERT=${_rollout_ep}
ROLLOUT_MESH_TP=${_rollout_tp}
VLLM_DATA_PARALLEL_SIZE=${VLLM_DATA_PARALLEL_SIZE:-1}
LIBTPU_INIT_ARGS=${LIBTPU_INIT_ARGS:-}
VLLM_MOE_CHUNK_SIZE=${VLLM_MOE_CHUNK_SIZE:-}
ONEHOT_MOE_PERMUTE_THRESHOLD=${ONEHOT_MOE_PERMUTE_THRESHOLD:-}
ATTN_CUSTOM_NUM_REQS_BUCKETS=${ATTN_CUSTOM_NUM_REQS_BUCKETS:-}
USE_BATCHED_RPA_KERNEL=${_use_batched_rpa}
EOF
)
    if command -v sha256sum >/dev/null 2>&1; then
      _setup_hash=$(printf '%s' "${_setup_payload}" | sha256sum | cut -c1-10)
    elif command -v shasum >/dev/null 2>&1; then
      _setup_hash=$(printf '%s' "${_setup_payload}" | shasum -a 256 | cut -c1-10)
    else
      _setup_hash=$(echo -n "${_setup_payload}" | cksum | awk '{print $1}')
    fi

    export ROLLOUT_JAX_CACHE_GCS_DIR="${_cache_bucket}/jax_cache/${_hw}/${_model_slug}/rollout_${_rollout_topo}_ep${_rollout_ep}_tp${_rollout_tp}_${_setup_hash}"
  fi
fi

export ROLLOUT_JAX_CACHE_GCS_DIR="${ROLLOUT_JAX_CACHE_GCS_DIR:-${JAX_CACHE_GCS_DIR:+${JAX_CACHE_GCS_DIR}/rollout}}"
export SAVE_JAX_CACHE="${SAVE_JAX_CACHE:-true}"
export SKIP_JAX_PRECOMPILE="${SKIP_JAX_PRECOMPILE:-1}"
