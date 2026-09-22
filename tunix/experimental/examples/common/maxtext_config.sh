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
# Shared MaxText configuration.
#
# Usage: source this after MODEL_NAME and TRAINER_BACKEND are set, and after
# any per-example default overrides:
#
#   export MODEL_NAME=${MODEL_NAME:-Qwen3-4B}
#   export TRAINER_BACKEND=${TRAINER_BACKEND:-tunix}
#   export REMAT_POLICY=${REMAT_POLICY:-decoder}   # per-example override
#   source "${LAUNCHER_DIR}/../common/maxtext_config.sh"
#
# Source it unconditionally: everything below is a no-op unless
# TRAINER_BACKEND=maxtext, and the launchers call maxtext_trainer_flags /
# maxtext_rollout_flags unconditionally, so the functions have to be defined on
# the tunix path too.

# MaxText model names are lowercase, e.g. Qwen3-4B -> qwen3-4b
export MAXTEXT_MODEL_NAME=${MAXTEXT_MODEL_NAME:-$(printf '%s' "${MODEL_NAME}" | tr '[:upper:]' '[:lower:]')}

# Orbax checkpoint
export MAXTEXT_CKPT=${MAXTEXT_CKPT:-}

# Output directory
export MAXTEXT_OUTPUT_DIR=${MAXTEXT_OUTPUT_DIR:-artifacts/maxtext}

# Padded MoE MLP intermediate dimension
export TRAINER_PADDED_MOE_MLP_DIM=${TRAINER_PADDED_MOE_MLP_DIM:-}
export TRAINER_BASE_NUM_KV_HEADS=${TRAINER_BASE_NUM_KV_HEADS:-${BASE_NUM_KV_HEADS:-}}

# Attention kernel overrides
export TRAINER_MAXTEXT_ATTENTION=${TRAINER_MAXTEXT_ATTENTION:-}
export ROLLOUT_MAXTEXT_ATTENTION=${ROLLOUT_MAXTEXT_ATTENTION:-}
export REMAT_POLICY=${REMAT_POLICY:-}
export LEARNING_RATE_FINAL_FRACTION=${LEARNING_RATE_FINAL_FRACTION:-}
export MAXTEXT_WARMUP_STEPS_FRACTION=${MAXTEXT_WARMUP_STEPS_FRACTION:-${WARMUP_STEPS_FRACTION:-}}

maxtext_require_ckpt() {
  [[ "${TRAINER_BACKEND}" == "maxtext" ]] || return 0
  if [[ -z "${MAXTEXT_CKPT}" ]]; then
    if [[ "${DRY_RUN}" == "true" ]]; then
      echo "Warning: TRAINER_BACKEND=maxtext without MAXTEXT_CKPT (Orbax params-only checkpoint)." >&2
      return 0
    else
      echo "Error: TRAINER_BACKEND=maxtext requires MAXTEXT_CKPT (Orbax params-only checkpoint)." >&2
      exit 1
    fi
  fi
}

convert_maxtext_ckpt() {
  if [[ "$TRAINER_BACKEND" != "maxtext" ]]; then
    return
  fi
  if [[ "$MAXTEXT_CKPT" =~ ^gs:// ]]; then
    echo "Using GCS MAXTEXT_CKPT: $MAXTEXT_CKPT"
    return
  fi
  if [[ -d "$MAXTEXT_CKPT" ]]; then
    echo "Found existing local MAXTEXT_CKPT: $MAXTEXT_CKPT"
    return
  fi
  local ckpt_base
  ckpt_base="$(dirname "$(dirname "$MAXTEXT_CKPT")")"
  echo "Converting HF checkpoint $MODEL_DIR to MaxText Orbax checkpoint at $ckpt_base..."
  mkdir -p "$ckpt_base"
  JAX_PLATFORMS=cpu PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "$PYTHON_BIN" \
    -m maxtext.checkpoint_conversion.to_maxtext \
    model_name="${MAXTEXT_MODEL_NAME}" \
    --hf_model_path="${MODEL_DIR}" \
    base_output_directory="${ckpt_base}" \
    scan_layers=True \
    skip_jax_distributed_system=True \
    checkpoint_storage_use_zarr3=false \
    checkpoint_storage_use_ocdbt=false
  if [[ ! -d "$MAXTEXT_CKPT" ]]; then
    echo "Error: MaxText checkpoint conversion did not produce expected directory: $MAXTEXT_CKPT"
    exit 1
  fi
}

# Trainer-side MaxText flags.
#
# Deliberately does NOT emit --mesh_fsdp/--mesh_tp/--mesh_expert: those build
# the generic trainer mesh in _create_mesh() for both backends, every launcher
# already passes them, and the launchers disagree on the variable names
# (TRAINER_TP vs TRAINER_MESH_TP), so emitting them here produced an empty
# --mesh_tp= that overrode the real one.
maxtext_trainer_flags() {
  [[ "${TRAINER_BACKEND}" == "maxtext" ]] || return 0
  local profiler_flags=""
  if [[ -n "$PROFILER_STEPS" ]]; then
    profiler_flags+=" --profiler_steps=${PROFILER_STEPS}"
  fi
  if [[ -n "$SKIP_FIRST_N_PROFILER_STEPS" ]]; then
    profiler_flags+=" --skip_first_n_profiler_steps=${SKIP_FIRST_N_PROFILER_STEPS}"
  fi
  if [[ -n "$PROFILER_PERIOD" ]]; then
    profiler_flags+=" --profiler_period=${PROFILER_PERIOD}"
  fi

  printf '%s' " \
    --maxtext_model_name=${MAXTEXT_MODEL_NAME} \
    ${MAXTEXT_CKPT:+--maxtext_ckpt_path=${MAXTEXT_CKPT}} \
    --maxtext_output_directory=${MAXTEXT_OUTPUT_DIR} \
    ${TRAINER_PADDED_MOE_MLP_DIM:+--maxtext_padded_moe_mlp_dim=${TRAINER_PADDED_MOE_MLP_DIM}} \
    ${MAXTEXT_WARMUP_STEPS_FRACTION:+--maxtext_warmup_steps_fraction=${MAXTEXT_WARMUP_STEPS_FRACTION}} \
    ${TRAINER_MAXTEXT_ATTENTION:+--maxtext_attention=${TRAINER_MAXTEXT_ATTENTION}} \
    ${TRAINER_BASE_NUM_KV_HEADS:+--base_num_kv_heads=${TRAINER_BASE_NUM_KV_HEADS}} \
    ${REMAT_POLICY:+--remat_policy=${REMAT_POLICY}} \
    ${LEARNING_RATE_FINAL_FRACTION:+--learning_rate_final_fraction=${LEARNING_RATE_FINAL_FRACTION}} \
    ${ROLLOUT_MESH_TP:+--rollout_mesh_tp=${ROLLOUT_MESH_TP}} \
    ${USE_WEIGHT_CONVERTER:+--use_weight_converter=${USE_WEIGHT_CONVERTER}} \
    ${MAX_SEQ_TOKEN_PER_TPU:+--max_seq_token_per_tpu=${MAX_SEQ_TOKEN_PER_TPU}} \
    ${profiler_flags} \
  "
}

# Rollout-side MaxText flags.
maxtext_rollout_flags() {
  [[ "${TRAINER_BACKEND}" == "maxtext" ]] || return 0
  printf '%s' " \
    --maxtext_model_name=${MAXTEXT_MODEL_NAME} \
    ${ROLLOUT_MAXTEXT_ATTENTION:+--maxtext_attention=${ROLLOUT_MAXTEXT_ATTENTION}} \
  "
}
