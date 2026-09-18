#!/bin/bash

# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Converts a Hugging Face checkpoint into the scanned Orbax checkpoint that
# `TRAINER_BACKEND=maxtext` loads (`--maxtext_ckpt_path`).
#
# The MaxText trainer backend cannot start from HF safetensors the way the Tunix
# backend can, so every MaxText recipe needs this step once. Conversion runs on
# CPU and does not touch the TPU, so it is safe to run while nothing else holds
# the chips -- but it is also slow enough to deserve its own CI step, which is
# why it lives here rather than inline in the recipe.
#
# Idempotent: if <MAXTEXT_CKPT_DIR>/0/items already exists, this is a no-op, so
# it can be called unconditionally from a recipe and again from CI.
#
# Prints nothing but progress on stdout; the resulting checkpoint path is
# <MAXTEXT_CKPT_DIR>/0/items.

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

MODEL_NAME=${MODEL_NAME:-Qwen3-0.6B}
MODEL_ID=${MODEL_ID:-Qwen/Qwen3-0.6B}
# MaxText config names are lowercase and must match the recipe's, because the
# rollout builds MaxTextForCausalLM from the same name and Raiden pairs the two
# sides' tensors by exact name.
MAXTEXT_MODEL_NAME=${MAXTEXT_MODEL_NAME:-$(printf '%s' "${MODEL_NAME}" | tr '[:upper:]' '[:lower:]')}

# Keep these two derivations identical to the recipe's.
ARTIFACT_ROOT=${ARTIFACT_ROOT:-"${ROOT_DIR}/artifacts/qwen3_dist_gsm8k"}
MODEL_DIR=${MODEL_DIR:-"${ARTIFACT_ROOT}/models/${MODEL_NAME}"}
MAXTEXT_CKPT_DIR=${MAXTEXT_CKPT_DIR:-"${ARTIFACT_ROOT}/maxtext_ckpt/${MAXTEXT_MODEL_NAME}"}

ITEMS_DIR="${MAXTEXT_CKPT_DIR}/0/items"

if [[ -d "${ITEMS_DIR}" ]]; then
  echo "MaxText checkpoint already present, skipping conversion: ${ITEMS_DIR}"
  exit 0
fi

# Download the HF weights into the same MODEL_DIR the recipe uses, so the
# launcher's own ensure_model_dir finds them and does not download twice.
if ! compgen -G "${MODEL_DIR}/*.safetensors" >/dev/null; then
  echo "Downloading ${MODEL_ID} to ${MODEL_DIR}..."
  PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" python3 - "${MODEL_ID}" "${MODEL_DIR}" <<'PY'
import os
import sys

from tunix.oss import utils as oss_utils

model_id, model_dir = sys.argv[1], sys.argv[2]
os.makedirs(model_dir, exist_ok=True)
oss_utils.hf_pipeline(model_id, model_dir)
PY
fi

# base.yml ships inside the installed maxtext package; there is no source tree
# to point at when MaxText comes from a git+https requirement.
BASE_YML=$(python3 -c 'import os.path, maxtext; print(os.path.join(os.path.dirname(maxtext.__file__), "configs", "base.yml"))')
if [[ ! -f "${BASE_YML}" ]]; then
  echo "Error: MaxText base.yml not found at ${BASE_YML}" >&2
  exit 1
fi

echo "Converting ${MODEL_ID} -> MaxText (${MAXTEXT_MODEL_NAME}) at ${MAXTEXT_CKPT_DIR}"

# scan_layers=True matches tunix/utils/maxtext_utils.py, which emits
# scan_layers=True for the training config; a checkpoint written unscanned would
# fail to restore into it.
#
# ocdbt and zarr3 are both off. Plain Orbax reads the non-ocdbt layout fine, so
# this test gives up nothing by it, and it keeps the artifact readable by the
# Pathways persistence handler, which cannot read either format. This test runs
# mcJAX and so does not need that today; it is one line to keep the checkpoint
# usable by a Pathways run and a silent incompatibility to drop it.
CONVERT_ARGS=(
  "${BASE_YML}"
  model_name="${MAXTEXT_MODEL_NAME}"
  base_output_directory="${MAXTEXT_CKPT_DIR}"
  hardware=cpu
  scan_layers=True
  skip_jax_distributed_system=True
  checkpoint_storage_use_ocdbt=false
  checkpoint_storage_use_zarr3=false
  log_config=False
  # A local path, so to_maxtext reuses the download above instead of resolving
  # HF_IDS[model_name] -- which for several models is the -it variant.
  --hf_model_path="${MODEL_DIR}"
)
if [[ -n "${HF_TOKEN:-}" ]]; then
  CONVERT_ARGS+=(hf_access_token="${HF_TOKEN}")
fi

JAX_PLATFORMS=cpu python3 -m maxtext.checkpoint_conversion.to_maxtext "${CONVERT_ARGS[@]}"

if [[ ! -d "${ITEMS_DIR}" ]]; then
  echo "Error: conversion reported success but ${ITEMS_DIR} does not exist." >&2
  find "${MAXTEXT_CKPT_DIR}" -maxdepth 3 -print >&2 || true
  exit 1
fi

echo "MaxText checkpoint ready: ${ITEMS_DIR}"
