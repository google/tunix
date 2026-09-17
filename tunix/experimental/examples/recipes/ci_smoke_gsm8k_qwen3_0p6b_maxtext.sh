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

# ==============================================================================
# CI Smoke Test Recipe: Qwen3-0.6B on GSM8K, production stack
# ==============================================================================
# The sibling recipe `ci_smoke_gsm8k_qwen3_0p6b.sh` exercises the Tunix-native
# stack. This one swaps in the three components the production recipe
# (`trellis_gsm8k_qwen3p5_35b.sh`) actually runs, on the same single 8-chip host:
#
#   - Trainer: MaxText's MaxTextTrainingEngine  (TRAINER_BACKEND=maxtext)
#   - Rollout: vLLM server-mode sampler running MaxTextForCausalLM
#              (SAMPLER=vllm + MAXTEXT_MODEL_NAME)
#   - Weight sync: Raiden native TPU sync       (WEIGHT_SYNC_MODE=raiden)
#
# Because both sides run MaxText's model code, Raiden's name-based tensor
# pairing is covered against the real MaxText parameter names -- the thing that
# breaks when either repository renames a weight.
#
# Both sides run mcJAX by default (TRAINER_PATHWAYS=0), which needs no daemons.
# TRAINER_PATHWAYS=1 runs the trainer as a Pathways proxy client instead --
# production's configuration, and the one that puts Raiden on its FFI transport
# on the trainer side while the rollout stays on TCP. See WHAT THIS DOES NOT
# COVER for why that is not the default.
#
# WHAT THIS DOES NOT COVER
# ------------------------
# Pathways, by default. It is the production configuration and this script
# supports it, but it depends on four things about the host that this script
# cannot establish for itself -- the daemons running, the TPU generation
# detected, the trainer's chips bound in PCI order, and the "proxy" JAX backend
# registered -- so as a default it would report host misconfiguration as test
# failure. mcJAX exercises MaxText, vLLM and Raiden without any of it.
#
# Production is also multi-host and multi-slice, and the 35B model is MoE, so
# MoE prefusion and the padded expert MLP dim are out of scope here: qwen3-0.6b
# is dense, and its 8 KV heads exceed the rollout's TP=4, so no KV-head
# replication is needed either. All of those knobs stay at their defaults.
# ==============================================================================

set -Ee

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${DIR}/../../../.." && pwd)"

# Model Configuration
export MODEL_NAME="${MODEL_NAME:-Qwen3-0.6B}"
export MODEL_ID="${MODEL_ID:-Qwen/Qwen3-0.6B}"
# Lowercase MaxText config name. Passed to the trainer AND the rollout, so both
# build the same model; Raiden pairs tensors by exact name and a mismatch here
# silently transfers nothing.
export MAXTEXT_MODEL_NAME="${MAXTEXT_MODEL_NAME:-qwen3-0.6b}"

# Scope MODEL_DIR and TOKENIZER_PATH by MODEL_NAME to avoid loading weights from
# mismatched models (e.g. Qwen3-1.7B) if previously downloaded to artifacts/
export ARTIFACT_ROOT="${ARTIFACT_ROOT:-${REPO_ROOT}/artifacts/qwen3_dist_gsm8k}"
export MODEL_DIR="${MODEL_DIR:-${ARTIFACT_ROOT}/models/${MODEL_NAME}}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-$MODEL_DIR}"

# Stack Configuration: MaxText trainer + vLLM server-mode rollout + Raiden
export TRAINER_BACKEND="${TRAINER_BACKEND:-maxtext}"
export SAMPLER="${SAMPLER:-vllm}"
export WEIGHT_SYNC_MODE="${WEIGHT_SYNC_MODE:-raiden}"

# The trainer restores a scanned Orbax checkpoint, so build one here (no-op if
# it already exists). CI calls the same script as its own step for clearer
# timing; the second call falls through immediately.
#
# Not because MaxText lacks HF support: model_creation_utils.from_pretrained
# converts safetensors on the fly, shelling out to checkpoint_conversion.to_maxtext
# in a CPU subprocess. Tunix just never reaches it -- launcher.sh refuses to
# start the maxtext backend without MAXTEXT_CKPT, tunix/utils/maxtext_utils.py
# hardcodes convert_checkpoint_if_possible=False, and MAXTEXT_CKPT becomes
# load_parameters_path, which suppresses the conversion branch anyway.
#
# Converting here rather than in-engine also shares one HF download with the
# rollout and the tokenizer, which the launcher requires regardless; the
# in-engine path passes no --hf_model_path and so would fetch the weights a
# second time.
export MAXTEXT_CKPT_DIR="${MAXTEXT_CKPT_DIR:-${ARTIFACT_ROOT}/maxtext_ckpt/${MAXTEXT_MODEL_NAME}}"
if [[ -z "${MAXTEXT_CKPT:-}" ]]; then
  bash "${REPO_ROOT}/scripts/build_maxtext_checkpoint.sh"
  export MAXTEXT_CKPT="${MAXTEXT_CKPT_DIR}/0/items"
fi

# MaxTextTrainingEngine does not implement per-token logps, so the learner has
# to take them from the rollout. USE_ROLLOUT_LOGPS=false would send it down
# `get_actor_per_token_logps`, which the engine cannot serve.
export USE_ROLLOUT_LOGPS="${USE_ROLLOUT_LOGPS:-true}"

# Chip allocation on a single 8-chip host (e.g. TPU v6e-8)
# Auto-detect TPU chips sorted by PCI topology order if running on a host with /dev/vfio.
# On multi-tray hosts (e.g. TPU v5e-8 or v6e-8), /dev/vfio device minor IDs can be
# enumerated out of order across PCIe switches. Slicing into 2x2 meshes requires
# chips on the same physical tray (same PCIe root complex) to be grouped together.
if [[ -z "${TRAINER_TPU_CHIPS:-}" || -z "${ROLLOUT_TPU_CHIPS:-}" ]]; then
  GET_SORTED_TPUS_SCRIPT="${REPO_ROOT}/scripts/get_sorted_tpu_ids.py"
  if [[ -f "$GET_SORTED_TPUS_SCRIPT" && -d /dev/vfio ]]; then
    DETECTED_TRAINER_CHIPS=$(python3 "$GET_SORTED_TPUS_SCRIPT" --count 4 2>/dev/null || true)
    DETECTED_ROLLOUT_CHIPS=$(python3 "$GET_SORTED_TPUS_SCRIPT" --count 4 --begin 4 2>/dev/null || true)
    if [[ -n "$DETECTED_TRAINER_CHIPS" && -n "$DETECTED_ROLLOUT_CHIPS" ]]; then
      export TRAINER_TPU_CHIPS="${TRAINER_TPU_CHIPS:-$DETECTED_TRAINER_CHIPS}"
      export ROLLOUT_TPU_CHIPS="${ROLLOUT_TPU_CHIPS:-$DETECTED_ROLLOUT_CHIPS}"
      echo "Discovered PCI-sorted TPU chips: Trainer=[$TRAINER_TPU_CHIPS], Rollout=[$ROLLOUT_TPU_CHIPS]"
    fi
  fi
fi

# Unlike the Tunix-backend recipe, which mirrors the rollout's TP=4 layout on the
# trainer, this one puts the trainer on FSDP=4 / TP=1 against a TP=4 rollout.
# That is the production shape (FSDP on the trainer, TP on the rollout) and it
# is the case Raiden has to resolve a cross-mesh resharding for; a matched
# TP=4/TP=4 pair would let a 1:1 copy pass for a working sync.
export TRAINER_TPU_CHIPS="${TRAINER_TPU_CHIPS:-0,1,2,3}"
export TRAINER_FSDP="${TRAINER_FSDP:-4}"
export TRAINER_TP="${TRAINER_TP:-1}"

export ROLLOUT_TPU_CHIPS="${ROLLOUT_TPU_CHIPS:-4,5,6,7}"
export ROLLOUT_FSDP="${ROLLOUT_FSDP:-1}"
export ROLLOUT_TP="${ROLLOUT_TP:-4}"

# TPU v6e-8 physical host bounds are 2,4,1. When partitioned into 4-chip slices
# (chips 0..3 and 4..7), each slice forms a 2x2 grid with bounds 2,2,1.
export TPU_CHIPS_PER_HOST_BOUNDS="${TPU_CHIPS_PER_HOST_BOUNDS:-2,2,1}"
export TPU_HOST_BOUNDS="${TPU_HOST_BOUNDS:-1,1,1}"
# Allow multiple concurrent libtpu loads across trainer and rollout processes on the same host
export ALLOW_MULTIPLE_LIBTPU_LOAD="${ALLOW_MULTIPLE_LIBTPU_LOAD:-1}"

# Run the trainer as a Pathways proxy client instead of mcJAX. This is the
# production pairing -- the trainer takes Raiden's FFI transport while the
# rollout stays on TCP -- and the only configuration that exercises the
# single-controller API the orchestrator depends on. It is off by default
# because of what it needs from the host; see WHAT THIS DOES NOT COVER above.
#
# Requires the Pathways daemons to be listening already -- run
# scripts/start_pathways_daemons.sh first. The default, TRAINER_PATHWAYS=0,
# runs mcJAX on both sides (TCP/TCP) and needs no daemons.
export TRAINER_PATHWAYS="${TRAINER_PATHWAYS:-0}"
if [[ "${TRAINER_PATHWAYS}" == "1" ]]; then
  # The Pathways persistence array handler cannot read OCDBT or zarr3; this also
  # makes tunix/utils/maxtext_utils.py emit the matching storage settings. The
  # checkpoint built above is already written in the compatible layout.
  export ENABLE_PATHWAYS_PERSISTENCE="${ENABLE_PATHWAYS_PERSISTENCE:-1}"
  # The IFRT proxy client and the daemons talk over loopback inside one
  # container, so they use insecure gRPC. Both sides have to agree, and the
  # daemon side is set by scripts/start_pathways_daemons.sh.
  export IFRT_PROXY_USE_INSECURE_GRPC_CREDENTIALS=true
  export PATHWAYS_UNSAFE_UNSAFE_OVERRIDE_GRPC_CREDENTIALS=grpc_insecure_override
fi

# Hyperparameters: 2-step run (validates rollout generation on step-1 synced weights)
export MAX_STEPS="${MAX_STEPS:-2}"
export BATCH_SIZE="${BATCH_SIZE:-4}"
export MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-4}"
export NUM_GENERATIONS="${NUM_GENERATIONS:-2}"
# MaxText shards the batch dimension of every loss input across the fsdp axis,
# so this must be a multiple of TRAINER_FSDP (the launcher rounds it up if not).
# Gradient accumulation steps = (MINI_BATCH_SIZE * NUM_GENERATIONS) /
# TRAIN_MICRO_BATCH_SIZE = (4 * 2) / 4 = 2. Keeping this above 1 is what
# exercises the accumulate-then-apply path: the orchestrator, not the engine,
# decides where the optimizer step falls, so a run with a single microbatch per
# step would never call train_step with apply_optimizer=False.
export TRAIN_MICRO_BATCH_SIZE="${TRAIN_MICRO_BATCH_SIZE:-4}"
export BETA="${BETA:-0}"

# Dataset & sequence lengths
export TFDS_SPLIT="${TFDS_SPLIT:-train[:16]}"
export SHUFFLE="${SHUFFLE:-false}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-512}"

# Checkpointing: verify step 1 checkpoint save
export CHECKPOINT_SAVE_INTERVAL_STEPS="${CHECKPOINT_SAVE_INTERVAL_STEPS:-1}"
export CHECKPOINT_MAX_TO_KEEP="${CHECKPOINT_MAX_TO_KEEP:-2}"

# Disable WandB logging and evaluation overhead
export WANDB_PROJECT="${WANDB_PROJECT:-ci-smoke-gsm8k-maxtext}"
export EVAL_EVERY_N_STEPS="${EVAL_EVERY_N_STEPS:-999}"

# Timeouts: more headroom than the Tunix-backend recipe. MaxText builds and
# compiles its own train step and restores an Orbax checkpoint, and the vLLM
# server-mode sampler starts an async engine, so first-step latency is
# dominated by compilation rather than by the 2 steps of actual work.
export WAIT_TIMEOUT_SECS="${WAIT_TIMEOUT_SECS:-1800}"

# Delegate to base launcher located in math_gsm8k_dist
exec "${DIR}/../math_gsm8k_dist/launcher.sh" "$@"
