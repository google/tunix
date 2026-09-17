#!/bin/bash
# =============================================================================
# raiden_smoke_test/config.sh -- SINGLE SOURCE OF TRUTH
# =============================================================================
# Sourced by build.sh and run.sh. Edit pins HERE only.
#
# Reproduces run `head-v19-0102` (2026-09-17): a 2-step GRPO smoke test on
# Qwen3.5-35B-A3B over Pathways-MaxText (Raiden FFI) + mcJAX-vLLM (Raiden TCP).
# That run passed 11/11 gates with EXIT_CODE=0.
#
# Override anything by exporting it before sourcing, e.g.
#   IMAGE_TAG=mytag bash raiden_smoke_test/build.sh
# =============================================================================
set -uo pipefail

# --- Layout -----------------------------------------------------------------
# This file lives at <projects>/tunix/raiden_smoke_test/config.sh, so:
#   TUNIX_DIR    = <projects>/tunix
#   PROJECTS_DIR = <projects>            <- the docker build context
# maxtext/ and tpu-inference/ must be siblings of tunix/ because the Dockerfile
# COPYs all three, and docker cannot COPY from outside the build context.
SMOKE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TUNIX_DIR="$(cd "${SMOKE_DIR}/.." && pwd)"
PROJECTS_DIR="${PROJECTS_DIR:-$(cd "${TUNIX_DIR}/.." && pwd)}"
MAXTEXT_DIR="${MAXTEXT_DIR:-${PROJECTS_DIR}/maxtext}"
TPU_INFERENCE_DIR="${TPU_INFERENCE_DIR:-${PROJECTS_DIR}/tpu-inference}"
# The Dockerfile lives in this directory, but the build CONTEXT is PROJECTS_DIR,
# because it does `COPY tunix ...`, `COPY maxtext ...`, `COPY tpu-inference ...`.
DOCKERFILE="${DOCKERFILE:-${SMOKE_DIR}/Dockerfile}"

# Logs. Deliberately NOT under any agent/conversation directory.
LOG_ROOT="${LOG_ROOT:-${HOME}/raiden_smoke_logs}"
LAST_RUN_POINTER="${LOG_ROOT}/.last_run"

# --- Pins -------------------------------------------------------------------
# Validated set as of 2026-09-17. tunix/maxtext/tpu-inference are read from the
# local working trees; these are recorded for provenance and asserted by build.sh.
EXPECT_TUNIX_COMMIT="${EXPECT_TUNIX_COMMIT:-032db7eaafec314f96a90ef206986dfb0d39bb8b}"
EXPECT_MAXTEXT_COMMIT="${EXPECT_MAXTEXT_COMMIT:-7c016bfe2a4968e436dde09a5ec881f1a125e2ec}"
EXPECT_TPU_INFERENCE_COMMIT="${EXPECT_TPU_INFERENCE_COMMIT:-9dc23977905017edfae14adbb413832c1d63a6c0}"

# Baked into the image by the Dockerfile ARGs.
# vLLM 51da0ca66 is also what upstream requirements/requirements.txt pins.
export VLLM_COMMIT="${VLLM_COMMIT:-51da0ca66c8065619c79e35dff97aa99aeaf5644}"
export TPU_INFERENCE_COMMIT="${TPU_INFERENCE_COMMIT:-9dc23977905017edfae14adbb413832c1d63a6c0}"

# --- Raiden wheel -----------------------------------------------------------
# NOTE: the distribution was RENAMED tpu_raiden_jax -> tpu_sync_jax, and the
# import is now `import tpu_sync`. Anything still matching tpu_raiden_jax-* is
# stale. (Upstream Dockerfile's Artifact Registry fallback still installs the
# old name and is therefore broken -- see README "Known upstream issues".)
WHEEL_NAME="${WHEEL_NAME:-tpu_sync_jax-0.0.1.dev20260914193202-cp312-cp312-manylinux_2_31_x86_64.whl}"
GCS_WHEEL_PATH="${GCS_WHEEL_PATH:-gs://cloud-tpu-inference-test-datenglin/${WHEEL_NAME}}"
RAIDEN_WHEELS_DIR="${RAIDEN_WHEELS_DIR:-${TUNIX_DIR}/raiden_wheels}"

# --- Image ------------------------------------------------------------------
IMAGE_TAG="${IMAGE_TAG:-yixuann-e2e-0917head-v19}"
IMAGE_REPO="${IMAGE_REPO:-gcr.io/cloud-tpu-multipod-dev/${USER}_google_com-runner}"
export TUNIX_IMAGE="${TUNIX_IMAGE:-${IMAGE_REPO}:${IMAGE_TAG}}"

# =============================================================================
# Cluster / run environment -- consumed by k8s_launcher.sh
# =============================================================================
export PROJECT="${PROJECT:-cloud-tpu-shared-capacity}"
export CLUSTER="${CLUSTER:-bodaborg-v5p-nap}"
export LOCATION_NAME="${LOCATION_NAME:-europe-west4}"
export CPU_MACHINE="${CPU_MACHINE:-n2d-standard-64}"
export QUEUE_NAME="${QUEUE_NAME:-default}"
export K8S_NAMESPACE="${K8S_NAMESPACE:-trellis}"

# Pathways images. The server tag carries the _fix suffix and the proxy tag does
# NOT. That asymmetry is intentional -- do not "correct" it.
# These 403 on `docker pull` from a cloudtop; GKE nodes pull with a different
# service account. A 403 here is expected, not a misconfiguration.
export PATHWAYS_SERVER_IMAGE="${PATHWAYS_SERVER_IMAGE:-us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/datenglin/unsanitized_server:raiden_20260914_fix}"
export PATHWAYS_PROXY_IMAGE="${PATHWAYS_PROXY_IMAGE:-us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/datenglin/unsanitized_proxy_server:raiden_20260914}"

# --- Model ------------------------------------------------------------------
export TRAINER_BACKEND="${TRAINER_BACKEND:-maxtext}"
export MAXTEXT_MODEL_NAME="${MAXTEXT_MODEL_NAME:-qwen3.5-35b-a3b}"
export MODEL_NAME="${MODEL_NAME:-Qwen3.5-35B-A3B}"
export MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-35B-A3B}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-Qwen/Qwen3.5-35B-A3B}"
export MAXTEXT_CKPT="${MAXTEXT_CKPT:-gs://hengtaoguo-maxtext-logs/checkpoints/qwen3.5-35b-a3b/scanned/2026-06-11-10-27/0/items}"

# --- Topology ---------------------------------------------------------------
# Trainer: Pathways, 8 chips, FSDP=8, Raiden FFI  (RAIDEN_USE_FFI=1 derived)
export TRAINER_JOBSET_YAML="${TRAINER_JOBSET_YAML:-jobset.pathways.yaml}"
export TRAINER_TPU_SLICE="${TRAINER_TPU_SLICE:-tpuv5p:2x2x2}"
export TRAINER_MESH_FSDP="${TRAINER_MESH_FSDP:-8}"
export TRAINER_MESH_TP="${TRAINER_MESH_TP:-1}"
export TRAINER_MESH_EXPERT="${TRAINER_MESH_EXPERT:-1}"

# Rollout: mcJAX, 4 chips, dp=2 tp=2, Raiden TCP (RAIDEN_USE_FFI=0 derived)
export ROLLOUT_JOBSET_YAML="${ROLLOUT_JOBSET_YAML:-jobset.tpu.yaml}"
export ROLLOUT_TPU_SLICE="${ROLLOUT_TPU_SLICE:-tpuv5p:2x2x1}"
export ROLLOUT_MESH_FSDP="${ROLLOUT_MESH_FSDP:-2}"
export ROLLOUT_MESH_TP="${ROLLOUT_MESH_TP:-2}"
export ROLLOUT_REPLICAS="${ROLLOUT_REPLICAS:-1}"

# --- Weight sync ------------------------------------------------------------
export SAMPLER="${SAMPLER:-vllm}"
export WEIGHT_SYNC_MODE="${WEIGHT_SYNC_MODE:-raiden}"
export USE_WEIGHT_CONVERTER="${USE_WEIGHT_CONVERTER:-true}"
export PREFUSE_MOE_WEIGHTS="${PREFUSE_MOE_WEIGHTS:-true}"
export ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-false}"
export ENABLE_PATHWAYS_PERSISTENCE="${ENABLE_PATHWAYS_PERSISTENCE:-1}"

# --- Memory -----------------------------------------------------------------
# Bare-HEAD defaults (190G/48G/70G/100G) are wrong for this topology.
export PATHWAYS_PROXY_MEMORY_LIMIT="${PATHWAYS_PROXY_MEMORY_LIMIT:-90G}"
export USER_CONTAINER_MEMORY="${USER_CONTAINER_MEMORY:-60G}"
export USER_CONTAINER_MEMORY_LIMIT="${USER_CONTAINER_MEMORY_LIMIT:-150G}"
export PATHWAYS_WORKER_MEMORY="${PATHWAYS_WORKER_MEMORY:-165G}"
export TRAINER_EXTRA_ENV="${TRAINER_EXTRA_ENV:-CKPT_D2H_CONCURRENT_GB=8}"

# --- Training ---------------------------------------------------------------
export MAX_STEPS="${MAX_STEPS:-2}"
export CHECKPOINT_SAVE_INTERVAL_STEPS="${CHECKPOINT_SAVE_INTERVAL_STEPS:-1}"  # every step
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
export BATCH_SIZE="${BATCH_SIZE:-2}"
export NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
export TRAIN_MICRO_BATCH_SIZE="${TRAIN_MICRO_BATCH_SIZE:-8}"

# BREAKING CHANGE vs pre-2026-09 configs: batch_size and mini_batch_size are both
# counted in PROMPT GROUPS now, and rl_program.py:202 enforces
# batch_size % mini_batch_size == 0. An old value of 8 (which meant trajectories)
# is rejected outright:
#   ValueError: batch_size must be divisible by mini_batch_size;
#               got batch_size=2, mini_batch_size=8
# MINI_BATCH_SIZE=2 reproduces the old effective arithmetic:
#   update_trajectories = mini_batch_size * num_generations = 2*4 = 8, and
#   8 % train_micro_batch_size(8) == 0 -> one optimizer step per full batch.
# The trainer's own validation would NOT catch a bad value (8*4=32, 32%8==0);
# only the orchestrator's check fires, ~2 min in, after TPUs are allocated.
export MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-2}"

# >>> WORKAROUND -- REMOVE ONCE UPSTREAM FIXES per_token_logps <<<<<<<<<<<<<<<<<
# Upstream commit 9b24d9df ("[Tunix] Add sampler-trainer agreement metrics and
# importance sampling to the RL orchestrator") added a token-level importance
# sampling (TIS) correction. It is DEFAULT ON
# (k8s_launcher.sh: USE_ROLLOUT_LOGPS=${USE_ROLLOUT_LOGPS:-true}) but it only
# implemented Trainer.per_token_logps for the tunix-native peft_trainer_v2
# backend. MaxTextTrainingEngine has no such method, so TRAINER_BACKEND=maxtext
# dies at step 0 with:
#   AttributeError: 'MaxTextTrainingEngine' object has no attribute 'per_token_logps'
# Chain: rl_program.py:986 gate -> :828 _apply_sampler_trainer_agreement
#        -> distributed_rl_engine.py:453 -> trainer_worker.py:260
#        -> _MeshBoundTrainer.__getattr__ (run_trainer_node.py:520)
#
# Setting this false renders --no-use_rollout_logps on the orchestrator, which
#   (a) skips _apply_sampler_trainer_agreement, and
#   (b) makes algorithm_adapter._extract_old_logps return None, so the gate is
#       false on its second clause too.
# The other per_token_logps call site (rl_program.py:972) is guarded by
# requires_reference_kl = (beta != 0 or force_compute_kl); BETA defaults to 0.
#
# COST: this run has NO off-policy TIS correction, which matters because the
# sampler (vLLM) and trainer (MaxText) are different backends -- exactly the
# regime TIS exists to correct. `perplexity: 1.0000` in the metrics is an
# artefact of this, not a model property. DELETE THIS LINE once upstream
# implements per_token_logps for the maxtext backend, and re-run to validate
# the agreement path.
export USE_ROLLOUT_LOGPS="${USE_ROLLOUT_LOGPS:-false}"
# >>> END WORKAROUND <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

export VERIFY_WEIGHTS="${VERIFY_WEIGHTS:-true}"
export DEBUG="${DEBUG:-1}"

# --- Output -----------------------------------------------------------------
GCS_OUTPUT_ROOT="${GCS_OUTPUT_ROOT:-gs://${USER}-maxtext-dataset/trellis/raiden_smoke}"

LAUNCHER="${LAUNCHER:-${TUNIX_DIR}/tunix/experimental/examples/math_gsm8k_dist/k8s_launcher.sh}"

# --- Shared helpers ---------------------------------------------------------
c_die()  { echo "FATAL: $*" >&2; exit 1; }
c_warn() { echo "WARN:  $*" >&2; }
c_info() { echo "  $*"; }

c_require_kubectl() {
  export PATH="${HOME}/.local/bin:${PATH}"
  command -v kubectl >/dev/null 2>&1 || c_die "kubectl not on PATH (try ~/.local/bin)."
}
