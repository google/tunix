#!/bin/bash
# Submits the Qwen3.5-35B-A3B GRPO run to bodaborg-v5p-nap.
#
#   WANDB_API_KEY=... bash docker/maz-q35/submit.sh <run-number> <max-steps> [start|stop]
#
# e.g. the 2-step probe:  bash docker/maz-q35/submit.sh 1 2
#      the 100-step run:  bash docker/maz-q35/submit.sh 2 100
#      tear one down:     bash docker/maz-q35/submit.sh 1 2 stop
#
# Set DRY_RUN=true to print the three rendered JobSet manifests instead of applying
# them. Nothing here is submitted without an explicit run.
#
# The API key is deliberately not in this file: it is a live credential and this file
# is committed. Export it in the shell, or `source` a file outside the repo.
set -euo pipefail

RUN_N="${1:?usage: submit.sh <run-number> <max-steps> [start|stop]}"
STEPS="${2:?usage: submit.sh <run-number> <max-steps> [start|stop]}"
ACTION="${3:-start}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LAUNCHER="${REPO_ROOT}/tunix/experimental/examples/math_gsm8k_dist/k8s_launcher.sh"

# --- Cluster -----------------------------------------------------------------
# The only cluster in the fleet with the v5p capacity these two topologies need.
export PROJECT=cloud-tpu-shared-capacity
export CLUSTER=bodaborg-v5p-nap
export LOCATION_NAME=europe-west4
export K8S_NAMESPACE=default
export QUEUE_NAME=default
export CPU_MACHINE=n2d-standard-64

# Pods at priority 0 are the first thing evicted when the cluster fills up, which for a
# multi-hour run means losing the trainer, the rollout or both partway through. `medium`
# (500) is what the rest of this cluster uses, and Kueue reads the same value for
# admission because no jobset here carries a kueue.x-k8s.io/priority-class label.
export PRIORITY_CLASS_NAME=medium

# --- Images ------------------------------------------------------------------
# Pinned by digest, not tag. A tag can be repointed between the three `kubectl apply`s,
# and weight sync requires the orchestrator, trainer and rollout to run identical code.
# This is Yixuan's yixuann-e2e-0912head-v8 plus one patch to
# maxtext_engine._prepare_batch (see docker/maz-q35/Dockerfile).
export TUNIX_IMAGE=gcr.io/cloud-tpu-multipod-dev/mazumdera-runner@sha256:163a7d21cc56c4cceec91e2fad0e87e7cf399b358c7bb6a09bded887d3d0f505
export PATHWAYS_SERVER_IMAGE=us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/datenglin/unsanitized_server:raiden_20260904
export PATHWAYS_PROXY_IMAGE=us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/datenglin/unsanitized_proxy_server:raiden_20260904

# --- Model -------------------------------------------------------------------
export TRAINER_BACKEND=maxtext
export MAXTEXT_MODEL_NAME=qwen3.5-35b-a3b
export MODEL_NAME=Qwen3.5-35B-A3B
export MODEL_ID=Qwen/Qwen3.5-35B-A3B
export TOKENIZER_PATH=Qwen/Qwen3.5-35B-A3B
export MAXTEXT_CKPT=gs://hengtaoguo-maxtext-logs/checkpoints/qwen3.5-35b-a3b/scanned/2026-06-11-10-27/0/items

# --- Trainer: 8 v5p chips, FSDP ----------------------------------------------
export TRAINER_JOBSET_YAML=jobset.pathways.yaml
export TRAINER_TPU_SLICE=tpuv5p:2x2x2
export TRAINER_MESH_FSDP=8
export TRAINER_MESH_TP=1
export TRAINER_MESH_EXPERT=1
export PATHWAYS_PROXY_MEMORY_LIMIT=120G
# TODO(maz): no-op. Carried over from the reference config, but nothing in the launcher,
# the generator or the templates reads it -- jobset.cpu.yaml has its memory limit
# commented out, so the orchestrator container is limited only by the node.
export USER_CONTAINER_MEMORY=260G

# --- Rollout: 4 v5p chips, tensor parallel -----------------------------------
# dp=1, tp=4. FSDP on the rollout all-gathers the weights on every decode step; TP leaves
# them sharded and splits the matmuls instead, which removes that per-step collective.
export ROLLOUT_JOBSET_YAML=jobset.tpu.yaml
export ROLLOUT_TPU_SLICE=tpuv5p:2x2x1
export ROLLOUT_MESH_FSDP=1
export ROLLOUT_MESH_TP=4
export ROLLOUT_REPLICAS=1
# Separate vLLM server process, not the in-process sampler.
export SAMPLER=vllm
export ENABLE_PREFIX_CACHING=false

# --- Weight sync -------------------------------------------------------------
export WEIGHT_SYNC_MODE=raiden
export USE_WEIGHT_CONVERTER=true
export PREFUSE_MOE_WEIGHTS=true
# Reads back and compares the synced weights each step. Worth keeping for the 2-step
# probe; reconsider for the 100-step run if it shows up in the step time.
export VERIFY_WEIGHTS=${VERIFY_WEIGHTS:-true}

# --- Batching and packing ----------------------------------------------------
export MAX_PROMPT_LENGTH=512
export MAX_RESPONSE_LENGTH=512
export BATCH_SIZE=4
export NUM_GENERATIONS=8
export MINI_BATCH_SIZE=32           # = BATCH_SIZE * NUM_GENERATIONS: one optimizer step per rollout batch
export TRAIN_MICRO_BATCH_SIZE=8     # 32 / 8 = 4 micro-batches per step
# Sequence packing. A row holds 4096 tokens against a 512+512 worst-case sequence, so
# up to 4 sequences share a row and the 32 sequences fill 8 rows -- one per FSDP shard.
# max_segments_per_packed_row is left unset so that max_packed_len alone bounds the row;
# a lower cap would produce more than 8 rows, which the mesh cannot place.
export MAX_SEQ_TOKEN_PER_TPU=4096

export MAX_STEPS="${STEPS}"

# --- Output ------------------------------------------------------------------
# Checkpointing off: saving is broken and being fixed separately. 0 means never.
export CHECKPOINT_SAVE_INTERVAL_STEPS=0
export MAXTEXT_OUTPUT_DIR=gs://mazumdera-bucket-cloud-tpu-multipod-dev/q35-runs/maz-q35-${RUN_N}/maxtext

export WANDB_PROJECT=${WANDB_PROJECT:-trellis-gsm8k}
export WANDB_RUN_NAME=maz-q35-${RUN_N}-${STEPS}steps
: "${WANDB_API_KEY:?export WANDB_API_KEY before running; it is not stored in this file}"
export WANDB_API_KEY

export DEBUG=1

# --- Job names ---------------------------------------------------------------
# The launcher derives every JobSet name from $USER: maz-q35-N-{orch,train,roll}.
export USER=maz-q35-${RUN_N}

echo "=== maz-q35-${RUN_N}: ${ACTION}, ${STEPS} steps"
echo "    image      ${TUNIX_IMAGE}"
echo "    trainer    ${TRAINER_TPU_SLICE} fsdp=${TRAINER_MESH_FSDP}"
echo "    rollout    ${ROLLOUT_TPU_SLICE} dp=${ROLLOUT_MESH_FSDP} tp=${ROLLOUT_MESH_TP} sampler=${SAMPLER}"
echo "    packing    ${MAX_SEQ_TOKEN_PER_TPU} tok/row, micro-batch ${TRAIN_MICRO_BATCH_SIZE}"
echo "    priority   ${PRIORITY_CLASS_NAME}"
echo "    output     ${MAXTEXT_OUTPUT_DIR}"
echo "    jobsets    ${USER}-orch ${USER}-train ${USER}-roll"
echo

# Deliberately not `gcloud container clusters get-credentials`. That command rewrites the
# context to use gke-gcloud-auth-plugin, which authenticates as gcloud's active account --
# on the launch VM the default compute service account, which cannot create jobsets here.
# The context is expected to already exist and to authenticate as a user with write
# access. See the reproduction notes for how to create one from Application Default
# Credentials.
CONTEXT="$(kubectl config current-context)"
if ! kubectl config view -o "jsonpath={.contexts[?(@.name=='${CONTEXT}')].context.cluster}" \
     | grep -q "${CLUSTER}"; then
  echo "ERROR: kubectl context '${CONTEXT}' does not point at ${CLUSTER}." >&2
  exit 1
fi
if [[ "${DRY_RUN:-false}" != "true" ]] \
   && [[ "$(kubectl auth can-i create jobsets.jobset.x-k8s.io -n "${K8S_NAMESPACE}")" != "yes" ]]; then
  echo "ERROR: context '${CONTEXT}' cannot create jobsets in ${K8S_NAMESPACE}." >&2
  echo "       Authenticating as: $(kubectl auth whoami -o jsonpath='{.status.userInfo.username}')" >&2
  exit 1
fi
echo "    context    ${CONTEXT} as $(kubectl auth whoami -o jsonpath='{.status.userInfo.username}')"
echo

bash "${LAUNCHER}" "${ACTION}"
