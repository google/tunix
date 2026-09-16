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
# This is Yixuan's yixuann-e2e-0912head-v8 plus six patches: MaxText PR 5219 and 5234,
# tunix PR 2229, the in-image part of tunix PR 2228 at head 3421417e, upstream tunix
# bf13cd2c for the trajectory reward key, and the packing budget fix (see
# docker/maz-q35/Dockerfile).
export TUNIX_IMAGE=gcr.io/cloud-tpu-multipod-dev/mazumdera-runner@sha256:e473f60f74c866e8f9f410c3692b4818c87bfdc6ff0696ecdb47b263cb808452
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
# The three containers of the trainer's `proc` pod share one n2d-standard-64 (245 GiB
# allocatable): pathways-rm, pathways-proxy and the trainer's own container. PR 2228 at
# head 3421417e adds explicit `requests` for the first two, which is what makes the
# limits below independent of scheduling: a container with only `limits` gets
# requests=limits, so before this the pod reserved 16+120+60 = 196 GiB and the proxy
# limit could not be raised without the pod becoming unschedulable. It now reserves
# 4+16+60 = 80 GiB and the limits are ceilings only.
#
# 120G on the proxy holds the ~70 GB Raiden stages device-to-host alongside the bounded
# checkpoint staging below. PR 2228's own default of 190G now fits as well, but 120G is
# the value the previous three runs used and nothing has shown it to be the constraint.
#
# 120G, not 70G, on the user container. The step-10 checkpoint save of maz-q35-2 was
# OOM-killed at 64.9 GiB against 70G with no weight sync running, so the save alone does
# not fit; a 35B bf16 model is ~70 GB, which is the obvious candidate for what it is
# staging. Requests still sum to 4+16+60 = 80 GiB, well under the node, so this changes
# nothing about scheduling -- it only stops the cgroup killing the save first. The three
# limits now sum to 256 GiB, above the node's 245 GiB, which is deliberate and is what
# limits are for: the proxy's own peak is the ~70 GB it stages for Raiden, so the
# realistic simultaneous worst case is 16 + 70 + 70 = 156 GiB.
export PATHWAYS_RM_MEMORY=4G
export PATHWAYS_PROXY_MEMORY=16G
export PATHWAYS_PROXY_MEMORY_LIMIT=120G
export USER_CONTAINER_MEMORY=60G
export USER_CONTAINER_MEMORY_LIMIT=120G
export PATHWAYS_WORKER_MEMORY=165G

# --- Rollout: 4 v5p chips, tp=2 x dp=2 ---------------------------------------
# TP rather than FSDP: FSDP on the rollout all-gathers the weights on every decode step,
# while TP leaves them sharded and splits the matmuls instead.
#
# TP is 2, not 4. At TP=4 `maxtext_utils` replicates `base_num_kv_heads` from 2 to 4,
# which MaxText then rejects because the qwen3.5-35b-a3b yml sets it too, and `gmm_v2`
# pads the MoE MLP dim from 512 to 1024 because 512/4 is not a multiple of 2x128. The
# experts are ~32B of this 35B model, so that padding roughly doubles what the rollout
# has to hold. At TP=2, 512/2 = 256 is already aligned and num_kv_heads is untouched.
# The remaining 2 chips go to data parallelism, which tunix PR 2229 plumbs through to
# the vLLM engine as `data_parallel_size`.
export ROLLOUT_JOBSET_YAML=jobset.tpu.yaml
export ROLLOUT_TPU_SLICE=tpuv5p:2x2x1
export ROLLOUT_MESH_FSDP=2
export ROLLOUT_MESH_TP=2
export ROLLOUT_REPLICAS=1
# Separate vLLM server process, not the in-process sampler. Before tunix PR 2229 this
# path could not weight-sync at all: tpu_inference publishes destination variable names
# bracketed (['base']['decoder']...) while the Raiden source side uses dotted keys, so
# preflight failed with all 633 source variables unmatched. 2229 canonicalises them.
export SAMPLER=vllm
export ENABLE_PREFIX_CACHING=false

# --- Weight sync -------------------------------------------------------------
export WEIGHT_SYNC_MODE=raiden
export USE_WEIGHT_CONVERTER=true
# Rollout only: the launcher passes --prefuse_moe_weights in start_rollout and nowhere
# else, and run_trainer_node defaults it to False. The two must differ. `[gate|up]`
# fusion is a global concatenation on the trainer and a per-shard interleave on the
# rollout, so fusing on both sides produces a permuted MoE weight -- and Raiden's
# checksums are permutation-invariant abs-sums, so weight-sync verification still
# passes. Confirm from the trainer log instead: `wi_0 shape=(256, 10, 2048, 512)`.
# A trailing 1024 means the trainer fused too.
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
# Saving at step 10 of maz-q35-2 OOM-killed the trainer's user container: the cgroup
# killed python at 64.9 GiB against the 70G limit, 36 s into the save, with no weight
# sync running. MaxText PR 5234's CKPT_D2H_CONCURRENT_GB=8 bound was already in force.
# maz-q35-3 ran 100 steps clean with saving off. Saving is back on here to exercise
# tunix PR 2228 at head 3421417e, which is the only thing that changed: that head
# removes the drain from prepare_weight_sync, so a save and the next step's Raiden
# transfer are now meant to overlap rather than serialise. Nothing in it lowers the
# peak of a save on its own, so this may reproduce the same OOM at step 10 -- which is
# the result worth having. See qwen35_report_v8.md section 6.1. Set to 0 to disable.
export CHECKPOINT_SAVE_INTERVAL_STEPS=${CHECKPOINT_SAVE_INTERVAL_STEPS:-10}
# PR 5234 already defaults `checkpoint_storage_device_host_concurrent_gb` to 8 in
# base.yml. Setting it explicitly in the trainer's environment makes the value visible
# in the trainer log ("CKPT_D2H_CONCURRENT_GB=8; overriding ..."), which is the only
# confirmation from outside the container that the bound is in force.
export TRAINER_EXTRA_ENV="CKPT_D2H_CONCURRENT_GB=8"
export MAXTEXT_OUTPUT_DIR=gs://mazumdera-bucket-cloud-tpu-multipod-dev/q35-runs/maz-q35-${RUN_N}/maxtext

export WANDB_PROJECT=${WANDB_PROJECT:-trellis-gsm8k}
export WANDB_RUN_NAME=maz-q35-${RUN_N}-${STEPS}steps
# Without an entity, wandb.init() raises (this key's viewer has no default entity),
# tunix downgrades that to one INFO line, and the run completes with no metrics at all.
export WANDB_ENTITY=${WANDB_ENTITY:-google-trellis}
: "${WANDB_API_KEY:?export WANDB_API_KEY before running; it is not stored in this file}"
export WANDB_API_KEY

# 0, not 1. DEBUG=1 turns on httpx wire-level logging, which floods the orchestrator log
# and makes the step-time and weight-sync lines hard to find over a 100-step run.
export DEBUG=0

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
# k8s_launcher.sh sources common/enter_kube_context.sh, which points KUBECONFIG at a
# per-cluster file of its own ($HOME/.kube/config.$PROJECT.$LOCATION_NAME.$CLUSTER) and
# runs `get-credentials` into it. That file authenticates as gcloud's active account and
# is not the kubeconfig validated below, so the launcher would discard whichever context
# this script just checked. The check below replaces it; set ENTER_KUBE_CONTEXT to the
# launcher's own script to restore the original behaviour.
export ENTER_KUBE_CONTEXT=${ENTER_KUBE_CONTEXT:-/dev/null}

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
