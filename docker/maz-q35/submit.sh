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
# This is Yixuan's yixuann-e2e-0912head-v8 plus seven patches: MaxText PR 5219 and 5234,
# tunix PR 2229, the in-image part of tunix PR 2228 at head 3421417e, upstream tunix
# bf13cd2c for the trajectory reward key, the packing budget fix, and upstream tunix
# babc1c70 + 0cfdab45 for per-rollout trajectory logging. Plus the tpu_sync_jax
# 2026-09-14 Raiden wheel in place of the base image's 2026-09-08 one
# (see docker/maz-q35/Dockerfile).
export TUNIX_IMAGE=gcr.io/cloud-tpu-multipod-dev/mazumdera-runner@sha256:428ef3033683eb50b4816672e0415c9e1c57ae5254ef1df1c3fc94d84ba160b7
# The two Pathways halves are deliberately on different tags: the server carries a fix
# on top of raiden_20260914 and the proxy does not have one. Both are the 2026-09-14
# Raiden generation, which is what the tpu_sync_jax wheel in the image above matches;
# runs up to maz-q35-4 used raiden_20260904 against the base image's 2026-09-08 wheel.
export PATHWAYS_SERVER_IMAGE=us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/datenglin/unsanitized_server:raiden_20260914_fix
export PATHWAYS_PROXY_IMAGE=us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/datenglin/unsanitized_proxy_server:raiden_20260914

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

# --- Rollout: 8 replicas x 4 v5p chips = 32 chips, tp=2 x dp=2 each ----------
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
# One rollout JobSet, back from the eight of maz-q35-6 and -7. Raiden cannot currently
# fan out: with 8 destinations the transfer group exceeds no K below 8, takes
# _execute_slice_broadcast, and hangs on the first hop of round 0 -- at K=1 (run 6,
# 684.18 s) and at K=4 (run 7, 694.83 s), both dying on the 600 s deadline hardcoded in
# tpu_sync's _send_rpc with no destination having received anything. See
# qwen35_report_v9.md. At 1 destination `len(unique_dst_units) > 1` is false, so the
# direct branch is taken and the 86.44 s sync of maz-q35-5 is the expected behaviour.
#
# The batch stays at 16 x 16 = 256 with 1024-token responses, so this replica does 8x
# the sequences and 2x the length of maz-q35-3's 4 x 8 at 512. Generation is the part
# that scales; weight sync (~60 s) and the optimizer step do not.
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
export MAX_RESPONSE_LENGTH=1024
export BATCH_SIZE=16
export NUM_GENERATIONS=16
export MINI_BATCH_SIZE=256          # = BATCH_SIZE * NUM_GENERATIONS: one optimizer step per rollout batch
export TRAIN_MICRO_BATCH_SIZE=8     # = TRAINER_MESH_FSDP: one packed row per FSDP shard per micro-batch
# Sequence packing. A row holds 4096 tokens against a 512+1024 worst-case sequence, so
# 2 sequences share a row in the worst case and the 256 sequences need at least 128 rows,
# or 16 micro-batches of 8. Short completions pack denser and cut that count.
# max_segments_per_packed_row is left unset so that max_packed_len alone bounds the row.
#
# Overridable so that an unpacked control arm can be run against the same seed and data
# order: MAX_SEQ_TOKEN_PER_TPU= bash submit.sh <n> <steps> start. k8s_launcher.sh omits
# --max_seq_token_per_tpu when this is empty, which selects PaddedBatchAssembler and
# leaves max_target_length at MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH. Note the `-`
# rather than `:-`: an explicitly empty value has to survive, not fall back to 4096.
export MAX_SEQ_TOKEN_PER_TPU="${MAX_SEQ_TOKEN_PER_TPU-4096}"

export MAX_STEPS="${STEPS}"

# --- Output ------------------------------------------------------------------
# Checkpointing is off, because the save's host footprint is unbounded. Two runs have
# now been OOM-killed writing the step-10 checkpoint, and each was killed at roughly 93%
# of whatever ceiling it was given:
#
#   maz-q35-2   70G limit    killed at  64.9 GiB   36 s into the save
#   maz-q35-4  120G limit    killed at 110.7 GiB   59 s into the save
#
# Both had MaxText PR 5234's CKPT_D2H_CONCURRENT_GB=8 in force and neither had a weight
# sync running, so neither that bound nor tunix PR 2228's overlap handling is reaching
# the path that allocates. Raising the limit again is not a fix: the node is 245 GiB and
# the Pathways proxy needs ~70 of it. maz-q35-3 ran 100 steps clean with saving off.
# See qwen35_report_v8.md section 6.1. Set to 10 to reproduce.
export CHECKPOINT_SAVE_INTERVAL_STEPS=${CHECKPOINT_SAVE_INTERVAL_STEPS:-0}
# PR 5234 already defaults `checkpoint_storage_device_host_concurrent_gb` to 8 in
# base.yml. Setting it explicitly in the trainer's environment makes the value visible
# in the trainer log ("CKPT_D2H_CONCURRENT_GB=8; overriding ..."), which is the only
# confirmation from outside the container that the bound is in force.
export TRAINER_EXTRA_ENV="CKPT_D2H_CONCURRENT_GB=8"

# RAIDEN_BROADCAST_K is deliberately unset. tpu_sync's RaidenController reads it at
# construction (rpc/raiden_controller.py:1416, default 64) and uses it twice: a transfer
# group only becomes a tree broadcast when its distinct destinations outnumber K, and K
# is then the fan-out passed to _execute_slice_broadcast. The one RaidenController in
# this run is built in the orchestrator process -- orchestrator.py calls
# create_default_handler, which builds RaidenHandler -> _RaidenTransport ->
# RaidenController -- which is why ORCHESTRATOR_EXTRA_ENV rather than TRAINER_EXTRA_ENV
# is the hook, should it be needed again.
#
# At ROLLOUT_REPLICAS=1 the value cannot matter: the tree-broadcast test also requires
# `len(unique_dst_units) > 1`, so a single destination always takes the direct branch.
# Setting it anyway would imply it was doing something.
# export ORCHESTRATOR_EXTRA_ENV="RAIDEN_BROADCAST_K=4"

export MAXTEXT_OUTPUT_DIR=gs://mazumdera-bucket-cloud-tpu-multipod-dev/q35-runs/maz-q35-${RUN_N}/maxtext

export WANDB_PROJECT=${WANDB_PROJECT:-trellis-gsm8k}
export WANDB_RUN_NAME=maz-q35-${RUN_N}-${STEPS}steps
# Without an entity, wandb.init() raises (this key's viewer has no default entity),
# tunix downgrades that to one INFO line, and the run completes with no metrics at all.
export WANDB_ENTITY=${WANDB_ENTITY:-google-trellis}
: "${WANDB_API_KEY:?export WANDB_API_KEY before running; it is not stored in this file}"
export WANDB_API_KEY

# 1 turns on httpx wire-level logging, which floods the orchestrator log, so read step
# times with `grep` rather than by scrolling. It is not what produces the trajectories --
# TRAJECTORY_LOG_DIR below is.
export DEBUG=1
# `env`, not `exact`. Both modes call the same vtc_completion_outcome, but they hand it
# different text: `env` scores action.action, the assistant turn alone, and `exact` scores
# metadata["text"], the whole chat transcript. The VTC prompt instructs the model to use
# <reasoning>...</reasoning> and <answer>\boxed{}</answer>, so the prompt itself carries
# every tag is_vtc_format_correct counts. It requires exactly one of each and sees two, so
# under `exact` format_ok is always false and a correct answer scores 0.5 instead of 1.0.
# Measured over steps 0-2: reward_mean 0.9375/0.8742/0.8871 under `env` against
# 0.4750/0.4582/0.4465 under `exact`, with answer extraction agreeing with gold on 99.2%
# of 930 trajectories in both. `env` is the correct one.
export REWARD_MODE=env

# One CSV row per consumed rollout: global_step, prompt_id, group_index, status, reward,
# question, prompt, completion, gold_answer. Written by StandardRLProgram, which reads
# traj["trajectory_reward"] rather than recomputing, so the reward column is the number
# the run actually trained on under either REWARD_MODE and DEBUG is not involved.
#
# This supersedes the gsm8k.make_gsm8k_reward_fn debug dump, which only exists under
# REWARD_MODE=exact -- maz-q35-8 ran ten steps with DEBUG=1 under `env` and logged zero
# "[Sampled Response]" lines, and maz-q35-9 got the dump under `exact` at the cost of the
# 0.5-capped reward above.
#
# TODO(tunix): trajectory_logger.log_item has no append path for gs:// (utils/
# trajectory_logger.py:122) -- each flush re-reads the whole CSV, concatenates and
# re-uploads, and each row embeds the full trajectory and token arrays, so the cost grows
# with the square of the run length. Watch the step time; a local dir plus a copy out is
# the fallback.
export LOG_DIR=gs://mazumdera-bucket-cloud-tpu-multipod-dev/q35-runs/maz-q35-${RUN_N}/logs
export TRAJECTORY_LOG_DIR=gs://mazumdera-bucket-cloud-tpu-multipod-dev/q35-runs/maz-q35-${RUN_N}/trajectories
# TODO(tunix): log_item mkdir()s log_path and then asserts is_dir() (utils/
# trajectory_logger.py:105), above the gs:// branch written to handle exactly this case.
# GCS has no directories, so on a bucket prefix mkdir is a no-op and is_dir is False until
# some object exists under it -- the logger can never write its own first file. maz-q35-10
# raised that assertion on all 100 steps and produced no CSV. One placeholder object is
# enough to satisfy it; measured in the image, log_item then appends correctly.
export TRAJECTORY_LOG_KEEP="${TRAJECTORY_LOG_DIR}/.keep"

# --- Job names ---------------------------------------------------------------
# The launcher derives every JobSet name from $USER: maz-q35-N-{orch,train,roll}.
export USER=maz-q35-${RUN_N}

echo "=== maz-q35-${RUN_N}: ${ACTION}, ${STEPS} steps"
echo "    image      ${TUNIX_IMAGE}"
echo "    trainer    ${TRAINER_TPU_SLICE} fsdp=${TRAINER_MESH_FSDP}"
echo "    rollout    ${ROLLOUT_REPLICAS} x ${ROLLOUT_TPU_SLICE} dp=${ROLLOUT_MESH_FSDP} tp=${ROLLOUT_MESH_TP} sampler=${SAMPLER}"
echo "    batch      ${BATCH_SIZE} prompts x ${NUM_GENERATIONS} gens = ${MINI_BATCH_SIZE}/step, resp<=${MAX_RESPONSE_LENGTH}"
if [[ -n "${MAX_SEQ_TOKEN_PER_TPU}" ]]; then
  echo "    packing    ${MAX_SEQ_TOKEN_PER_TPU} tok/row, micro-batch ${TRAIN_MICRO_BATCH_SIZE}"
else
  echo "    packing    OFF -- unpacked control, one sequence per $((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))-token row, micro-batch ${TRAIN_MICRO_BATCH_SIZE}"
fi
echo "    raiden     ${ORCHESTRATOR_EXTRA_ENV:-RAIDEN_BROADCAST_K unset (direct push)}"
echo "    reward     mode=${REWARD_MODE} debug=${DEBUG}"
echo "    traj csv   ${TRAJECTORY_LOG_DIR}"
echo "    priority   ${PRIORITY_CLASS_NAME}"
echo "    output     ${MAXTEXT_OUTPUT_DIR}"
if [ "${ROLLOUT_REPLICAS}" -gt 1 ]; then
  echo "    jobsets    ${USER}-orch ${USER}-train ${USER}-roll-0 .. -$((ROLLOUT_REPLICAS - 1))"
else
  echo "    jobsets    ${USER}-orch ${USER}-train ${USER}-roll"
fi
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

# Materialise the trajectory prefix before the orchestrator starts, so that log_item's
# is_dir assertion holds on its first flush. See the TODO above TRAJECTORY_LOG_KEEP.
if [[ "${ACTION}" == "start" ]] && [[ "${DRY_RUN:-false}" != "true" ]]; then
  if ! printf '' | gcloud storage cp - "${TRAJECTORY_LOG_KEEP}" --quiet; then
    echo "ERROR: could not create ${TRAJECTORY_LOG_KEEP}; the orchestrator would log no" >&2
    echo "       trajectories and fail the same assertion maz-q35-10 hit." >&2
    exit 1
  fi
  echo "    traj prefix created: ${TRAJECTORY_LOG_KEEP}"
  echo
fi

bash "${LAUNCHER}" "${ACTION}"
