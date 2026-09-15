#!/usr/bin/env bash
# Qwen3.5-35B-A3B distributed GRPO on TPU v5p -- reproduction launcher.
#
#   ./run_qwen35_repro.sh start     # launch orchestrator + trainer + 8 rollouts
#   ./run_qwen35_repro.sh stop      # tear everything down
#   DRY_RUN=true ./run_qwen35_repro.sh start   # print manifests, apply nothing
#
# Overridable knobs (all have defaults below):
#   MAX_STEPS, BATCH_SIZE, NUM_GENERATIONS, RUN_USER, RUN_TAG, TUNIX_IMAGE, DEBUG
#
# See qwen35_report_v7.md for the full rationale behind every value here.

set -euo pipefail

# ---------------------------------------------------------------------------
# 0. Identity
# ---------------------------------------------------------------------------
# k8s object names must be RFC-1123 labels, so no underscores. On corp GCE VMs
# $USER is e.g. "igorts_google_com", which produces jobset names the API server
# rejects. Everything downstream derives its job names from $USER, so sanitize
# it here rather than in a dozen places.
export USER="${RUN_USER:-$(echo "${USER:-$(whoami)}" | cut -d_ -f1 | tr -cd 'a-z0-9-')}"
if [[ -z "${USER}" || ! "${USER}" =~ ^[a-z0-9]([a-z0-9-]*[a-z0-9])?$ ]]; then
  echo "RUN_USER must be a valid k8s name component; got '${USER}'." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# 1. Cluster
# ---------------------------------------------------------------------------
export PROJECT="cloud-tpu-shared-capacity"
export CLUSTER="bodaborg-v5p-nap"
export LOCATION_NAME="europe-west4"
export REGION="europe-west4"
export ZONE="europe-west4-a"
export K8S_NAMESPACE="trellis"
export NAMESPACE="trellis"
export QUEUE_NAME="default"
export KUEUE_QUEUE_NAME="default"
export CPU_MACHINE="n2d-standard-64"
# Kueue WorkloadPriorityClass. Without it the workload sits at priority 0 and
# anything else in the tpu-shared-cohort evicts it; the first 100-step attempt
# died at step 43 that way. medium == 500.
export PRIORITY_CLASS="${PRIORITY_CLASS:-medium}"

# ---------------------------------------------------------------------------
# 2. Images
# ---------------------------------------------------------------------------
# Yixuan's e2e image plus a small file overlay; build it with
# ../../../../build_qwen35_overlay.sh (the file list lives in
# ../../../../Dockerfile.qwen35_overlay).
export TUNIX_IMAGE="${TUNIX_IMAGE:-gcr.io/cloud-tpu-multipod-dev/igorts_google_com-runner:qwen35-repro-v12}"
export PATHWAYS_SERVER_IMAGE="us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/datenglin/unsanitized_server:raiden_20260904"
export PATHWAYS_PROXY_IMAGE="us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/datenglin/unsanitized_proxy_server:raiden_20260904"
# The whole model is staged on the proxy host during Raiden D2H sync; the
# yaml generator's 100G default OOM-kills it for a 35B model.
export PATHWAYS_PROXY_MEMORY_LIMIT="250G"

# ---------------------------------------------------------------------------
# 3. Model and base checkpoint
# ---------------------------------------------------------------------------
export TRAINER_BACKEND="maxtext"
export MODEL_NAME="Qwen3.5-35B-A3B"
export MODEL_ID="Qwen/Qwen3.5-35B-A3B"
# Must match the MaxText pydantic literal, which is lowercase and dotted.
export MAXTEXT_MODEL_NAME="qwen3.5-35b-a3b"
export TOKENIZER_PATH="Qwen/Qwen3.5-35B-A3B"
export MAXTEXT_CKPT="gs://hengtaoguo-maxtext-logs/checkpoints/qwen3.5-35b-a3b/scanned/2026-06-11-10-27/0/items"
export MAXTEXT_OUTPUT_DIR="gs://yixuannwang-maxtext-dataset/${USER}/qwen35_repro"

# ---------------------------------------------------------------------------
# 4. Checkpointing: OFF
# ---------------------------------------------------------------------------
# 0 means "never save" on both sides: the orchestrator stops issuing save
# requests, and run_trainer_node hands MaxText a read-only CheckpointManager
# (restoring the base weights still works). Saving is being fixed separately;
# leaving it on here costs a 64 GiB write per interval.
export CHECKPOINT_SAVE_INTERVAL_STEPS=0
export ENABLE_PATHWAYS_PERSISTENCE=0

# ---------------------------------------------------------------------------
# 5. Trainer topology: one Pathways 2x2x2 slice, 8 chips, pure FSDP
# ---------------------------------------------------------------------------
export TRAINER_JOBSET_YAML="jobset.pathways.yaml"
# Scaling the trainer is about weight sync, not about training throughput.
# Measured at 8 chips: a 423s step is ~265s weight sync + ~150s rollout and
# only a few seconds of actual gradient work, so more chips buy nothing on the
# compute side. They only help if Raiden's D2H egress parallelizes across
# trainer hosts (v5p packs 4 chips per host, so 2x2x2 = 2 hosts, 2x2x4 = 4).
#
# TRAINER_MESH_FSDP must equal the chip count, and TRAIN_MICRO_BATCH_SIZE must
# be a multiple of it (maxtext_utils raises otherwise), so the three move
# together: 2x2x2/8/8, 2x2x4/16/16, 2x4x4/32/32.
export TRAINER_TPU_SLICE="${TRAINER_TPU_SLICE:-tpuv5p:2x2x2}"
export TRAINER_MESH_FSDP="${TRAINER_MESH_FSDP:-8}"
export TRAINER_MESH_TP=1
export TRAINER_MESH_EXPERT=1

# ---------------------------------------------------------------------------
# 6. Rollout topology: 8 independent 2x2x1 slices, 32 chips, pure TP
# ---------------------------------------------------------------------------
# Each slice is 4 chips, filled as TP=2 x DP=2. ROLLOUT_MESH_FSDP is a misnomer
# on the rollout side: nothing shards weights FSDP-style there, the value is
# handed to vLLM as data_parallel_size, i.e. whole engine replicas.
#
# TP must stay at 2. At TP=4, maxtext_utils replicates base_num_kv_heads 2 -> 4
# (which MaxText then rejects outright, since the model yml also sets it) and
# GMM_v2 pads the MoE MLP dim 512 -> 1024 because 512/4 is not a multiple of
# 2*128. The MoE experts are ~32B of this 35B model, so that padding roughly
# doubles what the trainer has to hold. At TP=2, 512/2 = 256 is already
# aligned and num_kv_heads is untouched.
export ROLLOUT_JOBSET_YAML="jobset.tpu.yaml"
export ROLLOUT_TPU_SLICE="tpuv5p:2x2x1"
export ROLLOUT_MESH_FSDP=2
export ROLLOUT_MESH_TP=2
export ROLLOUT_REPLICAS=8
# Must be inprocess_vllm. On the plain "vllm" path the Raiden destination
# registry is never populated with MaxText-named variables, so weight sync dies
# in preflight before step 0 with all 633 source variables unmatched
# ("source variable 'decoder.decoder_norm.scale' has no destination
# counterpart"). inprocess_vllm logs "Using local registration for destination
# metadata" and matches all 633. Same image, same mesh, same everything else.
export SAMPLER="${SAMPLER:-inprocess_vllm}"

# ---------------------------------------------------------------------------
# 7. Weight synchronization
# ---------------------------------------------------------------------------
export WEIGHT_SYNC_MODE="raiden"
export USE_WEIGHT_CONVERTER="${USE_WEIGHT_CONVERTER:-true}"
# This is the ROLLOUT's setting only. The trainer's is a separate variable
# (TRAINER_PREFUSE_MOE_WEIGHTS, defaulted false in k8s_launcher.sh) and the two
# are NOT supposed to agree -- `prefuse_moe_weights` means a different layout on
# each end. See the long comment in k8s_launcher.sh and report v7 section 6.
# Setting both true is what pinned reward at 0: the trainer's global-concat
# `wi` was copied verbatim, so at rollout TP=2 shard 0 got all-gate and shard 1
# all-up. That is a pure permutation, hence invisible to Raiden's abs-sum
# checksums.
export PREFUSE_MOE_WEIGHTS="${PREFUSE_MOE_WEIGHTS:-true}"
export TRAINER_PREFUSE_MOE_WEIGHTS="${TRAINER_PREFUSE_MOE_WEIGHTS:-false}"
# Deliberately overridable. VERIFY_WEIGHTS=true makes the Raiden delegate log
# destination checksums and transfer metrics after each h2d, which is the only
# handle we have on weight-sync correctness. A hardcoded `false` here would
# silently swallow `VERIFY_WEIGHTS=true ./run_qwen35_repro.sh start` --
# k8s_launcher.sh reads it long after this line.
export VERIFY_WEIGHTS="${VERIFY_WEIGHTS:-false}"
# Off: prompts in a GRPO batch share no prefix worth caching, and with it off
# the orchestrator is free to spread a prompt's generations across workers.
export ENABLE_PREFIX_CACHING=false

# ---------------------------------------------------------------------------
# 8. Batch shape
# ---------------------------------------------------------------------------
export MAX_STEPS="${MAX_STEPS:-100}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
export NUM_GENERATIONS="${NUM_GENERATIONS:-16}"
# One optimizer update per step over all BATCH_SIZE*NUM_GENERATIONS rollouts.
export MINI_BATCH_SIZE=$((BATCH_SIZE * NUM_GENERATIONS))
export MAX_PROMPT_LENGTH=512
# This budget is harsher than it looks, and it is implicated in reward being
# pinned at exactly 0. A trajectory that spends the whole budget in one turn is
# marked MAX_CONTEXT_LIMIT_REACHED, and trajectory_collect_engine.collect then
# skips _append_final_reward entirely -- so it scores a hard 0.0 no matter what
# the model wrote, rather than being graded and merely losing the format points.
# At 512 that was 27 of ~30 trajectories on rollout worker 0, which is why both
# reward_mean and reward_std were 0.0000: almost nothing was being graded at all.
# The prompt asks for "detailed step-by-step reasoning", so the budget has to
# cover the reasoning block *and* the closing tags with room to spare.
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
# Sequence packing. The packed row count per microbatch is
# trainer_fsdp * trainer_dp, and TRAIN_MICRO_BATCH_SIZE has to match it so
# MaxText gets per_device_batch_size = 1.
export TRAIN_MICRO_BATCH_SIZE="${TRAIN_MICRO_BATCH_SIZE:-${TRAINER_MESH_FSDP}}"
# Budget per packed row, and MaxText's max_target_length (the overlay makes the
# trainer take the larger of this and max_prompt+max_response).
#
# Left at the floor, packing is a no-op by construction: validate_packing_budget
# demands budget >= max_prompt+max_response, and stock maxtext_utils pins
# max_target_length to that same sum, so a row holds exactly one maximal
# trajectory. Measured that way: ~1.14 trajectories per row.
#
# At 4096 the same step packed 61/55/57/48/35 trajectories per microbatch
# (~7.6 per row), collapsing a step from 32 microbatches to 5. Note that this
# bought only ~4s of a 423s step -- gradient work was never the bottleneck --
# but it is strictly better and costs nothing.
export MAX_SEQ_TOKEN_PER_TPU="${MAX_SEQ_TOKEN_PER_TPU:-4096}"
export USE_ROLLOUT_LOGPS=true
# Qwen3 chat models close a turn with <|im_end|>, not the tokenizer's EOS, so
# without this every generation runs to MAX_RESPONSE_LENGTH and trips the
# overlong filter described above. k8s_launcher.sh now defaults it to the same
# value; it is repeated here because the measured 100-step run predates that
# default and the difference matters when comparing against report v7.
export EOS_TOKENS="${EOS_TOKENS-<|im_end|>}"

# ---------------------------------------------------------------------------
# 9. Optimizer
# ---------------------------------------------------------------------------
# Note: USE_LORA is not plumbed through to either node in this image, so this
# is a full-parameter fine-tune. Do not set USE_LORA and expect LoRA.
export LEARNING_RATE=2.0e-7
export ADAM_B1=0.9
export ADAM_B2=0.99
export ADAM_EPS=1.0e-8
export WEIGHT_DECAY=0.01
export OPT_CHAIN_TYPE=clip_by_global_norm
export MAX_GRAD_NORM=1.0
# Empty selects a constant learning rate. k8s_launcher.sh defaults to
# warmup_cosine_decay_schedule over LR_DECAY_STEPS=500, which for a 100-step
# run means never leaving the warmup ramp -- the run would train at an average
# LR far below 2e-7 and the measured steps in report v7 would not be
# comparable. Set SCHEDULE_TYPE=warmup_cosine_decay_schedule for real training
# runs long enough for the schedule to make sense.
export SCHEDULE_TYPE=""
export BETA=0
export EPSILON=0.2

# ---------------------------------------------------------------------------
# 10. Metrics
# ---------------------------------------------------------------------------
# .bashrc returns early for non-interactive shells, so sourcing it is not
# enough to pick the key up; read the export line directly.
if [[ -z "${WANDB_API_KEY:-}" && -f "${HOME}/.bashrc" ]]; then
  eval "$(grep -m1 '^export WANDB_API_KEY=' "${HOME}/.bashrc" || true)"
fi
export WANDB_API_KEY="${WANDB_API_KEY:-}"
if [[ -z "${WANDB_API_KEY}" ]]; then
  echo "WARNING: WANDB_API_KEY is unset; the run will not log to Weights & Biases." >&2
fi
# Required whenever the key's viewer has no defaultEntity, which is the common
# case for a team key: wandb.init() then refuses to start and tunix logs a
# single INFO line before carrying on with metrics silently off.
export WANDB_ENTITY="${WANDB_ENTITY:-google-trellis}"
export WANDB_PROJECT="${WANDB_PROJECT:-qwen35-35b-a3b-grpo}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-${USER}-${RUN_TAG:-repro}-b${BATCH_SIZE}g${NUM_GENERATIONS}-s${MAX_STEPS}}"
export FLUSH_METRICS_EVERY_N_STEPS=1

export DEBUG="${DEBUG:-0}"
export DRY_RUN="${DRY_RUN:-false}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/k8s_launcher.sh" "${1:-start}"
