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

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

# ==============================================================================
# Hardware & Cluster Profiles
# ==============================================================================
# Supported profiles:
#   - bodaborg: TPU v7x slice (2x2x1) on cluster bodaborg-tpu7x-nap (us-central1)
#   - v5p:      TPU v5p slice (2x2x2 trainer, 2x2x1 rollout) on trellis-demo-0810 / mlperf-v5p
PROFILE="bodaborg"

# Workload configuration defaults
STEPS=100
WEIGHT_SYNC_MODE="raiden"
BATCH_SIZE=2
NUM_GENERATIONS=2
TRAIN_MICRO_BATCH_SIZE=1
MAX_PROMPT_LENGTH=512
MAX_RESPONSE_LENGTH=128
DEBUG=1

# Model & Checkpoint defaults
MODEL_NAME="Qwen3-1.7B"
MODEL_ID="Qwen/Qwen3-1.7B"
TOKENIZER_PATH="Qwen/Qwen3-1.7B"
TRAINER_BACKEND="maxtext"
MAXTEXT_MODEL_NAME="qwen3-1.7b"
MAXTEXT_CKPT="gs://cloud-tpu-multipod-dev-bucket-europe-west4/users/atwigg/checkpoints/qwen3_1.7b_fixed/0/items"

# Container Image & Git Sync defaults
TUNIX_IMAGE="gcr.io/cloud-tpu-multipod-dev/wuhaotest/tunix-maxtext-rlvllm"
DEFAULT_BRANCH="$(git -C "${REPO_ROOT}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "atwigg/gsm8k-dist-fixes")"
SYNC_GIT_BRANCH="${DEFAULT_BRANCH}"

# WandB defaults
WANDB_PROJECT="trellis-gsm8k"
WANDB_RUN_NAME=""
WANDB_API_KEY="${WANDB_API_KEY:-}"

# Flag override trackers
USER_SET_CLUSTER=""
USER_SET_PROJECT=""
USER_SET_REGION=""
USER_SET_CPU_MACHINE=""
USER_SET_TRAINER_TPU=""
USER_SET_ROLLOUT_TPU=""
USER_SET_TRAINER_FSDP=""
USER_SET_ROLLOUT_TP=""

USER_PREFIX="${USER:-$(whoami)}"
ORCHESTRATOR_ID="${USER_PREFIX}-orch"
TRAINER_ID="${USER_PREFIX}-train"
ROLLOUT_ID="${USER_PREFIX}-roll"

DRY_RUN=0
FOLLOW_LOGS=0
TAIL_LINES=100
SUBCOMMAND="start"

# ==============================================================================
# Helper: Print Usage
# ==============================================================================
usage() {
  cat <<EOF
Usage: $(basename "$0") [SUBCOMMAND] [OPTIONS]

Clean, modular submission and management tool for Distributed GSM8K GRPO training.

SUBCOMMANDS:
  start            Start distributed training jobsets (Default)
  stop             Stop and clean up active jobsets
  restart          Restart active jobsets (stop then start)
  status           Show live status of JobSets, Pods, and Kueue Workloads
  logs <TARGET>    Stream logs (-f) for a component: orch, train, or roll
  watch            Continuously monitor pods and Kueue reservation
  help, -h         Show this help message

KEY OPTIONS:
  --steps <N>              Total training steps [Default: 100]
  --weight-sync <MODE>     Weight sync: raiden (real sync), noop (dry test), none [Default: raiden]
  --profile <PROFILE>      Hardware profile preset: bodaborg (v7x) or v5p [Default: bodaborg]
  --cluster <NAME>         Target cluster [Default: bodaborg-tpu7x-nap]
  --project <ID>           GCP Project ID [Default: cloud-tpu-shared-capacity]
  --region <REGION>        GCP Region [Default: us-central1]
  --branch <BRANCH>        Git branch to sync inside pods [Default: current git branch]
  --ckpt <GCS_PATH>        Orbax checkpoint path for MaxText
  --image <IMAGE>          Container image [Default: tunix-maxtext-rlvllm]
  --batch-size <N>         Prompt groups per step [Default: 2]
  --num-generations <N>    Completions per prompt [Default: 2]
  --micro-batch <N>        Micro batch size [Default: 1]
  --wandb-project <NAME>   Weights & Biases project name [Default: trellis-gsm8k]
  --wandb-run <NAME>       Weights & Biases run name
  --dry-run                Display resolved configuration without executing commands

EXAMPLES:
  # Launch 100-step training with real Raiden weight sync on bodaborg-tpu7x-nap:
  $(basename "$0") start --steps 100 --weight-sync raiden

  # Launch a rapid 10-step dry-run with noop weight sync:
  $(basename "$0") start --steps 10 --weight-sync noop

  # Check pod and Kueue admission status:
  $(basename "$0") status

  # Stream orchestrator logs:
  $(basename "$0") logs orch

  # Stop and clean up all jobsets:
  $(basename "$0") stop

EOF
}

# ==============================================================================
# Parse Subcommand and Flags
# ==============================================================================
if [[ $# -gt 0 ]]; then
  case "$1" in
    start|stop|restart|status|logs|watch)
      SUBCOMMAND="$1"
      shift
      ;;
    help|-h|--help)
      usage
      exit 0
      ;;
    -*)
      SUBCOMMAND="start"
      ;;
  esac
fi

LOG_TARGET=""
if [[ "$SUBCOMMAND" == "logs" && $# -gt 0 && ! "$1" =~ ^- ]]; then
  LOG_TARGET="$1"
  shift
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      PROFILE="$2"
      shift 2
      ;;
    --profile=*)
      PROFILE="${1#*=}"
      shift
      ;;
    --steps)
      STEPS="$2"
      shift 2
      ;;
    --steps=*)
      STEPS="${1#*=}"
      shift
      ;;
    --weight-sync)
      WEIGHT_SYNC_MODE="$2"
      shift 2
      ;;
    --weight-sync=*)
      WEIGHT_SYNC_MODE="${1#*=}"
      shift
      ;;
    --cluster)
      USER_SET_CLUSTER="$2"
      shift 2
      ;;
    --cluster=*)
      USER_SET_CLUSTER="${1#*=}"
      shift
      ;;
    --project)
      USER_SET_PROJECT="$2"
      shift 2
      ;;
    --project=*)
      USER_SET_PROJECT="${1#*=}"
      shift
      ;;
    --region)
      USER_SET_REGION="$2"
      shift 2
      ;;
    --region=*)
      USER_SET_REGION="${1#*=}"
      shift
      ;;
    --branch)
      SYNC_GIT_BRANCH="$2"
      shift 2
      ;;
    --branch=*)
      SYNC_GIT_BRANCH="${1#*=}"
      shift
      ;;
    --ckpt)
      MAXTEXT_CKPT="$2"
      shift 2
      ;;
    --ckpt=*)
      MAXTEXT_CKPT="${1#*=}"
      shift
      ;;
    --image)
      TUNIX_IMAGE="$2"
      shift 2
      ;;
    --image=*)
      TUNIX_IMAGE="${1#*=}"
      shift
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --batch-size=*)
      BATCH_SIZE="${1#*=}"
      shift
      ;;
    --num-generations)
      NUM_GENERATIONS="$2"
      shift 2
      ;;
    --num-generations=*)
      NUM_GENERATIONS="${1#*=}"
      shift
      ;;
    --micro-batch)
      TRAIN_MICRO_BATCH_SIZE="$2"
      shift 2
      ;;
    --micro-batch=*)
      TRAIN_MICRO_BATCH_SIZE="${1#*=}"
      shift
      ;;
    --wandb-project)
      WANDB_PROJECT="$2"
      shift 2
      ;;
    --wandb-project=*)
      WANDB_PROJECT="${1#*=}"
      shift
      ;;
    --wandb-run)
      WANDB_RUN_NAME="$2"
      shift 2
      ;;
    --wandb-run=*)
      WANDB_RUN_NAME="${1#*=}"
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -f|--follow)
      FOLLOW_LOGS=1
      shift
      ;;
    --tail|-n)
      TAIL_LINES="$2"
      shift 2
      ;;
    --tail=*|-n=*)
      TAIL_LINES="${1#*=}"
      shift
      ;;
    help|-h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown flag: $1" >&2
      usage
      exit 1
      ;;
  esac
done

# ==============================================================================
# Resolve Profile Settings
# ==============================================================================
if [[ "$PROFILE" == "bodaborg" ]]; then
  PROJECT="cloud-tpu-shared-capacity"
  CLUSTER="bodaborg-tpu7x-nap"
  REGION="us-central1"
  CPU_MACHINE="e2-standard-16"
  TRAINER_JOBSET_YAML="jobset.tpu.yaml"
  TRAINER_TPU_SLICE="tpu7x:2x2x1"
  TRAINER_MESH_FSDP=4
  ROLLOUT_TPU_SLICE="tpu7x:2x2x1"
  ROLLOUT_MESH_TP=4
elif [[ "$PROFILE" == "v5p" ]]; then
  PROJECT="cloud-tpu-multipod-dev"
  CLUSTER="trellis-demo-0810"
  REGION="europe-west4"
  CPU_MACHINE="n2-standard-64"
  TRAINER_JOBSET_YAML="jobset.tpu.yaml"
  TRAINER_TPU_SLICE="tpuv5:2x2x2"
  TRAINER_MESH_FSDP=8
  ROLLOUT_TPU_SLICE="tpuv5:2x2x1"
  ROLLOUT_MESH_TP=4
else
  echo "Error: Unknown profile '$PROFILE'. Supported profiles: 'bodaborg', 'v5p'." >&2
  exit 1
fi

# Apply explicit flag overrides
if [[ -n "$USER_SET_CLUSTER" ]]; then CLUSTER="$USER_SET_CLUSTER"; fi
if [[ -n "$USER_SET_PROJECT" ]]; then PROJECT="$USER_SET_PROJECT"; fi
if [[ -n "$USER_SET_REGION" ]]; then REGION="$USER_SET_REGION"; fi
if [[ -n "$USER_SET_CPU_MACHINE" ]]; then CPU_MACHINE="$USER_SET_CPU_MACHINE"; fi
if [[ -n "$USER_SET_TRAINER_TPU" ]]; then TRAINER_TPU_SLICE="$USER_SET_TRAINER_TPU"; fi
if [[ -n "$USER_SET_ROLLOUT_TPU" ]]; then ROLLOUT_TPU_SLICE="$USER_SET_ROLLOUT_TPU"; fi
if [[ -n "$USER_SET_TRAINER_FSDP" ]]; then TRAINER_MESH_FSDP="$USER_SET_TRAINER_FSDP"; fi
if [[ -n "$USER_SET_ROLLOUT_TP" ]]; then ROLLOUT_MESH_TP="$USER_SET_ROLLOUT_TP"; fi

# ==============================================================================
# Helper Functions
# ==============================================================================
ensure_kube_context() {
  local context_name="gke_${PROJECT}_${REGION}_${CLUSTER}"
  if kubectl config get-contexts "$context_name" &>/dev/null; then
    kubectl config use-context "$context_name" >/dev/null
  else
    echo "Configuring credentials for cluster '${CLUSTER}' in project '${PROJECT}'..."
    gcloud container clusters get-credentials "${CLUSTER}" \
      --region="${REGION}" \
      --project="${PROJECT}" \
      --dns-endpoint >/dev/null
    kubectl config use-context "$context_name" >/dev/null
  fi
  kubectl config set-context --current --namespace=default >/dev/null
}

show_status() {
  echo "================================================================================"
  echo "  Active Distributed GSM8K JobSets (${USER_PREFIX})"
  echo "================================================================================"
  kubectl get jobsets "${ORCHESTRATOR_ID}" "${TRAINER_ID}" "${ROLLOUT_ID}" 2>/dev/null || true

  echo
  echo "================================================================================"
  echo "  Pods Status"
  echo "================================================================================"
  kubectl get pods -l "jobset.sigs.k8s.io/jobset-name in (${ORCHESTRATOR_ID},${TRAINER_ID},${ROLLOUT_ID})" -o wide 2>/dev/null || true

  echo
  echo "================================================================================"
  echo "  Kueue Workloads"
  echo "================================================================================"
  kubectl get workloads -A -o wide 2>/dev/null | grep -E "${USER_PREFIX}|NAME" || true

  echo
  echo "================================================================================"
  echo "  TPU Cohort Quota Status"
  echo "================================================================================"
  kubectl get clusterqueues -o json 2>/dev/null | jq -r '
    .items[] | 
    "\(.metadata.name): " +
    "Nominal Quota=\(.spec.resourceGroups[0].flavors[]? | select(.name=="tpu7x-flavor") | .resources[]? | select(.name=="google.com/tpu") | .nominalQuota // 0) | " +
    "Reserved=\(.status.flavorsReservation[]? | select(.name=="tpu7x-flavor") | .resources[]? | select(.name=="google.com/tpu") | .total // 0)"
  ' || true
  echo "================================================================================"
}

stream_logs() {
  local target="${1:-orch}"
  local pod_label=""
  case "$target" in
    orch|orchestrator)
      pod_label="jobset.sigs.k8s.io/jobset-name=${ORCHESTRATOR_ID}"
      ;;
    train|trainer)
      pod_label="jobset.sigs.k8s.io/jobset-name=${TRAINER_ID}"
      ;;
    roll|rollout)
      pod_label="jobset.sigs.k8s.io/jobset-name=${ROLLOUT_ID}"
      ;;
    *)
      echo "Unknown target '$target'. Use 'orch', 'train', or 'roll'." >&2
      exit 1
      ;;
  esac

  echo "Finding pod for ${target} (${pod_label})..."
  local pod_name
  pod_name="$(kubectl get pods -l "${pod_label}" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
  if [[ -z "$pod_name" ]]; then
    echo "No pod found matching label: ${pod_label}" >&2
    exit 1
  fi

  if [[ "$FOLLOW_LOGS" -eq 1 ]]; then
    echo "Streaming logs from pod: ${pod_name} (container: main, follow=true)..."
    kubectl logs -f "${pod_name}" -c main
  else
    echo "Fetching last ${TAIL_LINES} lines from pod: ${pod_name} (container: main)..."
    echo "Tip: add -f to stream/follow logs continuously."
    echo "--------------------------------------------------------------------------------"
    kubectl logs --tail="${TAIL_LINES}" "${pod_name}" -c main
    echo "--------------------------------------------------------------------------------"
  fi
}

stop_workloads() {
  echo "Stopping distributed GSM8K jobsets for user '${USER_PREFIX}'..."
  kubectl delete jobset "${ORCHESTRATOR_ID}" "${TRAINER_ID}" "${ROLLOUT_ID}" 2>/dev/null || true
  echo "Cleanup complete."
}

start_workloads() {
  echo "================================================================================"
  echo "  Launching Distributed GSM8K GRPO Training (${STEPS} steps)"
  echo "================================================================================"
  echo "  Profile:             ${PROFILE}"
  echo "  Cluster:             ${CLUSTER} (${PROJECT} / ${REGION})"
  echo "  Steps:               ${STEPS}"
  echo "  Weight Sync:         ${WEIGHT_SYNC_MODE}"
  echo "  Git Branch Sync:     ${SYNC_GIT_BRANCH}"
  echo "  Trainer Backend:     ${TRAINER_BACKEND} (${TRAINER_TPU_SLICE}, FSDP=${TRAINER_MESH_FSDP})"
  echo "  Rollout Slice:       ${ROLLOUT_TPU_SLICE} (TP=${ROLLOUT_MESH_TP})"
  echo "  Batch / Num Gen:     ${BATCH_SIZE} prompts / ${NUM_GENERATIONS} completions per prompt"
  echo "  Micro Batch Size:    ${TRAIN_MICRO_BATCH_SIZE}"
  echo "  Max Prompt / Resp:   ${MAX_PROMPT_LENGTH} / ${MAX_RESPONSE_LENGTH}"
  echo "  MaxText Checkpoint:  ${MAXTEXT_CKPT}"
  echo "  Image:               ${TUNIX_IMAGE}"
  echo "  WandB Project:       ${WANDB_PROJECT}"
  if [[ -n "${WANDB_RUN_NAME}" ]]; then
    echo "  WandB Run:           ${WANDB_RUN_NAME}"
  fi
  echo "================================================================================"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[Dry Run] Commands will not be executed."
    return 0
  fi

  # Stop any existing runs first to prevent duplicate state
  stop_workloads

  echo "Deploying Orchestrator, Trainer, and Rollout JobSets..."
  PROJECT="${PROJECT}" \
  CLUSTER="${CLUSTER}" \
  REGION="${REGION}" \
  CPU_MACHINE="${CPU_MACHINE}" \
  TRAINER_JOBSET_YAML="${TRAINER_JOBSET_YAML}" \
  TRAINER_TPU_SLICE="${TRAINER_TPU_SLICE}" \
  TRAINER_MESH_FSDP="${TRAINER_MESH_FSDP}" \
  ROLLOUT_TPU_SLICE="${ROLLOUT_TPU_SLICE}" \
  ROLLOUT_MESH_TP="${ROLLOUT_MESH_TP}" \
  SYNC_GIT_BRANCH="${SYNC_GIT_BRANCH}" \
  WEIGHT_SYNC_MODE="${WEIGHT_SYNC_MODE}" \
  TRAINER_BACKEND="${TRAINER_BACKEND}" \
  MAXTEXT_MODEL_NAME="${MAXTEXT_MODEL_NAME}" \
  MAXTEXT_CKPT="${MAXTEXT_CKPT}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  NUM_GENERATIONS="${NUM_GENERATIONS}" \
  TRAIN_MICRO_BATCH_SIZE="${TRAIN_MICRO_BATCH_SIZE}" \
  MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH}" \
  MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH}" \
  MAX_STEPS="${STEPS}" \
  DEBUG="${DEBUG}" \
  WANDB_PROJECT="${WANDB_PROJECT}" \
  WANDB_RUN_NAME="${WANDB_RUN_NAME}" \
  WANDB_API_KEY="${WANDB_API_KEY}" \
  bash "${SCRIPT_DIR}/k8s_launcher.sh" --command=start --image="${TUNIX_IMAGE}"

  echo
  echo "JobSets submitted successfully!"
  echo
  echo "Useful follow-up commands:"
  echo "  Check status:        $(basename "$0") status"
  echo "  Stream Orchestrator: $(basename "$0") logs orch"
  echo "  Stream Trainer:      $(basename "$0") logs train"
  echo "  Stream Rollout:      $(basename "$0") logs roll"
  echo "  Stop training:       $(basename "$0") stop"
}

# ==============================================================================
# Main Dispatcher
# ==============================================================================
if [[ "$DRY_RUN" -eq 0 ]]; then
  ensure_kube_context
fi

case "$SUBCOMMAND" in
  start)
    start_workloads
    ;;
  stop)
    stop_workloads
    ;;
  restart)
    stop_workloads
    sleep 3
    start_workloads
    ;;
  status)
    show_status
    ;;
  logs)
    stream_logs "${LOG_TARGET:-orch}"
    ;;
  watch)
    watch -n 3 "$(cd "${SCRIPT_DIR}" && pwd)/$(basename "$0") status"
    ;;
  *)
    echo "Unknown subcommand: $SUBCOMMAND" >&2
    usage
    exit 1
    ;;
esac
