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
# Dynamic Context & Cluster Detection (No Hardcoded Cluster Names)
# ==============================================================================
# Infer target cluster, project, and region from active kubectl context
CURRENT_CONTEXT="$(kubectl config current-context 2>/dev/null || true)"
DETECTED_PROJECT=""
DETECTED_REGION=""
DETECTED_CLUSTER=""

if [[ "$CURRENT_CONTEXT" =~ ^gke_([^_]+)_([^_]+)_(.+)$ ]]; then
  DETECTED_PROJECT="${BASH_REMATCH[1]}"
  DETECTED_REGION="${BASH_REMATCH[2]}"
  DETECTED_CLUSTER="${BASH_REMATCH[3]}"
fi

# Flag overrides
USER_SET_CLUSTER=""
USER_SET_PROJECT=""
USER_SET_REGION=""
USER_SET_CPU_MACHINE=""
USER_SET_TRAINER_TPU=""
USER_SET_ROLLOUT_TPU=""
USER_SET_TRAINER_FSDP=""
USER_SET_ROLLOUT_TP=""
USER_SET_TRAINER_YAML=""

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
GCS_SCRATCH_LOCATION="${GCS_SCRATCH_LOCATION:-gs://cloud-tpu-multipod-dev-bucket-europe-west4/tmp}"

# Container Image & Git Sync defaults
TUNIX_IMAGE="gcr.io/cloud-tpu-multipod-dev/wuhaotest/tunix-maxtext-rlvllm"
DEFAULT_BRANCH="$(git -C "${REPO_ROOT}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "atwigg/gsm8k-dist-fixes")"
SYNC_GIT_BRANCH="${DEFAULT_BRANCH}"

# WandB defaults
WANDB_PROJECT="trellis-gsm8k"
WANDB_RUN_NAME=""
WANDB_API_KEY="${WANDB_API_KEY:-}"

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

Modular submission and management tool for Distributed GSM8K GRPO training.
Automatically detects the target cluster from your active kubectl context or CLI flags.

SUBCOMMANDS:
  start            Start distributed training jobsets (Default)
  stop             Stop and clean up active jobsets
  restart          Restart active jobsets (stop then start)
  status           Show live status of JobSets, Pods, and Kueue Workloads
  logs <TARGET>    Stream logs for a component: orch, train, or roll (add -f to follow)
  watch            Continuously monitor pods and Kueue reservation
  help, -h         Show this help message

CLUSTER OPTIONS:
  --cluster <NAME>         Target cluster name (default: detected from kubectl context)
  --project <ID>           GCP Project ID (default: detected from kubectl context)
  --region <REGION>        GCP Region (default: detected from kubectl context)
  --cpu-machine <TYPE>     Instance type for CPU orchestrator node (default: auto-detected)
  --tpu-slice <SLICE>      TPU slice topology, e.g. tpu7x:2x2x1 or tpuv5:2x2x1 (default: auto-detected)
  --trainer-tpu-slice <S>  Custom TPU slice for trainer
  --rollout-tpu-slice <S>  Custom TPU slice for rollout

TRAINING OPTIONS:
  --steps <N>              Total training steps [Default: 100]
  --weight-sync <MODE>     Weight sync: raiden (real sync), noop (dry test), none [Default: raiden]
  --batch-size <N>         Prompt groups per step [Default: 2]
  --num-generations <N>    Completions per prompt [Default: 2]
  --micro-batch <N>        Micro batch size [Default: 1]
  --branch <BRANCH>        Git branch to sync inside pods [Default: current git branch]
  --ckpt <GCS_PATH>        Orbax checkpoint path for MaxText
  --image <IMAGE>          Container image [Default: tunix-maxtext-rlvllm]
  --wandb-project <NAME>   Weights & Biases project name [Default: trellis-gsm8k]
  --wandb-run <NAME>       Weights & Biases run name
  --dry-run                Display resolved configuration without executing commands

LOGGING OPTIONS:
  -f, --follow             Follow/stream logs continuously (with 'logs' subcommand)
  -n, --tail <N>           Number of log lines to show (default: 100)

EXAMPLES:
  # Launch 100-step training with real Raiden weight sync on the active cluster:
  $(basename "$0") start --steps 100 --weight-sync raiden

  # Target a specific cluster explicitly:
  $(basename "$0") start --cluster my-cluster --project my-project --region us-central1

  # Launch a rapid 10-step dry-run with noop weight sync:
  $(basename "$0") start --steps 10 --weight-sync noop

  # Check pod and Kueue admission status:
  $(basename "$0") status

  # Stream orchestrator logs:
  $(basename "$0") logs orch -f

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
    --cpu-machine)
      USER_SET_CPU_MACHINE="$2"
      shift 2
      ;;
    --cpu-machine=*)
      USER_SET_CPU_MACHINE="${1#*=}"
      shift
      ;;
    --tpu-slice)
      USER_SET_TRAINER_TPU="$2"
      USER_SET_ROLLOUT_TPU="$2"
      shift 2
      ;;
    --tpu-slice=*)
      USER_SET_TRAINER_TPU="${1#*=}"
      USER_SET_ROLLOUT_TPU="${1#*=}"
      shift
      ;;
    --trainer-tpu-slice)
      USER_SET_TRAINER_TPU="$2"
      shift 2
      ;;
    --trainer-tpu-slice=*)
      USER_SET_TRAINER_TPU="${1#*=}"
      shift
      ;;
    --rollout-tpu-slice)
      USER_SET_ROLLOUT_TPU="$2"
      shift 2
      ;;
    --rollout-tpu-slice=*)
      USER_SET_ROLLOUT_TPU="${1#*=}"
      shift
      ;;
    --trainer-mesh-fsdp)
      USER_SET_TRAINER_FSDP="$2"
      shift 2
      ;;
    --trainer-mesh-fsdp=*)
      USER_SET_TRAINER_FSDP="${1#*=}"
      shift
      ;;
    --rollout-mesh-tp)
      USER_SET_ROLLOUT_TP="$2"
      shift 2
      ;;
    --rollout-mesh-tp=*)
      USER_SET_ROLLOUT_TP="${1#*=}"
      shift
      ;;
    --trainer-jobset-yaml)
      USER_SET_TRAINER_YAML="$2"
      shift 2
      ;;
    --trainer-jobset-yaml=*)
      USER_SET_TRAINER_YAML="${1#*=}"
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
    --gcs-scratch)
      GCS_SCRATCH_LOCATION="$2"
      shift 2
      ;;
    --gcs-scratch=*)
      GCS_SCRATCH_LOCATION="${1#*=}"
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
# Resolve Target Cluster & Context
# ==============================================================================
CLUSTER="${USER_SET_CLUSTER:-${DETECTED_CLUSTER:-${CLUSTER:-}}}"
PROJECT="${USER_SET_PROJECT:-${DETECTED_PROJECT:-${PROJECT:-}}}"
REGION="${USER_SET_REGION:-${DETECTED_REGION:-${REGION:-}}}"

if [[ -z "$CLUSTER" ]]; then
  echo "Error: Could not detect target cluster from active kubectl context or CLI flags." >&2
  echo "Please connect to a cluster or pass --cluster <NAME> --project <PROJECT_ID> --region <REGION>." >&2
  exit 1
fi

ensure_kube_context() {
  local context_name="gke_${PROJECT}_${REGION}_${CLUSTER}"
  if kubectl config get-contexts "$context_name" &>/dev/null; then
    kubectl config use-context "$context_name" >/dev/null
  elif [[ -n "$PROJECT" && -n "$REGION" ]]; then
    echo "Configuring credentials for cluster '${CLUSTER}' in project '${PROJECT}'..."
    gcloud container clusters get-credentials "${CLUSTER}" \
      --region="${REGION}" \
      --project="${PROJECT}" \
      --dns-endpoint >/dev/null
    kubectl config use-context "$context_name" >/dev/null
  else
    # Keep current context if valid
    local active="$(kubectl config current-context 2>/dev/null || true)"
    if [[ -z "$active" ]]; then
      echo "Error: Unable to determine or switch kubectl context." >&2
      exit 1
    fi
  fi
  kubectl config set-context --current --namespace=default >/dev/null
}

if [[ "$DRY_RUN" -eq 0 ]]; then
  ensure_kube_context
fi

# ==============================================================================
# Auto-Detect Hardware from Cluster Nodes
# ==============================================================================
# 1. CPU Machine Type (select an untainted node instance type)
if [[ -n "$USER_SET_CPU_MACHINE" ]]; then
  CPU_MACHINE="$USER_SET_CPU_MACHINE"
else
  DETECTED_CPU="$(kubectl get nodes -o json 2>/dev/null | jq -r '.items[] | select(.metadata.labels["cloud.google.com/gke-tpu-accelerator"] == null) | select((.spec.taints // []) | length == 0) | .metadata.labels["node.kubernetes.io/instance-type"]' | head -n 1 || true)"
  CPU_MACHINE="${DETECTED_CPU:-e2-standard-16}"
fi

# 2. TPU Accelerator & Topology
DETECTED_ACCEL="$(kubectl get nodes -o json 2>/dev/null | jq -r '.items[].metadata.labels["cloud.google.com/gke-tpu-accelerator"] // empty' | sort -u | head -n 1 || true)"
DETECTED_TOPOLOGY="$(kubectl get nodes -o json 2>/dev/null | jq -r '.items[].metadata.labels["cloud.google.com/gke-tpu-topology"] // empty' | sort -V | head -n 1 || true)"

AUTO_TPU_SLICE=""
if [[ -n "$DETECTED_ACCEL" && -n "$DETECTED_TOPOLOGY" ]]; then
  AUTO_TPU_SLICE="${DETECTED_ACCEL}:${DETECTED_TOPOLOGY}"
elif [[ -n "$DETECTED_ACCEL" ]]; then
  AUTO_TPU_SLICE="${DETECTED_ACCEL}:2x2x1"
else
  AUTO_TPU_SLICE="tpu7x:2x2x1"
fi

TRAINER_TPU_SLICE="${USER_SET_TRAINER_TPU:-$AUTO_TPU_SLICE}"
ROLLOUT_TPU_SLICE="${USER_SET_ROLLOUT_TPU:-$AUTO_TPU_SLICE}"

# 3. Mesh dimensions (derive chip count from topology if not specified)
# Parse topology e.g. 2x2x1 -> 4 chips, 2x2x2 -> 8 chips
chip_count_from_slice() {
  local slice="$1"
  local topo="${slice##*:}"
  if [[ "$topo" =~ ^([0-9]+)x([0-9]+)x([0-9]+)$ ]]; then
    echo $(( BASH_REMATCH[1] * BASH_REMATCH[2] * BASH_REMATCH[3] ))
  else
    echo 4
  fi
}

TRAINER_CHIPS="$(chip_count_from_slice "$TRAINER_TPU_SLICE")"
ROLLOUT_CHIPS="$(chip_count_from_slice "$ROLLOUT_TPU_SLICE")"

TRAINER_MESH_FSDP="${USER_SET_TRAINER_FSDP:-$TRAINER_CHIPS}"
ROLLOUT_MESH_TP="${USER_SET_ROLLOUT_TP:-$ROLLOUT_CHIPS}"
TRAINER_JOBSET_YAML="${USER_SET_TRAINER_YAML:-jobset.tpu.yaml}"
# MaxText requires train_micro_batch_size to be a multiple of mesh_fsdp
TRAIN_MICRO_BATCH_SIZE="${USER_SET_MICRO_BATCH:-$TRAINER_MESH_FSDP}"
if (( TRAIN_MICRO_BATCH_SIZE < TRAINER_MESH_FSDP )); then
  TRAIN_MICRO_BATCH_SIZE="$TRAINER_MESH_FSDP"
fi

# ==============================================================================
# Helper Functions
# ==============================================================================
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
    "Nominal Quota=\(.spec.resourceGroups[0].flavors[]? | select(.name | test("tpu")) | .resources[]? | select(.name=="google.com/tpu") | .nominalQuota // 0) | " +
    "Reserved=\(.status.flavorsReservation[]? | select(.name | test("tpu")) | .resources[]? | select(.name=="google.com/tpu") | .total // 0)"
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
  echo "  Cluster:             ${CLUSTER} (${PROJECT:-<default>} / ${REGION:-<default>})"
  echo "  Steps:               ${STEPS}"
  echo "  Weight Sync:         ${WEIGHT_SYNC_MODE}"
  echo "  Git Branch Sync:     ${SYNC_GIT_BRANCH}"
  echo "  Trainer Backend:     ${TRAINER_BACKEND} (${TRAINER_TPU_SLICE}, FSDP=${TRAINER_MESH_FSDP})"
  echo "  Rollout Slice:       ${ROLLOUT_TPU_SLICE} (TP=${ROLLOUT_MESH_TP})"
  echo "  CPU Node Machine:    ${CPU_MACHINE}"
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
  GCS_SCRATCH_LOCATION="${GCS_SCRATCH_LOCATION}" \
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
  echo "  Stream Orchestrator: $(basename "$0") logs orch -f"
  echo "  Stream Trainer:      $(basename "$0") logs train -f"
  echo "  Stream Rollout:      $(basename "$0") logs roll -f"
  echo "  Stop training:       $(basename "$0") stop"
}

# ==============================================================================
# Main Dispatcher
# ==============================================================================
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
