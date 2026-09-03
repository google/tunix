#!/bin/bash
# Copyright 2026 Google LLC
#
# Unified Kubernetes JobSet Launcher for Distributed RL Workloads
# (MaxText Trainer + Tunix GRPO Orchestrator + vLLM Rollout + Raiden Weight Sync)
#
# Designed for easy, one-command reproducibility with pre-built container images.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Smart resolution of TUNIX repository root and related project directories
if [ -d "${SCRIPT_DIR}/tunix/experimental" ]; then
  TUNIX_DIR="${SCRIPT_DIR}"
elif [ -d "${SCRIPT_DIR}/../../../../tunix/experimental" ]; then
  TUNIX_DIR="$(cd "${SCRIPT_DIR}/../../../../" && pwd)"
elif [ -d "${SCRIPT_DIR}/tunix" ]; then
  TUNIX_DIR="$(cd "${SCRIPT_DIR}/tunix" && pwd)"
else
  TUNIX_DIR="${TUNIX_DIR:-$(pwd)}"
fi

YAML_GENERATOR="${YAML_GENERATOR:-${TUNIX_DIR}/tunix/experimental/distributed/deployment/yaml_generator.py}"
YAMLS_DIR="${YAMLS_DIR:-${TUNIX_DIR}/tunix/experimental/distributed/deployment/yamls}"
MAXTEXT_DIR="${MAXTEXT_DIR:-${TUNIX_DIR}/../maxtext}"
TPU_INFERENCE_DIR="${TPU_INFERENCE_DIR:-${TUNIX_DIR}/../tpu-inference}"

# ==============================================================================
# Helper: Print Usage
# ==============================================================================
print_usage() {
  cat << 'HELP_EOF'
Usage: ./launch_raiden.sh [COMMAND] [OPTIONS]

Unified script to launch, monitor, triage, and stop distributed RL training 
with MaxText, Tunix, vLLM, and Raiden weight sync on GKE TPU clusters.

COMMANDS:
  start              (Default) Launch orchestrator, trainer, and rollout JobSets
  stop               Cleanly delete all JobSets for the current run
  restart            Stop existing workload and immediately re-launch
  status             Show running JobSets, pod phases, and node placements
  logs [ROLE] [-f]   View or follow logs. ROLE: 'trainer' (default), 'rollout', or 'orch'
  triage             Fetch trainer logs and run automated error diagnosis
  dry-run | render   Print generated Kubernetes YAML manifests without applying
  help, -h, --help   Show this help message

OPTIONS:
  --image <IMAGE>           Container image to run
                            Default: gcr.io/cloud-tpu-multipod-dev/yixuannwang_google_com-runner:yixuann-debug-raiden-0903-1
  --model, --preset <NAME>  Model preset to use:
                              - 'qwen3-0.6b' (Default: Qwen3-0.6B, tpuv5:2x2x2 train, tpuv5:2x2x1 roll)
                              - 'qwen3.5-35b' (Qwen3.5-35B-A3B with scanned ckpt)
                              - 'qwen3-1.7b' (Qwen3-1.7B standard baseline)
  --run-id <ID>             Custom workload / JobSet prefix
                            Default: <username>-raiden-<model_tag>
  --random-id               Append random suffix to run ID for isolated scratch runs
  --sync-code               Package local tunix/maxtext code to GCS (for rapid live dev)
  --no-sync-code            Run container image directly as-is (Default: recommended for reproducibility)
  --ckpt <GCS_PATH>         Custom MaxText checkpoint path (overrides preset default)
  --cluster <NAME>          GKE cluster name (default: bodaborg-v5p-nap)
  --region <REGION>         GCP region (default: europe-west4)
  --zone <ZONE>             GCP zone (default: europe-west4-b)
  --project <PROJECT>       GCP project (default: cloud-tpu-shared-capacity)
  --cpu-machine <TYPE>      CPU machine type for orchestrator (default: n2d-standard-64)
  --scratch <GCS_PATH>      GCS scratch location
                            Default: gs://mohitkhatwani_multipods/pathways_scratch/<run_id>
  --max-steps <N>           Max training steps (default: 2)
  --batch-size <N>          Batch size (default: 4)
  --rollout-replicas <N>    Number of rollout TPU workers (default: 2)
  --trainer-slice <SLICE>   TPU slice for trainer (default: tpuv5:2x2x2)
  --rollout-slice <SLICE>   TPU slice for rollout (default: tpuv5:2x2x1)
  --no-cluster-connect      Skip automatic gcloud cluster authentication check

EXAMPLES:
  # 1. Quick reproduction with verified default image:
  ./launch_raiden.sh

  # 2. Launch with a specific given container image:
  ./launch_raiden.sh --image gcr.io/cloud-tpu-multipod-dev/my-team-image:tag

  # 3. Check status of running pods:
  ./launch_raiden.sh status

  # 4. Stream trainer Python logs:
  ./launch_raiden.sh logs trainer -f

  # 5. Automatically diagnose errors in trainer log:
  ./launch_raiden.sh triage

  # 6. Stop and delete all JobSets for this run:
  ./launch_raiden.sh stop
HELP_EOF
}

# ==============================================================================
# Model & Workload Presets
# ==============================================================================
load_preset_defaults() {
  local preset="$1"
  case "$preset" in
    qwen3-0.6b|0.6b|0.6B)
      PRESET_NAME="qwen3-0.6b"
      PRESET_MODEL_TAG="06b"
      PRESET_MODEL_NAME="Qwen3-0.6B"
      PRESET_MODEL_ID="Qwen/Qwen3-0.6B"
      PRESET_MAXTEXT_MODEL_NAME="qwen3-0.6b"
      PRESET_TRAINER_BACKEND="maxtext"
      PRESET_MAXTEXT_CKPT="gs://maxtext-model-checkpoints/qwen3-0.6b/2025-10-27/scanned/0/items"
      PRESET_TRAINER_TPU_SLICE="tpuv5:2x2x2"
      PRESET_TRAINER_MESH_FSDP=8
      PRESET_TRAIN_MICRO_BATCH_SIZE=8
      PRESET_BATCH_SIZE=4
      PRESET_NUM_GENERATIONS=2
      PRESET_ROLLOUT_TPU_SLICE="tpuv5:2x2x1"
      PRESET_ROLLOUT_TENSOR_PARALLEL_SIZE=2
      PRESET_ROLLOUT_MESH_TP=2
      PRESET_ROLLOUT_REPLICAS=2
      PRESET_SAMPLER="vllm"
      PRESET_WEIGHT_SYNC_MODE="raiden"
      PRESET_USE_WEIGHT_CONVERTER=1
      PRESET_ROLLOUT_BACKEND="maxtext"
      PRESET_VERIFY_WEIGHTS="true"
      PRESET_DISABLE_CHECKPOINTING="true"
      PRESET_MAX_STEPS=2
      PRESET_DEFAULT_IMAGE="gcr.io/cloud-tpu-multipod-dev/yixuannwang_google_com-runner:yixuann-debug-raiden-0903-1"
      ;;
    qwen3.5-35b|35b|35B)
      PRESET_NAME="qwen3.5-35b"
      PRESET_MODEL_TAG="35b"
      PRESET_MODEL_NAME="Qwen3.5-35B-A3B"
      PRESET_MODEL_ID="Qwen/Qwen3.5-35B-A3B"
      PRESET_MAXTEXT_MODEL_NAME="qwen3.5-35b-a3b"
      PRESET_TRAINER_BACKEND="maxtext"
      PRESET_MAXTEXT_CKPT="gs://hengtaoguo-maxtext-logs/checkpoints/qwen3.5-35b-a3b/scanned/2026-06-11-10-27/0/items"
      PRESET_TRAINER_TPU_SLICE="tpuv5:2x2x2"
      PRESET_TRAINER_MESH_FSDP=8
      PRESET_TRAIN_MICRO_BATCH_SIZE=8
      PRESET_BATCH_SIZE=4
      PRESET_NUM_GENERATIONS=2
      PRESET_ROLLOUT_TPU_SLICE="tpuv5:2x2x1"
      PRESET_ROLLOUT_TENSOR_PARALLEL_SIZE=2
      PRESET_ROLLOUT_MESH_TP=2
      PRESET_ROLLOUT_REPLICAS=2
      PRESET_SAMPLER="vllm"
      PRESET_WEIGHT_SYNC_MODE="raiden"
      PRESET_USE_WEIGHT_CONVERTER=1
      PRESET_ROLLOUT_BACKEND="maxtext"
      PRESET_VERIFY_WEIGHTS="true"
      PRESET_DISABLE_CHECKPOINTING="true"
      PRESET_MAX_STEPS=2
      PRESET_DEFAULT_IMAGE="gcr.io/cloud-tpu-multipod-dev/yixuannwang_google_com-runner:yixuann-raiden-debug"
      ;;
    qwen3-1.7b|1.7b|1.7B)
      PRESET_NAME="qwen3-1.7b"
      PRESET_MODEL_TAG="17b"
      PRESET_MODEL_NAME="Qwen3-1.7B"
      PRESET_MODEL_ID="Qwen/Qwen3-1.7B"
      PRESET_MAXTEXT_MODEL_NAME="qwen3-1.7b"
      PRESET_TRAINER_BACKEND="tunix"
      PRESET_MAXTEXT_CKPT=""
      PRESET_TRAINER_TPU_SLICE="tpuv5:2x2x2"
      PRESET_TRAINER_MESH_FSDP=8
      PRESET_TRAIN_MICRO_BATCH_SIZE=1
      PRESET_BATCH_SIZE=4
      PRESET_NUM_GENERATIONS=2
      PRESET_ROLLOUT_TPU_SLICE="tpuv5:2x2x1"
      PRESET_ROLLOUT_TENSOR_PARALLEL_SIZE=1
      PRESET_ROLLOUT_MESH_TP=4
      PRESET_ROLLOUT_REPLICAS=1
      PRESET_SAMPLER="vllm"
      PRESET_WEIGHT_SYNC_MODE="raiden"
      PRESET_USE_WEIGHT_CONVERTER=1
      PRESET_ROLLOUT_BACKEND="maxtext"
      PRESET_VERIFY_WEIGHTS="true"
      PRESET_DISABLE_CHECKPOINTING="true"
      PRESET_MAX_STEPS=1
      PRESET_DEFAULT_IMAGE="gcr.io/cloud-tpu-multipod-dev/yixuannwang_google_com-runner:yixuann-debug-raiden-0903-1"
      ;;
    *)
      echo "Error: Unknown preset '$preset'. Available: qwen3-0.6b, qwen3.5-35b, qwen3-1.7b" >&2
      exit 1
      ;;
  esac
}

# ==============================================================================
# Parse Command & CLI Options
# ==============================================================================
COMMAND=""
TARGET_PRESET="${MODEL_PRESET:-${PRESET:-qwen3-0.6b}}"

USER_IMAGE=""
USER_CKPT=""
USER_RUN_ID=""
USER_MAX_STEPS=""
USER_BATCH_SIZE=""
USER_TRAIN_MICRO_BATCH_SIZE=""
USER_ROLLOUT_REPLICAS=""
USER_TRAINER_TPU_SLICE=""
USER_ROLLOUT_TPU_SLICE=""
USER_SYNC_CODE=""
USER_SCRATCH=""
USER_CLUSTER=""
USER_REGION=""
USER_ZONE=""
USER_PROJECT=""
USER_CPU_MACHINE=""

RANDOMIZE_ID=false
DRY_RUN="${DRY_RUN:-false}"
CONNECT_CLUSTER="${CONNECT_CLUSTER:-true}"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    start|stop|restart|status|logs|triage|dry-run|render)
      COMMAND="$1"
      shift
      ;;
    help|-h|--help)
      print_usage
      exit 0
      ;;
    --command=*)
      COMMAND="${1#*=}"
      shift
      ;;
    --command)
      COMMAND="$2"
      shift 2
      ;;
    --image=*)
      USER_IMAGE="${1#*=}"
      shift
      ;;
    --image)
      USER_IMAGE="$2"
      shift 2
      ;;
    --model=*|--preset=*)
      TARGET_PRESET="${1#*=}"
      shift
      ;;
    --model|--preset)
      TARGET_PRESET="$2"
      shift 2
      ;;
    --ckpt=*|--maxtext-ckpt=*)
      USER_CKPT="${1#*=}"
      shift
      ;;
    --ckpt|--maxtext-ckpt)
      USER_CKPT="$2"
      shift 2
      ;;
    --run-id=*)
      USER_RUN_ID="${1#*=}"
      shift
      ;;
    --run-id)
      USER_RUN_ID="$2"
      shift 2
      ;;
    --random-id)
      RANDOMIZE_ID=true
      shift
      ;;
    --sync-code)
      USER_SYNC_CODE=true
      shift
      ;;
    --no-sync-code)
      USER_SYNC_CODE=false
      shift
      ;;
    --cluster=*)
      USER_CLUSTER="${1#*=}"
      shift
      ;;
    --cluster)
      USER_CLUSTER="$2"
      shift 2
      ;;
    --region=*)
      USER_REGION="${1#*=}"
      shift
      ;;
    --region)
      USER_REGION="$2"
      shift 2
      ;;
    --zone=*)
      USER_ZONE="${1#*=}"
      shift
      ;;
    --zone)
      USER_ZONE="$2"
      shift 2
      ;;
    --project=*)
      USER_PROJECT="${1#*=}"
      shift
      ;;
    --project)
      USER_PROJECT="$2"
      shift 2
      ;;
    --cpu-machine=*)
      USER_CPU_MACHINE="${1#*=}"
      shift
      ;;
    --cpu-machine)
      USER_CPU_MACHINE="$2"
      shift 2
      ;;
    --scratch=*)
      USER_SCRATCH="${1#*=}"
      shift
      ;;
    --scratch)
      USER_SCRATCH="$2"
      shift 2
      ;;
    --max-steps=*)
      USER_MAX_STEPS="${1#*=}"
      shift
      ;;
    --max-steps)
      USER_MAX_STEPS="$2"
      shift 2
      ;;
    --batch-size=*)
      USER_BATCH_SIZE="${1#*=}"
      shift
      ;;
    --batch-size)
      USER_BATCH_SIZE="$2"
      shift 2
      ;;
    --train-micro-batch-size=*)
      USER_TRAIN_MICRO_BATCH_SIZE="${1#*=}"
      shift
      ;;
    --train-micro-batch-size)
      USER_TRAIN_MICRO_BATCH_SIZE="$2"
      shift 2
      ;;
    --rollout-replicas=*)
      USER_ROLLOUT_REPLICAS="${1#*=}"
      shift
      ;;
    --rollout-replicas)
      USER_ROLLOUT_REPLICAS="$2"
      shift 2
      ;;
    --trainer-slice=*)
      USER_TRAINER_TPU_SLICE="${1#*=}"
      shift
      ;;
    --trainer-slice)
      USER_TRAINER_TPU_SLICE="$2"
      shift 2
      ;;
    --rollout-slice=*)
      USER_ROLLOUT_TPU_SLICE="${1#*=}"
      shift
      ;;
    --rollout-slice)
      USER_ROLLOUT_TPU_SLICE="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --no-cluster-connect)
      CONNECT_CLUSTER=false
      shift
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

# Load selected preset values
load_preset_defaults "${TARGET_PRESET}"

# Resolve all final configuration values: CLI flag > Env Var > Preset Default
MODEL_NAME="${MODEL_NAME:-${PRESET_MODEL_NAME}}"
MODEL_ID="${MODEL_ID:-${PRESET_MODEL_ID}}"
MODEL_TAG="${MODEL_TAG:-${PRESET_MODEL_TAG}}"
MAXTEXT_MODEL_NAME="${MAXTEXT_MODEL_NAME:-${PRESET_MAXTEXT_MODEL_NAME}}"
TRAINER_BACKEND="${TRAINER_BACKEND:-${PRESET_TRAINER_BACKEND}}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_ID}}"

MAXTEXT_CKPT="${USER_CKPT:-${MAXTEXT_CKPT:-${PRESET_MAXTEXT_CKPT}}}"
TUNIX_IMAGE="${USER_IMAGE:-${TUNIX_IMAGE:-${PRESET_DEFAULT_IMAGE}}}"
MAX_STEPS="${USER_MAX_STEPS:-${MAX_STEPS:-${PRESET_MAX_STEPS}}}"
BATCH_SIZE="${USER_BATCH_SIZE:-${BATCH_SIZE:-${PRESET_BATCH_SIZE}}}"
NUM_GENERATIONS="${NUM_GENERATIONS:-${PRESET_NUM_GENERATIONS}}"
TRAIN_MICRO_BATCH_SIZE="${USER_TRAIN_MICRO_BATCH_SIZE:-${TRAIN_MICRO_BATCH_SIZE:-${PRESET_TRAIN_MICRO_BATCH_SIZE}}}"
ROLLOUT_REPLICAS="${USER_ROLLOUT_REPLICAS:-${ROLLOUT_REPLICAS:-${PRESET_ROLLOUT_REPLICAS}}}"
TRAINER_TPU_SLICE="${USER_TRAINER_TPU_SLICE:-${TRAINER_TPU_SLICE:-${PRESET_TRAINER_TPU_SLICE}}}"
ROLLOUT_TPU_SLICE="${USER_ROLLOUT_TPU_SLICE:-${ROLLOUT_TPU_SLICE:-${PRESET_ROLLOUT_TPU_SLICE}}}"
TRAINER_MESH_FSDP="${TRAINER_MESH_FSDP:-${PRESET_TRAINER_MESH_FSDP}}"
ROLLOUT_MESH_TP="${ROLLOUT_MESH_TP:-${PRESET_ROLLOUT_MESH_TP}}"
SAMPLER="${SAMPLER:-${PRESET_SAMPLER}}"
WEIGHT_SYNC_MODE="${WEIGHT_SYNC_MODE:-${PRESET_WEIGHT_SYNC_MODE}}"
USE_WEIGHT_CONVERTER="${USE_WEIGHT_CONVERTER:-${PRESET_USE_WEIGHT_CONVERTER}}"
ROLLOUT_BACKEND="${ROLLOUT_BACKEND:-${PRESET_ROLLOUT_BACKEND}}"
VERIFY_WEIGHTS="${VERIFY_WEIGHTS:-${PRESET_VERIFY_WEIGHTS}}"
DISABLE_CHECKPOINTING="${DISABLE_CHECKPOINTING:-${PRESET_DISABLE_CHECKPOINTING}}"

SYNC_CODE="${USER_SYNC_CODE:-${SYNC_CODE:-false}}"  # Default: false for given-image reproducibility
PROJECT="${USER_PROJECT:-${PROJECT:-cloud-tpu-shared-capacity}}"
REGION="${USER_REGION:-${REGION:-europe-west4}}"
ZONE="${USER_ZONE:-${ZONE:-europe-west4-b}}"
CLUSTER="${USER_CLUSTER:-${CLUSTER:-bodaborg-v5p-nap}}"
CPU_MACHINE="${USER_CPU_MACHINE:-${CPU_MACHINE:-n2d-standard-64}}"

CURRENT_SYSTEM_USER="${USER:-$(whoami 2>/dev/null || echo "user")}"
if [[ -n "${USER_RUN_ID}" ]]; then
  RUN_ID="${USER_RUN_ID}"
elif [[ "${RANDOMIZE_ID}" == "true" ]]; then
  RUN_ID="${CURRENT_SYSTEM_USER}-r$((RANDOM % 90000 + 10000))"
else
  RUN_ID="${CURRENT_SYSTEM_USER}-raiden-${MODEL_TAG}"
fi

export USER="${RUN_ID}"
GCS_SCRATCH_LOCATION="${USER_SCRATCH:-${GCS_SCRATCH_LOCATION:-gs://mohitkhatwani_multipods/pathways_scratch/${RUN_ID}}}"
GCS_SYNC_TAR="${GCS_SCRATCH_LOCATION}/code_sync/${USER}.tar.gz"

ORCHESTRATOR_ID="${RUN_ID}-orch"
TRAINER_ID="${RUN_ID}-train"
ROLLOUT_ID="${RUN_ID}-roll"

MODEL_DIR="${MODEL_DIR:-}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-128}"
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-$((BATCH_SIZE * NUM_GENERATIONS))}"
EVAL_EVERY_N_STEPS="${EVAL_EVERY_N_STEPS:-1000000}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-16.0}"
DEBUG="${DEBUG:-0}"

MAXTEXT_OUTPUT_DIR="${MAXTEXT_OUTPUT_DIR:-/tmp/artifacts/math_gsm8k_dist/maxtext}"
TRAINER_MESH_TP="${TRAINER_MESH_TP:-1}"
TRAINER_MESH_EXPERT="${TRAINER_MESH_EXPERT:-1}"
TRAINER_PADDED_MOE_MLP_DIM="${TRAINER_PADDED_MOE_MLP_DIM:-}"
ROLLOUT_USE_BATCHED_RPA="${ROLLOUT_USE_BATCHED_RPA:-}"
ROLLOUT_MAXTEXT_ATTENTION="${ROLLOUT_MAXTEXT_ATTENTION:-}"
TRAINER_JOBSET_YAML="${TRAINER_JOBSET_YAML:-jobset.pathways.yaml}"

WANDB_PROJECT="${WANDB_PROJECT:-trellis-gsm8k}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"

ORCHESTRATOR_PORT="${ORCHESTRATOR_PORT:-20000}"
ROLLOUT_PORT="${ROLLOUT_PORT:-20001}"
TRAINER_PORT="${TRAINER_PORT:-20002}"

COMMAND="${COMMAND:-start}"
if [[ "$COMMAND" == "dry-run" || "$COMMAND" == "render" ]]; then
  COMMAND="start"
  DRY_RUN=true
fi

# ==============================================================================
# Cluster Connection
# ==============================================================================
connect_cluster() {
  if [[ "${CONNECT_CLUSTER}" != "true" || "${DRY_RUN}" == "true" ]]; then
    return 0
  fi

  local target_context="gke_${PROJECT}_${REGION}_${CLUSTER}"
  local current_context
  current_context=$(kubectl config current-context 2>/dev/null || true)

  if [[ "$current_context" == "$target_context" ]]; then
    return 0
  fi

  echo "================================================================="
  echo "Connecting to GKE cluster: ${CLUSTER} in ${REGION} (${PROJECT})..."
  echo "================================================================="
  gcloud container clusters get-credentials "${CLUSTER}" \
    --region "${REGION}" \
    --project "${PROJECT}" \
    --dns-endpoint
  kubectl config use-context "${target_context}" >/dev/null 2>&1 || true
}

# ==============================================================================
# Code Sync Logic (Optional for given-image runs)
# ==============================================================================
sync_code_to_gcs() {
  if [[ "${SYNC_CODE}" != "true" ]]; then
    echo "ℹ️  Code sync disabled (SYNC_CODE=false). Using container image as-is."
    return 0
  fi

  if [ ! -d "${TUNIX_DIR}" ] || [ ! -d "${MAXTEXT_DIR}" ]; then
    echo "Error: Local tunix/ or maxtext/ directories not found for code sync." >&2
    echo "Expected at: ${TUNIX_DIR} and ${MAXTEXT_DIR}" >&2
    exit 1
  fi

  local pkg_msg="tunix/, maxtext/"
  local extra_dirs=()
  if [ -d "${MAXTEXT_DIR}" ]; then
    local maxtext_parent
    maxtext_parent="$(cd "${MAXTEXT_DIR}/.." && pwd)"
    extra_dirs+=("-C" "${maxtext_parent}" "maxtext")
  fi
  if [[ "${SYNC_TPU_INFERENCE:-false}" == "true" ]] && [ -d "${TPU_INFERENCE_DIR}" ]; then
    local tpu_inf_parent
    tpu_inf_parent="$(cd "${TPU_INFERENCE_DIR}/.." && pwd)"
    extra_dirs+=("-C" "${tpu_inf_parent}" "tpu-inference")
    pkg_msg="tunix/, maxtext/, tpu-inference/"
  fi

  echo "================================================================="
  echo "📦 Packaging local changes from ${pkg_msg}..."
  echo "================================================================="
  local tar_file="/tmp/code_sync_${USER}.tar.gz"
  rm -f "${tar_file}"

  tar --exclude=".git" \
      --exclude=".venv" \
      --exclude="venv*" \
      --exclude="myenv" \
      --exclude="__pycache__" \
      --exclude=".pytest_cache" \
      --exclude="docs" \
      --exclude="benchmarks" \
      --exclude="tests" \
      --exclude="artifacts" \
      --exclude="checkpoints" \
      -czf "${tar_file}" \
      -C "${TUNIX_DIR}" . \
      "${extra_dirs[@]}"

  local tar_size
  tar_size=$(ls -lh "${tar_file}" | awk '{print $5}')
  echo "📦 Archive created (${tar_size}). Uploading to ${GCS_SYNC_TAR}..."

  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "[DRY RUN] Would upload ${tar_file} to ${GCS_SYNC_TAR}"
  else
    if command -v gcloud &>/dev/null; then
      gcloud storage cp "${tar_file}" "${GCS_SYNC_TAR}" --quiet || gsutil cp "${tar_file}" "${GCS_SYNC_TAR}"
    else
      gsutil cp "${tar_file}" "${GCS_SYNC_TAR}"
    fi
    echo "✅ Local code synced to GCS: ${GCS_SYNC_TAR}"
  fi

  rm -f "${tar_file}"
  echo "================================================================="
}

get_sync_prefix() {
  if [[ "${SYNC_CODE}" == "true" ]]; then
    local tpu_inf_cmd=""
    local python_path="/app/maxtext/src:/app:\${PYTHONPATH:-}"
    if [[ "${SYNC_TPU_INFERENCE:-false}" == "true" ]]; then
      tpu_inf_cmd="(pip install --no-deps -e /app/tpu-inference 2>/dev/null || true) && (cp -rf /app/tpu-inference/tpu_inference /opt/venv/lib/python3.12/site-packages/ 2>/dev/null || true) && "
      python_path="/app/tpu-inference:/app/maxtext/src:/app:\${PYTHONPATH:-}"
    fi
    echo "echo '==> Syncing code from ${GCS_SYNC_TAR}...'; (gcloud storage cp ${GCS_SYNC_TAR} /tmp/code_sync.tar.gz 2>/dev/null || gsutil cp ${GCS_SYNC_TAR} /tmp/code_sync.tar.gz 2>/dev/null || python3 -c \"import json, urllib.request, urllib.parse; token = json.loads(urllib.request.urlopen(urllib.request.Request('http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token', headers={'Metadata-Flavor': 'Google'})).read())['access_token']; b, p = '${GCS_SYNC_TAR}'.replace('gs://', '').split('/', 1); req = urllib.request.Request(f'https://storage.googleapis.com/storage/v1/b/{b}/o/{urllib.parse.quote(p, safe=\\\"\\\")}?alt=media', headers={'Authorization': f'Bearer {token}'}); open('/tmp/code_sync.tar.gz', 'wb').write(urllib.request.urlopen(req).read())\" 2>/dev/null || python3 -c \"import gcsfs; gcsfs.GCSFileSystem().get('${GCS_SYNC_TAR}', '/tmp/code_sync.tar.gz')\") && tar -xzf /tmp/code_sync.tar.gz -C /app && rm -f /tmp/code_sync.tar.gz && (pip install --force-reinstall --no-deps /app/raiden_wheels/*.whl 2>/dev/null || true) && ${tpu_inf_cmd}(cp -rf /app/tunix /opt/venv/lib/python3.12/site-packages/ 2>/dev/null || true) && (cp -rf /app/maxtext/src/maxtext /opt/venv/lib/python3.12/site-packages/ 2>/dev/null || true) && export PYTHONPATH=\"${python_path}\" && echo '==> Code sync and Raiden wheel applied to /app and site-packages.';"
  else
    echo ""
  fi
}

apply_or_print() {
  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "---"
    cat
  else
    kubectl apply -f -
  fi
}

# ==============================================================================
# JobSet Lifecycle Operations
# ==============================================================================
stop_workload() {
  local jobsets=("${ORCHESTRATOR_ID}" "${TRAINER_ID}" "${ROLLOUT_ID}")
  for (( i=0; i<8; i++ )); do
    jobsets+=("${ROLLOUT_ID}-${i}")
  done

  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "[DRY RUN] Would delete jobsets: ${jobsets[*]}"
    return 0
  fi

  echo "Stopping workload (${RUN_ID})..."
  for js in "${jobsets[@]}"; do
    if kubectl get jobset "$js" &>/dev/null; then
      echo "  Deleting JobSet: $js..."
      kubectl delete jobset "$js" --ignore-not-found=true --wait=false 2>/dev/null || true
    fi
  done
  echo "✅ Teardown command sent for ${RUN_ID}."
}

start_orchestrator() {
  local sync_prefix
  sync_prefix=$(get_sync_prefix)

  echo "Rendering & Starting orchestrator (${ORCHESTRATOR_ID})..."
  python3 "${YAML_GENERATOR}" \
    "${YAMLS_DIR}/jobset.cpu.yaml" \
    --jobset_name="${ORCHESTRATOR_ID}" \
    --cpu_machine="${CPU_MACHINE}" \
    --worker_container_image="${TUNIX_IMAGE}" \
    --worker_container_port="${ORCHESTRATOR_PORT}" \
    --worker_startup_command=" \
      ${sync_prefix} \
      ${WANDB_PROJECT:+WANDB_PROJECT=\"${WANDB_PROJECT}\"} \
      ${WANDB_RUN_NAME:+WANDB_RUN_NAME=\"${WANDB_RUN_NAME}\"} \
      PYTHONUNBUFFERED=1 python -m tunix.experimental.distributed.runtime.main \
        --discovery_id=${ORCHESTRATOR_ID} \
        --discovery_port=${ORCHESTRATOR_PORT} \
        --process_main=tunix.experimental.examples.math_gsm8k_dist.run_gsm8k_dist_grpo.main \
        --model_id=${MODEL_ID} \
        --tokenizer_path=${TOKENIZER_PATH} \
        --batch_size=${BATCH_SIZE} \
        --num_generations=${NUM_GENERATIONS} \
        --max_steps=${MAX_STEPS} \
        --max_prompt_length=${MAX_PROMPT_LENGTH} \
        --max_response_length=${MAX_RESPONSE_LENGTH} \
        --train_micro_batch_size=${TRAIN_MICRO_BATCH_SIZE} \
        --wandb_project=\"${WANDB_PROJECT}\" \
        --wandb_run_name=\"${WANDB_RUN_NAME}\" \
        --weight_sync_mode=${WEIGHT_SYNC_MODE} \
        --stop_workers_on_exit \
        ${DEBUG:+--debug} \
    " \
    | apply_or_print
}

start_trainer() {
  local maxtext_args=""
  if [[ "${TRAINER_BACKEND}" == "maxtext" ]]; then
    maxtext_args=" \
      --maxtext_model_name=${MAXTEXT_MODEL_NAME} \
      ${TRAINER_PADDED_MOE_MLP_DIM:+--maxtext_padded_moe_mlp_dim=${TRAINER_PADDED_MOE_MLP_DIM}} \
      --maxtext_ckpt_path=${MAXTEXT_CKPT} \
      --maxtext_output_directory=${MAXTEXT_OUTPUT_DIR} \
      --mesh_tp=${TRAINER_MESH_TP} \
      --mesh_expert=${TRAINER_MESH_EXPERT} \
    "
  fi

  local sync_prefix
  sync_prefix=$(get_sync_prefix)

  echo "Rendering & Starting trainer (${TRAINER_ID})..."
  python3 "${YAML_GENERATOR}" \
    "${YAMLS_DIR}/${TRAINER_JOBSET_YAML}" \
    --jobset_name="${TRAINER_ID}" \
    --tpu_slice="${TRAINER_TPU_SLICE}" \
    --cpu_machine="${CPU_MACHINE}" \
    --pathways_gcs_scratch_location="${GCS_SCRATCH_LOCATION}" \
    --worker_container_image="${TUNIX_IMAGE}" \
    --worker_container_port="${TRAINER_PORT}" \
    --worker_startup_command=" \
      ${sync_prefix} \
      PYTHONUNBUFFERED=1 DISABLE_CHECKPOINTING=${DISABLE_CHECKPOINTING} VERIFY_WEIGHTS=${VERIFY_WEIGHTS} USE_WEIGHT_CONVERTER=${USE_WEIGHT_CONVERTER} ROLLOUT_BACKEND=${ROLLOUT_BACKEND} python -m tunix.experimental.distributed.runtime.main \
        --discovery_addrs=${ORCHESTRATOR_ID}:${ORCHESTRATOR_PORT} \
        --process_executor=tunix.experimental.distributed.runtime.executor.K8sExecutor \
        --process_main=tunix.experimental.examples.math_gsm8k_dist.run_trainer_node.main \
        --worker_id=${TRAINER_ID} \
        --port=${TRAINER_PORT} \
        --mesh_fsdp=${TRAINER_MESH_FSDP} \
        --trainer_backend=${TRAINER_BACKEND} \
        --model_name=${MODEL_NAME} \
        --model_id=${MODEL_ID} \
        --model_dir=${MODEL_DIR} \
        --tokenizer_path=${TOKENIZER_PATH} \
        --max_prompt_length=${MAX_PROMPT_LENGTH} \
        --max_response_length=${MAX_RESPONSE_LENGTH} \
        --mini_batch_size=${MINI_BATCH_SIZE} \
        --train_micro_batch_size=${TRAIN_MICRO_BATCH_SIZE} \
        --eval_every_n_steps=${EVAL_EVERY_N_STEPS} \
        --lora_rank=${LORA_RANK} \
        --lora_alpha=${LORA_ALPHA} \
        ${ROLLOUT_MESH_TP:+--rollout_mesh_tp=${ROLLOUT_MESH_TP}} \
        ${maxtext_args} \
    " \
    | apply_or_print
}

start_rollout_instance() {
  local target_id="$1"
  local maxtext_args=""
  if [[ "${TRAINER_BACKEND}" == "maxtext" ]]; then
    maxtext_args=" \
      --maxtext_model_name=${MAXTEXT_MODEL_NAME} \
      ${ROLLOUT_MAXTEXT_ATTENTION:+--maxtext_attention=${ROLLOUT_MAXTEXT_ATTENTION}} \
    "
  fi
  local vllm_args=""
  if [[ "$SAMPLER" == "vllm" ]]; then
    vllm_args="\
    --sampler=vllm \
    --sampler_mesh_tp=${ROLLOUT_MESH_TP} \
    --mesh_tp=${ROLLOUT_MESH_TP} \
    "
  fi

  local sync_prefix
  sync_prefix=$(get_sync_prefix)

  echo "Rendering & Starting rollout (${target_id})..."
  python3 "${YAML_GENERATOR}" \
    "${YAMLS_DIR}/jobset.tpu.yaml" \
    --jobset_name="${target_id}" \
    --tpu_slice="${ROLLOUT_TPU_SLICE}" \
    --pathways_gcs_scratch_location="${GCS_SCRATCH_LOCATION}" \
    --worker_container_image="${TUNIX_IMAGE}" \
    --worker_container_port="${ROLLOUT_PORT}" \
    --worker_startup_command=" \
      ${sync_prefix} \
      PYTHONUNBUFFERED=1 SKIP_JAX_PRECOMPILE=1 VERIFY_WEIGHTS=${VERIFY_WEIGHTS} ${ROLLOUT_USE_BATCHED_RPA:+USE_BATCHED_RPA_KERNEL=1} python -m tunix.experimental.distributed.runtime.main \
        --discovery_addrs=${ORCHESTRATOR_ID}:${ORCHESTRATOR_PORT} \
        --process_executor=tunix.experimental.distributed.runtime.executor.K8sExecutor \
        --process_main=tunix.experimental.examples.math_gsm8k_dist.run_rollout_node.main \
        --worker_id=${target_id} \
        --port=${ROLLOUT_PORT} \
        --model_name=${MODEL_NAME} \
        --model_id=${MODEL_ID} \
        --model_dir=${MODEL_DIR} \
        --tokenizer_path=${TOKENIZER_PATH} \
        --max_prompt_length=${MAX_PROMPT_LENGTH} \
        --max_response_length=${MAX_RESPONSE_LENGTH} \
        --lora_rank=${LORA_RANK} \
        --lora_alpha=${LORA_ALPHA} \
        --weight_sync_mode=${WEIGHT_SYNC_MODE} \
        ${maxtext_args} \
        ${vllm_args} \
        ${DEBUG:+--debug} \
    " \
    | apply_or_print
}

start_all_rollouts() {
  if [[ "${ROLLOUT_REPLICAS}" -gt 1 ]]; then
    for (( i=0; i<ROLLOUT_REPLICAS; i++ )); do
      start_rollout_instance "${ROLLOUT_ID}-${i}"
    done
  else
    start_rollout_instance "${ROLLOUT_ID}"
  fi
}

# ==============================================================================
# Status, Logs & Triage
# ==============================================================================
show_status() {
  connect_cluster
  echo "================================================================================"
  echo "JobSets for Run: ${RUN_ID}"
  echo "================================================================================"
  local js_all
  js_all=$(kubectl get jobset 2>/dev/null || true)
  local js_output
  js_output=$(echo "$js_all" | grep "${RUN_ID}" || true)
  if [[ -n "$js_output" ]]; then
    echo "$js_all" | { head -n 1 || true; }
    echo "$js_output"
  else
    echo "No active JobSets found for ${RUN_ID}."
  fi
  echo ""
  echo "================================================================================"
  echo "Pods for Run: ${RUN_ID}"
  echo "================================================================================"
  local pod_all
  pod_all=$(kubectl get pods -o wide 2>/dev/null || true)
  local pod_output
  pod_output=$(echo "$pod_all" | grep "${RUN_ID}" || true)
  if [[ -n "$pod_output" ]]; then
    echo "$pod_all" | { head -n 1 || true; }
    echo "$pod_output"
  else
    echo "No active Pods found for ${RUN_ID}."
  fi
}

show_logs() {
  connect_cluster
  local role="${1:-trainer}"
  shift || true
  local log_args=("$@")

  local target_jobset=""
  local container_flag="-c main"

  case "$role" in
    trainer|train)
      target_jobset="${TRAINER_ID}"
      ;;
    rollout|roll)
      if [[ "${ROLLOUT_REPLICAS}" -gt 1 ]]; then
        target_jobset="${ROLLOUT_ID}-0"
      else
        target_jobset="${ROLLOUT_ID}"
      fi
      ;;
    orchestrator|orch)
      target_jobset="${ORCHESTRATOR_ID}"
      ;;
    *)
      echo "Unknown role '$role'. Valid options: trainer, rollout, orch" >&2
      exit 1
      ;;
  esac

  local pod_name
  pod_name=$(kubectl get pods -l "jobset.sigs.k8s.io/jobset-name=${target_jobset},jobset.sigs.k8s.io/replicatedjob-name=proc" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
  if [[ -z "$pod_name" ]]; then
    pod_name=$(kubectl get pods -l "jobset.sigs.k8s.io/jobset-name=${target_jobset}" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
  fi

  if [[ -z "$pod_name" ]]; then
    echo "Error: No pod found for JobSet ${target_jobset}." >&2
    echo "Check current status with: $0 status" >&2
    exit 1
  fi

  echo "Fetching logs from pod ${pod_name} (${container_flag})..."
  kubectl logs "${pod_name}" ${container_flag} "${log_args[@]}"
}

triage_workload() {
  connect_cluster
  local pod_name
  pod_name=$(kubectl get pods -l "jobset.sigs.k8s.io/jobset-name=${TRAINER_ID},jobset.sigs.k8s.io/replicatedjob-name=proc" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)

  if [[ -z "$pod_name" ]]; then
    echo "Error: Trainer proc pod not found for ${TRAINER_ID}." >&2
    echo "Workload might not be running. Run '$0 status' to inspect." >&2
    exit 1
  fi

  local log_file="/tmp/triage-${RUN_ID}-trainer.log"
  echo "Downloading recent trainer logs from ${pod_name} to ${log_file}..."
  kubectl logs "${pod_name}" -c main --tail=1000 > "${log_file}" 2>&1 || true

  echo "================================================================================"
  echo "TRIAGE REPORT FOR: ${RUN_ID}"
  echo "================================================================================"

  if grep -E "(WeightConverterError|UnscanShapeMismatch|WeightSyncBindError|ConversionPlanError|ShapeMismatchError|Push range out of bounds)" "${log_file}"; then
    echo "--------------------------------------------------------------------------------"
    echo "❌ [CONVERSION_FAILURE] Weight conversion logic failed."
    echo "Inspect conversion trace in ${log_file}:"
    grep -n -C 3 -E "(WeightConverterError|UnscanShapeMismatch|WeightSyncBindError|ConversionPlanError|ShapeMismatchError)" "${log_file}" | head -n 30
    echo "--------------------------------------------------------------------------------"
    return 1
  elif grep -E "(TPU driver fault|ICI timeout|DMA error|Heartbeat timeout|OutOfMemoryError|SIGKILL|Killed)" "${log_file}"; then
    echo "--------------------------------------------------------------------------------"
    echo "⚠️  [INFRA_ERROR] Hardware / Driver / Network infrastructure failure detected."
    grep -n -C 2 -E "(TPU driver fault|ICI timeout|DMA error|Heartbeat timeout|OutOfMemoryError|SIGKILL|Killed)" "${log_file}" | head -n 20
    echo "--------------------------------------------------------------------------------"
    return 101
  elif grep -E "Traceback \(most recent call last\):" "${log_file}"; then
    echo "--------------------------------------------------------------------------------"
    echo "⚠️  [PYTHON_EXCEPTION] Non-conversion Python exception occurred:"
    grep -A 15 "Traceback (most recent call last):" "${log_file}" | tail -n 20
    echo "--------------------------------------------------------------------------------"
    return 2
  else
    echo "✅ No critical errors detected in recent trainer logs."
    echo "Recent output lines:"
    tail -n 10 "${log_file}"
    return 0
  fi
}

# ==============================================================================
# Dispatch Command
# ==============================================================================
case "$COMMAND" in
  start)
    connect_cluster
    echo "================================================================="
    echo "Workload ID:  ${RUN_ID}"
    echo "Model Preset: ${PRESET_NAME} (${MODEL_NAME})"
    echo "Image:        ${TUNIX_IMAGE}"
    echo "Code Sync:    ${SYNC_CODE}"
    echo "TPU Slices:   Trainer=${TRAINER_TPU_SLICE}, Rollout=${ROLLOUT_TPU_SLICE} (x${ROLLOUT_REPLICAS})"
    echo "Cluster:      ${CLUSTER} (${REGION} / ${PROJECT})"
    echo "================================================================="

    sync_code_to_gcs
    stop_workload
    start_orchestrator
    start_trainer
    start_all_rollouts

    if [[ "${DRY_RUN}" != "true" ]]; then
      echo "================================================================="
      echo "🎉 Workload launched successfully!"
      echo ""
      echo "Useful Commands:"
      echo "  Status:  $0 status --run-id=${RUN_ID}"
      echo "  Logs:    $0 logs trainer -f --run-id=${RUN_ID}"
      echo "  Triage:  $0 triage --run-id=${RUN_ID}"
      echo "  Stop:    $0 stop --run-id=${RUN_ID}"
      echo "================================================================="
    fi
    ;;

  stop)
    connect_cluster
    stop_workload
    ;;

  restart)
    connect_cluster
    stop_workload
    sleep 3
    "$0" start --run-id="${RUN_ID}" --model="${PRESET_NAME}" --image="${TUNIX_IMAGE}" "${EXTRA_ARGS[@]}"
    ;;

  status)
    show_status
    ;;

  logs)
    show_logs "${EXTRA_ARGS[@]}"
    ;;

  triage)
    triage_workload
    ;;

  *)
    echo "Error: Unknown command '$COMMAND'." >&2
    print_usage
    exit 1
    ;;
esac
