#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../../.." && pwd)

TARGET_CLUSTER=${TARGET_CLUSTER:-inference-v5e}
TUNIX_IMAGE=${TUNIX_IMAGE:-}

PROJECT=${PROJECT:-cloud-tpu-inference-test}
REGION=${REGION:-us-west1}
ZONE=${ZONE:-us-west1-c}
CLUSTER=${CLUSTER:-lancewang-pw-v5e-4slice}
CLUSTER_LOCATION=${CLUSTER_LOCATION:-${ZONE}}

MODEL_NAME=${MODEL_NAME:-Qwen3-1.7B}
MODEL_ID=${MODEL_ID:-Qwen/Qwen3-1.7B}
MODEL_DIR=${MODEL_DIR:-artifacts/qwen3_dist_gsm8k/models/${MODEL_NAME}}
TOKENIZER_PATH=${TOKENIZER_PATH:-${MODEL_ID}}

MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-128}
LORA_RANK=${LORA_RANK:-16}
LORA_ALPHA=${LORA_ALPHA:-16.0}
SAMPLER=${SAMPLER:-vllm}
WEIGHT_SYNC_MODE=${WEIGHT_SYNC_MODE:-none}
STANDALONE_INIT_WITH_RANDOM_WEIGHTS=${STANDALONE_INIT_WITH_RANDOM_WEIGHTS:-0}
VERIFY_WEIGHTS=${VERIFY_WEIGHTS:-false}
DEBUG=${DEBUG:-0}

CPU_MACHINE=${CPU_MACHINE:-n2-standard-64}
GCS_SCRATCH_LOCATION=${GCS_SCRATCH_LOCATION:-gs://cloud-pathways-staging/tmp}
PATHWAYS_SERVER_IMAGE=${PATHWAYS_SERVER_IMAGE:-}
PATHWAYS_PROXY_IMAGE=${PATHWAYS_PROXY_IMAGE:-}
HF_TOKEN_SECRET_NAME=${HF_TOKEN_SECRET_NAME:-}
PYTHON_BIN=${PYTHON_BIN:-python3}

JOBSET_NAME=${JOBSET_NAME:-${USER}-rollout-smoke}
JOBSET_YAML=${JOBSET_YAML:-jobset.pathways.yaml}
TPU_SLICE=${TPU_SLICE:-tpuv5e:4x4}
ROLLOUT_PORT=${ROLLOUT_PORT:-20001}
ROLLOUT_MESH_TP=${ROLLOUT_MESH_TP:-4}
WAIT_TIMEOUT=${WAIT_TIMEOUT:-30m}
KEEP_JOBSET=${KEEP_JOBSET:-0}

PROMPT_ONE=${PROMPT_ONE:-Reply with exactly OK.}
PROMPT_TWO=${PROMPT_TWO:-What is 2 + 3? Reply with digits only.}

apply_cluster_target() {
  case "$1" in
    inference-v5e)
      PROJECT=cloud-tpu-inference-test
      REGION=us-west1
      ZONE=us-west1-c
      CLUSTER_LOCATION=us-west1-c
      CLUSTER=lancewang-pw-v5e-4slice
      TPU_SLICE=${TPU_SLICE:-tpuv5e:4x4}
      ;;
    multipod-rl-v5e)
      PROJECT=cloud-tpu-multipod-dev
      REGION=us-central1
      ZONE=us-central1-a
      CLUSTER_LOCATION=us-central1
      CLUSTER=rl-v5e-16-cluster-v2
      TPU_SLICE=${TPU_SLICE:-tpuv5e:4x4}
      ;;
    multipod-rl-scaffolding)
      PROJECT=cloud-tpu-multipod-dev
      REGION=us-central1
      ZONE=us-central1-a
      CLUSTER_LOCATION=us-central1
      CLUSTER=rl-scaffolding
      TPU_SLICE=${TPU_SLICE:-tpuv5e:4x4}
      ;;
    bodaborg-v5p-nap | bodaborg-v5p)
      PROJECT=cloud-tpu-shared-capacity
      REGION=europe-west4
      ZONE=europe-west4-b
      CLUSTER_LOCATION=europe-west4
      CLUSTER=bodaborg-v5p-nap
      TPU_SLICE=${TPU_SLICE:-tpuv5:2x2x1}
      CPU_MACHINE=n2d-standard-64
      PATHWAYS_SERVER_IMAGE=${PATHWAYS_SERVER_IMAGE:-us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/shauryag/unsanitized_server:raiden_20260812}
      PATHWAYS_PROXY_IMAGE=${PATHWAYS_PROXY_IMAGE:-us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/shauryag/unsanitized_proxy_server:raiden_20260812}
      ;;
    *)
      echo "Unsupported target '${1}'." >&2
      exit 1
      ;;
  esac
}

slice_total_devices() {
  local slice="$1"
  local topology="${slice#*:}"
  local total=1
  local dim
  IFS='x' read -ra dims <<< "${topology}"
  for dim in "${dims[@]}"; do
    total=$((total * dim))
  done
  echo "${total}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target=*)
      TARGET_CLUSTER="${1#*=}"
      shift
      ;;
    --target)
      TARGET_CLUSTER="$2"
      shift 2
      ;;
    --image=*)
      TUNIX_IMAGE="${1#*=}"
      shift
      ;;
    --image)
      TUNIX_IMAGE="$2"
      shift 2
      ;;
    --jobset-name=*)
      JOBSET_NAME="${1#*=}"
      shift
      ;;
    --jobset-name)
      JOBSET_NAME="$2"
      shift 2
      ;;
    --keep-jobset)
      KEEP_JOBSET=1
      shift
      ;;
    --help)
      cat <<EOF
Usage: $(basename "$0") [--target TARGET] [--image IMAGE] [--jobset-name NAME] [--keep-jobset]

Launches a standalone rollout worker on Pathways without the GRPO orchestrator,
then probes the worker with deterministic prompts over its gRPC interface.

Key env vars:
  TUNIX_IMAGE             User container image that already contains the current repo state.
  WEIGHT_SYNC_MODE        Defaults to none for a pure rollout smoke test.
  SAMPLER                 Defaults to vllm.
  PATHWAYS_SERVER_IMAGE   Optional Pathways server image override.
  PATHWAYS_PROXY_IMAGE    Optional Pathways proxy image override.
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

apply_cluster_target "${TARGET_CLUSTER}"

if [[ -z "${TUNIX_IMAGE}" ]]; then
  echo "TUNIX_IMAGE must be set to an image containing the current repo state." >&2
  exit 1
fi

total_devices=$(slice_total_devices "${TPU_SLICE}")
if (( ROLLOUT_MESH_TP <= 0 || total_devices % ROLLOUT_MESH_TP != 0 )); then
  echo "Invalid rollout mesh: slice=${TPU_SLICE} total_devices=${total_devices} mesh_tp=${ROLLOUT_MESH_TP}" >&2
  exit 1
fi
ROLLOUT_MESH_FSDP=$((total_devices / ROLLOUT_MESH_TP))

debug_flag=""
if [[ "${DEBUG}" != "0" ]]; then
  debug_flag="--debug"
fi

cd "${REPO_ROOT}"
source tunix/experimental/examples/math_gsm8k_dist/enter_kube_context.sh

cleanup() {
  if [[ "${KEEP_JOBSET}" != "1" ]]; then
    kubectl delete jobset "${JOBSET_NAME}" --ignore-not-found >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

kubectl delete jobset "${JOBSET_NAME}" --ignore-not-found >/dev/null 2>&1 || true

${PYTHON_BIN} tunix/experimental/distributed/deployment/yaml_generator.py \
  tunix/experimental/distributed/deployment/yamls/${JOBSET_YAML} \
  --jobset_name="${JOBSET_NAME}" \
  --tpu_slice="${TPU_SLICE}" \
  --cpu_machine="${CPU_MACHINE}" \
  --pathways_gcs_scratch_location="${GCS_SCRATCH_LOCATION}" \
  ${PATHWAYS_SERVER_IMAGE:+--pathways_server_image=${PATHWAYS_SERVER_IMAGE}} \
  ${PATHWAYS_PROXY_IMAGE:+--pathways_proxy_server_image=${PATHWAYS_PROXY_IMAGE}} \
  ${HF_TOKEN_SECRET_NAME:+--worker_hf_token_secret_name=${HF_TOKEN_SECRET_NAME}} \
  --worker_container_image="${TUNIX_IMAGE}" \
  --worker_container_port="${ROLLOUT_PORT}" \
  --worker_startup_command=" \
    SKIP_JAX_PRECOMPILE=1 \
    STANDALONE_INIT_WITH_RANDOM_WEIGHTS=${STANDALONE_INIT_WITH_RANDOM_WEIGHTS} \
    VERIFY_WEIGHTS=${VERIFY_WEIGHTS} \
    /opt/venv/bin/python3 -m tunix.experimental.examples.math_gsm8k_dist.run_rollout_probe_server \
      --worker_id=${JOBSET_NAME} \
      --port=${ROLLOUT_PORT} \
      --model_name='${MODEL_NAME}' \
      --model_id='${MODEL_ID}' \
      --model_dir='${MODEL_DIR}' \
      --tokenizer_path='${TOKENIZER_PATH}' \
      --mesh_fsdp=${ROLLOUT_MESH_FSDP} \
      --mesh_tp=${ROLLOUT_MESH_TP} \
      --max_prompt_length=${MAX_PROMPT_LENGTH} \
      --max_response_length=${MAX_RESPONSE_LENGTH} \
      --lora_rank=${LORA_RANK} \
      --lora_alpha=${LORA_ALPHA} \
      --sampler=${SAMPLER} \
      --weight_sync_mode=${WEIGHT_SYNC_MODE} \
      ${debug_flag} \
  " \
  | kubectl apply -f -

selector="jobset.sigs.k8s.io/jobset-name=${JOBSET_NAME},jobset.sigs.k8s.io/replicatedjob-name=proc"
kubectl wait -n default --for=condition=Ready pod -l "${selector}" --timeout="${WAIT_TIMEOUT}"
proc_pod=$(kubectl get pods -n default -l "${selector}" -o jsonpath='{.items[0].metadata.name}')

echo "Probing rollout worker in pod ${proc_pod} on port ${ROLLOUT_PORT}"
kubectl exec -n default "${proc_pod}" -c main -- \
  /opt/venv/bin/python3 -m tunix.experimental.examples.math_gsm8k_dist.probe_rollout_worker \
  --address "grpc://127.0.0.1:${ROLLOUT_PORT}" \
  --max-generation-steps 32 \
  --temperature 0.0 \
  --top-p 1.0 \
  --top-k 1 \
  --seed 0 \
  --startup-timeout-s 900 \
  --prompt "${PROMPT_ONE}" \
  --prompt "${PROMPT_TWO}"