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

MODEL_ID=${MODEL_ID:-Qwen/Qwen3-1.7B}
TOKENIZER_PATH=${TOKENIZER_PATH:-${MODEL_ID}}
SERVE_MODEL=${SERVE_MODEL:-${MODEL_ID}}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-${MODEL_ID}}
MODEL_NAME=${MODEL_NAME:-Qwen3-1.7B}

MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-128}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}
SKIP_JAX_PRECOMPILE=${SKIP_JAX_PRECOMPILE:-1}
SERVE_MESH_DP=${SERVE_MESH_DP:-4}
SERVE_MESH_TP=${SERVE_MESH_TP:-4}
SEED=${SEED:-0}

CPU_MACHINE=${CPU_MACHINE:-n2-standard-64}
GCS_SCRATCH_LOCATION=${GCS_SCRATCH_LOCATION:-gs://cloud-pathways-staging/tmp}
PATHWAYS_SERVER_IMAGE=${PATHWAYS_SERVER_IMAGE:-}
PATHWAYS_PROXY_IMAGE=${PATHWAYS_PROXY_IMAGE:-}
HF_TOKEN_SECRET_NAME=${HF_TOKEN_SECRET_NAME:-}
PYTHON_BIN=${PYTHON_BIN:-python3}

JOBSET_NAME=${JOBSET_NAME:-${USER}-vllm-serve-smoke}
JOBSET_YAML=${JOBSET_YAML:-jobset.pathways.yaml}
TPU_SLICE=${TPU_SLICE:-tpuv5e:4x4}
VLLM_PORT=${VLLM_PORT:-8000}
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

Launches a standalone Pathways vLLM OpenAI server on a TPU v5e slice and probes
it in-cluster with two deterministic prompts.

Key env vars:
  TUNIX_IMAGE             User container image that already contains this repo state.
  PATHWAYS_SERVER_IMAGE   Optional Pathways server image override.
  PATHWAYS_PROXY_IMAGE    Optional Pathways proxy image override.
  HF_TOKEN_SECRET_NAME    Optional Kubernetes secret exposing HF_TOKEN.
  SERVE_MODEL             Defaults to MODEL_ID to avoid stale local snapshots.
  TOKENIZER_PATH          Defaults to MODEL_ID.
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
  --worker_container_port="${VLLM_PORT}" \
  --worker_startup_command=" \
    SKIP_JAX_PRECOMPILE=${SKIP_JAX_PRECOMPILE} \
    VLLM_TARGET_DEVICE=tpu \
    TPU_BACKEND_TYPE=jax \
    VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    VLLM_TPU_RPA_VERSION=2 \
    DISABLE_MOSAIC_ATTN=1 \
    VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    /opt/venv/bin/python3 -m vllm.entrypoints.openai.api_server \
      --host=0.0.0.0 \
      --port=${VLLM_PORT} \
      --model='${SERVE_MODEL}' \
      --served-model-name='${SERVED_MODEL_NAME}' \
      --tokenizer='${TOKENIZER_PATH}' \
      --data-parallel-size=${SERVE_MESH_DP} \
      --tensor-parallel-size=${SERVE_MESH_TP} \
      --max-model-len=${MAX_MODEL_LEN} \
      --seed=${SEED} \
  " \
  | kubectl apply -f -

selector="jobset.sigs.k8s.io/jobset-name=${JOBSET_NAME},jobset.sigs.k8s.io/replicatedjob-name=proc"
kubectl wait -n default --for=condition=Ready pod -l "${selector}" --timeout="${WAIT_TIMEOUT}"
proc_pod=$(kubectl get pods -n default -l "${selector}" -o jsonpath='{.items[0].metadata.name}')

echo "Probing vLLM server in pod ${proc_pod} on port ${VLLM_PORT}"
kubectl exec -i -n default "${proc_pod}" -c main -- env \
  API_PORT="${VLLM_PORT}" \
  SERVED_MODEL_NAME="${SERVED_MODEL_NAME}" \
  MAX_TOKENS="${MAX_RESPONSE_LENGTH}" \
  PROMPT_ONE="${PROMPT_ONE}" \
  PROMPT_TWO="${PROMPT_TWO}" \
  /opt/venv/bin/python3 - <<'PY'
import json
import os
import time
import urllib.request

api_port = int(os.environ["API_PORT"])
base_url = f"http://127.0.0.1:{api_port}"
model = os.environ["SERVED_MODEL_NAME"]
max_tokens = int(os.environ["MAX_TOKENS"])
prompts = [os.environ["PROMPT_ONE"], os.environ["PROMPT_TWO"]]

def request(path: str, payload: dict | None = None) -> tuple[int, str]:
  data = None if payload is None else json.dumps(payload).encode("utf-8")
  req = urllib.request.Request(
      base_url + path,
      data=data,
      headers={"Content-Type": "application/json"},
      method="POST" if payload is not None else "GET",
  )
  with urllib.request.urlopen(req, timeout=30) as resp:
    return resp.status, resp.read().decode("utf-8")

deadline = time.time() + 900
last_error = None
while time.time() < deadline:
  try:
    status, health = request("/health")
    _, models_raw = request("/v1/models")
    models = json.loads(models_raw) if models_raw else {}
    print("Health status:", status)
    print("Health body:", health)
    print("Models:", json.dumps(models, indent=2, sort_keys=True))
    break
  except Exception as exc:  # pylint: disable=broad-except
    last_error = exc
    time.sleep(5)
else:
  raise RuntimeError(f"vLLM server did not become ready: {last_error}")

for index, prompt in enumerate(prompts, start=1):
  _, response_raw = request(
      "/v1/chat/completions",
      {
          "model": model,
          "messages": [{"role": "user", "content": prompt}],
          "temperature": 0.0,
          "top_p": 1.0,
          "max_tokens": max_tokens,
          "seed": 0,
      },
  )
  response = json.loads(response_raw) if response_raw else {}
  print(f"\\nPrompt {index}: {prompt}")
  print(json.dumps(response, indent=2, sort_keys=True))
PY