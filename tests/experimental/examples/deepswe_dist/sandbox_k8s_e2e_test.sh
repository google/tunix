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

# DeepSWE Kubernetes Job E2E Launcher
TUNIX_IMAGE=${TUNIX_IMAGE:-"us-central1-docker.pkg.dev/cloud-tpu-multipod-dev/yangmu/tunix/tunix_base_image:trellis-demo-0813"}
NAMESPACE=${NAMESPACE:-"trellis"}
SERVICE_ACCOUNT=${SERVICE_ACCOUNT:-"xpk-sa"}
IMAGE_REWRITE_PREFIX=${IMAGE_REWRITE_PREFIX:-"europe-west4-docker.pkg.dev/cloud-tpu-multipod-dev/tunix"}
DATASET_NAME=${DATASET_NAME:-"R2E-Gym/R2E-Gym-Subset"}
BATCH_SIZE=${BATCH_SIZE:-1}
NUM_GENERATIONS=${NUM_GENERATIONS:-2}
MAX_STEPS=${MAX_STEPS:-3}
MINIMUM_STEP_TIME_IN_SECOND=${MINIMUM_STEP_TIME_IN_SECOND:-0}
SAMPLING_RATE=${SAMPLING_RATE:-1.0}
SHUFFLE=${SHUFFLE:-true}
SEED=${SEED:-42}
CPU_MACHINE=${CPU_MACHINE:-"n2-standard-64"}
NODE_SELECTOR_KEY=${NODE_SELECTOR_KEY:-"cloud.google.com/gke-nodepool"}
NODE_SELECTOR_VAL=${NODE_SELECTOR_VAL:-"sandbox-cpu-pool"}
JOB_NODEPOOL=${JOB_NODEPOOL:-"default-pool"}
SCAFFOLD=${SCAFFOLD:-"r2egym"}
DRY_RUN=${DRY_RUN:-0}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry_run)
      DRY_RUN=1
      shift
      ;;
    --scaffold=*)
      SCAFFOLD="${1#*=}"
      shift
      ;;
    --image=*)
      TUNIX_IMAGE="${1#*=}"
      shift
      ;;
    --namespace=*)
      NAMESPACE="${1#*=}"
      shift
      ;;
    --batch_size=*)
      BATCH_SIZE="${1#*=}"
      shift
      ;;
    --num_generations=*)
      NUM_GENERATIONS="${1#*=}"
      shift
      ;;
    --max_steps=*)
      MAX_STEPS="${1#*=}"
      shift
      ;;
    --minimum_step_time_in_second=*)
      MINIMUM_STEP_TIME_IN_SECOND="${1#*=}"
      shift
      ;;
    --sampling_rate=*)
      SAMPLING_RATE="${1#*=}"
      shift
      ;;
    --shuffle=*)
      SHUFFLE="${1#*=}"
      shift
      ;;
    --seed=*)
      SEED="${1#*=}"
      shift
      ;;
    --node_selector_key=*)
      NODE_SELECTOR_KEY="${1#*=}"
      shift
      ;;
    --node_selector_val=*)
      NODE_SELECTOR_VAL="${1#*=}"
      shift
      ;;
    --job_nodepool=*)
      JOB_NODEPOOL="${1#*=}"
      shift
      ;;
    --service_account=*)
      SERVICE_ACCOUNT="${1#*=}"
      shift
      ;;
    --image_rewrite_prefix=*)
      IMAGE_REWRITE_PREFIX="${1#*=}"
      shift
      ;;
    *)
      shift
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
if [[ "${DRY_RUN}" == "1" || "${DRY_RUN}" == "true" ]]; then
  echo "=== Running DeepSWE E2E Test locally (Dry Run / Mock Mode) ==="
  COMPILED_BIN=$(ls "${SCRIPT_DIR}"/../../../../../../*-bin/third_party/py/tunix/experimental/examples/deepswe_dist/sandbox_k8s_e2e_test 2>/dev/null | head -n 1 || true)
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    ${PYTHON_BIN} -m tunix.experimental.examples.deepswe_dist.sandbox_k8s_e2e_test \
      --dry_run \
      --scaffold="${SCAFFOLD}" \
      --batch_size="${BATCH_SIZE}" \
      --num_generations="${NUM_GENERATIONS}" \
      --max_steps="${MAX_STEPS}" \
      --minimum_step_time_in_second="${MINIMUM_STEP_TIME_IN_SECOND}" \
      --sampling_rate="${SAMPLING_RATE}" \
      --shuffle="${SHUFFLE}" \
      --seed="${SEED}"
  elif [[ -n "${COMPILED_BIN}" && -x "${COMPILED_BIN}" ]]; then
    "${COMPILED_BIN}" \
      --dry_run \
      --scaffold="${SCAFFOLD}" \
      --batch_size="${BATCH_SIZE}" \
      --num_generations="${NUM_GENERATIONS}" \
      --max_steps="${MAX_STEPS}" \
      --minimum_step_time_in_second="${MINIMUM_STEP_TIME_IN_SECOND}" \
      --sampling_rate="${SAMPLING_RATE}" \
      --shuffle="${SHUFFLE}" \
      --seed="${SEED}"
  else
    python3 -m tunix.experimental.examples.deepswe_dist.sandbox_k8s_e2e_test \
      --dry_run \
      --scaffold="${SCAFFOLD}" \
      --batch_size="${BATCH_SIZE}" \
      --num_generations="${NUM_GENERATIONS}" \
      --max_steps="${MAX_STEPS}" \
      --minimum_step_time_in_second="${MINIMUM_STEP_TIME_IN_SECOND}" \
      --sampling_rate="${SAMPLING_RATE}" \
      --shuffle="${SHUFFLE}" \
      --seed="${SEED}"
  fi
  exit 0
fi

# Configure Kubernetes context
if [[ "${SWITCH_KUBE_CONTEXT:-0}" == "1" ]] && [[ -f "${SCRIPT_DIR}/../common/enter_kube_context.sh" ]]; then
  source "${SCRIPT_DIR}/../common/enter_kube_context.sh"
fi

RANDOM_SUFFIX=$(head /dev/urandom | tr -dc a-z0-9 | head -c 6 ; echo '')
JOB_NAME="deepswe-e2e-${USER:-wuhao}-${RANDOM_SUFFIX}"
CONFIGMAP_NAME="code-${JOB_NAME}"

DEEPSWE_SRC_DIR="${REPO_ROOT}/examples/deepswe"
if [[ ! -f "${DEEPSWE_SRC_DIR}/sandbox_utils.py" ]]; then
  DEEPSWE_SRC_DIR="${SCRIPT_DIR}/../../../oss/examples/deepswe"
fi

DEEPSWE_DIST_DIR="${REPO_ROOT}/tunix/experimental/examples/deepswe_dist"
if [[ ! -f "${DEEPSWE_DIST_DIR}/deepswe.py" ]]; then
  DEEPSWE_DIST_DIR="${SCRIPT_DIR}"
fi

echo "=== Creating ConfigMap ${CONFIGMAP_NAME} with local patched files in namespace ${NAMESPACE} ==="
kubectl create configmap "${CONFIGMAP_NAME}" \
  --namespace="${NAMESPACE}" \
  --from-file=sandbox_utils.py="${DEEPSWE_SRC_DIR}/sandbox_utils.py" \
  --from-file=swe_env.py="${DEEPSWE_SRC_DIR}/swe_env.py" \
  --from-file=openhands_utils.py="${DEEPSWE_SRC_DIR}/openhands_utils.py" \
  --from-file=template.py="${DEEPSWE_SRC_DIR}/template.py" \
  --from-file=deepswe.py="${DEEPSWE_DIST_DIR}/deepswe.py" \
  --from-file=sandbox_k8s_e2e_test.py="${SCRIPT_DIR}/sandbox_k8s_e2e_test.py"

echo "=== Submitting Dedicated Kubernetes Job: ${JOB_NAME} in namespace ${NAMESPACE} ==="

cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: ${JOB_NAME}
  namespace: ${NAMESPACE}
spec:
  backoffLimit: 0
  ttlSecondsAfterFinished: 600
  template:
    metadata:
      labels:
        app: deepswe-e2e-test
    spec:
      restartPolicy: Never
      serviceAccountName: ${SERVICE_ACCOUNT}
      nodeSelector:
        cloud.google.com/gke-nodepool: ${JOB_NODEPOOL}
      tolerations:
      - key: "cloud.google.com/gke-nodepool"
        operator: "Exists"
        effect: "NoSchedule"
      volumes:
      - name: code-volume
        configMap:
          name: ${CONFIGMAP_NAME}
      containers:
      - name: e2e-tester
        image: ${TUNIX_IMAGE}
        imagePullPolicy: IfNotPresent
        securityContext:
          privileged: true
        volumeMounts:
        - name: code-volume
          mountPath: /e2e_code
        env:
        - name: NAMESPACE
          value: "${NAMESPACE}"
        - name: NODE_SELECTOR_KEY
          value: "${NODE_SELECTOR_KEY}"
        - name: NODE_SELECTOR_VAL
          value: "${NODE_SELECTOR_VAL}"
        - name: IMAGE_REWRITE_PREFIX
          value: "${IMAGE_REWRITE_PREFIX}"
        - name: SANDBOX_TOLERATIONS
          value: '${SANDBOX_TOLERATIONS:-[{"key":"workload","operator":"Equal","value":"sandbox","effect":"NoSchedule"}]}'
        - name: OPENHANDS_SUPPRESS_BANNER
          value: "1"
        command:
        - bash
        - -c
        - |
          echo "=== DeepSWE Sandbox E2E Job Started at \$(date) ==="
          pip install -q --no-cache-dir gym docker 'swebench==3.0.2' 'openhands-sdk>=1.44.1' 'k8s-agent-sandbox>=0.5.1' httpx
          rm -rf /tmp/agent-sandbox
          git clone --depth 1 https://github.com/kubernetes-sigs/agent-sandbox.git /tmp/agent-sandbox
          pip install -q --no-cache-dir /tmp/agent-sandbox/examples/agent-sandbox-rl
          SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0 pip install -q --no-cache-dir /tmp/agent-sandbox/clients/integrations/openhands 2>/dev/null || true
          mkdir -p /opt/venv/lib/python3.12/site-packages
          cp -r /tmp/agent-sandbox/clients/integrations/openhands/openhands_k8s_agent_sandbox /opt/venv/lib/python3.12/site-packages/ 2>/dev/null || true
          pip install -q --no-deps 'git+https://github.com/r2e-gym/r2e-gym.git@0d94c4eb9431cd195c55a7ea3abd54006c9a1735'
          sed -i 's/create_repo, upload_folder, HfFolder/create_repo, upload_folder/' /opt/venv/lib/python3.12/site-packages/r2egym/agenthub/utils/utils.py 2>/dev/null || true
          sed -i 's/self.commit = ParsedCommit(\*\*json.loads(self.commit_json))/self.commit = ParsedCommit(\*\*(json.loads(self.commit_json) if isinstance(self.commit_json, str) else self.commit_json))/' /opt/venv/lib/python3.12/site-packages/r2egym/agenthub/runtime/docker.py 2>/dev/null || true

          mkdir -p /app/examples/deepswe
          mkdir -p /app/tunix/oss/examples/deepswe
          mkdir -p /app/tunix/experimental/examples/deepswe_dist
          cp -r /tmp/agent-sandbox/clients/integrations/openhands/openhands_k8s_agent_sandbox /app/ 2>/dev/null || true

          cp /e2e_code/sandbox_utils.py /app/examples/deepswe/
          cp /e2e_code/sandbox_utils.py /app/tunix/oss/examples/deepswe/
          cp /e2e_code/swe_env.py /app/examples/deepswe/
          cp -r /app/examples/deepswe/* /app/tunix/oss/examples/deepswe/ 2>/dev/null || true
          cp /e2e_code/swe_env.py /app/tunix/oss/examples/deepswe/
          cp /e2e_code/openhands_utils.py /app/examples/deepswe/ 2>/dev/null || true
          cp /e2e_code/openhands_utils.py /app/tunix/oss/examples/deepswe/ 2>/dev/null || true
          cp /e2e_code/template.py /app/examples/deepswe/ 2>/dev/null || true
          cp /e2e_code/template.py /app/tunix/oss/examples/deepswe/ 2>/dev/null || true
          cp /e2e_code/deepswe.py /app/tunix/experimental/examples/deepswe_dist/
          cp /e2e_code/sandbox_k8s_e2e_test.py /app/tunix/experimental/examples/deepswe_dist/
          cp /e2e_code/*.py /app/

          export PYTHONPATH="/app:/e2e_code:\${PYTHONPATH:-}"

          python3 /app/tunix/experimental/examples/deepswe_dist/sandbox_k8s_e2e_test.py \
            --run_as_job \
            --dataset_name="${DATASET_NAME}" \
            --scaffold="${SCAFFOLD}" \
            --batch_size=${BATCH_SIZE} \
            --num_generations=${NUM_GENERATIONS} \
            --max_steps=${MAX_STEPS} \
            --minimum_step_time_in_second=${MINIMUM_STEP_TIME_IN_SECOND} \
            --sampling_rate=${SAMPLING_RATE} \
            --shuffle=${SHUFFLE} \
            --seed=${SEED} \
            --namespace="${NAMESPACE}" \
            --node_selector_key="${NODE_SELECTOR_KEY}" \
            --node_selector_val="${NODE_SELECTOR_VAL}"
          EXIT_CODE=\$?
          echo "=== DeepSWE Sandbox E2E Job Finished at \$(date) with code \${EXIT_CODE} ==="
          exit \${EXIT_CODE}
EOF

echo "Job ${JOB_NAME} created. Waiting for pod startup..."
kubectl wait --namespace="${NAMESPACE}" --for=condition=Ready pod -l job-name="${JOB_NAME}" --timeout=120s || true

echo "Streaming logs for ${JOB_NAME}:"
kubectl logs --namespace="${NAMESPACE}" -f "job/${JOB_NAME}" || true

echo "Waiting for Job completion..."
if kubectl wait --namespace="${NAMESPACE}" --for=condition=complete --timeout=15m "job/${JOB_NAME}"; then
  echo "🎉 Job ${JOB_NAME} completed successfully!"
  kubectl delete job "${JOB_NAME}" --namespace="${NAMESPACE}" || true
  kubectl delete configmap "${CONFIGMAP_NAME}" --namespace="${NAMESPACE}" || true
  exit 0
else
  echo "❌ Job ${JOB_NAME} failed or timed out."
  kubectl describe job "${JOB_NAME}" --namespace="${NAMESPACE}" || true
  kubectl delete job "${JOB_NAME}" --namespace="${NAMESPACE}" || true
  kubectl delete configmap "${CONFIGMAP_NAME}" --namespace="${NAMESPACE}" || true
  exit 1
fi
