#!/bin/bash
# Runner script for DeepSWE Sandbox Pre-warming Benchmark on GKE.
# Runs on bodaborg-v5p-nap in namespace default.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TUNIX_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Pin kubeconfig and cluster env for bodaborg-v5p-nap
export KUBECONFIG="${HOME}/.kube/config.cloud-tpu-shared-capacity.europe-west4.bodaborg-v5p-nap"
if [[ ! -s "${KUBECONFIG}" ]]; then
  export KUBECONFIG="${HOME}/.kube/config"
fi

NAMESPACE="${SANDBOX_NAMESPACE:-default}"
JOB_NAME="deepswe-prewarm-benchmark"
YAML_FILE="${SCRIPT_DIR}/benchmark_prewarm_job.yaml"

echo ">>> Cleaning up any previous benchmark jobs..."
kubectl --kubeconfig "${KUBECONFIG}" delete job "${JOB_NAME}" -n "${NAMESPACE}" --ignore-not-found

echo ">>> Submitting Benchmark Job to namespace '${NAMESPACE}'..."
kubectl --kubeconfig "${KUBECONFIG}" apply -f "${YAML_FILE}" -n "${NAMESPACE}"

echo ">>> Waiting for benchmark pod to start..."
while ! kubectl --kubeconfig "${KUBECONFIG}" get pod -l app=prewarm-benchmark -n "${NAMESPACE}" --no-headers 2>/dev/null | grep -q "Running\|Completed\|Error"; do
  sleep 2
done

echo ">>> Streaming live logs from benchmark pod..."
kubectl --kubeconfig "${KUBECONFIG}" logs -f -l app=prewarm-benchmark -n "${NAMESPACE}"

echo ">>> Benchmark completed."
