#!/bin/bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# End-to-end GKE multi-worker OpenTelemetry verification script for dist_rl.
#
# Deploys the standalone OTel Collector Service (${USER}-otel:4317) from
# yamls/jobset.otel.yaml and launches 4 in-cluster worker pods (rollout-0,
# rollout-1, trainer-0, orchestrator-0) that resolve http://${USER}-otel:4317
# via Kubernetes DNS, initialize otel_setup.setup_metrics(), and export OTLP
# gRPC metrics.
#
# Usage:
#   # On a machine with kubectl configured (e.g. sizhi-dev -> trellis-demo-0810):
#   bash third_party/py/tunix/dist_rl/distributed/deployment/verify_otel_gke.sh
#
#   # Or from Cloudtop via SSH to a VM with GKE credentials:
#   KUBECTL_SSH_HOST=sizhi_google_com@sizhi-dev \
#     bash third_party/py/tunix/dist_rl/distributed/deployment/verify_otel_gke.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
YAML_GEN="${SCRIPT_DIR}/yaml_generator.py"
OTEL_TEMPLATE="${SCRIPT_DIR}/yamls/jobset.otel.yaml"
OTEL_SETUP_PY="${SCRIPT_DIR}/../../common/otel_setup.py"

export USER_PREFIX="${USER_PREFIX:-${USER:-sizhi}}"
export OTEL_ID="${OTEL_ID:-${USER_PREFIX}-otel}"
export OTEL_PORT="${OTEL_PORT:-4317}"
export NAMESPACE="${K8S_NAMESPACE:-default}"
export WORKER_IMAGE="${WORKER_IMAGE:-us-west1-docker.pkg.dev/supercomputer-testing/lancewang/tunix_base_image:tunix_0904}"
export CLEANUP="${CLEANUP:-false}"

# Auto-detect whether local kubectl has an active authenticated cluster context;
# if not and KUBECTL_SSH_HOST is set (or sizhi-dev is reachable), execute
# against the remote kubectl host so any agent on Cloudtop can reproduce the
# test with one command.
KUBECTL_SSH_HOST="${KUBECTL_SSH_HOST:-}"
if [[ -z "${KUBECTL_SSH_HOST}" ]]; then
  if ! kubectl get nodes --request-timeout=5s >/dev/null 2>&1; then
    if ssh -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=no "${USER}_google_com@sizhi-dev" "kubectl get nodes --request-timeout=5s" >/dev/null 2>&1; then
      KUBECTL_SSH_HOST="${USER}_google_com@sizhi-dev"
    fi
  fi
fi

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT

# 1. Render jobset.otel.yaml via yaml_generator.py and set debug verbosity to
#    detailed so kubectl logs deploy/${OTEL_ID} prints every ResourceMetrics item.
python3 "${YAML_GEN}" "${OTEL_TEMPLATE}" \
  --jobset_name="${OTEL_ID}" \
  --namespace="${NAMESPACE}" \
  --worker_container_port="${OTEL_PORT}" \
  | sed 's/verbosity: basic/verbosity: detailed/' \
  > "${WORK_DIR}/rendered_otel_detailed.yaml"

cp "${OTEL_SETUP_PY}" "${WORK_DIR}/otel_setup.py"

cat > "${WORK_DIR}/multi_worker_test.yaml" <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: ${USER_PREFIX}-rollout-worker-0
  namespace: ${NAMESPACE}
  labels:
    app: ${USER_PREFIX}-dist-rl-test
    role: rollout
spec:
  restartPolicy: Never
  tolerations:
  - key: "google.com/tpu"
    operator: "Exists"
    effect: "NoSchedule"
  - key: "cloud.google.com/gke-nodepool"
    operator: "Exists"
    effect: "NoSchedule"
  containers:
  - name: rollout-worker
    image: ${WORKER_IMAGE}
    imagePullPolicy: IfNotPresent
    env:
    - name: OTEL_EXPORTER_OTLP_ENDPOINT
      value: "http://${OTEL_ID}:${OTEL_PORT}"
    - name: OTEL_SERVICE_NAME
      value: "tunix-rollout"
    - name: OTEL_METRIC_EXPORT_INTERVAL
      value: "1000"
    - name: WORKER_ID
      value: "rollout-0"
    command: ["bash", "-c"]
    args:
    - |
      pip install -q opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp-proto-grpc 2>/dev/null || true
      PYTHONPATH=/otel_src python3 - << "PY"
      import os, socket, urllib.parse
      import otel_setup
      from opentelemetry import metrics
      ep = os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"]
      wid = os.environ["WORKER_ID"]
      parsed = urllib.parse.urlparse(ep)
      ip = socket.gethostbyname(parsed.hostname)
      with socket.create_connection((parsed.hostname, parsed.port), timeout=2.0) as s:
          print(f"[{wid}] K8s DNS resolved {parsed.hostname} -> {ip}:{parsed.port}, TCP connected: pod_ip={s.getsockname()[0]}:{s.getsockname()[1]} -> collector_vip={s.getpeername()[0]}:{s.getpeername()[1]}")
      ok = otel_setup.setup_metrics(
          default_service_name=os.environ["OTEL_SERVICE_NAME"],
          resource_attributes={
              "tunix.worker_id": wid,
              "tunix.role": "rollout",
              "tunix.run_id": "trellis-gke-e2e",
          },
      )
      print(f"[{wid}] otel_setup.setup_metrics() -> {ok}")
      meter = metrics.get_meter("tunix.worker")
      c = meter.create_counter("tunix.worker.trajectories.completed", unit="{trajectory}")
      h = meter.create_histogram("tunix.worker.execution_sec", unit="s")
      c.add(4, {"worker_id": wid, "status": "ok", "task_name": "gsm8k"})
      h.record(1.35, {"worker_id": wid, "task_name": "gsm8k"})
      metrics.get_meter_provider().force_flush()
      metrics.get_meter_provider().shutdown()
      print(f"[{wid}] Exported metrics to {ep} -> OK")
      PY
    volumeMounts:
    - name: otel-src
      mountPath: /otel_src
  volumes:
  - name: otel-src
    configMap:
      name: ${USER_PREFIX}-otel-py
---
apiVersion: v1
kind: Pod
metadata:
  name: ${USER_PREFIX}-rollout-worker-1
  namespace: ${NAMESPACE}
  labels:
    app: ${USER_PREFIX}-dist-rl-test
    role: rollout
spec:
  restartPolicy: Never
  tolerations:
  - key: "google.com/tpu"
    operator: "Exists"
    effect: "NoSchedule"
  - key: "cloud.google.com/gke-nodepool"
    operator: "Exists"
    effect: "NoSchedule"
  containers:
  - name: rollout-worker
    image: ${WORKER_IMAGE}
    imagePullPolicy: IfNotPresent
    env:
    - name: OTEL_EXPORTER_OTLP_ENDPOINT
      value: "http://${OTEL_ID}:${OTEL_PORT}"
    - name: OTEL_SERVICE_NAME
      value: "tunix-rollout"
    - name: OTEL_METRIC_EXPORT_INTERVAL
      value: "1000"
    - name: WORKER_ID
      value: "rollout-1"
    command: ["bash", "-c"]
    args:
    - |
      pip install -q opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp-proto-grpc 2>/dev/null || true
      PYTHONPATH=/otel_src python3 - << "PY"
      import os, socket, urllib.parse
      import otel_setup
      from opentelemetry import metrics
      ep = os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"]
      wid = os.environ["WORKER_ID"]
      parsed = urllib.parse.urlparse(ep)
      ip = socket.gethostbyname(parsed.hostname)
      with socket.create_connection((parsed.hostname, parsed.port), timeout=2.0) as s:
          print(f"[{wid}] K8s DNS resolved {parsed.hostname} -> {ip}:{parsed.port}, TCP connected: pod_ip={s.getsockname()[0]}:{s.getsockname()[1]} -> collector_vip={s.getpeername()[0]}:{s.getpeername()[1]}")
      ok = otel_setup.setup_metrics(
          default_service_name=os.environ["OTEL_SERVICE_NAME"],
          resource_attributes={
              "tunix.worker_id": wid,
              "tunix.role": "rollout",
              "tunix.run_id": "trellis-gke-e2e",
          },
      )
      print(f"[{wid}] otel_setup.setup_metrics() -> {ok}")
      meter = metrics.get_meter("tunix.worker")
      c = meter.create_counter("tunix.worker.trajectories.completed", unit="{trajectory}")
      h = meter.create_histogram("tunix.worker.execution_sec", unit="s")
      c.add(4, {"worker_id": wid, "status": "ok", "task_name": "gsm8k"})
      h.record(2.18, {"worker_id": wid, "task_name": "gsm8k"})
      metrics.get_meter_provider().force_flush()
      metrics.get_meter_provider().shutdown()
      print(f"[{wid}] Exported metrics to {ep} -> OK")
      PY
    volumeMounts:
    - name: otel-src
      mountPath: /otel_src
  volumes:
  - name: otel-src
    configMap:
      name: ${USER_PREFIX}-otel-py
---
apiVersion: v1
kind: Pod
metadata:
  name: ${USER_PREFIX}-trainer-worker-0
  namespace: ${NAMESPACE}
  labels:
    app: ${USER_PREFIX}-dist-rl-test
    role: trainer
spec:
  restartPolicy: Never
  tolerations:
  - key: "google.com/tpu"
    operator: "Exists"
    effect: "NoSchedule"
  - key: "cloud.google.com/gke-nodepool"
    operator: "Exists"
    effect: "NoSchedule"
  containers:
  - name: trainer-worker
    image: ${WORKER_IMAGE}
    imagePullPolicy: IfNotPresent
    env:
    - name: OTEL_EXPORTER_OTLP_ENDPOINT
      value: "http://${OTEL_ID}:${OTEL_PORT}"
    - name: OTEL_SERVICE_NAME
      value: "tunix-trainer"
    - name: OTEL_METRIC_EXPORT_INTERVAL
      value: "1000"
    - name: WORKER_ID
      value: "trainer-0"
    command: ["bash", "-c"]
    args:
    - |
      pip install -q opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp-proto-grpc 2>/dev/null || true
      PYTHONPATH=/otel_src python3 - << "PY"
      import os, socket, urllib.parse
      import otel_setup
      from opentelemetry import metrics
      ep = os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"]
      wid = os.environ["WORKER_ID"]
      parsed = urllib.parse.urlparse(ep)
      ip = socket.gethostbyname(parsed.hostname)
      with socket.create_connection((parsed.hostname, parsed.port), timeout=2.0) as s:
          print(f"[{wid}] K8s DNS resolved {parsed.hostname} -> {ip}:{parsed.port}, TCP connected: pod_ip={s.getsockname()[0]}:{s.getsockname()[1]} -> collector_vip={s.getpeername()[0]}:{s.getpeername()[1]}")
      ok = otel_setup.setup_metrics(
          default_service_name=os.environ["OTEL_SERVICE_NAME"],
          resource_attributes={
              "tunix.worker_id": wid,
              "tunix.role": "trainer",
              "tunix.run_id": "trellis-gke-e2e",
          },
      )
      print(f"[{wid}] otel_setup.setup_metrics() -> {ok}")
      meter = metrics.get_meter("tunix.trainer")
      h = meter.create_histogram("tunix.trainer.step.duration", unit="s")
      h.record(0.84, {"worker_id": wid, "phase": "fwd_bwd"})
      metrics.get_meter_provider().force_flush()
      metrics.get_meter_provider().shutdown()
      print(f"[{wid}] Exported metrics to {ep} -> OK")
      PY
    volumeMounts:
    - name: otel-src
      mountPath: /otel_src
  volumes:
  - name: otel-src
    configMap:
      name: ${USER_PREFIX}-otel-py
---
apiVersion: v1
kind: Pod
metadata:
  name: ${USER_PREFIX}-orchestrator-0
  namespace: ${NAMESPACE}
  labels:
    app: ${USER_PREFIX}-dist-rl-test
    role: orchestrator
spec:
  restartPolicy: Never
  tolerations:
  - key: "google.com/tpu"
    operator: "Exists"
    effect: "NoSchedule"
  - key: "cloud.google.com/gke-nodepool"
    operator: "Exists"
    effect: "NoSchedule"
  containers:
  - name: orchestrator
    image: ${WORKER_IMAGE}
    imagePullPolicy: IfNotPresent
    env:
    - name: OTEL_EXPORTER_OTLP_ENDPOINT
      value: "http://${OTEL_ID}:${OTEL_PORT}"
    - name: OTEL_SERVICE_NAME
      value: "tunix-orchestrator"
    - name: OTEL_METRIC_EXPORT_INTERVAL
      value: "1000"
    - name: WORKER_ID
      value: "orchestrator-0"
    command: ["bash", "-c"]
    args:
    - |
      pip install -q opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp-proto-grpc 2>/dev/null || true
      PYTHONPATH=/otel_src python3 - << "PY"
      import os, socket, urllib.parse
      import otel_setup
      from opentelemetry import metrics
      ep = os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"]
      wid = os.environ["WORKER_ID"]
      parsed = urllib.parse.urlparse(ep)
      ip = socket.gethostbyname(parsed.hostname)
      with socket.create_connection((parsed.hostname, parsed.port), timeout=2.0) as s:
          print(f"[{wid}] K8s DNS resolved {parsed.hostname} -> {ip}:{parsed.port}, TCP connected: pod_ip={s.getsockname()[0]}:{s.getsockname()[1]} -> collector_vip={s.getpeername()[0]}:{s.getpeername()[1]}")
      ok = otel_setup.setup_metrics(
          default_service_name=os.environ["OTEL_SERVICE_NAME"],
          resource_attributes={
              "tunix.worker_id": wid,
              "tunix.role": "orchestrator",
              "tunix.run_id": "trellis-gke-e2e",
          },
      )
      print(f"[{wid}] otel_setup.setup_metrics() -> {ok}")
      meter = metrics.get_meter("tunix.orchestrator")
      c = meter.create_counter("tunix.goodput.tokens.total", unit="{token}")
      c.add(512, {"fate": "effective"})
      metrics.get_meter_provider().force_flush()
      metrics.get_meter_provider().shutdown()
      print(f"[{wid}] Exported metrics to {ep} -> OK")
      PY
    volumeMounts:
    - name: otel-src
      mountPath: /otel_src
  volumes:
  - name: otel-src
    configMap:
      name: ${USER_PREFIX}-otel-py
EOF

cat > "${WORK_DIR}/run_cluster_steps.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail

STAGE_DIR="\${1:-/tmp/otel_gke_verify}"

run_cmd() {
  echo "$ \$*"
  "\$@"
  echo ""
}

run_cmd kubectl config current-context
run_cmd kubectl get nodes -o wide

kubectl delete pod -n "${NAMESPACE}" -l "app=${USER_PREFIX}-dist-rl-test" --ignore-not-found >/dev/null 2>&1 || true
kubectl delete configmap -n "${NAMESPACE}" "${USER_PREFIX}-otel-py" --ignore-not-found >/dev/null 2>&1 || true
kubectl delete -f "\${STAGE_DIR}/rendered_otel_detailed.yaml" --ignore-not-found >/dev/null 2>&1 || true

run_cmd kubectl create configmap "${USER_PREFIX}-otel-py" -n "${NAMESPACE}" --from-file=otel_setup.py="\${STAGE_DIR}/otel_setup.py"
run_cmd kubectl apply -f "\${STAGE_DIR}/rendered_otel_detailed.yaml"
run_cmd kubectl rollout status "deployment/${OTEL_ID}" -n "${NAMESPACE}" --timeout=60s
run_cmd kubectl get deploy,pods,svc,endpoints -n "${NAMESPACE}" -l "app=${OTEL_ID}" -o wide

run_cmd kubectl apply -f "\${STAGE_DIR}/multi_worker_test.yaml"

for pod in "${USER_PREFIX}-rollout-worker-0" "${USER_PREFIX}-rollout-worker-1" "${USER_PREFIX}-trainer-worker-0" "${USER_PREFIX}-orchestrator-0"; do
  for _ in \$(seq 1 90); do
    phase=\$(kubectl get pod "\$pod" -n "${NAMESPACE}" -o jsonpath='{.status.phase}' 2>/dev/null || echo "")
    if [[ "\$phase" == "Succeeded" || "\$phase" == "Failed" ]]; then
      break
    fi
    sleep 1
  done
done

run_cmd kubectl get pods -n "${NAMESPACE}" -l "app=${USER_PREFIX}-dist-rl-test" -o wide
run_cmd kubectl logs -n "${NAMESPACE}" "${USER_PREFIX}-rollout-worker-0"
run_cmd kubectl logs -n "${NAMESPACE}" "${USER_PREFIX}-rollout-worker-1"
run_cmd kubectl logs -n "${NAMESPACE}" "${USER_PREFIX}-trainer-worker-0"
run_cmd kubectl logs -n "${NAMESPACE}" "${USER_PREFIX}-orchestrator-0"

sleep 6
run_cmd kubectl logs -n "${NAMESPACE}" "deploy/${OTEL_ID}" --tail=90

COLLECTOR_LOGS="\$(kubectl logs -n "${NAMESPACE}" "deploy/${OTEL_ID}")"
for expected in "rollout-0" "rollout-1" "trainer-0" "orchestrator-0" "tunix.worker.trajectories.completed" "tunix.trainer.step.duration" "tunix.goodput.tokens.total"; do
  if ! grep -q "\${expected}" <<< "\${COLLECTOR_LOGS}"; then
    echo "ERROR: Missing expected token '\${expected}' in collector logs!" >&2
    exit 1
  fi
done
echo "VERIFICATION PASSED: All 4 workers connected to http://${OTEL_ID}:${OTEL_PORT} and exported OTLP metrics."

if [[ "${CLEANUP}" == "true" ]]; then
  kubectl delete pod -n "${NAMESPACE}" -l "app=${USER_PREFIX}-dist-rl-test" --ignore-not-found
  kubectl delete configmap -n "${NAMESPACE}" "${USER_PREFIX}-otel-py" --ignore-not-found
  kubectl delete -f "\${STAGE_DIR}/rendered_otel_detailed.yaml" --ignore-not-found
fi
EOF
chmod +x "${WORK_DIR}/run_cluster_steps.sh"

if [[ -n "${KUBECTL_SSH_HOST}" ]]; then
  echo "Using remote kubectl host: ${KUBECTL_SSH_HOST}"
  ssh -o StrictHostKeyChecking=no "${KUBECTL_SSH_HOST}" "mkdir -p /tmp/otel_gke_verify"
  scp -q -o StrictHostKeyChecking=no \
    "${WORK_DIR}/otel_setup.py" \
    "${WORK_DIR}/rendered_otel_detailed.yaml" \
    "${WORK_DIR}/multi_worker_test.yaml" \
    "${WORK_DIR}/run_cluster_steps.sh" \
    "${KUBECTL_SSH_HOST}:/tmp/otel_gke_verify/"
  ssh -o StrictHostKeyChecking=no "${KUBECTL_SSH_HOST}" "bash /tmp/otel_gke_verify/run_cluster_steps.sh /tmp/otel_gke_verify"
else
  bash "${WORK_DIR}/run_cluster_steps.sh" "${WORK_DIR}"
fi
