#!/usr/bin/env bash
set -euo pipefail

namespace="${P58_NAMESPACE:-default}"
queue_name="${P58_QUEUE_NAME:-multislice-queue}"
sandbox_nodepool="${P58_SANDBOX_NODEPOOL:-deepswe-cpu-pool-2}"
probe_pod="${P58_SANDBOX_PROBE_POD:?set P58_SANDBOX_PROBE_POD to the live probe Pod name}"

case "$namespace:$queue_name:$sandbox_nodepool:$probe_pod" in
  *[!a-z0-9:.-]*|*::*|:*|*:)
    echo "P58_SANDBOX_CAPACITY_BLOCKED reason=invalid_identifier" >&2
    exit 2
    ;;
esac
if [[ "$queue_name" != "multislice-queue" || "$sandbox_nodepool" != "deepswe-cpu-pool-2" ]]; then
  echo "P58_SANDBOX_CAPACITY_BLOCKED reason=unsigned_queue_or_nodepool" >&2
  exit 2
fi
command -v kubectl >/dev/null || {
  echo "P58_SANDBOX_CAPACITY_BLOCKED reason=kubectl_missing" >&2
  exit 2
}

local_queue="$(kubectl -n "$namespace" get localqueue.kueue.x-k8s.io "$queue_name" -o jsonpath='{.spec.clusterQueue}')"
local_active="$(kubectl -n "$namespace" get localqueue.kueue.x-k8s.io "$queue_name" -o jsonpath='{range .status.conditions[?(@.type=="Active")]}{.status}{end}')"
cluster_active="$(kubectl get clusterqueue.kueue.x-k8s.io "$local_queue" -o jsonpath='{range .status.conditions[?(@.type=="Active")]}{.status}{end}')"
if [[ -z "$local_queue" || "$local_active" != "True" || "$cluster_active" != "True" ]]; then
  echo "P58_SANDBOX_CAPACITY_BLOCKED reason=queue_inactive local_queue=$queue_name cluster_queue=${local_queue:-missing} local_active=${local_active:-missing} cluster_active=${cluster_active:-missing}" >&2
  exit 3
fi

ready_nodes="$(kubectl get nodes -l "cloud.google.com/gke-nodepool=$sandbox_nodepool" -o jsonpath='{range .items[*]}{.metadata.name}{"|"}{.spec.unschedulable}{"|"}{range .status.conditions[?(@.type=="Ready")]}{.status}{end}{"\n"}{end}' | awk -F'|' '$2 != "true" && $3 == "True" {count += 1} END {print count + 0}')"
if [[ "$ready_nodes" -lt 1 ]]; then
  echo "P58_SANDBOX_CAPACITY_BLOCKED reason=no_ready_cpu_node nodepool=$sandbox_nodepool" >&2
  exit 3
fi

phase="$(kubectl -n "$namespace" get pod "$probe_pod" -o jsonpath='{.status.phase}')"
gates="$(kubectl -n "$namespace" get pod "$probe_pod" -o jsonpath='{range .spec.schedulingGates[*]}{.name}{" "}{end}')"
pod_queue="$(kubectl -n "$namespace" get pod "$probe_pod" -o jsonpath='{.metadata.labels.kueue\.x-k8s\.io/queue-name}')"
pod_managed="$(kubectl -n "$namespace" get pod "$probe_pod" -o jsonpath='{.metadata.labels.kueue\.x-k8s\.io/managed}')"
pod_pool="$(kubectl -n "$namespace" get pod "$probe_pod" -o jsonpath='{.spec.nodeSelector.cloud\.google\.com/gke-nodepool}')"
node_name="$(kubectl -n "$namespace" get pod "$probe_pod" -o jsonpath='{.spec.nodeName}')"
if [[ "$phase" != "Running" || -n "$gates" || "$pod_queue" != "$queue_name" || "$pod_managed" != "true" || "$pod_pool" != "$sandbox_nodepool" || -z "$node_name" ]]; then
  echo "P58_SANDBOX_CAPACITY_BLOCKED reason=probe_not_admitted pod=$probe_pod phase=${phase:-missing} gates=${gates:-none} queue=${pod_queue:-missing} kueue_managed=${pod_managed:-missing} nodepool=${pod_pool:-missing} node=${node_name:-missing}" >&2
  exit 3
fi
actual_pool="$(kubectl get node "$node_name" -o jsonpath='{.metadata.labels.cloud\.google\.com/gke-nodepool}')"
if [[ "$actual_pool" != "$sandbox_nodepool" ]]; then
  echo "P58_SANDBOX_CAPACITY_BLOCKED reason=probe_wrong_nodepool pod=$probe_pod node=$node_name actual_pool=${actual_pool:-missing}" >&2
  exit 3
fi

echo "P58_SANDBOX_CAPACITY_PASS scope=one-sandbox-admission-only namespace=$namespace local_queue=$queue_name cluster_queue=$local_queue kueue_managed=true nodepool=$sandbox_nodepool ready_nodes=$ready_nodes pod=$probe_pod node=$node_name"
