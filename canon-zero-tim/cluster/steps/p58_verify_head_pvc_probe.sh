#!/usr/bin/env bash
set -euo pipefail

namespace="${P58_NAMESPACE:-default}"
queue_name="${P58_QUEUE_NAME:-multislice-queue}"
head_nodepool="${P58_HEAD_NODEPOOL:-canon-cpu-pool}"
model_pvc="${P58_MODEL_PVC:-haoyugao-cpu-np-pvc}"
probe_pod="${P58_HEAD_PVC_PROBE_POD:?set P58_HEAD_PVC_PROBE_POD to the live probe Pod name}"

case "$namespace:$queue_name:$head_nodepool:$model_pvc:$probe_pod" in
  *[!a-z0-9:.-]*|*::*|:*|*:)
    echo "P58_HEAD_PVC_BLOCKED reason=invalid_identifier" >&2
    exit 2
    ;;
esac
if [[ "$queue_name" != "multislice-queue" || \
      "$head_nodepool" != "canon-cpu-pool" || \
      "$model_pvc" != "haoyugao-cpu-np-pvc" ]]; then
  echo "P58_HEAD_PVC_BLOCKED reason=unsigned_contract" >&2
  exit 2
fi
command -v kubectl >/dev/null || {
  echo "P58_HEAD_PVC_BLOCKED reason=kubectl_missing" >&2
  exit 2
}

phase="$(kubectl -n "$namespace" get pod "$probe_pod" -o jsonpath='{.status.phase}')"
gates="$(kubectl -n "$namespace" get pod "$probe_pod" -o jsonpath='{range .spec.schedulingGates[*]}{.name}{" "}{end}')"
pod_queue="$(kubectl -n "$namespace" get pod "$probe_pod" -o jsonpath='{.metadata.labels.kueue\.x-k8s\.io/queue-name}')"
pod_managed="$(kubectl -n "$namespace" get pod "$probe_pod" -o jsonpath='{.metadata.labels.kueue\.x-k8s\.io/managed}')"
pod_pool="$(kubectl -n "$namespace" get pod "$probe_pod" -o jsonpath='{.spec.nodeSelector.cloud\.google\.com/gke-nodepool}')"
pod_claim="$(kubectl -n "$namespace" get pod "$probe_pod" -o jsonpath='{.spec.volumes[?(@.name=="model-pvc")].persistentVolumeClaim.claimName}')"
pod_read_only="$(kubectl -n "$namespace" get pod "$probe_pod" -o jsonpath='{.spec.volumes[?(@.name=="model-pvc")].persistentVolumeClaim.readOnly}')"
node_name="$(kubectl -n "$namespace" get pod "$probe_pod" -o jsonpath='{.spec.nodeName}')"
if [[ "$phase" != "Succeeded" || -n "$gates" || \
      "$pod_queue" != "$queue_name" || "$pod_managed" != "true" || \
      "$pod_pool" != "$head_nodepool" || "$pod_claim" != "$model_pvc" || \
      "$pod_read_only" != "true" || -z "$node_name" ]]; then
  echo "P58_HEAD_PVC_BLOCKED reason=probe_not_complete pod=$probe_pod phase=${phase:-missing} gates=${gates:-none} queue=${pod_queue:-missing} kueue_managed=${pod_managed:-missing} nodepool=${pod_pool:-missing} pvc=${pod_claim:-missing} read_only=${pod_read_only:-missing} node=${node_name:-missing}" >&2
  exit 3
fi
actual_pool="$(kubectl get node "$node_name" -o jsonpath='{.metadata.labels.cloud\.google\.com/gke-nodepool}')"
if [[ "$actual_pool" != "$head_nodepool" ]]; then
  echo "P58_HEAD_PVC_BLOCKED reason=probe_wrong_nodepool pod=$probe_pod node=$node_name actual_pool=${actual_pool:-missing}" >&2
  exit 3
fi
log="$(kubectl -n "$namespace" logs "$probe_pod" -c head-pvc-probe)"
if [[ "$log" != "P58_HEAD_PVC_MOUNT_PASS path=/mnt/disks/linchai_data/models/Qwen3-4B-Instruct-2507" ]]; then
  echo "P58_HEAD_PVC_BLOCKED reason=missing_mount_marker pod=$probe_pod" >&2
  exit 3
fi

echo "P58_HEAD_PVC_PASS scope=canon-head-read-only-mount namespace=$namespace kueue_managed=true nodepool=$head_nodepool pvc=$model_pvc pod=$probe_pod node=$node_name"
