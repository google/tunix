#!/bin/bash

set -euo pipefail

export PATHWAYS_SERVER_IMAGE=${PATHWAYS_SERVER_IMAGE:-us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/datenglin/unsanitized_server:raiden_20260908}
export PATHWAYS_PROXY_IMAGE=${PATHWAYS_PROXY_IMAGE:-us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/datenglin/unsanitized_proxy_server:raiden_20260908}
export TUNIX_IMAGE=${TUNIX_IMAGE:-us-west1-docker.pkg.dev/supercomputer-testing/lancewang/tunix_base_image:tunix_0908}
export REPRO_PY=${REPRO_PY:-/usr/local/google/home/lancewang/github/tunix/raiden_pathways_ffi_two_controller_repro.py}

fatal_log_pattern='Traceback \(most recent call last\)|AssertionError:|RuntimeError:|AttributeError:|ValueError:|FATAL|Fatal Python error:'
cleanup_jobsets=(lwraidenapisrc lwraidenapidst lwraidenffisrc lwraidenffidst)

wait_for_no_pods() {
  local jobset_name="$1"

  while kubectl get pods \
    -l jobset.sigs.k8s.io/jobset-name="$jobset_name",jobset.sigs.k8s.io/replicatedjob-name=proc \
    --no-headers 2>/dev/null | grep -q .; do
    sleep 2
  done
}

remote_log_has_fatal() {
  local pod="$1"
  local log_path="$2"

  kubectl exec "$pod" -c proc -- sh -lc "
    if [ -f '$log_path' ]; then
      grep -Eq '$fatal_log_pattern' '$log_path'
    else
      exit 1
    fi
  " >/dev/null 2>&1
}

remote_log_has_text() {
  local pod="$1"
  local log_path="$2"
  local pattern="$3"

  kubectl exec "$pod" -c proc -- sh -lc "
    if [ -f '$log_path' ]; then
      grep -Eq '$pattern' '$log_path'
    else
      exit 1
    fi
  " >/dev/null 2>&1
}

copy_logs() {
  local src_pod="$1"
  local dst_pod="$2"
  local log_dir="$3"

  kubectl cp "$src_pod":/tmp/direct_api_controller_src.log "$log_dir/direct_api_controller_src.log" -c proc >/dev/null 2>&1 || true
  kubectl cp "$dst_pod":/tmp/direct_api_controller_dst.log "$log_dir/direct_api_controller_dst.log" -c proc >/dev/null 2>&1 || true
  kubectl cp "$src_pod":/tmp/direct_api_source.log "$log_dir/direct_api_source.log" -c proc >/dev/null 2>&1 || true
  kubectl cp "$dst_pod":/tmp/direct_api_destination.log "$log_dir/direct_api_destination.log" -c proc >/dev/null 2>&1 || true
}

cd /usr/local/google/home/lancewang/github/tunix

RUN_ID="raiden_pw_ffi_repro_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$PWD/raiden_bug_logs/$RUN_ID"
mkdir -p "$LOG_DIR"

export PROJECT=cloud-tpu-inference-test
export CLUSTER=lancewang-pw-v5e-4slice
export CLUSTER_LOCATION=us-west1-c
export REGION=us-west1
export ZONE=us-west1-c
source /usr/local/google/home/lancewang/github/tunix/tunix/experimental/examples/common/enter_kube_context.sh

kubectl delete jobset "${cleanup_jobsets[@]}" --ignore-not-found=true --wait=true
for jobset_name in "${cleanup_jobsets[@]}"; do
  wait_for_no_pods "$jobset_name"
done
mkdir -p /tmp/lwraidenlogs_ffi_src/default /tmp/lwraidenlogs_ffi_dst/default
python3 /usr/local/google/home/lancewang/github/tunix/tunix/experimental/distributed/deployment/yaml_generator.py \
  /usr/local/google/home/lancewang/github/tunix/tunix/experimental/distributed/deployment/yamls/jobset.pathways.yaml \
  --jobset_name=lwraidenffisrc \
  --tpu_slice=tpuv5e:4x4 \
  --pathways_server_image="$PATHWAYS_SERVER_IMAGE" \
  --pathways_proxy_server_image="$PATHWAYS_PROXY_IMAGE" \
  --worker_container_name=proc \
  --worker_container_image="$TUNIX_IMAGE" \
  --worker_container_port=29003 \
  --worker_startup_command='sleep infinity' \
  > /tmp/lwraidenlogs_ffi_src/default/jobset.yaml
python3 /usr/local/google/home/lancewang/github/tunix/tunix/experimental/distributed/deployment/yaml_generator.py \
  /usr/local/google/home/lancewang/github/tunix/tunix/experimental/distributed/deployment/yamls/jobset.pathways.yaml \
  --jobset_name=lwraidenffidst \
  --tpu_slice=tpuv5e:4x4 \
  --pathways_server_image="$PATHWAYS_SERVER_IMAGE" \
  --pathways_proxy_server_image="$PATHWAYS_PROXY_IMAGE" \
  --worker_container_name=proc \
  --worker_container_image="$TUNIX_IMAGE" \
  --worker_container_port=29003 \
  --worker_startup_command='sleep infinity' \
  > /tmp/lwraidenlogs_ffi_dst/default/jobset.yaml

kubectl create -f /tmp/lwraidenlogs_ffi_dst/default/jobset.yaml
kubectl create -f /tmp/lwraidenlogs_ffi_src/default/jobset.yaml
kubectl wait --for=condition=Ready pod -l jobset.sigs.k8s.io/jobset-name=lwraidenffidst,jobset.sigs.k8s.io/replicatedjob-name=proc --timeout=20m
kubectl wait --for=condition=Ready pod -l jobset.sigs.k8s.io/jobset-name=lwraidenffisrc,jobset.sigs.k8s.io/replicatedjob-name=proc --timeout=20m

dst_pod=$(kubectl get pods -l jobset.sigs.k8s.io/jobset-name=lwraidenffidst,jobset.sigs.k8s.io/replicatedjob-name=proc --field-selector=status.phase=Running --sort-by=.metadata.creationTimestamp -o name | tail -n 1 | cut -d/ -f2)
src_pod=$(kubectl get pods -l jobset.sigs.k8s.io/jobset-name=lwraidenffisrc,jobset.sigs.k8s.io/replicatedjob-name=proc --field-selector=status.phase=Running --sort-by=.metadata.creationTimestamp -o name | tail -n 1 | cut -d/ -f2)
src_ip=$(kubectl get pod "$src_pod" -o jsonpath='{.status.podIP}')
dst_ip=$(kubectl get pod "$dst_pod" -o jsonpath='{.status.podIP}')

printf 'run_id=%s\nlog_dir=%s\nsrc_pod=%s\nsrc_ip=%s\ndst_pod=%s\ndst_ip=%s\n' "$RUN_ID" "$LOG_DIR" "$src_pod" "$src_ip" "$dst_pod" "$dst_ip" | tee "$LOG_DIR/run_metadata.txt"

kubectl cp "$REPRO_PY" "$src_pod":/tmp/raiden_pathways_ffi_two_controller_repro.py -c proc
kubectl cp "$REPRO_PY" "$dst_pod":/tmp/raiden_pathways_ffi_two_controller_repro.py -c proc
kubectl exec "$src_pod" -c proc -- mkdir -p /tmp/raiden_patch
kubectl exec "$dst_pod" -c proc -- mkdir -p /tmp/raiden_patch
kubectl cp /usr/local/google/home/lancewang/github/tpu-sync/tpu_sync "$src_pod":/tmp/raiden_patch -c proc
kubectl cp /usr/local/google/home/lancewang/github/tpu-sync/tpu_sync "$dst_pod":/tmp/raiden_patch -c proc

kubectl exec "$dst_pod" -c proc -- sh -lc '
  rm -f /tmp/direct_api_*.log
  export PYTHONPATH=/tmp/raiden_patch:$PYTHONPATH
  export RAIDEN_FAIL_ON_IDENTICAL_SLICE_PLANS=0
  export RAIDEN_LOG_DETAILED_SLICE_PLANS=1
  nohup /opt/venv/bin/python /tmp/raiden_pathways_ffi_two_controller_repro.py \
    --role=controller_dst \
    --controller_address=0.0.0.0:10020 \
    --num_dst_hosts=4 \
    > /tmp/direct_api_controller_dst.log 2>&1 &
  nohup /opt/venv/bin/python /tmp/raiden_pathways_ffi_two_controller_repro.py \
    --role=destination \
    --controller_address=127.0.0.1:10020 \
    --num_src_hosts=4 \
    --num_dst_hosts=4 \
    > /tmp/direct_api_destination.log 2>&1 &
'

kubectl exec "$src_pod" -c proc -- sh -lc '
  rm -f /tmp/direct_api_*.log
  export PYTHONPATH=/tmp/raiden_patch:$PYTHONPATH
  export RAIDEN_FAIL_ON_IDENTICAL_SLICE_PLANS=0
  export RAIDEN_LOG_DETAILED_SLICE_PLANS=1
  nohup /opt/venv/bin/python /tmp/raiden_pathways_ffi_two_controller_repro.py \
    --role=source \
    --controller_address=127.0.0.1:10019 \
    --num_src_hosts=4 \
    --num_dst_hosts=4 \
    > /tmp/direct_api_source.log 2>&1 &
'

controller_stream="$LOG_DIR/controller_src_stream.txt"
kubectl exec "$src_pod" -c proc -- sh -lc "
  export PYTHONPATH=/tmp/raiden_patch:\$PYTHONPATH
  export RAIDEN_FAIL_ON_IDENTICAL_SLICE_PLANS=0
  export RAIDEN_LOG_DETAILED_SLICE_PLANS=1
  /opt/venv/bin/python /tmp/raiden_pathways_ffi_two_controller_repro.py \
    --role=controller_src \
    --controller_address=0.0.0.0:10019 \
    --dst_controller_address=${dst_ip}:10020 \
    --num_src_hosts=4 \
    --num_dst_hosts=4 \
    --log_level=INFO \
    2>&1 | tee /tmp/direct_api_controller_src.log
" | tee "$controller_stream" &
controller_pid=$!
controller_status=0
while kill -0 "$controller_pid" 2>/dev/null; do
  if remote_log_has_fatal "$src_pod" /tmp/direct_api_source.log; then
    kubectl exec "$src_pod" -c proc -- sh -lc 'tail -n 80 /tmp/direct_api_source.log' >&2 || true
    kill "$controller_pid" 2>/dev/null || true
    wait "$controller_pid" || true
    copy_logs "$src_pod" "$dst_pod" "$LOG_DIR"
    exit 1
  fi
  if remote_log_has_fatal "$dst_pod" /tmp/direct_api_destination.log; then
    kubectl exec "$dst_pod" -c proc -- sh -lc 'tail -n 80 /tmp/direct_api_destination.log' >&2 || true
    kill "$controller_pid" 2>/dev/null || true
    wait "$controller_pid" || true
    copy_logs "$src_pod" "$dst_pod" "$LOG_DIR"
    exit 1
  fi
  sleep 2
done

wait "$controller_pid" || controller_status=$?

if [[ "$controller_status" -ne 0 ]]; then
  copy_logs "$src_pod" "$dst_pod" "$LOG_DIR"
  exit "$controller_status"
fi

while true; do
  if remote_log_has_fatal "$dst_pod" /tmp/direct_api_destination.log; then
    kubectl exec "$dst_pod" -c proc -- sh -lc 'tail -n 120 /tmp/direct_api_destination.log' >&2 || true
    copy_logs "$src_pod" "$dst_pod" "$LOG_DIR"
    exit 1
  fi
  if remote_log_has_text "$dst_pod" /tmp/direct_api_destination.log 'Destination verification succeeded'; then
    break
  fi
  sleep 2
done

copy_logs "$src_pod" "$dst_pod" "$LOG_DIR"

echo "$LOG_DIR"
