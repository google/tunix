#!/bin/bash

set -euo pipefail

export TUNIX_IMAGE=${TUNIX_IMAGE:-us-west1-docker.pkg.dev/supercomputer-testing/lancewang/tunix_base_image:tunix_0906}
export REPRO_PY=${REPRO_PY:-/usr/local/google/home/lancewang/github/tunix/raiden_weight_synchronizer_construct_repro.py}

export DST_PROJECT=${DST_PROJECT:-cloud-tpu-inference-test}
export DST_REGION=${DST_REGION:-us-west1}
export DST_CLUSTER=${DST_CLUSTER:-lancewang-mcjax-v5e-2slice}
export DST_TPU_SLICE=${DST_TPU_SLICE:-tpuv5e:4x4}
export DST_JOBSET_NAME=${DST_JOBSET_NAME:-lwraidenmcjaxws}

cd /usr/local/google/home/lancewang/github/tunix && \
RUN_ID="mcjax_ws_construct_repro_$(date +%Y%m%d_%H%M%S)" && \
LOG_DIR="$PWD/raiden_bug_logs/$RUN_ID" && \
mkdir -p "$LOG_DIR" /tmp/lwraidenlogs_mcjax_ws/default && \
gcloud container clusters get-credentials "$DST_CLUSTER" --region "$DST_REGION" --project "$DST_PROJECT" >/dev/null && \
dst_context=$(kubectl config current-context) && \
kubectl --context="$dst_context" delete jobset "$DST_JOBSET_NAME" --ignore-not-found=true --wait=true && \
python3 /usr/local/google/home/lancewang/github/tunix/tunix/experimental/distributed/deployment/yaml_generator.py \
  /usr/local/google/home/lancewang/github/tunix/tunix/experimental/distributed/deployment/yamls/jobset.mcjax.slice_pinned.yaml \
  --jobset_name="$DST_JOBSET_NAME" \
  --tpu_slice="$DST_TPU_SLICE" \
  --worker_container_name=proc \
  --worker_container_image="$TUNIX_IMAGE" \
  --worker_container_port=29003 \
  --worker_startup_command='sleep infinity' \
  > /tmp/lwraidenlogs_mcjax_ws/default/jobset.yaml && \
kubectl --context="$dst_context" create -f /tmp/lwraidenlogs_mcjax_ws/default/jobset.yaml && \
kubectl --context="$dst_context" wait --for=condition=Ready pod -l jobset.sigs.k8s.io/jobset-name="$DST_JOBSET_NAME",jobset.sigs.k8s.io/replicatedjob-name=proc --timeout=20m && \
dst_pods=$(kubectl --context="$dst_context" get pods -l jobset.sigs.k8s.io/jobset-name="$DST_JOBSET_NAME",jobset.sigs.k8s.io/replicatedjob-name=proc --field-selector=status.phase=Running --sort-by=.metadata.creationTimestamp -o name | cut -d/ -f2 | tr '\n' ' ' | sed 's/[[:space:]]*$//') && \
printf 'run_id=%s\nlog_dir=%s\ndst_context=%s\ndst_cluster=%s\ndst_pods=%s\ndst_tpu_slice=%s\n' "$RUN_ID" "$LOG_DIR" "$dst_context" "$DST_CLUSTER" "$dst_pods" "$DST_TPU_SLICE" | tee "$LOG_DIR/run_metadata.txt" && \
for dst_pod in $dst_pods; do \
  kubectl --context="$dst_context" cp "$REPRO_PY" "$dst_pod":/tmp/raiden_weight_synchronizer_construct_repro.py -c proc; \
done && \
status=0 && \
pids='' && \
for dst_pod in $dst_pods; do \
  kubectl --context="$dst_context" exec "$dst_pod" -c proc -- sh -lc '/opt/venv/bin/python /tmp/raiden_weight_synchronizer_construct_repro.py > /tmp/ws_construct_repro.log 2>&1' & \
  pids="$pids $!"; \
done && \
for pid in $pids; do \
  wait "$pid" || status=$?; \
done && \
for dst_pod in $dst_pods; do \
  kubectl --context="$dst_context" cp "$dst_pod":/tmp/ws_construct_repro.log "$LOG_DIR/ws_construct_repro.${dst_pod}.log" -c proc; \
done && \
echo "$LOG_DIR" && \
exit "$status"
