#!/bin/bash

set -euo pipefail

export PATHWAYS_SERVER_IMAGE=${PATHWAYS_SERVER_IMAGE:-us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/datenglin/unsanitized_server:raiden_20260904}
export PATHWAYS_PROXY_IMAGE=${PATHWAYS_PROXY_IMAGE:-us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/datenglin/unsanitized_proxy_server:raiden_20260904}
export TUNIX_IMAGE=${TUNIX_IMAGE:-us-west1-docker.pkg.dev/supercomputer-testing/lancewang/tunix_base_image:tunix_0906}
export REPRO_PY=${REPRO_PY:-/usr/local/google/home/lancewang/github/tunix/raiden_pathways_direct_api_repro.py}

export SRC_PROJECT=${SRC_PROJECT:-cloud-tpu-inference-test}
export SRC_REGION=${SRC_REGION:-us-west1}
export SRC_ZONE=${SRC_ZONE:-us-west1-c}
export SRC_CLUSTER=${SRC_CLUSTER:-lancewang-pw-v5e-4slice}
export SRC_TPU_SLICE=${SRC_TPU_SLICE:-tpuv5e:4x4}

export DST_PROJECT=${DST_PROJECT:-cloud-tpu-inference-test}
export DST_REGION=${DST_REGION:-us-west1}
export DST_CLUSTER=${DST_CLUSTER:-lancewang-mcjax-v5e-2slice}
export DST_TPU_SLICE=${DST_TPU_SLICE:-tpuv5e:4x4}

export SRC_JOBSET_NAME=${SRC_JOBSET_NAME:-lwraidenpwsrc}
export DST_JOBSET_NAME=${DST_JOBSET_NAME:-lwraidenmcjaxdst}

cd /usr/local/google/home/lancewang/github/tunix && \
RUN_ID="raiden_pw_mcjax_repro_$(date +%Y%m%d_%H%M%S)" && \
LOG_DIR="$PWD/raiden_bug_logs/$RUN_ID" && \
mkdir -p "$LOG_DIR" && \
gcloud container clusters get-credentials "$SRC_CLUSTER" --zone "$SRC_ZONE" --project "$SRC_PROJECT" >/dev/null && \
src_context=$(kubectl config current-context) && \
gcloud container clusters get-credentials "$DST_CLUSTER" --region "$DST_REGION" --project "$DST_PROJECT" >/dev/null && \
dst_context=$(kubectl config current-context) && \
kubectl --context="$src_context" delete jobset "$SRC_JOBSET_NAME" --ignore-not-found=true --wait=true && \
kubectl --context="$dst_context" delete jobset "$DST_JOBSET_NAME" --ignore-not-found=true --wait=true && \
mkdir -p /tmp/lwraidenlogs_pw_src/default /tmp/lwraidenlogs_mcjax_dst/default && \
python3 /usr/local/google/home/lancewang/github/tunix/tunix/experimental/distributed/deployment/yaml_generator.py \
  /usr/local/google/home/lancewang/github/tunix/tunix/experimental/distributed/deployment/yamls/jobset.pathways.yaml \
  --jobset_name="$SRC_JOBSET_NAME" \
  --tpu_slice="$SRC_TPU_SLICE" \
  --pathways_server_image="$PATHWAYS_SERVER_IMAGE" \
  --pathways_proxy_server_image="$PATHWAYS_PROXY_IMAGE" \
  --worker_container_name=proc \
  --worker_container_image="$TUNIX_IMAGE" \
  --worker_container_port=29003 \
  --worker_startup_command='sleep infinity' \
  > /tmp/lwraidenlogs_pw_src/default/jobset.yaml && \
python3 /usr/local/google/home/lancewang/github/tunix/tunix/experimental/distributed/deployment/yaml_generator.py \
  /usr/local/google/home/lancewang/github/tunix/tunix/experimental/distributed/deployment/yamls/jobset.mcjax.slice_pinned.yaml \
  --jobset_name="$DST_JOBSET_NAME" \
  --tpu_slice="$DST_TPU_SLICE" \
  --worker_container_name=proc \
  --worker_container_image="$TUNIX_IMAGE" \
  --worker_container_port=29003 \
  --worker_startup_command='sleep infinity' \
  > /tmp/lwraidenlogs_mcjax_dst/default/jobset.yaml && \
kubectl --context="$dst_context" create -f /tmp/lwraidenlogs_mcjax_dst/default/jobset.yaml && \
kubectl --context="$src_context" create -f /tmp/lwraidenlogs_pw_src/default/jobset.yaml && \
kubectl --context="$dst_context" wait --for=condition=Ready pod -l jobset.sigs.k8s.io/jobset-name="$DST_JOBSET_NAME",jobset.sigs.k8s.io/replicatedjob-name=proc --timeout=20m && \
kubectl --context="$src_context" wait --for=condition=Ready pod -l jobset.sigs.k8s.io/jobset-name="$SRC_JOBSET_NAME",jobset.sigs.k8s.io/replicatedjob-name=proc --timeout=20m && \
dst_pods=$(kubectl --context="$dst_context" get pods -l jobset.sigs.k8s.io/jobset-name="$DST_JOBSET_NAME",jobset.sigs.k8s.io/replicatedjob-name=proc --field-selector=status.phase=Running --sort-by=.metadata.creationTimestamp -o name | cut -d/ -f2 | tr '\n' ' ' | sed 's/[[:space:]]*$//') && \
dst_pod=$(printf '%s\n' "$dst_pods" | awk '{print $NF}') && \
src_pod=$(kubectl --context="$src_context" get pods -l jobset.sigs.k8s.io/jobset-name="$SRC_JOBSET_NAME",jobset.sigs.k8s.io/replicatedjob-name=proc --field-selector=status.phase=Running --sort-by=.metadata.creationTimestamp -o name | tail -n 1 | cut -d/ -f2) && \
src_ip=$(kubectl --context="$src_context" get pod "$src_pod" -o jsonpath='{.status.podIP}') && \
dst_ip=$(kubectl --context="$dst_context" get pod "$dst_pod" -o jsonpath='{.status.podIP}') && \
printf 'run_id=%s\nlog_dir=%s\nsrc_context=%s\nsrc_cluster=%s\nsrc_pod=%s\nsrc_ip=%s\ndst_context=%s\ndst_cluster=%s\ndst_pods=%s\ndst_pod=%s\ndst_ip=%s\nsrc_tpu_slice=%s\ndst_tpu_slice=%s\n' "$RUN_ID" "$LOG_DIR" "$src_context" "$SRC_CLUSTER" "$src_pod" "$src_ip" "$dst_context" "$DST_CLUSTER" "$dst_pods" "$dst_pod" "$dst_ip" "$SRC_TPU_SLICE" "$DST_TPU_SLICE" | tee "$LOG_DIR/run_metadata.txt" && \
kubectl --context="$src_context" cp "$REPRO_PY" "$src_pod":/tmp/raiden_pathways_direct_api_repro.py -c proc && \
kubectl --context="$src_context" exec "$src_pod" -c proc -- mkdir -p /tmp/raiden_patch && \
kubectl --context="$src_context" cp /usr/local/google/home/lancewang/github/tpu-sync/tpu_sync "$src_pod":/tmp/raiden_patch -c proc && \
for dst_pod in $dst_pods; do \
  kubectl --context="$dst_context" cp "$REPRO_PY" "$dst_pod":/tmp/raiden_pathways_direct_api_repro.py -c proc; \
  kubectl --context="$dst_context" exec "$dst_pod" -c proc -- mkdir -p /tmp/raiden_patch; \
  kubectl --context="$dst_context" cp /usr/local/google/home/lancewang/github/tpu-sync/tpu_sync "$dst_pod":/tmp/raiden_patch -c proc; \
  kubectl --context="$dst_context" exec "$dst_pod" -c proc -- sh -lc "
    rm -f /tmp/direct_api_*.log
    export PYTHONPATH=/tmp/raiden_patch:\$PYTHONPATH
    export RAIDEN_FAIL_ON_IDENTICAL_SLICE_PLANS=0
    export RAIDEN_LOG_DETAILED_SLICE_PLANS=1
    nohup /opt/venv/bin/python /tmp/raiden_pathways_direct_api_repro.py \
      --role=destination \
      --controller_address=${src_ip}:10019 \
      --num_src_hosts=4 \
      --num_dst_hosts=4 \
      > /tmp/direct_api_destination.log 2>&1 &
  "; \
done && \
kubectl --context="$src_context" exec "$src_pod" -c proc -- sh -lc '
  rm -f /tmp/direct_api_*.log
  export PYTHONPATH=/tmp/raiden_patch:$PYTHONPATH
  export RAIDEN_FAIL_ON_IDENTICAL_SLICE_PLANS=0
  export RAIDEN_LOG_DETAILED_SLICE_PLANS=1
  nohup /opt/venv/bin/python /tmp/raiden_pathways_direct_api_repro.py \
    --role=source \
    --controller_address=127.0.0.1:10019 \
    --num_src_hosts=4 \
    --num_dst_hosts=4 \
    > /tmp/direct_api_source.log 2>&1 &
' && \
kubectl --context="$src_context" exec "$src_pod" -c proc -- sh -lc "
  export PYTHONPATH=/tmp/raiden_patch:\$PYTHONPATH
  export RAIDEN_FAIL_ON_IDENTICAL_SLICE_PLANS=0
  export RAIDEN_LOG_DETAILED_SLICE_PLANS=1
  /opt/venv/bin/python /tmp/raiden_pathways_direct_api_repro.py \
    --role=controller_src \
    --controller_address=0.0.0.0:10019 \
    --num_src_hosts=4 \
    --num_dst_hosts=4 \
    --log_level=INFO \
    2>&1 | tee /tmp/direct_api_controller_src.log
" | tee "$LOG_DIR/controller_src_stream.txt" && \
kubectl --context="$src_context" cp "$src_pod":/tmp/direct_api_controller_src.log "$LOG_DIR/direct_api_controller_src.log" -c proc && \
kubectl --context="$src_context" cp "$src_pod":/tmp/direct_api_source.log "$LOG_DIR/direct_api_source.log" -c proc && \
for dst_pod in $dst_pods; do \
  kubectl --context="$dst_context" cp "$dst_pod":/tmp/direct_api_destination.log "$LOG_DIR/direct_api_destination.${dst_pod}.log" -c proc; \
done && \
echo "$LOG_DIR"
