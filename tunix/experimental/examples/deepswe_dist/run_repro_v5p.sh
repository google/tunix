#!/bin/bash
# Minimal standalone v5p launcher for reproducing Mamba prefix caching (align mode) NaN / '!' collapse.
# Runs a single 4-chip TPU v5p JobSet (tpuv5:2x2x1) on bodaborg-v5p-nap without trainer or sandboxes.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TUNIX_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
MAXTEXT_ROOT="$(cd "${TUNIX_ROOT}/../maxtext" && pwd)"
TPU_INFERENCE_ROOT="$(cd "${TUNIX_ROOT}/../tpu-inference" && pwd)"

CLUSTER_CONTEXT="${CLUSTER_CONTEXT:-gke_cloud-tpu-shared-capacity_europe-west4_bodaborg-v5p-nap}"
NAMESPACE="${NAMESPACE:-trellis}"
QUEUE_NAME="${QUEUE_NAME:-multislice-queue}"
JOB_NAME="${JOB_NAME:-atwigg-mamba-repro}"
DOCKER_IMAGE="${DOCKER_IMAGE:-gcr.io/cloud-tpu-multipod-dev/atwigg/trellis:latest}"
RESERVATION="${RESERVATION:-cloudtpu-20260902214500-1810493672}"

ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-true}"
MAMBA_CACHE_MODE="${MAMBA_CACHE_MODE:-align}"
NUM_PROMPTS="${NUM_PROMPTS:-1}"
NUM_GENERATIONS="${NUM_GENERATIONS:-16}"
MAX_TURNS="${MAX_TURNS:-5}"
MAX_TOKENS_PER_TURN="${MAX_TOKENS_PER_TURN:-1024}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"
SYNC_LOCAL_PATCHES="${SYNC_LOCAL_PATCHES:-true}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

KEEP_POD_ALIVE="${KEEP_POD_ALIVE:-true}"
RECREATE_POD="${RECREATE_POD:-false}"

KUBECTL="kubectl --context=${CLUSTER_CONTEXT} -n ${NAMESPACE}"

if [[ "${ENABLE_PREFIX_CACHING}" == "true" ]]; then
  PREFIX_FLAG="--enable_prefix_caching"
else
  PREFIX_FLAG="--no-enable_prefix_caching"
fi

EXISTING_POD="$(${KUBECTL} get pods -l jobset.sigs.k8s.io/jobset-name=${JOB_NAME} -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
EXISTING_PHASE=""
if [[ -n "${EXISTING_POD}" ]]; then
  EXISTING_PHASE="$(${KUBECTL} get pod "${EXISTING_POD}" -o jsonpath='{.status.phase}' 2>/dev/null || true)"
fi

if [[ "${RECREATE_POD}" == "true" || "${EXISTING_PHASE}" != "Running" ]]; then
  echo "Deleting any existing JobSet ${JOB_NAME}..."
  ${KUBECTL} delete jobset "${JOB_NAME}" --ignore-not-found=true --wait=true

  cat <<EOF | ${KUBECTL} create -f -
apiVersion: jobset.x-k8s.io/v1alpha2
kind: JobSet
metadata:
  name: ${JOB_NAME}
  namespace: ${NAMESPACE}
  labels:
    kueue.x-k8s.io/queue-name: ${QUEUE_NAME}
spec:
  failurePolicy:
    maxRestarts: 0
  network:
    enableDNSHostnames: true
    publishNotReadyAddresses: true
  replicatedJobs:
  - name: proc
    replicas: 1
    template:
      spec:
        backoffLimit: 0
        completions: 1
        parallelism: 1
        template:
          metadata:
            annotations:
              cluster-autoscaler.kubernetes.io/safe-to-evict: "false"
          spec:
            dnsPolicy: ClusterFirstWithHostNet
            hostNetwork: true
            priorityClassName: medium
            restartPolicy: Never
            serviceAccountName: xpk-sa
            terminationGracePeriodSeconds: 10
            nodeSelector:
              cloud.google.com/gke-tpu-accelerator: tpu-v5p-slice
              cloud.google.com/gke-tpu-topology: 2x2x1
              cloud.google.com/reservation-name: ${RESERVATION}
            tolerations:
            - effect: NoSchedule
              key: cloud.google.com/reservation-name
              operator: Equal
              value: ${RESERVATION}
            - effect: NoSchedule
              key: google.com/tpu
              operator: Exists
            volumes:
            - hostPath:
                path: /tmp
                type: DirectoryOrCreate
              name: shared-tmp
            containers:
            - name: main
              image: ${DOCKER_IMAGE}
              imagePullPolicy: IfNotPresent
              securityContext:
                privileged: true
              resources:
                limits:
                  google.com/tpu: "4"
                requests:
                  google.com/tpu: "4"
              volumeMounts:
              - mountPath: /tmp
                name: shared-tmp
              env:
              - name: JAX_PLATFORMS
                value: "tpu,cpu"
              - name: VLLM_ENABLE_V1_MULTIPROCESSING
                value: "0"
              command:
              - bash
              - -c
              - |
                echo "Pod ready on \$(hostname) at \$(date)"
                sleep 7200
EOF

  echo "Waiting for pod of ${JOB_NAME} to start..."
  for _ in $(seq 1 60); do
    EXISTING_POD="$(${KUBECTL} get pods -l jobset.sigs.k8s.io/jobset-name=${JOB_NAME} -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
    if [[ -n "${EXISTING_POD}" ]]; then
      EXISTING_PHASE="$(${KUBECTL} get pod "${EXISTING_POD}" -o jsonpath='{.status.phase}' 2>/dev/null || true)"
      echo "  Pod ${EXISTING_POD} phase: ${EXISTING_PHASE}"
      if [[ "${EXISTING_PHASE}" == "Running" ]]; then
        break
      fi
    fi
    sleep 3
  done
else
  echo "Reusing existing Running pod ${EXISTING_POD}..."
fi

echo "Syncing repro_mamba_prefix_cache.py to ${EXISTING_POD}..."
${KUBECTL} exec -i "${EXISTING_POD}" -- bash -c "cat > /tmp/repro_mamba_prefix_cache.py" < "${SCRIPT_DIR}/repro_mamba_prefix_cache.py"

if [[ "${SYNC_LOCAL_PATCHES}" == "true" ]]; then
  echo "Syncing local maxtext and tpu-inference files to ${EXISTING_POD}..."
  tar -czf - \
    -C "${MAXTEXT_ROOT}/src/maxtext" \
      models/qwen3.py \
      integration/vllm/maxtext_vllm_adapter/adapter.py \
    -C "${TPU_INFERENCE_ROOT}" \
      tpu_inference/core/hybrid_coordinator.py \
      tpu_inference/core/sched/dp_scheduler.py \
      tpu_inference/runner/kv_cache_manager.py \
      tpu_inference/runner/persistent_batch_manager.py \
      tpu_inference/layers/vllm/custom_ops/gdn_attention_op.py \
      tpu_inference/layers/common/gdn_attention.py \
      tpu_inference/layers/common/fused_moe_gmm.py \
      tpu_inference/kernels/gdn/v3/wrapper.py \
      tpu_inference/kernels/gdn/v3/metadata.py \
      tpu_inference/kernels/gdn/v3/memory_ref.py \
    | ${KUBECTL} exec -i "${EXISTING_POD}" -- bash -c '
      mkdir -p /tmp/patch_stage && tar -xzf - -C /tmp/patch_stage
      python3 -c "import importlib.util, pathlib, shutil; mt_root = pathlib.Path(importlib.util.find_spec(\"maxtext\").origin).resolve().parent; ti_root = pathlib.Path(importlib.util.find_spec(\"tpu_inference\").origin).resolve().parent; stage = pathlib.Path(\"/tmp/patch_stage\"); [shutil.copy2(stage / rel, mt_root / rel) for rel in (\"models/qwen3.py\", \"integration/vllm/maxtext_vllm_adapter/adapter.py\")]; [shutil.copy2(stage / \"tpu_inference\" / rel, ti_root / rel) for rel in (\"core/hybrid_coordinator.py\", \"core/sched/dp_scheduler.py\", \"runner/kv_cache_manager.py\", \"runner/persistent_batch_manager.py\", \"layers/vllm/custom_ops/gdn_attention_op.py\", \"layers/common/gdn_attention.py\", \"layers/common/fused_moe_gmm.py\", \"kernels/gdn/v3/wrapper.py\", \"kernels/gdn/v3/metadata.py\", \"kernels/gdn/v3/memory_ref.py\")]; print(\"Synced local maxtext + tpu_inference files.\")"
    '
fi

echo "Running reproduction on ${EXISTING_POD}..."
${KUBECTL} exec -i "${EXISTING_POD}" -- bash <<EOF
set -euo pipefail
python3 -c '
import os, signal
my_pid = os.getpid()
for entry in os.listdir("/proc"):
    if entry.isdigit():
        pid = int(entry)
        if pid in (1, my_pid):
            continue
        try:
            cmdline = open(f"/proc/{pid}/cmdline", "rb").read().decode("utf-8", "replace").replace("\x00", " ").strip()
            if "repro_mamba_prefix_cache" in cmdline or "EngineCore" in cmdline or "multiprocessing" in cmdline:
                os.kill(pid, signal.SIGKILL)
        except Exception:
            pass
'
export PYTHONUNBUFFERED=1
export JAX_COMPILATION_CACHE_DIR="/tmp/jax_cache"
export TUNIX_IS_INTERNAL_ENV=false
export SCAFFOLD="openhands"
export USE_RAIDEN_FFI=false
export RAIDEN_USE_FFI=0
export RAIDEN_DEVICES_PER_HOST=4
export ROLLOUT_PREFUSE_MOE_WEIGHTS=true
export ROLLOUT_MESH_TP=1
export ROLLOUT_TENSOR_PARALLEL_SIZE=1
export PREFUSE_MOE_WEIGHTS=true
export ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING}"
export VLLM_MAMBA_CACHE_MODE="${MAMBA_CACHE_MODE}"
export ROLLOUT_FREE_KV_CACHE=false
export VLLM_MAX_NUM_SEQS=16
export VLLM_GPU_MEMORY_UTILIZATION=0.9
export NUM_PRECOMPILE_WORKERS=8
export NEW_MODEL_DESIGN=1
export ATTN_BUCKETIZED_NUM_REQS=true
export ATTN_CUSTOM_NUM_REQS_BUCKETS=4
export ONEHOT_MOE_PERMUTE_THRESHOLD=32768
export VLLM_MOE_CHUNK_SIZE=256
export SLICE_ROPE_CACHE=1
export DP_SCHED_BATCH_PREFILL=false
export LIBTPU_INIT_ARGS=" --xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false --xla_tpu_check_legacy_constraints_in_reduce_scatter_legalizer=false"
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_LOGGING_LEVEL=INFO
export SKIP_JAX_PRECOMPILE=1

python3 /tmp/repro_mamba_prefix_cache.py \
  ${PREFIX_FLAG} \
  --mamba_cache_mode="${MAMBA_CACHE_MODE}" \
  --num_prompts="${NUM_PROMPTS}" \
  --num_generations="${NUM_GENERATIONS}" \
  --max_turns="${MAX_TURNS}" \
  --max_tokens_per_turn="${MAX_TOKENS_PER_TURN}" \
  --max_response_length="${MAX_RESPONSE_LENGTH}" \
  ${EXTRA_ARGS}
EOF

if [[ "${KEEP_POD_ALIVE}" != "true" ]]; then
  ${KUBECTL} delete jobset "${JOB_NAME}" --ignore-not-found=true
fi

