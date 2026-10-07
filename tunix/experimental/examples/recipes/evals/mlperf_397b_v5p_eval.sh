#!/bin/bash
set -e

# ==============================================================================
# MLPerf DeepSWE evaluation recipe: Qwen3.5-397B-A17B on TPU v5p
# ==============================================================================
# - Inherits 397B eval settings from mlperf_397b_v7x_eval.sh -> mlperf_397b_256_v7x.sh
# - Cluster: bodaborg-v5p-nap in europe-west4
# - Rollout on 16 chips per replica (tpuv5p:2x2x4, DP=1, EP=16, TP=1; no Trainer)
# - Sandboxes on sandbox-cpu-pool
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export TUNIX_IMAGE="${TUNIX_IMAGE:-gcr.io/cloud-tpu-multipod-dev/sanbao/tunix_stack:eval}"

# Cluster & GCS paths
export REGION="${REGION:-europe-west4}"
export CLUSTER="${CLUSTER:-bodaborg-v5p-nap}"
export K8S_NAMESPACE="${K8S_NAMESPACE:-trellis}"
export BUCKET="${BUCKET:-gs://atwigg-trellis-europe-west4-dev}"
export TPU_RESERVATION="${TPU_RESERVATION-}"
export USE_DYNAMIC_SLICING="${USE_DYNAMIC_SLICING:-false}"
export MAXTEXT_CKPT="${MAXTEXT_CKPT:-gs://sanbao-europe/qwen35_397b/scanned_reshard_fsdp32_tp2/0/items}"

# v5p Rollout Topology (16 chips = 16 devices = 4 hosts per replica, DP=1, EP=16, TP=1)
export ROLLOUT_JOBSET_YAML="${ROLLOUT_JOBSET_YAML:-jobset.pathways.yaml}"
export ROLLOUT_TPU_SLICE="${ROLLOUT_TPU_SLICE:-tpuv5p:2x2x4}"
export VLLM_DATA_PARALLEL_SIZE="${VLLM_DATA_PARALLEL_SIZE:-1}"
_rollout_dims="${ROLLOUT_TPU_SLICE#*:}"
export ROLLOUT_MESH_EXPERT="${ROLLOUT_MESH_EXPERT:-$(( ${_rollout_dims//x/*} / ${VLLM_DATA_PARALLEL_SIZE:-1} ))}"

# v5p Sandbox
export SANDBOX_NODE_SELECTOR_VAL="${SANDBOX_NODE_SELECTOR_VAL:-sandbox-cpu-pool}"
export SANDBOX_TOLERATIONS="${SANDBOX_TOLERATIONS-}"
export IMAGE_REWRITE_PREFIX="${IMAGE_REWRITE_PREFIX:-europe-west4-docker.pkg.dev/cloud-tpu-multipod-dev/tunix/}"

source "${SCRIPT_DIR}/mlperf_397b_v7x_eval.sh" "$@"
