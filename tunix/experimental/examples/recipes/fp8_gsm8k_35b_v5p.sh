#!/bin/bash
# FP8 MoE comparison on v5p (GSM8K, Qwen3.5-35B-A3B): bf16 = bf16 rollout and trainer,
# r8 = FP8 rollout experts, r8t8 = FP8 rollout experts + FP8 fake-quant trainer experts.
# Usage: fp8_gsm8k_35b_v5p.sh <bf16|r8|r8t8> [launcher args, e.g. --dry-run | stop]
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
variant=$1; shift
case "$variant" in
  bf16) export ROLLOUT_FP8=false TRAINER_FP8=false ;;
  r8)   export ROLLOUT_FP8=true  TRAINER_FP8=false ;;
  r8t8) export ROLLOUT_FP8=true  TRAINER_FP8=true ;;
  *) echo "unknown variant $variant"; exit 1 ;;
esac
export JOB_PREFIX="${RUN_PREFIX:-igfp8v2}-${variant}"
# The gsm8k launcher names its pods $USER-{orch,train,roll}; keep the three variants apart.
export USER="${JOB_PREFIX}"
export TUNIX_IMAGE="${TUNIX_IMAGE:-gcr.io/cloud-tpu-multipod-dev/igorts/trellis:fp8-v2}"
# bodaborg-v5p-nap (europe-west4), trellis queue: 64-chip trainer + 8 x 4-chip rollouts = 96 chips per variant.
export TPU_RESERVATION=cloudtpu-20260902214500-1810493672
export WANDB_RUN_NAME="${JOB_PREFIX}-gsm8k-v5p"
export ROLLOUT_REPLICAS=8 MAX_STEPS=${MAX_STEPS:-30}
# Match the Pathways/raiden setup the DeepSWE runs synced with (mlperf_pathways_config.sh): the
# gsm8k default server image failed to compile jit__shard_init in the trainer raiden FFI bind.
export PATHWAYS_SERVER_IMAGE=us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/datenglin/unsanitized_server:raiden_988370191
export PATHWAYS_PROXY_IMAGE=us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/datenglin/unsanitized_proxy_server:raiden_988370191
export TRAINER_EXTRA_ENV="RAIDEN_FFI_USE_DIRECT_DEVICE_BUFFER=0 TPU_RAIDEN_DATA_NICS=eth0"
exec "${DIR}/gsm_8k_35b_256.sh" "$@"
