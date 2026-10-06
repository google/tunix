#!/bin/bash
set -e

# =====================================================
# Dropless MoE variant of mlperf_gbs1024_submission.sh
# =====================================================
# ragged_buffer_factor=-1 sizes the ring-of-experts buffers for the worst case, so
# no tokens are dropped. On its own it does not fit the trainer's HBM; this is the
# combination that ran 10 steps on bodaborg-tpu7x-gsc-elm (2026-10-06):
#   - 4 MoE token chunks and the attention output (context) offloaded to host;
#   - --xla_tpu_max_hbm_size_mib=84992 for the trainer. About 14.5 GiB of HBM is
#     resident when the fwd_bwd program loads, and without the cap XLA schedules
#     against all 94.74 GiB (program 82.27 GiB vs 80.26 GiB free). With it the
#     program compiles to 72.29 GiB.
# The image must retry vLLM engine start (vllm_sampler_v2 _START_MAX_ATTEMPTS):
# rollout engine cores occasionally segfault at init, and FAIL_FAST stops the run
# on the first unrecovered one.
# MAXTEXT_EXTRA_FLAGS replaces the recipe's MaxText flag list, so it must be unset;
# use MAXTEXT_USER_EXTRA_FLAGS to append overrides.
#
# Usage: mlperf_gbs1024_dropless.sh [start|stop]   (default: start)

COMMAND="${1:-start}"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${COMMAND}" == "stop" ]]; then
  # mlperf_gbs1024_submission.sh always starts; stop through the 1024 recipe with
  # the submission's replica count (its own default is 64).
  export ROLLOUT_REPLICAS="${ROLLOUT_REPLICAS:-128}"
  exec bash "${DIR}/mlperf_397b_1024_v7x.sh" stop
elif [[ "${COMMAND}" != "start" ]]; then
  echo "ERROR: unknown command '${COMMAND}'; use start or stop." >&2
  exit 1
fi

if [[ -n "${MAXTEXT_EXTRA_FLAGS:-}" ]]; then
  echo "ERROR: unset MAXTEXT_EXTRA_FLAGS (it replaces the recipe's MaxText flags);" \
    "use MAXTEXT_USER_EXTRA_FLAGS to append overrides." >&2
  exit 1
fi

export TUNIX_IMAGE="${TUNIX_IMAGE:-gcr.io/cloud-tpu-multipod-dev/atwigg/trellis-experimental:1005}"
export RAGGED_BUFFER_FACTOR="${RAGGED_BUFFER_FACTOR:--1.0}"
export NUM_MOE_TOKEN_CHUNKS="${NUM_MOE_TOKEN_CHUNKS:-4}"
export CONTEXT_REMAT_POLICY="${CONTEXT_REMAT_POLICY:-offload}"
export TRAINER_EXTRA_LIBTPU_INIT_ARGS="${TRAINER_EXTRA_LIBTPU_INIT_ARGS:---xla_tpu_max_hbm_size_mib=84992}"

bash "${DIR}/mlperf_gbs1024_submission.sh"
