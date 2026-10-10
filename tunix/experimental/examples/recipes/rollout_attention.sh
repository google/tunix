#!/bin/bash
# ==============================================================================
# Rollout (vLLM / MaxText) attention kernel switch, shared by the recipe
# launchers. Sourced before the recipe builds VLLM_ADDITIONAL_CONFIG; exports
# ROLLOUT_MAXTEXT_ATTENTION, which the recipe plugs into
# maxtext_config.attention. Override from the command line:
#   USE_BATCHED_RPA_LONG_CTX_KERNEL=false bash mlperf_397b_256_v7x.sh start
#
# USE_BATCHED_RPA_LONG_CTX_KERNEL=true   (default) attention=vllm_batched_rpa_long_ctx.
#                    MaxText sets USE_BATCHED_RPA_LONG_CTX_KERNEL=1 for
#                    tpu-inference, routing serving attention to the wide-block
#                    batched ragged-paged-attention kernel. The variable is also
#                    forwarded to the rollout workers (ROLLOUT_EXTRA_ENV /
#                    VLLM_RAY_EXTRA_ENV_VARS_TO_COPY) so tpu-inference sees it
#                    from process start.
# USE_BATCHED_RPA_LONG_CTX_KERNEL=false  attention=vllm_rpa (stock RPA kernel).
#
# Needs a MaxText with the vllm_batched_rpa_long_ctx attention option
# (https://github.com/AI-Hypercomputer/maxtext/pull/5621).
# ==============================================================================

export USE_BATCHED_RPA_LONG_CTX_KERNEL="${USE_BATCHED_RPA_LONG_CTX_KERNEL:-true}"
case "${USE_BATCHED_RPA_LONG_CTX_KERNEL}" in
  true)
    export ROLLOUT_MAXTEXT_ATTENTION="${ROLLOUT_MAXTEXT_ATTENTION:-vllm_batched_rpa_long_ctx}"
    ;;
  false)
    export ROLLOUT_MAXTEXT_ATTENTION="${ROLLOUT_MAXTEXT_ATTENTION:-vllm_rpa}"
    ;;
  *)
    echo "ERROR: USE_BATCHED_RPA_LONG_CTX_KERNEL must be true or false, got '${USE_BATCHED_RPA_LONG_CTX_KERNEL}'" >&2
    exit 1
    ;;
esac
