#!/bin/bash
set -e

# ==============================================================================
# MLPerf DeepSWE Shared Base Configuration & Dispatcher
# ==============================================================================
# This script defines shared defaults for MLPerf distributed recipes (35B, 397B)
# on TPU clusters (v5p, v7x) and dispatches execution to deepswe_dist/k8s_launcher.sh.
#
# Recipe scripts specify model, cluster, topology, and hardware-specific flags,
# then source this base script at the end:
#   source "${DIR}/mlperf_base.sh" "$@"
# ==============================================================================

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ==============================================================================
# Container Image, User Identity & WandB
# ==============================================================================
export TUNIX_IMAGE="${TUNIX_IMAGE:-gcr.io/cloud-tpu-multipod-dev/atwigg/trellis:latest}"
export JOB_PREFIX="${JOB_PREFIX:-${USER}}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export ORCHESTRATOR_PORT="${ORCHESTRATOR_PORT:-20000}"
export ROLLOUT_PORT="${ROLLOUT_PORT:-20001}"
export TRAINER_PORT="${TRAINER_PORT:-20002}"
export PROFILER_STEPS=${PROFILER_STEPS:-0}
export SKIP_FIRST_N_PROFILER_STEPS=${SKIP_FIRST_N_PROFILER_STEPS:--1}

# TPU advanced profiling. Appended rather than assigned so the per-recipe
# MAXTEXT_EXTRA_FLAGS is preserved and non-profiling runs are unchanged.
#
# The 397B v7x trace captured 814 ms, 4.5% of one 17.95 s fwd_bwd micro step,
# and recorded zero Steps events. The limit is a per-chip SparseCore
# trace-entry budget, not the 2 GB XSpace proto cap: all four chips recorded
# equal entry counts to within 0.06% but stopped at different wall-clock times,
# each with a nonzero dropped_traces counter. Capturing a full micro step
# therefore requires the per-chip entry rate to fall by 22x. Measured factors:
#
#   tpu_num_sparse_core_tiles_to_trace=1   15.7x   12.8 s   insufficient
#   tpu_num_sparse_cores_to_trace=1         2.0x    1.6 s   insufficient
#   both                                   31.5x   25.6 s   sufficient
#
# Cost: tiles=1 retains one TEC line per plane, representative to 11.4% across
# all 256; sparse_cores=1 drops the "SparseCore 1" plane on every device.
# tpu_num_chips_to_profile_per_task=1 does not extend the window, since the
# budget is per chip; it bounds output size at ~0.4 GB rather than ~1.2 GB.
# Re-measure all three on a new TPU generation.
#
# TODO(profiling): if the budget is a host-wide pool partitioned across the
# profiled chips, chips=1 would also extend the window and one of the two
# SparseCore levers could be relaxed.
export TPU_PROFILE_CHIPS_PER_TASK="${TPU_PROFILE_CHIPS_PER_TASK:-1}"
export TPU_PROFILE_SPARSE_CORES="${TPU_PROFILE_SPARSE_CORES:-1}"
export TPU_PROFILE_SPARSE_CORE_TILES="${TPU_PROFILE_SPARSE_CORE_TILES:-1}"
if [[ "${PROFILER_STEPS}" =~ ^[0-9]+$ && "${PROFILER_STEPS}" -gt 0 ]]; then
  tpu_profiling_flags=(
    "enable_tpu_profiling_options=true"
    "tpu_num_chips_to_profile_per_task=${TPU_PROFILE_CHIPS_PER_TASK}"
    "tpu_num_sparse_cores_to_trace=${TPU_PROFILE_SPARSE_CORES}"
    "tpu_num_sparse_core_tiles_to_trace=${TPU_PROFILE_SPARSE_CORE_TILES}"
    "upload_all_profiler_results=false"
  )
  export MAXTEXT_EXTRA_FLAGS="${MAXTEXT_EXTRA_FLAGS:+$MAXTEXT_EXTRA_FLAGS }${tpu_profiling_flags[*]}"
  unset tpu_profiling_flags
fi

# ==============================================================================
# Cluster Context & Kueue / Priority
# ==============================================================================
export PROJECT="${PROJECT:-cloud-tpu-shared-capacity}"
if [[ -n "${REGION:-}" && -n "${CLUSTER:-}" ]]; then
  kubectl config use-context "gke_${PROJECT}_${REGION}_${CLUSTER}" || true
  if [[ -n "${K8S_NAMESPACE:-}" ]]; then
    kubectl config set-context --current --namespace="${K8S_NAMESPACE}" || true
  fi
fi

export KUEUE_QUEUE="${KUEUE_QUEUE:-multislice-queue}"
export PREEMPTIBLE="${PREEMPTIBLE:-${preemptible:-false}}"
export GANG_ID="${GANG_ID:-${JOB_PREFIX}}"
export PRIORITY_CLASS="${PRIORITY_CLASS:-medium}"
# yaml_generator reads KUEUE_PRIORITY_CLASS (not PRIORITY_CLASS) to render
# ${PRIORITY_CLASS_LINE}; without it the trainer admits at priority 0 and is
# evictable by any prioritised workload.
export KUEUE_PRIORITY_CLASS="${KUEUE_PRIORITY_CLASS:-${PRIORITY_CLASS}}"
export SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-xpk-sa}"
export CPU_MACHINE="${CPU_MACHINE:-n2d-standard-64}"

# ==============================================================================
# Fail-fast (see k8s_launcher.sh). `true`: a trainer/rollout worker dying
# after registration fails its JobSet and cluster_reaper tears down the run,
# instead of the run deadlocking. `false`: legacy in-place pod restarts.
# ==============================================================================
export FAIL_FAST="${FAIL_FAST:-true}"
# JobSet recreations allowed for worker failures before registration.
export FT_STARTUP_RETRIES="${FT_STARTUP_RETRIES:-3}"
# Sandbox side (FAIL_FAST=true only): max wait for a sandbox/warm pool to
# become ready, and fleet.acquire attempts per episode (legacy: SDK 900s x 5
# attempts).
export FT_SANDBOX_READY_TIMEOUT_S="${FT_SANDBOX_READY_TIMEOUT_S:-600}"
export FT_SANDBOX_ACQUIRE_RETRIES="${FT_SANDBOX_ACQUIRE_RETRIES:-2}"

# ==============================================================================
# Pathways & Raiden Weight Sync Defaults
# ==============================================================================
source "${DIR}/mlperf_pathways_config.sh"

export USE_WEIGHT_CONVERTER="true"
export PREFUSE_MOE_WEIGHTS="true"
export TRAINER_PREFUSE_MOE_WEIGHTS="true"
export ROLLOUT_PREFUSE_MOE_WEIGHTS="true"
export VERIFY_WEIGHTS="true"
export TRAINER_PADDED_MOE_MLP_DIM=""
export WEIGHT_SYNC_MODE="${WEIGHT_SYNC_MODE:-raiden}"
export WEIGHT_SYNC_DISABLE_TIMEOUTS="${WEIGHT_SYNC_DISABLE_TIMEOUTS:-${DISABLE_WEIGHT_SYNC_TIMEOUTS:-0}}"

export TPU_RAIDEN_DATA_NICS="${TPU_RAIDEN_DATA_NICS:-eth0}"
export RAIDEN_FFI_USE_DIRECT_DEVICE_BUFFER="${RAIDEN_FFI_USE_DIRECT_DEVICE_BUFFER:-0}"
export ENABLE_MULTI_NUMA="${ENABLE_MULTI_NUMA:-0}"

# ==============================================================================
# WandB Configuration
# ==============================================================================
export WANDB_ENTITY="${WANDB_ENTITY:-google-trellis}"
export WANDB_PROJECT="${WANDB_PROJECT:-trellis-deepswe}"

# ==============================================================================
# Common Model & Backend Configuration
# ==============================================================================
export TRAINER_BACKEND="maxtext"
export SAMPLER="vllm"
export TRAINABLE_PARAMETERS_MASK='^(?!.*routed_experts/gate/kernel).*'
# Qwen3.5 <|im_end|>, <|endoftext|> (generation_config.json eos_token_id).
# 151645/151643 are the Qwen2.5/Qwen3 ids and are ordinary tokens in the
# Qwen3.5 vocab.
export EOS_TOKENS="${EOS_TOKENS:-248046,248044}"
export TRAINER_BASE_NUM_KV_HEADS=2
export ROLLOUT_MESH_FSDP="${ROLLOUT_MESH_FSDP:-1}"
export ROLLOUT_MESH_TP="${ROLLOUT_MESH_TP:-1}"

# ==============================================================================
# MLPerf RCP Logging
# ==============================================================================
export RCP_LOGGING="${RCP_LOGGING:-false}"
if [[ -n "${MAXTEXT_OUTPUT_DIR:-}" ]]; then
  export METRIC_LOGGER_DIR="${METRIC_LOGGER_DIR:-${MAXTEXT_OUTPUT_DIR}/mllog}"
fi
export TARGET_ACCURACY="${TARGET_ACCURACY:-0.69}"

# ==============================================================================
# vLLM Rollout Configuration
# ==============================================================================
export VLLM_LOGGING_LEVEL="INFO"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-65536}"
export VLLM_MAX_NUM_BATCHED_TOKENS=2048
export VLLM_MAX_NUM_SEQS=16
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.9}"

# Sharding Configs
export VLLM_DATA_PARALLEL_SIZE="${VLLM_DATA_PARALLEL_SIZE:-1}"
export VLLM_ENABLE_EXPERT_PARALLEL="true"

# Prefix Caching Configs
export ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-true}"
export VLLM_PREFIX_CACHE_RETENTION_INTERVAL="${VLLM_PREFIX_CACHE_RETENTION_INTERVAL:-0}"
if [[ "${ENABLE_PREFIX_CACHING}" == "true" ]]; then
  export MAMBA_CACHE_MODE="${MAMBA_CACHE_MODE:-align}"
else
  export MAMBA_CACHE_MODE="${MAMBA_CACHE_MODE:-none}"
fi
export VLLM_MAMBA_CACHE_MODE="${VLLM_MAMBA_CACHE_MODE:-${MAMBA_CACHE_MODE}}"

# FP8 MoE rollout/trainer (ROLLOUT_FP8, TRAINER_FP8); see fp8_moe.sh.
source "${DIR}/fp8_moe.sh"

# Router replay
export RETURN_ROUTED_EXPERTS="${RETURN_ROUTED_EXPERTS:-true}"

# KV Cache Configs
export ROLLOUT_FREE_KV_CACHE="false"
export PARTIAL_ROLLOUT="${PARTIAL_ROLLOUT:-false}"
export VLLM_KV_CACHE_DTYPE="bfloat16"
export VLLM_BLOCK_SIZE=256

# Engine Configs
export VLLM_ASYNC_SCHEDULING="true"
export VLLM_ENABLE_CHUNKED_PREFILL="true"

# Model Configs
export VLLM_LANGUAGE_MODEL_ONLY="true"
export VLLM_REASONING_PARSER="qwen3"
export VLLM_LIMIT_MM_PER_PROMPT='{"image": 0, "video": 0}'

# ==============================================================================
# Rollout Worker Environment Flags (Optimizations & Runtime Settings)
# ==============================================================================
export NUM_PRECOMPILE_WORKERS=8
export NEW_MODEL_DESIGN=1
export ATTN_BUCKETIZED_NUM_REQS=true
export ATTN_CUSTOM_NUM_REQS_BUCKETS=4
export ONEHOT_MOE_PERMUTE_THRESHOLD="${ONEHOT_MOE_PERMUTE_THRESHOLD:-32768}"
export VLLM_MOE_CHUNK_SIZE=256
export SLICE_ROPE_CACHE=1
export DP_SCHED_BATCH_PREFILL=false
export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:- --xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false --xla_tpu_check_legacy_constraints_in_reduce_scatter_legalizer=false}"
export VLLM_ENABLE_V1_MULTIPROCESSING=0

# ==============================================================================
# Hyperparameters & DeepSWE Pipeline Configuration
# ==============================================================================
export MAX_STEPS=${MAX_STEPS:-50}
export BATCH_SIZE=${BATCH_SIZE:-16}
export MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-${BATCH_SIZE}}
export NUM_GENERATIONS="${NUM_GENERATIONS:-16}"
export TRAIN_MICRO_BATCH_SIZE="${TRAIN_MICRO_BATCH_SIZE:-32}"
export CHECKPOINT_SAVE_INTERVAL_STEPS=${CHECKPOINT_SAVE_INTERVAL_STEPS:-0}
export CHECKPOINT_MAX_TO_KEEP="${CHECKPOINT_MAX_TO_KEEP:-10}"
# Every checkpoint has the params eval reads; only every Nth (and the last)
# adds the optimizer state, which is ~6x larger and slower to write than a step.
export CHECKPOINT_OPTIMIZER_INTERVAL_STEPS="${CHECKPOINT_OPTIMIZER_INTERVAL_STEPS:-5}"
export CHECKPOINT_ASYNC=${CHECKPOINT_ASYNC:-true}
export ENABLE_PATHWAYS_PERSISTENCE=${ENABLE_PATHWAYS_PERSISTENCE:-1}
export MAX_STALENESS=${MAX_STALENESS:-1}
# Overlap each step's weight sync with the next step's training. Safe only
# while the trainer transfers from host staging, since the next step rewrites
# the device buffers mid-transfer.
export ASYNC_WEIGHT_SYNC=${ASYNC_WEIGHT_SYNC:-false}
if [[ "${ASYNC_WEIGHT_SYNC}" == "true" && "${WEIGHT_SYNC_MODE}" == "raiden" \
      && "${RAIDEN_FFI_USE_DIRECT_DEVICE_BUFFER}" != "0" ]]; then
  echo "ASYNC_WEIGHT_SYNC=true requires RAIDEN_FFI_USE_DIRECT_DEVICE_BUFFER=0" >&2
  exit 1
fi
# Pack the next microbatch while the trainer runs the current one.
export PIPELINE_TRAIN_MICROBATCHES=${PIPELINE_TRAIN_MICROBATCHES:-false}
export TRAJECTORY_GROUP_ORDER=${TRAJECTORY_GROUP_ORDER:-prompt_batch}

# Sequence packing
export MAX_SEQ_TOKEN_PER_TPU=${MAX_SEQ_TOKEN_PER_TPU:-65536}
export MAX_SEGMENTS_PER_PACKED_ROW=${MAX_SEGMENTS_PER_PACKED_ROW:-16}

# Sampling Parameters (explicitly disable top-k, set top-p 1.0 and temperature 1.0)
export TEMPERATURE="${TEMPERATURE:-1.0}"
export TOP_P="${TOP_P:-1.0}"
export TOP_K="-1"

# Algorithmic & Loss Hyperparameters
export BETA=0.0
export EPSILON=0.2
export EPSILON_HIGH=0.28
export USE_ROLLOUT_LOGPS="false"
export EXACT_TOKEN_CONTINUITY="${EXACT_TOKEN_CONTINUITY:-true}"
# Per the MLPerf Qwen3.5 DeepSWE NeMo-RL reference (`overlong_filtering: true`),
# truncated/overlong trajectories have their token loss masked out
# (`OVERLONG_LOSS_MASKING="true"`), while their `0.0` reward remains in the
# `grpo-loo` group baseline (`OVERLONG_FILTER="false"`) so groups where all
# unsolved trajectories are truncated still produce positive advantages for
# solved trajectories.
export OVERLONG_FILTER="${OVERLONG_FILTER:-false}"
export OVERLONG_LOSS_MASKING="${OVERLONG_LOSS_MASKING:-true}"
export SEQ_LOGPROB_ERROR_THRESHOLD=2.0
export TRUNCATED_IMPORTANCE_SAMPLING_TYPE="seq-mask-tis"
export TRUNCATED_IMPORTANCE_SAMPLING_RATIO_MIN=0.999
export TRUNCATED_IMPORTANCE_SAMPLING_RATIO=1.002
export ADVANTAGE_ESTIMATOR="grpo-loo"
export LOSS_AGG_MODE="token-mean"
export FLOAT32_GATE_LOGITS="true"
export FLOAT32_LOGITS="true"

# Optimizer Hyperparameters
export LEARNING_RATE="${LEARNING_RATE:-1e-6}"
export ADAM_B1=0.9
export ADAM_B2=0.999
export WEIGHT_DECAY=0.0
export MAX_GRAD_NORM="${MAX_GRAD_NORM:-0.125}"
# The maxtext trainer clips only via clip_by_global_norm; an empty chain type
# would leave it at base.yml's 1.0.
export OPT_CHAIN_TYPE="clip_by_global_norm"
export WARMUP_STEPS_FRACTION=0.0
export LEARNING_RATE_FINAL_FRACTION=1.0
export SKIP_STEP_ON_SPIKES="${SKIP_STEP_ON_SPIKES:-false}"
export SKIP_STEP_ON_NAN="${SKIP_STEP_ON_NAN:-true}"

# Architecture & Rematerialization
export REMAT_POLICY="${REMAT_POLICY:-full}"
export TRAINER_MAXTEXT_ATTENTION="flash"
export COMPUTE_LOGPS_CHUNK_SIZE=512

export EPISODE_TIMEOUT_SECS=1800
export DEBUG=${DEBUG:-1}

# ==============================================================================
# DeepSWE Environment & Agent Sandbox
# ==============================================================================
export DATASET_PATH="${DATASET_PATH:-gs://mlperf_dataset/benchmark-r2e-gym-easy-curriculum-v2}"
export SHUFFLE="${SHUFFLE:-false}"
export USE_AGENT_SANDBOX=1
export SCAFFOLD="openhands"
export SANDBOX_NAMESPACE="${SANDBOX_NAMESPACE:-${K8S_NAMESPACE:-trellis}}"
export POOL_NAME_FORMAT="${POOL_NAME_FORMAT:-}"
export TEMPLATE_NAME_PREFIX="${TEMPLATE_NAME_PREFIX:-}"
export SANDBOX_NODE_SELECTOR_KEY="cloud.google.com/gke-nodepool"
export SANDBOX_NODE_SELECTOR_VAL="${SANDBOX_NODE_SELECTOR_VAL:-sandbox-np}"
export MAX_WARMPOOL_REPLICAS="${MAX_WARMPOOL_REPLICAS:-2}"
export ROLLOUT_MAX_CONCURRENCY="${ROLLOUT_MAX_CONCURRENCY:-256}"
export MAX_CONCURRENCY="${MAX_CONCURRENCY:-256}"
export STEP_TIMEOUT_SECS="${STEP_TIMEOUT_SECS:-60}"
export REWARD_TIMEOUT_SECS="${REWARD_TIMEOUT_SECS:-60}"
export FLUSH_EVERY_N_STEPS=1
export MAX_TURNS=30
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"

# ==============================================================================
# Execution Dispatch
# ==============================================================================
if [[ -z "${LAUNCHER:-}" ]]; then
  if [ -f "${DIR}/../deepswe_dist/k8s_launcher.sh" ]; then
    LAUNCHER="${DIR}/../deepswe_dist/k8s_launcher.sh"
  elif [ -f "${DIR}/tunix/experimental/examples/deepswe_dist/k8s_launcher.sh" ]; then
    LAUNCHER="${DIR}/tunix/experimental/examples/deepswe_dist/k8s_launcher.sh"
  elif [ -f "${DIR}/../../../../third_party/py/tunix/experimental/examples/deepswe_dist/k8s_launcher.sh" ]; then
    LAUNCHER="${DIR}/../../../../third_party/py/tunix/experimental/examples/deepswe_dist/k8s_launcher.sh"
  elif [ -f "${HOME}/github/tunix_build/tunix/experimental/examples/deepswe_dist/k8s_launcher.sh" ]; then
    LAUNCHER="${HOME}/github/tunix_build/tunix/experimental/examples/deepswe_dist/k8s_launcher.sh"
  else
    echo "Error: k8s_launcher.sh not found relative to ${DIR}"
    exit 1
  fi
fi

if [[ "${MLPERF_NO_LAUNCH:-0}" != "1" ]]; then
  if [[ "${1:-}" == --* ]]; then
    if [[ -n "${CHECKPOINT_MANIFEST_FILE:-}" ]]; then
      COMMAND="eval"
    else
      COMMAND="start"
    fi
  else
    COMMAND="${1:-start}"
    shift || true
  fi
  if [[ "${COMMAND}" == "eval" && -n "${CHECKPOINT_MANIFEST_FILE:-}" ]]; then
    # MLPerf offline evaluation (see evals/README.md): evaluate the training
    # run's checkpoints in step order, appending eval_* events to its MLLOG,
    # and stop at the first checkpoint that reaches TARGET_ACCURACY. The eval
    # container emits run_stop backdated to that checkpoint's manifest
    # timestamp_ms, which is taken before the checkpoint is written.
    echo "Running sequential offline evaluation from manifest: ${CHECKPOINT_MANIFEST_FILE}"
    export RCP_LOGGING=true
    RCP_EVAL="${DIR}/evals/rcp_eval.py"
    # check needs mlperf_logging; point RCP_PYTHON at a venv that has it.
    RCP_PYTHON="${RCP_PYTHON:-python3}"
    MAX_EVAL_ATTEMPTS="${MAX_EVAL_ATTEMPTS:-3}"
    for arg in "$@"; do
      case "${arg}" in
        --dry-run|--dry_run|--render) export DRY_RUN=true ;;
      esac
    done
    # Validates the manifest against the MLLOG and lists the checkpoints that
    # still need an eval_accuracy (already-evaluated ones are skipped).
    PLAN_TSV="$("${RCP_PYTHON}" "${RCP_EVAL}" plan --manifest "${CHECKPOINT_MANIFEST_FILE}")" || exit 1
    MANIFEST_ROWS=()
    if [[ -n "${PLAN_TSV}" ]]; then
      mapfile -t MANIFEST_ROWS <<< "${PLAN_TSV}"
    fi
    BASE_EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${MAXTEXT_OUTPUT_DIR}/eval_results}"
    EVAL_JOBSET_NAME="${EVAL_JOBSET_NAME:-${JOB_PREFIX}-eval}"
    for row in "${MANIFEST_ROWS[@]}"; do
      IFS=$'\t' read -r STEP SAMPLES TS_MS CKPT_PATH IS_LAST MLLOG_FILE <<< "${row}"
      export MAXTEXT_CKPT="${CKPT_PATH}"
      export CHECKPOINT_STEP="${STEP}"
      export SAMPLES_COUNT="${SAMPLES}"
      export CHECKPOINT_TIMESTAMP_MS="${TS_MS}"
      export IS_LAST_CHECKPOINT="${IS_LAST}"
      # Append eval_* / run_stop to the training run's MLLOG.
      export METRIC_LOGGER_DIR="${MLLOG_FILE}"
      export EVAL_OUTPUT_DIR="${BASE_EVAL_OUTPUT_DIR%/}/step_${STEP}"
      if [[ -z "${SCAN_LAYERS_OVERRIDE:-}" ]]; then
        # Checkpoints saved during RL training are unscanned.
        export SCAN_LAYERS="false"
      fi

      STEP_STATUS="missing"
      for (( attempt = 1; attempt <= MAX_EVAL_ATTEMPTS; attempt++ )); do
        echo "=== Evaluating checkpoint step=${STEP} samples=${SAMPLES} is_last=${IS_LAST} attempt=${attempt}/${MAX_EVAL_ATTEMPTS} path=${CKPT_PATH} ==="
        if [[ "${DRY_RUN:-false}" == "true" ]]; then
          "${LAUNCHER}" --command eval --image "${TUNIX_IMAGE}" "$@"
          STEP_STATUS="dry_run"
          break
        fi
        # Clear any JobSet left over from an earlier attempt or run.
        "${LAUNCHER}" --command stop_eval --image "${TUNIX_IMAGE}" &>/dev/null || true
        if ! "${LAUNCHER}" --command eval --image "${TUNIX_IMAGE}" "$@"; then
          echo "Launching the eval JobSet failed for step ${STEP}. Retrying in ${EVAL_RETRY_DELAY_SECS:-30}s..."
          sleep "${EVAL_RETRY_DELAY_SECS:-30}"
          continue
        fi
        HEAD_JOBSET="${EVAL_JOBSET_NAME}"
        if [[ "${ROLLOUT_REPLICAS:-1}" -gt 1 ]]; then
          HEAD_JOBSET="${EVAL_JOBSET_NAME}-0"
        fi
        echo "Waiting for evaluation JobSet ${HEAD_JOBSET} (main container) in namespace ${K8S_NAMESPACE}..."
        while true; do
          if ! kubectl get jobset "${HEAD_JOBSET}" -n "${K8S_NAMESPACE}" &>/dev/null; then
            echo "JobSet ${HEAD_JOBSET} no longer exists."
            break
          fi
          JS_COND="$(kubectl get jobset "${HEAD_JOBSET}" -n "${K8S_NAMESPACE}" -o jsonpath='{.status.conditions[?(@.status=="True")].type}' 2>/dev/null || true)"
          if [[ "${JS_COND}" =~ (Failed|Completed) ]]; then
            echo "JobSet ${HEAD_JOBSET} reached terminal condition: ${JS_COND}."
            break
          fi
          PROC_SELECTOR="jobset.sigs.k8s.io/jobset-name=${HEAD_JOBSET},jobset.sigs.k8s.io/replicatedjob-name=proc,batch.kubernetes.io/job-completion-index=0"
          MAIN_EXIT="$(kubectl get pods -n "${K8S_NAMESPACE}" -l "${PROC_SELECTOR}" -o jsonpath='{.items[0].status.containerStatuses[?(@.name=="main")].state.terminated.exitCode}' 2>/dev/null || true)"
          if [[ -n "${MAIN_EXIT}" ]]; then
            echo "Main evaluation container finished with exit code ${MAIN_EXIT}."
            break
          fi
          WAIT_REASON="$(kubectl get pods -n "${K8S_NAMESPACE}" -l "${PROC_SELECTOR}" -o jsonpath='{.items[0].status.containerStatuses[?(@.name=="main")].state.waiting.reason}' 2>/dev/null || true)"
          RESTARTS="$(kubectl get pods -n "${K8S_NAMESPACE}" -l "${PROC_SELECTOR}" -o jsonpath='{.items[0].status.containerStatuses[?(@.name=="main")].restartCount}' 2>/dev/null || true)"
          if [[ "${WAIT_REASON}" == "CrashLoopBackOff" ]] || [[ "${RESTARTS:-0}" -ge 2 ]]; then
            LAST_EXIT="$(kubectl get pods -n "${K8S_NAMESPACE}" -l "${PROC_SELECTOR}" -o jsonpath='{.items[0].status.containerStatuses[?(@.name=="main")].lastState.terminated.exitCode}' 2>/dev/null || true)"
            echo "Main evaluation container is crashing (reason=${WAIT_REASON}, restarts=${RESTARTS}, lastExit=${LAST_EXIT})."
            break
          fi
          sleep 10
        done
        "${LAUNCHER}" --command stop_eval --image "${TUNIX_IMAGE}" || true
        # The MLLOG, not the container exit code, decides whether this
        # checkpoint was evaluated: retrying after eval_accuracy was logged
        # would log the checkpoint twice. If the MLLOG cannot be read, stop
        # rather than risk that.
        STEP_STATUS="$("${RCP_PYTHON}" "${RCP_EVAL}" status --mllog "${MLLOG_FILE}" --samples_count "${SAMPLES}")" || exit 1
        if [[ "${STEP_STATUS}" == done* ]]; then
          break
        fi
        echo "No eval_accuracy logged for step ${STEP} (attempt ${attempt}/${MAX_EVAL_ATTEMPTS})."
      done

      if [[ "${STEP_STATUS}" == "dry_run" ]]; then
        continue
      fi
      if [[ "${STEP_STATUS}" != done* ]]; then
        # Skipping a checkpoint would make a later one look like the first to
        # reach the target, which invalidates the result.
        echo "ERROR: step ${STEP} was not evaluated after ${MAX_EVAL_ATTEMPTS} attempts. Fix the failure and rerun; evaluated steps are skipped." >&2
        exit 1
      fi
      read -r _ STEP_ACC RUN_STOP_STATUS <<< "${STEP_STATUS}"
      echo ">>> step ${STEP}: pass@4=${STEP_ACC} run_stop=${RUN_STOP_STATUS}"
      if [[ "${RUN_STOP_STATUS}" != "none" ]]; then
        echo "run_stop(${RUN_STOP_STATUS}) logged at step ${STEP}. Stopping offline evaluation loop."
        break
      fi
    done

    if [[ "${DRY_RUN:-false}" != "true" ]]; then
      echo "=== Final state ==="
      "${RCP_PYTHON}" "${RCP_EVAL}" inspect --manifest "${CHECKPOINT_MANIFEST_FILE}" || true
      if [[ "${RUN_COMPLIANCE_CHECK:-true}" == "true" ]]; then
        "${RCP_PYTHON}" "${RCP_EVAL}" check --manifest "${CHECKPOINT_MANIFEST_FILE}" \
          || echo "WARNING: compliance check did not pass (see above)." >&2
      fi
    fi
    exit 0
  fi
  exec "${LAUNCHER}" --command "${COMMAND}" --image "${TUNIX_IMAGE}" "$@"
fi
