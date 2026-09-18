#!/bin/bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Available options: 'start', 'stop', 'orchestrator', 'trainer', 'rollout'.
COMMAND=""
TUNIX_IMAGE=${TUNIX_IMAGE:-}

LAUNCHER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" &>/dev/null; then
  PYTHON="python"
fi

YAML_GEN=${YAML_GEN:-"${LAUNCHER_DIR}/../../distributed/deployment/yaml_generator.py"}
YAML_DIR=${YAML_DIR:-"${LAUNCHER_DIR}/../../distributed/deployment/yamls"}

export MODEL_NAME=${MODEL_NAME:-Qwen3-1.7B}
export MODEL_ID=${MODEL_ID:-Qwen/Qwen3-1.7B}
# Must be model-specific: vLLM prioritizes non-empty local snapshot directories,
# which can cause stale config/shape mismatches if shared across models.
export MODEL_DIR=${MODEL_DIR:-artifacts/qwen3_dist_gsm8k/models/${MODEL_NAME}}
# Defaults to MODEL_ID so AutoTokenizer downloads directly from HuggingFace
# instead of failing on an initially empty local MODEL_DIR.
export TOKENIZER_PATH=${TOKENIZER_PATH:-${MODEL_ID}}

export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-128}
export BATCH_SIZE=${BATCH_SIZE:-2}
export NUM_GENERATIONS=${NUM_GENERATIONS:-2}
export MAX_STEPS=${MAX_STEPS:-1}
export TRAIN_MICRO_BATCH_SIZE=${TRAIN_MICRO_BATCH_SIZE:-1}
export MAX_SEQ_TOKEN_PER_TPU=${MAX_SEQ_TOKEN_PER_TPU:-}
export MAX_SEGMENTS_PER_PACKED_ROW=${MAX_SEGMENTS_PER_PACKED_ROW:-}

# Set to tunix to run Tunix's PeftTrainer, and maxtext to run MaxText's MaxTextTrainingEngine
export TRAINER_BACKEND=${TRAINER_BACKEND:-tunix}
# NOTE: this is trajectories, not prompt groups, and that contradicts
# run_trainer_node.py's own help text ("Number of prompt groups per optimizer
# update"). It is deliberate and must stay until the image is re-synced.
#
# The image's trainer computes grad_accumulation_steps as
# ceil(mini_batch_size / train_micro_batch_size), omitting num_generations --
# bug A of the dense-GRPO bringup, which that report fixed for the Tunix
# backend and flagged as untested on MaxText. The launcher's extra
# * NUM_GENERATIONS here cancels the trainer's missing one, so the pair
# produces the right answer. Correcting only one side breaks a working config.
#
# The proper fix (mini_batch_size = BATCH_SIZE, trainer multiplies by
# num_generations via _gradient_accumulation_steps) is already written in this
# repo's run_trainer_node.py, but that file cannot be injected -- see the
# trainer payload comment below. Land both together.
export MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-$((BATCH_SIZE * NUM_GENERATIONS))}
export EVAL_EVERY_N_STEPS=${EVAL_EVERY_N_STEPS:-1000000}
export MAX_GRAD_NORM=${MAX_GRAD_NORM:-0.125}
export ADAM_B1=${ADAM_B1:-0.9}
export ADAM_B2=${ADAM_B2:-0.999}
export WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}
export LEARNING_RATE=${LEARNING_RATE:-1.0e-6}
export LORA_RANK=${LORA_RANK:-16}
export LORA_ALPHA=${LORA_ALPHA:-16.0}
export USE_LORA=${USE_LORA:-0}
export REWARD_MODE=${REWARD_MODE:-env}
export BETA=${BETA:-0}
export EPSILON=${EPSILON:-0.2}
export DEBUG=${DEBUG:-0}
export SAMPLER=${SAMPLER:-inprocess_vllm}
export WEIGHT_SYNC_MODE=${WEIGHT_SYNC_MODE:-none}
export USE_ROLLOUT_LOGPS=${USE_ROLLOUT_LOGPS:-true}

# Truncated importance sampling (TIS) and sequence loss masking. All empty by
# default: an empty value emits no flag at all, so the orchestrator keeps its
# own default and the rendered command is unchanged from before these existed.
# TIS_RATIO_MIN and TIS_RATIO_MAX are the keep-band; the orchestrator rejects
# TIS_TYPE without both, so a half-configured band fails at startup rather than
# silently masking nothing.
export TIS_TYPE=${TIS_TYPE:-}
export TIS_RATIO_MIN=${TIS_RATIO_MIN:-}
export TIS_RATIO_MAX=${TIS_RATIO_MAX:-}
export SEQ_LOGPROB_ERROR_THRESHOLD=${SEQ_LOGPROB_ERROR_THRESHOLD:-}
export OVERLONG_LOSS_MASKING=${OVERLONG_LOSS_MASKING:-false}
export CHECKPOINT_SAVE_INTERVAL_STEPS=${CHECKPOINT_SAVE_INTERVAL_STEPS:-1}
export CHECKPOINT_MAX_TO_KEEP=${CHECKPOINT_MAX_TO_KEEP:-10}
export CHECKPOINT_ROOT_DIRECTORY=${CHECKPOINT_ROOT_DIRECTORY:-checkpoints}
export ENABLE_PATHWAYS_PERSISTENCE=${ENABLE_PATHWAYS_PERSISTENCE:-0}

# MaxText trainer configuration: only consulted when TRAINER_BACKEND=maxtext
export MAXTEXT_MODEL_NAME=${MAXTEXT_MODEL_NAME:-qwen3-1.7b}
export MAXTEXT_CKPT=${MAXTEXT_CKPT:-}
export MAXTEXT_OUTPUT_DIR=${MAXTEXT_OUTPUT_DIR:-artifacts/math_gsm8k_dist/maxtext}
# Padded MoE MLP intermediate dimension; must match rollout TP padding for MoE models.
export TRAINER_PADDED_MOE_MLP_DIM=${TRAINER_PADDED_MOE_MLP_DIM:-}
# Optional: enable experimental batched-RPA attention kernel for rollout.
export ROLLOUT_USE_BATCHED_RPA=${ROLLOUT_USE_BATCHED_RPA:-}
export ROLLOUT_MAXTEXT_ATTENTION=${ROLLOUT_MAXTEXT_ATTENTION:-}

# MoE & Weight Sync Flags
export PREFUSE_MOE_WEIGHTS=${PREFUSE_MOE_WEIGHTS:-true}
export USE_WEIGHT_CONVERTER=${USE_WEIGHT_CONVERTER:-true}
export ENABLE_PREFIX_CACHING=${ENABLE_PREFIX_CACHING:-false}
export RETURN_ROUTED_EXPERTS=${RETURN_ROUTED_EXPERTS:-false}

# Number of KV heads the ROLLOUT replicates to; must equal the rollout's
# tp * ep. Empty by default, which emits no env var at all, so the trainer keeps
# its rollout_mesh_tp fallback and ep==1 runs are unchanged. Set it to tp*ep
# whenever ROLLOUT_MESH_EP > 1, or Raiden weight sync fails preflight with a
# global-shape mismatch on decoder.layers.N.attention.attention.key.kernel.
export KV_TP_SIZE=${KV_TP_SIZE:-}

# Logs source/destination Raiden tensor checksums on both the trainer and
# rollout sides during weight sync, for cross-verification of a real run.
export VERIFY_WEIGHTS=${VERIFY_WEIGHTS:-false}

export WANDB_PROJECT=${WANDB_PROJECT:-trellis-gsm8k}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-}
export WANDB_API_KEY=${WANDB_API_KEY:-}
export TFDS_DATA_DIR=${TFDS_DATA_DIR:-"artifacts/data"}
export TFDS_SPLIT=${TFDS_SPLIT:-train}
export FLUSH_METRICS_EVERY_N_STEPS=${FLUSH_METRICS_EVERY_N_STEPS:-1}

export ORCHESTRATOR_ID=$USER-orch
export ORCHESTRATOR_PORT=20000

export ROLLOUT_ID=$USER-roll
export ROLLOUT_PORT=20001
export ROLLOUT_REPLICAS=${ROLLOUT_REPLICAS:-1}

export TRAINER_ID=$USER-train
export TRAINER_PORT=20002

export CPU_MACHINE=${CPU_MACHINE:-n2-standard-64}
export GCS_SCRATCH_LOCATION=${GCS_SCRATCH_LOCATION:-gs://cloud-pathways-staging/tmp}

export TRAINER_JOBSET_YAML=${TRAINER_JOBSET_YAML:-jobset.pathways.yaml}
export TRAINER_TPU_SLICE=${TRAINER_TPU_SLICE:-tpuv5e:4x4}
export TRAINER_MESH_FSDP=${TRAINER_MESH_FSDP:-16}
export TRAINER_MESH_TP=${TRAINER_MESH_TP:-1}
export TRAINER_MESH_EXPERT=${TRAINER_MESH_EXPERT:-1}

export PATHWAYS_SERVER_IMAGE=${PATHWAYS_SERVER_IMAGE:-us-docker.pkg.dev/cloud-tpu-v2-images/pathways/server:latest}
export PATHWAYS_PROXY_IMAGE=${PATHWAYS_PROXY_IMAGE:-us-docker.pkg.dev/cloud-tpu-v2-images/pathways/proxy_server:latest}

export ROLLOUT_JOBSET_YAML=${ROLLOUT_JOBSET_YAML:-leaderworkerset.mcjax.ray.yaml}
export ROLLOUT_TPU_SLICE=${ROLLOUT_TPU_SLICE:-tpuv5e:4x4}
export ROLLOUT_MESH_FSDP=${ROLLOUT_MESH_FSDP:-1}
export ROLLOUT_MESH_TP=${ROLLOUT_MESH_TP:-16}
# Sampler-side expert parallelism. Carved out of the rollout slice, so
# ROLLOUT_MESH_FSDP * ROLLOUT_MESH_TP * ROLLOUT_MESH_EP must equal the rollout
# device count. Defaults to 1 to leave existing runs untouched.
export ROLLOUT_MESH_EP=${ROLLOUT_MESH_EP:-1}

# Kubernetes Cluster & Scheduling Options
export K8S_NAMESPACE=${K8S_NAMESPACE:-${NAMESPACE:-default}}
export KUEUE_QUEUE_NAME=${KUEUE_QUEUE_NAME:-${QUEUE_NAME:-}}
export DRY_RUN=${DRY_RUN:-false}

# Raiden weight sync rides the Pathways proxy, and `start_trainer` turns on
# RAIDEN_USE_FFI purely from WEIGHT_SYNC_MODE. The stock `pathways/server` and
# `pathways/proxy_server` images do not carry the Raiden FFI, so pairing them
# with WEIGHT_SYNC_MODE=raiden gets you a run that comes up, trains, and never
# syncs a weight. Fail here instead, where the cause is still visible.
if [[ "${WEIGHT_SYNC_MODE}" == "raiden" ]]; then
  for var in PATHWAYS_SERVER_IMAGE PATHWAYS_PROXY_IMAGE; do
    if [[ "${!var}" != *raiden* ]]; then
      echo "Error: WEIGHT_SYNC_MODE=raiden requires a Raiden-capable Pathways" >&2
      echo "       image, but ${var}=${!var} is not one." >&2
      echo "       Known good (validated on the 35B v5p stack):" >&2
      echo "         us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/datenglin/unsanitized_server:raiden_20260904" >&2
      echo "         us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/datenglin/unsanitized_proxy_server:raiden_20260904" >&2
      exit 1
    fi
  done
fi

# Sampler expert parallelism changes the KV head count the rollout builds.
# vLLM replicates KV heads up to tp*ep when the model has fewer, and Raiden
# pairs tensors by name, so the trainer must be told the same number or weight
# sync dies in preflight -- several minutes in, with a message that names a
# tensor shape and not this config. Check it here, where the cause is visible.
if [[ "${ROLLOUT_MESH_EP:-1}" -gt 1 ]]; then
  expected_kv_tp=$((ROLLOUT_MESH_TP * ROLLOUT_MESH_EP))
  if [[ -z "${KV_TP_SIZE}" ]]; then
    echo "Error: ROLLOUT_MESH_EP=${ROLLOUT_MESH_EP} > 1 requires KV_TP_SIZE to be" >&2
    echo "       set, because the rollout replicates KV heads to tp*ep and the" >&2
    echo "       trainer must build the same shape." >&2
    echo "       Set: export KV_TP_SIZE=${expected_kv_tp}" >&2
    exit 1
  fi
  if [[ "${KV_TP_SIZE}" -ne "${expected_kv_tp}" ]]; then
    echo "Error: KV_TP_SIZE=${KV_TP_SIZE} does not equal ROLLOUT_MESH_TP *" >&2
    echo "       ROLLOUT_MESH_EP = ${ROLLOUT_MESH_TP} * ${ROLLOUT_MESH_EP} =" >&2
    echo "       ${expected_kv_tp}. A mismatch here is a silent weight-sync" >&2
    echo "       shape error, so it is rejected rather than guessed at." >&2
    exit 1
  fi
fi

apply_manifest() {
  if [[ "$DRY_RUN" == "true" ]]; then
    echo "---"
    cat
  else
    kubectl apply -f -
  fi
}

stop_orchestrator() {
  if [[ "$DRY_RUN" == "true" ]]; then
    echo "kubectl delete jobset ${ORCHESTRATOR_ID} -n ${K8S_NAMESPACE}"
  else
    kubectl delete jobset "${ORCHESTRATOR_ID}" -n "${K8S_NAMESPACE}" --ignore-not-found --wait=true
    while kubectl get jobset "${ORCHESTRATOR_ID}" -n "${K8S_NAMESPACE}" &>/dev/null; do
      sleep 2
    done
  fi
}

start_orchestrator() {
  local debug_flag=""
  if [[ "${DEBUG}" == "1" || "${DEBUG}" == "true" || "${DEBUG}" == "True" ]]; then
    debug_flag="--debug"
  fi
  local router_replay_b64 datatypes_b64 agent_types_b64 collect_engine_b64 algorithm_adapter_b64 batch_assembly_b64 rl_program_b64 run_gsm8k_b64 collector_b64
  router_replay_b64="$(gzip -9c "${LAUNCHER_DIR}/../../../common/router_replay.py" | base64 -w0)"
  datatypes_b64="$(gzip -9c "${LAUNCHER_DIR}/../../common/datatypes.py" | base64 -w0)"
  collector_b64="$(gzip -9c "${LAUNCHER_DIR}/../../rollout/collector.py" | base64 -w0)"
  agent_types_b64="$(gzip -9c "${LAUNCHER_DIR}/../../../rl/agentic/agents/agent_types.py" | base64 -w0)"
  collect_engine_b64="$(gzip -9c "${LAUNCHER_DIR}/../../../rl/agentic/trajectory/trajectory_collect_engine.py" | base64 -w0)"
  algorithm_adapter_b64="$(gzip -9c "${LAUNCHER_DIR}/../../orchestrator/algorithm_adapter.py" | base64 -w0)"
  batch_assembly_b64="$(gzip -9c "${LAUNCHER_DIR}/../../orchestrator/batch_assembly.py" | base64 -w0)"
  rl_program_b64="$(gzip -9c "${LAUNCHER_DIR}/../../orchestrator/rl_program.py" | base64 -w0)"
  run_gsm8k_b64="$(gzip -9c "${LAUNCHER_DIR}/run_gsm8k_dist_grpo.py" | base64 -w0)"

  "$PYTHON" "$YAML_GEN" \
    "$YAML_DIR/jobset.cpu.yaml" \
    --jobset_name="${ORCHESTRATOR_ID}" \
    --namespace="${K8S_NAMESPACE}" \
    ${KUEUE_QUEUE_NAME:+--queue_name="${KUEUE_QUEUE_NAME}"} \
    --cpu_machine=${CPU_MACHINE} \
    --worker_container_image="${TUNIX_IMAGE}" \
    --worker_container_port="${ORCHESTRATOR_PORT}" \
    --worker_startup_command=" \
      mkdir -p /app/tunix/tunix/common /app/tunix/tunix/experimental/rollout; \
      echo '${router_replay_b64}' | base64 -d | gunzip > /app/tunix/tunix/common/router_replay.py; \
      echo '${datatypes_b64}' | base64 -d | gunzip > /app/tunix/tunix/experimental/common/datatypes.py; \
      echo '${collector_b64}' | base64 -d | gunzip > /app/tunix/tunix/experimental/rollout/collector.py; \
      echo '${agent_types_b64}' | base64 -d | gunzip > /app/tunix/tunix/rl/agentic/agents/agent_types.py; \
      echo '${collect_engine_b64}' | base64 -d | gunzip > /app/tunix/tunix/rl/agentic/trajectory/trajectory_collect_engine.py; \
      echo '${algorithm_adapter_b64}' | base64 -d | gunzip > /app/tunix/tunix/experimental/orchestrator/algorithm_adapter.py; \
      echo '${batch_assembly_b64}' | base64 -d | gunzip > /app/tunix/tunix/experimental/orchestrator/batch_assembly.py; \
      echo '${rl_program_b64}' | base64 -d | gunzip > /app/tunix/tunix/experimental/orchestrator/rl_program.py; \
      echo '${run_gsm8k_b64}' | base64 -d | gunzip > /app/tunix/tunix/experimental/examples/math_gsm8k_dist/run_gsm8k_dist_grpo.py; \
      ${HF_TOKEN:+HF_TOKEN=\"${HF_TOKEN}\"} \
      ${WANDB_API_KEY:+WANDB_API_KEY=\"${WANDB_API_KEY}\"} \
      ${WANDB_MODE:+WANDB_MODE=\"${WANDB_MODE}\"} \
      ${WANDB_ENTITY:+WANDB_ENTITY=\"${WANDB_ENTITY}\"} \
      WANDB_PROJECT=\"${WANDB_PROJECT}\" \
      WANDB_RUN_NAME=\"${WANDB_RUN_NAME}\" \
      python -m tunix.experimental.distributed.runtime.main \
        --discovery_id=${ORCHESTRATOR_ID} \
        --discovery_port=${ORCHESTRATOR_PORT} \
        --process_main=tunix.experimental.examples.math_gsm8k_dist.run_gsm8k_dist_grpo.main \
        --model_id=${MODEL_ID} \
        --tokenizer_path=${TOKENIZER_PATH} \
        --batch_size=${BATCH_SIZE} \
        --num_generations=${NUM_GENERATIONS} \
        --reward_mode=${REWARD_MODE} \
        --max_steps=${MAX_STEPS} \
        --max_prompt_length=${MAX_PROMPT_LENGTH} \
        --max_response_length=${MAX_RESPONSE_LENGTH} \
        --train_micro_batch_size=${TRAIN_MICRO_BATCH_SIZE} \
        --rollout_replicas=${ROLLOUT_REPLICAS} \
        --wandb_project=\"${WANDB_PROJECT}\" \
        --wandb_run_name=\"${WANDB_RUN_NAME}\" \
        ${WANDB_ENTITY:+--wandb_entity=\"${WANDB_ENTITY}\"} \
        --flush_metrics_every_n_steps=${FLUSH_METRICS_EVERY_N_STEPS} \
        --weight_sync_mode=${WEIGHT_SYNC_MODE} \
        --stop_workers_on_exit \
        $([[ "${RETURN_ROUTED_EXPERTS}" == "false" || "${RETURN_ROUTED_EXPERTS}" == "False" || "${RETURN_ROUTED_EXPERTS}" == "0" ]] && echo --no-return_routed_experts || echo --return_routed_experts) \
        $([[ "${USE_ROLLOUT_LOGPS}" == "false" || "${USE_ROLLOUT_LOGPS}" == "False" || "${USE_ROLLOUT_LOGPS}" == "0" ]] && echo --no-use_rollout_logps || echo --use_rollout_logps) \
        ${MAX_SEQ_TOKEN_PER_TPU:+--max_seq_token_per_tpu=${MAX_SEQ_TOKEN_PER_TPU}} \
        ${MAX_SEGMENTS_PER_PACKED_ROW:+--max_segments_per_packed_row=${MAX_SEGMENTS_PER_PACKED_ROW}} \
        ${TRAINER_MESH_FSDP:+--trainer_fsdp=${TRAINER_MESH_FSDP}} \
        ${TIS_TYPE:+--truncated_importance_sampling_type=${TIS_TYPE}} \
        ${TIS_RATIO_MIN:+--truncated_importance_sampling_ratio_min=${TIS_RATIO_MIN}} \
        ${TIS_RATIO_MAX:+--truncated_importance_sampling_ratio=${TIS_RATIO_MAX}} \
        ${SEQ_LOGPROB_ERROR_THRESHOLD:+--seq_logprob_error_threshold=${SEQ_LOGPROB_ERROR_THRESHOLD}} \
        $([[ "${OVERLONG_LOSS_MASKING}" == "true" ]] && echo --overlong_loss_masking || echo "") \
        ${LOSS_AGG_MODE:+--loss_agg_mode=${LOSS_AGG_MODE}} \
        ${EPSILON_HIGH:+--epsilon_high=${EPSILON_HIGH}} \
        ${debug_flag} \
    " \
    | apply_manifest
}

stop_trainer() {
  if [[ "$DRY_RUN" == "true" ]]; then
    echo "kubectl delete jobset ${TRAINER_ID} -n ${K8S_NAMESPACE}"
  else
    kubectl delete jobset "${TRAINER_ID}" -n "${K8S_NAMESPACE}" --ignore-not-found --wait=true
    while kubectl get jobset "${TRAINER_ID}" -n "${K8S_NAMESPACE}" &>/dev/null; do
      sleep 2
    done
  fi
}

start_trainer() {
  local extra_flags=""
  local debug_flag=""
  if [[ "${DEBUG}" == "1" || "${DEBUG}" == "true" || "${DEBUG}" == "True" ]]; then
    debug_flag="--debug"
  fi

  if [[ "${TRAINER_JOBSET_YAML}" == "jobset.pathways.yaml" ]]; then
    echo "Trainer Pathways images: server=${PATHWAYS_SERVER_IMAGE} proxy=${PATHWAYS_PROXY_IMAGE}"
  fi

  if [[ "${TRAINER_BACKEND}" == "maxtext" ]]; then
    if [[ -z "${MAXTEXT_CKPT}" ]]; then
      if [[ "${DRY_RUN}" == "true" ]]; then
        echo "Warning: TRAINER_BACKEND=maxtext without MAXTEXT_CKPT (Orbax params-only checkpoint)." >&2
      else
        echo "Error: TRAINER_BACKEND=maxtext requires MAXTEXT_CKPT (Orbax params-only checkpoint)." >&2
        exit 1
      fi
    fi
    extra_flags+=" \
      --maxtext_model_name=${MAXTEXT_MODEL_NAME} \
      ${TRAINER_PADDED_MOE_MLP_DIM:+--maxtext_padded_moe_mlp_dim=${TRAINER_PADDED_MOE_MLP_DIM}} \
      ${MAXTEXT_CKPT:+--maxtext_ckpt_path=${MAXTEXT_CKPT}} \
      --maxtext_output_directory=${MAXTEXT_OUTPUT_DIR} \
      --mesh_tp=${TRAINER_MESH_TP} \
      --mesh_expert=${TRAINER_MESH_EXPERT} \
      ${ROLLOUT_MESH_TP:+--rollout_mesh_tp=${ROLLOUT_MESH_TP}} \
      --use_weight_converter=${USE_WEIGHT_CONVERTER} \
    "
    # NOT passed: --kv_tp_size. The flag exists in this repo's
    # run_trainer_node.py but NOT in the one baked into TUNIX_IMAGE, and we
    # deliberately do not inject that file -- see the comment on the trainer
    # payload below. Re-enable together with the injection once the two are
    # reconciled; it is only needed when the rollout runs ep>1.
  fi

  local raiden_env=""
  if [[ "${WEIGHT_SYNC_MODE}" == "raiden" ]]; then
    if [[ "${TRAINER_JOBSET_YAML}" == "jobset.pathways.yaml" ]]; then
      raiden_env+=" RAIDEN_USE_FFI=1"
    fi
  fi

  # Gzipped before base64, matching the rollout payload below. Uncompressed
  # these render to ~87KB of argv and execve caps a SINGLE argument at
  # MAX_ARG_STRLEN = 128KB (not ARG_MAX = 2MB, which is never the binding limit
  # here), so there is headroom either way -- but gzip keeps it that way.
  #
  # DO NOT add run_trainer_node.py here without first reconciling it against
  # the copy inside TUNIX_IMAGE. They have diverged: the image's copy accepts
  # --max_grad_norm / --adam_b1 / --adam_b2 / --weight_decay / --learning_rate,
  # which this launcher passes, and this repo's copy has only the --optimizer_*
  # spellings. Injecting ours killed the trainer in 11s with
  #   main.py: error: unrecognized arguments: --max_grad_norm=1.0
  # and argparse exit code 2, which surfaces as a jobset crash-loop rather than
  # anything that names the real problem.
  #
  # The same hazard applies to every file here: injection silently downgrades
  # the container to whatever this checkout happens to contain. maxtext_utils.py
  # and algo_core.py are known-good because they have been injected for many
  # runs.
  local router_replay_b64 datatypes_b64 rl_common_b64 maxtext_utils_b64 algo_core_b64
  router_replay_b64="$(gzip -9c "${LAUNCHER_DIR}/../../../common/router_replay.py" | base64 -w0)"
  datatypes_b64="$(gzip -9c "${LAUNCHER_DIR}/../../common/datatypes.py" | base64 -w0)"
  rl_common_b64="$(gzip -9c "${LAUNCHER_DIR}/../../../rl/common.py" | base64 -w0)"
  maxtext_utils_b64="$(gzip -9c "${LAUNCHER_DIR}/../../../utils/maxtext_utils.py" | base64 -w0)"
  algo_core_b64="$(gzip -9c "${LAUNCHER_DIR}/../../../rl/algo_core.py" | base64 -w0)"

  "$PYTHON" "$YAML_GEN" \
    "$YAML_DIR/${TRAINER_JOBSET_YAML}" \
    --jobset_name="${TRAINER_ID}" \
    --namespace="${K8S_NAMESPACE}" \
    ${KUEUE_QUEUE_NAME:+--queue_name="${KUEUE_QUEUE_NAME}"} \
    --tpu_slice=${TRAINER_TPU_SLICE} \
    --cpu_machine=${CPU_MACHINE} \
    --pathways_server_image="${PATHWAYS_SERVER_IMAGE}" \
    --pathways_proxy_server_image="${PATHWAYS_PROXY_IMAGE}" \
    --pathways_gcs_scratch_location=${GCS_SCRATCH_LOCATION} \
    --worker_container_image="${TUNIX_IMAGE}" \
    --worker_container_port="${TRAINER_PORT}" \
    --worker_startup_command=" \
      sed -i 's/if (self.load_parameters_path or self.load_full_state_path) and not self.enable_checkpointing:/if self.load_full_state_path and not self.enable_checkpointing:/g' /app/maxtext/src/maxtext/configs/types.py /opt/venv/lib/python3.12/site-packages/maxtext/configs/types.py 2>/dev/null || true; \
      mkdir -p /app/tunix/tunix/common; \
      echo '${router_replay_b64}' | base64 -d | gunzip > /app/tunix/tunix/common/router_replay.py; \
      echo '${datatypes_b64}' | base64 -d | gunzip > /app/tunix/tunix/experimental/common/datatypes.py; \
      echo '${rl_common_b64}' | base64 -d | gunzip > /app/tunix/tunix/rl/common.py; \
      echo '${maxtext_utils_b64}' | base64 -d | gunzip > /app/tunix/tunix/utils/maxtext_utils.py; \
      PYTHONPATH=/app/tunix python -c 'from tunix.utils.maxtext_utils import apply_gdn_conv_padding_fix; apply_gdn_conv_padding_fix()'; \
      echo '${algo_core_b64}' | base64 -d | gunzip > /app/tunix/tunix/rl/algo_core.py; \
      ${HF_TOKEN:+HF_TOKEN=\"${HF_TOKEN}\"} VERIFY_WEIGHTS=${VERIFY_WEIGHTS} ENABLE_PATHWAYS_PERSISTENCE=${ENABLE_PATHWAYS_PERSISTENCE}${raiden_env} ADAM_B1=${ADAM_B1} ADAM_B2=${ADAM_B2} WEIGHT_DECAY=${WEIGHT_DECAY} MAX_GRAD_NORM=${MAX_GRAD_NORM} ${KV_TP_SIZE:+KV_TP_SIZE=${KV_TP_SIZE}} ${TRAINER_ACT_DTYPE:+TRAINER_ACT_DTYPE=${TRAINER_ACT_DTYPE}} ${TRAINER_MATMUL_PRECISION:+TRAINER_MATMUL_PRECISION=${TRAINER_MATMUL_PRECISION}} python -m tunix.experimental.distributed.runtime.main \
        --discovery_addrs=${ORCHESTRATOR_ID}:${ORCHESTRATOR_PORT} \
        --process_executor=tunix.experimental.distributed.runtime.executor.K8sExecutor \
        --process_main=tunix.experimental.examples.common.run_trainer_node.main \
        --worker_id=${TRAINER_ID} \
        --port=${TRAINER_PORT} \
        --mesh_fsdp=${TRAINER_MESH_FSDP} \
        --trainer_backend=${TRAINER_BACKEND} \
        --model_name=${MODEL_NAME} \
        --model_id=${MODEL_ID} \
        --model_dir=${MODEL_DIR} \
        --sampler_type=${SAMPLER} \
        --tokenizer_path=${TOKENIZER_PATH} \
        --max_prompt_length=${MAX_PROMPT_LENGTH} \
        --max_response_length=${MAX_RESPONSE_LENGTH} \
        --mini_batch_size=${MINI_BATCH_SIZE} \
        --train_micro_batch_size=${TRAIN_MICRO_BATCH_SIZE} \
        --eval_every_n_steps=${EVAL_EVERY_N_STEPS} \
        --max_grad_norm=${MAX_GRAD_NORM} \
        --adam_b1=${ADAM_B1} \
        --adam_b2=${ADAM_B2} \
        --weight_decay=${WEIGHT_DECAY} \
        --learning_rate=${LEARNING_RATE} \
        --lora_rank=${LORA_RANK} \
        --lora_alpha=${LORA_ALPHA} \
        --checkpoint_save_interval_steps=${CHECKPOINT_SAVE_INTERVAL_STEPS} \
        --checkpoint_max_to_keep=${CHECKPOINT_MAX_TO_KEEP} \
        --checkpoint_root_directory=${CHECKPOINT_ROOT_DIRECTORY} \
        ${extra_flags} \
        ${debug_flag} \
    " \
    | apply_manifest
}

stop_rollout_instance() {
  local target_id="$1"
  if [[ "$DRY_RUN" == "true" ]]; then
    echo "kubectl delete leaderworkerset ${target_id} -n ${K8S_NAMESPACE}"
    echo "kubectl delete jobset ${target_id} -n ${K8S_NAMESPACE}"
  else
    kubectl delete leaderworkerset "${target_id}" -n "${K8S_NAMESPACE}" --ignore-not-found --wait=true 2>/dev/null || true
    kubectl delete jobset "${target_id}" -n "${K8S_NAMESPACE}" --ignore-not-found --wait=true 2>/dev/null || true
    while kubectl get leaderworkerset "${target_id}" -n "${K8S_NAMESPACE}" &>/dev/null || kubectl get jobset "${target_id}" -n "${K8S_NAMESPACE}" &>/dev/null; do
      sleep 2
    done
  fi
}

stop_rollout() {
  for ((i = 0; i < ROLLOUT_REPLICAS; i++)); do
    local target_id="${ROLLOUT_ID}"
    if [[ $ROLLOUT_REPLICAS -gt 1 ]]; then
      target_id="${ROLLOUT_ID}-${i}"
    fi
    stop_rollout_instance "${target_id}"
  done
}

start_rollout_instance() {
  local target_id="$1"
  local extra_flags=""
  local debug_flag=""
  if [[ "${DEBUG}" == "1" || "${DEBUG}" == "true" || "${DEBUG}" == "True" ]]; then
    debug_flag="--debug"
  fi

  if [[ "${ROLLOUT_JOBSET_YAML}" == "jobset.pathways.yaml" ]]; then
    echo "Rollout Pathways images: server=${PATHWAYS_SERVER_IMAGE} proxy=${PATHWAYS_PROXY_IMAGE}"
  fi

  if [[ "${TRAINER_BACKEND}" == "maxtext" ]]; then
    extra_flags+="\
      --maxtext_model_name=${MAXTEXT_MODEL_NAME} \
      ${ROLLOUT_MAXTEXT_ATTENTION:+--maxtext_attention=${ROLLOUT_MAXTEXT_ATTENTION}} \
    "
  fi

  local raiden_env=""
  if [[ "${WEIGHT_SYNC_MODE}" == "raiden" ]]; then
    # mcJax rollout uses TCP transport (FFI disabled)
    raiden_env+=" RAIDEN_USE_FFI=0"
  fi

  # Payloads are gzipped before base64: five uncompressed sources overflow the
  # execve argv limit ("Argument list too long") when rendered into the startup
  # command. Python source compresses ~4-5x, which fits comfortably.
  local router_replay_b64 datatypes_b64 maxtext_utils_b64
  router_replay_b64="$(gzip -9c "${LAUNCHER_DIR}/../../../common/router_replay.py" | base64 -w0)"
  datatypes_b64="$(gzip -9c "${LAUNCHER_DIR}/../../common/datatypes.py" | base64 -w0)"
  maxtext_utils_b64="$(gzip -9c "${LAUNCHER_DIR}/../../../utils/maxtext_utils.py" | base64 -w0)"
  # Router replay lives on the rollout side of the wire: the worker must attach
  # the sampler's routing to the RolloutOutput, and the collect engine must
  # carry it through turn assembly. The v10 image predates both.
  local rollout_worker_b64 agent_types_b64 collect_engine_b64 rollout_node_b64 collector_b64
  collector_b64="$(gzip -9c "${LAUNCHER_DIR}/../../rollout/collector.py" | base64 -w0)"
  rollout_worker_b64="$(gzip -9c "${LAUNCHER_DIR}/../../worker/rollout_worker.py" | base64 -w0)"
  agent_types_b64="$(gzip -9c "${LAUNCHER_DIR}/../../../rl/agentic/agents/agent_types.py" | base64 -w0)"
  collect_engine_b64="$(gzip -9c "${LAUNCHER_DIR}/../../../rl/agentic/trajectory/trajectory_collect_engine.py" | base64 -w0)"
  # Carries the new --mesh_ep flag (sampler-side expert parallelism).
  rollout_node_b64="$(gzip -9c "${LAUNCHER_DIR}/../common/run_rollout_node.py" | base64 -w0)"
  # Stops vLLM inheriting top_k=20 / top_p=0.95 from the model's
  # generation_config.json. Without this the sampler draws from a top-20
  # truncated distribution while the trainer scores full-vocab, so every
  # importance ratio in the run is comparing two different policies.
  local vllm_sampler_b64 inprocess_adapter_b64
  vllm_sampler_b64="$(gzip -9c "${LAUNCHER_DIR}/../../../generate/vllm_sampler.py" | base64 -w0)"
  inprocess_adapter_b64="$(gzip -9c "${LAUNCHER_DIR}/../../rollout/inprocess_vllm_sampler_adapter.py" | base64 -w0)"

  # The routed-experts slot-group patch rewrites tpu-inference's capture path so
  # the write side keys slots off the same KV-cache group the read side uses.
  # It is only meaningful when the sampler actually captures routing, and it
  # fails closed by design if the image's tpu_runner.py does not match the
  # source it was written against -- which would otherwise abort a run that has
  # no use for it. So it is applied ONLY when replay is on.
  #
  # The GDN conv-padding fix is unconditional: it corrects sampler numerics on
  # this hybrid model regardless of replay.
  local rollout_patch_call
  if [[ "${RETURN_ROUTED_EXPERTS:-false}" == "true" ]]; then
    rollout_patch_call="from tunix.utils.maxtext_utils import apply_gdn_conv_padding_fix, apply_routed_experts_slot_group_fix; apply_gdn_conv_padding_fix(); apply_routed_experts_slot_group_fix()"
    echo "Rollout patches: gdn_conv_padding + routed_experts_slot_group (replay ON)"
  else
    rollout_patch_call="from tunix.utils.maxtext_utils import apply_gdn_conv_padding_fix; apply_gdn_conv_padding_fix()"
    echo "Rollout patches: gdn_conv_padding only (replay OFF, slot-group fix not needed)"
  fi


  "$PYTHON" "$YAML_GEN" \
    "$YAML_DIR/${ROLLOUT_JOBSET_YAML}" \
    --jobset_name="${target_id}" \
    --namespace="${K8S_NAMESPACE}" \
    ${KUEUE_QUEUE_NAME:+--queue_name="${KUEUE_QUEUE_NAME}"} \
    --tpu_slice="${ROLLOUT_TPU_SLICE}" \
    --pathways_server_image="${PATHWAYS_SERVER_IMAGE}" \
    --pathways_proxy_server_image="${PATHWAYS_PROXY_IMAGE}" \
    --pathways_gcs_scratch_location=${GCS_SCRATCH_LOCATION} \
    --worker_container_image="${TUNIX_IMAGE}" \
    --worker_container_port="${ROLLOUT_PORT}" \
    --worker_startup_command=" \
      mkdir -p /app/tunix/tunix/common /app/tunix/tunix/experimental/rollout; \
      echo '${router_replay_b64}' | base64 -d | gunzip > /app/tunix/tunix/common/router_replay.py; \
      echo '${datatypes_b64}' | base64 -d | gunzip > /app/tunix/tunix/experimental/common/datatypes.py; \
      echo '${collector_b64}' | base64 -d | gunzip > /app/tunix/tunix/experimental/rollout/collector.py; \
      echo '${maxtext_utils_b64}' | base64 -d | gunzip > /app/tunix/tunix/utils/maxtext_utils.py; \
      PYTHONPATH=/app/tunix python -c '${rollout_patch_call}' || exit 1; \
      echo '${rollout_worker_b64}' | base64 -d | gunzip > /app/tunix/tunix/experimental/worker/rollout_worker.py; \
      echo '${inprocess_adapter_b64}' | base64 -d | gunzip > /app/tunix/tunix/experimental/rollout/inprocess_vllm_sampler_adapter.py; \
      echo '${agent_types_b64}' | base64 -d | gunzip > /app/tunix/tunix/rl/agentic/agents/agent_types.py; \
      echo '${collect_engine_b64}' | base64 -d | gunzip > /app/tunix/tunix/rl/agentic/trajectory/trajectory_collect_engine.py; \
      echo '${rollout_node_b64}' | base64 -d | gunzip > /app/tunix/tunix/experimental/examples/common/run_rollout_node.py; \
      echo '${vllm_sampler_b64}' | base64 -d | gunzip > /app/tunix/tunix/generate/vllm_sampler.py; \
      ${HF_TOKEN:+HF_TOKEN=\"${HF_TOKEN}\"} SKIP_JAX_PRECOMPILE=1 VERIFY_WEIGHTS=${VERIFY_WEIGHTS}${raiden_env} ${ROLLOUT_USE_BATCHED_RPA:+USE_BATCHED_RPA_KERNEL=1} ${ROLLOUT_MAXTEXT_CKPT:+ROLLOUT_MAXTEXT_CKPT=${ROLLOUT_MAXTEXT_CKPT}} python -m tunix.experimental.distributed.runtime.main \
        --discovery_addrs=${ORCHESTRATOR_ID}:${ORCHESTRATOR_PORT} \
        --process_executor=tunix.experimental.distributed.runtime.executor.K8sExecutor \
        --process_main=tunix.experimental.examples.common.run_rollout_node.main \
        --worker_id=${target_id} \
        --port=${ROLLOUT_PORT} \
        --mesh_fsdp=${ROLLOUT_MESH_FSDP} \
        --mesh_tp=${ROLLOUT_MESH_TP} \
        --mesh_ep=${ROLLOUT_MESH_EP} \
        --model_name=${MODEL_NAME} \
        --model_id=${MODEL_ID} \
        --model_dir=${MODEL_DIR} \
        --tokenizer_path=${TOKENIZER_PATH} \
        --max_prompt_length=${MAX_PROMPT_LENGTH} \
        --max_response_length=${MAX_RESPONSE_LENGTH} \
        --sampler=${SAMPLER} \
        --lora_rank=${LORA_RANK} \
        --lora_alpha=${LORA_ALPHA} \
        --weight_sync_mode=${WEIGHT_SYNC_MODE} \
        --prefuse_moe_weights=${PREFUSE_MOE_WEIGHTS} \
        --enable_prefix_caching=${ENABLE_PREFIX_CACHING} \
        --return_routed_experts=${RETURN_ROUTED_EXPERTS} \
        --chat_parser=${CHAT_PARSER:-raw} \
        ${extra_flags} \
        ${debug_flag} \
    " \
    | apply_manifest
}

start_rollout() {
  for ((i = 0; i < ROLLOUT_REPLICAS; i++)); do
    local target_id="${ROLLOUT_ID}"
    if [[ $ROLLOUT_REPLICAS -gt 1 ]]; then
      target_id="${ROLLOUT_ID}-${i}"
    fi
    start_rollout_instance "${target_id}"
  done
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    start|stop|orchestrator|trainer|rollout)
      COMMAND="$1"
      shift
      ;;
    --debug)
      DEBUG=1
      shift
      ;;
    --no-debug)
      DEBUG=0
      shift
      ;;
    --command)
      COMMAND="$2"
      shift 2
      ;;
    --command=*)
      COMMAND="${1#*=}"
      shift
      ;;
    --image)
      TUNIX_IMAGE="$2"
      shift 2
      ;;
    --image=*)
      TUNIX_IMAGE="${1#*=}"
      shift
      ;;
    --namespace)
      K8S_NAMESPACE="$2"
      shift 2
      ;;
    --namespace=*)
      K8S_NAMESPACE="${1#*=}"
      shift
      ;;
    --queue)
      KUEUE_QUEUE_NAME="$2"
      shift 2
      ;;
    --queue=*)
      KUEUE_QUEUE_NAME="${1#*=}"
      shift
      ;;
    --dry-run|--render)
      DRY_RUN=true
      shift
      ;;
    --scratch=*|--gcs-scratch=*)
      GCS_SCRATCH_LOCATION="${1#*=}"
      shift
      ;;
    --scratch|--gcs-scratch)
      GCS_SCRATCH_LOCATION="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [start|stop|orchestrator|trainer|rollout] [options]"
      echo "Options:"
      echo "  --command <cmd>          Command to run (start, stop, orchestrator, trainer, rollout)"
      echo "  --namespace <ns>         Kubernetes namespace (default: default)"
      echo "  --queue <name>           Kueue local queue name (optional)"
      echo "  --image <image>          Container image to use"
      echo "  --dry-run, --render      Print generated YAMLs without applying"
      echo "  --scratch, --gcs-scratch GCS scratch location"
      echo "  --debug, --no-debug      Toggle debug logging (default: disabled)"
      exit 0
      ;;
    *)
      shift
      ;;
  esac
done

if [[ "$DRY_RUN" != "true" ]]; then
  ENTER_KUBE_CONTEXT=${ENTER_KUBE_CONTEXT:-"${LAUNCHER_DIR}/../common/enter_kube_context.sh"}
  source "${ENTER_KUBE_CONTEXT}"
fi

if [[ -z "$TUNIX_IMAGE" ]]; then
  echo "Error: no image set. Build one with tunix, maxtext, and" \
       "tpu-inference installed, then pass it via TUNIX_IMAGE=... or" \
       "--image=..."
  exit 1
fi

if [[ "$COMMAND" == "start" ]]; then
  stop_orchestrator
  stop_trainer
  stop_rollout
  start_orchestrator
  start_trainer
  start_rollout
elif [[ "$COMMAND" == "stop" ]]; then
  stop_orchestrator
  stop_trainer
  stop_rollout
elif [[ "$COMMAND" == "orchestrator" ]]; then
  stop_orchestrator; start_orchestrator
elif [[ "$COMMAND" == "trainer" ]]; then
  stop_trainer; start_trainer
elif [[ "$COMMAND" == "rollout" ]]; then
  stop_rollout; start_rollout
else
  echo "Error: Invalid command '$COMMAND'. Available commands: 'start', 'stop', 'orchestrator', 'trainer', 'rollout'."
  exit 1
fi
