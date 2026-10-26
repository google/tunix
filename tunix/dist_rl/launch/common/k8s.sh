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

# Shared Kubernetes / Pathways JobSet & LeaderWorkerSet launcher for
# the distributed RL launchers.
#
# Invoked by `launch.sh` (never directly) as `k8s.sh <command> [config args]`.
# Consumes the cluster/deployment environment variables exported by
# `resolve_config.py` and forwards YAML + `key=value` arguments directly to
# `run_trainer_node`, `run_rollout_node`, and the distributed orchestrator via
# `tunix.dist_rl.distributed.runtime.main`.

set -Eeuo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
require_cluster_target k8s

YAML_GEN="${YAML_GEN:-${REPO_ROOT}/tunix/dist_rl/distributed/deployment/yaml_generator.py}"
YAML_DIR="${YAML_DIR:-${REPO_ROOT}/tunix/dist_rl/distributed/deployment/yamls}"

COMMAND="$1"
shift
CONFIG_ARGS=("$@")

run_yaml_gen() {
  if [[ -n "${YAML_GEN_BIN:-}" ]]; then
    "${YAML_GEN_BIN}" "$@"
  else
    "$PYTHON_BIN" "$YAML_GEN" "$@"
  fi
}

# Normalize host-absolute `override_config_file` paths under REPO_ROOT to
# repo-relative paths (`tunix/...`) so container startup commands resolve them
# inside the GKE image. `cluster_config.dry_run` only affects this launcher and
# is not forwarded to the containers.
CONTAINER_CONFIG_ARGS=("cluster_config.run_id=${RUN_ID}")
for arg in "${CONFIG_ARGS[@]}"; do
  case "$arg" in
    override_config_file="${REPO_ROOT}/"*)
      CONTAINER_CONFIG_ARGS+=("override_config_file=${arg#"override_config_file=${REPO_ROOT}/"}")
      ;;
    cluster_config.dry_run=*)
      ;;
    *)
      CONTAINER_CONFIG_ARGS+=("$arg")
      ;;
  esac
done

format_args() {
  local out=""
  local item
  for item in "$@"; do
    out+=" $(printf '%q' "$item")"
  done
  printf '%s' "${out# }"
}

CONTAINER_ARGS_STR="$(format_args "${CONTAINER_CONFIG_ARGS[@]}")"

is_jax_cache_disabled() {
  is_true "${K8S_JAX_CACHE_DISABLE:-false}"
}

if ! is_jax_cache_disabled && [[ -z "${K8S_JAX_CACHE_ROLLOUT_GCS_DIR}" ]]; then
  K8S_JAX_CACHE_ROLLOUT_GCS_DIR=$("$PYTHON_BIN" -m tunix.dist_rl.common.gcs_cache resolve-uri 2>/dev/null || true)
fi

apply_manifest() {
  if is_true "${DRY_RUN}"; then
    echo "---"
    cat
  else
    kubectl apply -f -
  fi
}

stop_orchestrator() {
  if is_true "${DRY_RUN}"; then
    echo "kubectl delete jobset ${K8S_ORCHESTRATOR_ID} -n ${K8S_NAMESPACE}"
  else
    kubectl delete jobset "${K8S_ORCHESTRATOR_ID}" -n "${K8S_NAMESPACE}" --ignore-not-found --wait=true
    while kubectl get jobset "${K8S_ORCHESTRATOR_ID}" -n "${K8S_NAMESPACE}" &>/dev/null; do
      sleep 2
    done
  fi
}

start_orchestrator() {
  local env_prefix=""
  if [[ -n "${HF_TOKEN:-}" ]]; then
    env_prefix+="HF_TOKEN=$(printf '%q' "${HF_TOKEN}") "
  fi
  if [[ -n "${WANDB_API_KEY:-}" ]]; then
    env_prefix+="WANDB_API_KEY=$(printf '%q' "${WANDB_API_KEY}") "
  fi

  local startup_cmd="${env_prefix}python -m tunix.dist_rl.distributed.runtime.main --discovery_id=${K8S_ORCHESTRATOR_ID} --discovery_port=${ORCHESTRATOR_PORT} --process_main=${ORCHESTRATOR_MAIN} ${CONTAINER_ARGS_STR}"

  run_yaml_gen \
    "${YAML_DIR}/jobset.cpu.yaml" \
    --jobset_name="${K8S_ORCHESTRATOR_ID}" \
    --namespace="${K8S_NAMESPACE}" \
    ${K8S_KUEUE_QUEUE_NAME:+--queue_name="${K8S_KUEUE_QUEUE_NAME}"} \
    --cpu_machine="${K8S_CPU_MACHINE}" \
    --worker_container_image="${K8S_TUNIX_IMAGE}" \
    --worker_container_port="${ORCHESTRATOR_PORT}" \
    --worker_startup_command="${startup_cmd}" \
    | apply_manifest
}

stop_trainer() {
  if is_true "${DRY_RUN}"; then
    echo "kubectl delete jobset ${K8S_TRAINER_ID} -n ${K8S_NAMESPACE}"
  else
    kubectl delete jobset "${K8S_TRAINER_ID}" -n "${K8S_NAMESPACE}" --ignore-not-found --wait=true
    while kubectl get jobset "${K8S_TRAINER_ID}" -n "${K8S_NAMESPACE}" &>/dev/null; do
      sleep 2
    done
  fi
}

start_trainer() {
  if [[ "${K8S_TRAINER_JOBSET_YAML}" == "jobset.pathways.yaml" ]]; then
    echo "Trainer Pathways images: server=${K8S_PATHWAYS_SERVER_IMAGE} proxy=${K8S_PATHWAYS_PROXY_IMAGE}" >&2
  fi

  local env_prefix="VERIFY_WEIGHTS=${WEIGHT_SYNC_VERIFY_WEIGHTS} ENABLE_PATHWAYS_PERSISTENCE=${K8S_ENABLE_PATHWAYS_PERSISTENCE}"
  if [[ -n "${HF_TOKEN:-}" ]]; then
    env_prefix="HF_TOKEN=$(printf '%q' "${HF_TOKEN}") ${env_prefix}"
  fi
  if [[ -n "${CKPT_D2H_CONCURRENT_GB}" ]]; then
    env_prefix+=" CKPT_D2H_CONCURRENT_GB=${CKPT_D2H_CONCURRENT_GB}"
  fi
  if [[ "${WEIGHT_SYNC_MODE}" == "raiden" ]]; then
    if [[ -n "${WEIGHT_SYNC_RAIDEN_USE_FFI}" ]]; then
      if is_true "${WEIGHT_SYNC_RAIDEN_USE_FFI}"; then
        env_prefix+=" RAIDEN_USE_FFI=1"
      else
        env_prefix+=" RAIDEN_USE_FFI=0"
      fi
    elif [[ "${K8S_TRAINER_JOBSET_YAML}" == "jobset.pathways.yaml" ]]; then
      env_prefix+=" RAIDEN_USE_FFI=1"
    fi
  fi
  if [[ -n "${K8S_TRAINER_EXTRA_ENV}" ]]; then
    env_prefix+=" ${K8S_TRAINER_EXTRA_ENV}"
  fi

  local startup_cmd="${env_prefix} python -m tunix.dist_rl.distributed.runtime.main --discovery_addrs=${K8S_ORCHESTRATOR_ID}:${ORCHESTRATOR_PORT} --process_executor=tunix.dist_rl.distributed.runtime.executor.K8sExecutor --process_main=${TRAINER_MAIN} ${CONTAINER_ARGS_STR}"

  run_yaml_gen \
    "${YAML_DIR}/${K8S_TRAINER_JOBSET_YAML}" \
    --jobset_name="${K8S_TRAINER_ID}" \
    --namespace="${K8S_NAMESPACE}" \
    ${K8S_KUEUE_QUEUE_NAME:+--queue_name="${K8S_KUEUE_QUEUE_NAME}"} \
    --tpu_slice="${K8S_TRAINER_TPU_SLICE}" \
    --cpu_machine="${K8S_CPU_MACHINE}" \
    --pathways_server_image="${K8S_PATHWAYS_SERVER_IMAGE}" \
    --pathways_proxy_server_image="${K8S_PATHWAYS_PROXY_IMAGE}" \
    --pathways_proxy_memory_limit="${K8S_PATHWAYS_PROXY_MEMORY_LIMIT}" \
    --pathways_proxy_memory="${K8S_PATHWAYS_PROXY_MEMORY}" \
    --pathways_rm_memory="${K8S_PATHWAYS_RM_MEMORY}" \
    --user_container_memory="${K8S_USER_CONTAINER_MEMORY}" \
    --user_container_memory_limit="${K8S_USER_CONTAINER_MEMORY_LIMIT}" \
    --pathways_worker_memory="${K8S_PATHWAYS_WORKER_MEMORY}" \
    ${TRAINER_IFRT_PROXY_LARGE_TRANSFER_THRESHOLD_BYTES:+--ifrt_proxy_large_transfer_threshold_bytes="${TRAINER_IFRT_PROXY_LARGE_TRANSFER_THRESHOLD_BYTES}"} \
    --pathways_gcs_scratch_location="${K8S_GCS_SCRATCH_LOCATION}" \
    --worker_container_image="${K8S_TUNIX_IMAGE}" \
    --worker_container_port="${TRAINER_PORT}" \
    --worker_startup_command="${startup_cmd}" \
    | apply_manifest
}

stop_rollout_instance() {
  local target_id="$1"
  local resource_kind="jobset"
  if [[ "${K8S_ROLLOUT_JOBSET_YAML}" =~ ^leaderworkerset ]]; then
    resource_kind="leaderworkerset"
  fi
  if is_true "${DRY_RUN}"; then
    echo "kubectl delete ${resource_kind} ${target_id} -n ${K8S_NAMESPACE}"
  else
    kubectl delete "${resource_kind}" "${target_id}" -n "${K8S_NAMESPACE}" --ignore-not-found --wait=true
    while kubectl get "${resource_kind}" "${target_id}" -n "${K8S_NAMESPACE}" &>/dev/null; do
      sleep 2
    done
  fi
}

stop_rollout() {
  local i
  for ((i = 0; i < K8S_ROLLOUT_REPLICAS; i++)); do
    local target_id="${K8S_ROLLOUT_ID}"
    if (( K8S_ROLLOUT_REPLICAS > 1 )); then
      target_id="${K8S_ROLLOUT_ID}-${i}"
    fi
    stop_rollout_instance "${target_id}"
  done
}

start_rollout_instance() {
  local target_id="$1"
  if [[ "${K8S_ROLLOUT_JOBSET_YAML}" == "jobset.pathways.yaml" ]]; then
    echo "Rollout Pathways images: server=${K8S_PATHWAYS_SERVER_IMAGE} proxy=${K8S_PATHWAYS_PROXY_IMAGE}" >&2
  fi

  local env_prefix="SKIP_JAX_PRECOMPILE=${K8S_ROLLOUT_SKIP_JAX_PRECOMPILE} VERIFY_WEIGHTS=${WEIGHT_SYNC_VERIFY_WEIGHTS} ROLLOUT_TPU_SLICE=$(printf '%q' "${K8S_ROLLOUT_TPU_SLICE}")"
  if [[ -n "${HF_TOKEN:-}" ]]; then
    env_prefix="HF_TOKEN=$(printf '%q' "${HF_TOKEN}") ${env_prefix}"
  fi
  if [[ "${WEIGHT_SYNC_MODE}" == "raiden" ]]; then
    env_prefix+=" RAIDEN_USE_FFI=0"
  fi
  if [[ -n "${VLLM_ADDITIONAL_CONFIG_JSON}" ]]; then
    env_prefix+=" VLLM_ADDITIONAL_CONFIG_JSON=$(printf '%q' "${VLLM_ADDITIONAL_CONFIG_JSON}")"
  fi
  if [[ -n "${SANDBOX_NAMESPACE}" ]]; then
    env_prefix+=" NAMESPACE=$(printf '%q' "${SANDBOX_NAMESPACE}")"
    if [[ -n "${SANDBOX_NODE_SELECTOR_KEY}" && -n "${SANDBOX_NODE_SELECTOR_VAL}" ]]; then
      env_prefix+=" NODE_SELECTOR_KEY=$(printf '%q' "${SANDBOX_NODE_SELECTOR_KEY}") NODE_SELECTOR_VAL=$(printf '%q' "${SANDBOX_NODE_SELECTOR_VAL}")"
    fi
  fi

  if is_jax_cache_disabled; then
    env_prefix+=" DISABLE_JAX_CACHE=1"
  else
    if [[ -n "${K8S_JAX_CACHE_ROLLOUT_GCS_DIR}" ]]; then
      env_prefix+=" ROLLOUT_JAX_CACHE_GCS_DIR=$(printf '%q' "${K8S_JAX_CACHE_ROLLOUT_GCS_DIR}")"
    fi
    if [[ -n "${K8S_JAX_CACHE_GCS_DIR}" ]]; then
      env_prefix+=" JAX_CACHE_GCS_DIR=$(printf '%q' "${K8S_JAX_CACHE_GCS_DIR}")"
    fi
    if [[ -n "${K8S_JAX_CACHE_BUCKET}" ]]; then
      env_prefix+=" JAX_CACHE_BUCKET=$(printf '%q' "${K8S_JAX_CACHE_BUCKET}")"
    fi
    if [[ -n "${K8S_JAX_CACHE_SAVE}" ]]; then
      env_prefix+=" SAVE_JAX_CACHE=$(printf '%q' "${K8S_JAX_CACHE_SAVE}")"
    fi
  fi

  local ray_head_pre_init_cmd="export JAX_COMPILATION_CACHE_DIR=\"${K8S_JAX_CACHE_LOCAL_DIR}\" VLLM_XLA_CACHE_PATH=\"${K8S_JAX_CACHE_LOCAL_DIR}\" && mkdir -p \"${K8S_JAX_CACHE_LOCAL_DIR}\""
  local ray_worker_pre_init_cmd="${ray_head_pre_init_cmd}"
  if ! is_jax_cache_disabled && [[ -n "${K8S_JAX_CACHE_ROLLOUT_GCS_DIR}" ]]; then
    ray_worker_pre_init_cmd+=" && (python3 -m tunix.dist_rl.common.gcs_cache download \"${K8S_JAX_CACHE_LOCAL_DIR}\" \"${K8S_JAX_CACHE_ROLLOUT_GCS_DIR}\" || true)"
  fi

  local startup_cmd="${env_prefix} python -m tunix.dist_rl.distributed.runtime.main --discovery_addrs=${K8S_ORCHESTRATOR_ID}:${ORCHESTRATOR_PORT} --process_executor=tunix.dist_rl.distributed.runtime.executor.K8sExecutor --process_main=${ROLLOUT_MAIN} ${CONTAINER_ARGS_STR} cluster_config.k8s.rollout_id=$(printf '%q' "${target_id}")"

  run_yaml_gen \
    "${YAML_DIR}/${K8S_ROLLOUT_JOBSET_YAML}" \
    --jobset_name="${target_id}" \
    --namespace="${K8S_NAMESPACE}" \
    ${K8S_KUEUE_QUEUE_NAME:+--queue_name="${K8S_KUEUE_QUEUE_NAME}"} \
    --tpu_slice="${K8S_ROLLOUT_TPU_SLICE}" \
    --pathways_server_image="${K8S_PATHWAYS_SERVER_IMAGE}" \
    --pathways_proxy_server_image="${K8S_PATHWAYS_PROXY_IMAGE}" \
    --pathways_gcs_scratch_location="${K8S_GCS_SCRATCH_LOCATION}" \
    --worker_container_image="${K8S_TUNIX_IMAGE}" \
    --worker_container_port="${ROLLOUT_PORT}" \
    --ray_head_pre_init_command="${ray_head_pre_init_cmd}" \
    --ray_worker_pre_init_command="${ray_worker_pre_init_cmd}" \
    --worker_startup_command="${startup_cmd}" \
    | apply_manifest
}

start_rollout() {
  if ! is_jax_cache_disabled && [[ -n "${K8S_JAX_CACHE_ROLLOUT_GCS_DIR}" ]]; then
    echo "[launcher] Rollout JAX cache GCS: ${K8S_JAX_CACHE_ROLLOUT_GCS_DIR} (save=${K8S_JAX_CACHE_SAVE:-true})" >&2
  fi
  local i
  for ((i = 0; i < K8S_ROLLOUT_REPLICAS; i++)); do
    local target_id="${K8S_ROLLOUT_ID}"
    if (( K8S_ROLLOUT_REPLICAS > 1 )); then
      target_id="${K8S_ROLLOUT_ID}-${i}"
    fi
    start_rollout_instance "${target_id}"
  done
}

if ! is_true "${DRY_RUN}"; then
  ENTER_KUBE_CONTEXT="${ENTER_KUBE_CONTEXT:-${LAUNCH_DIR}/enter_kube_context.sh}"
  if [[ -f "${ENTER_KUBE_CONTEXT}" ]]; then
    source "${ENTER_KUBE_CONTEXT}"
  fi
fi

if [[ -z "${K8S_TUNIX_IMAGE}" ]]; then
  echo "Error: cluster_config.k8s.tunix_image is empty." >&2
  exit 1
fi

case "${COMMAND}" in
  start)
    stop_orchestrator
    stop_trainer
    stop_rollout
    start_orchestrator
    start_trainer
    start_rollout
    ;;
  stop)
    stop_orchestrator
    stop_trainer
    stop_rollout
    ;;
  orchestrator)
    stop_orchestrator
    start_orchestrator
    ;;
  trainer)
    stop_trainer
    start_trainer
    ;;
  rollout)
    stop_rollout
    start_rollout
    ;;
  *)
    echo "Error: Invalid command '${COMMAND}'. Available commands: 'start', 'stop', 'orchestrator', 'trainer', 'rollout'." >&2
    exit 1
    ;;
esac
