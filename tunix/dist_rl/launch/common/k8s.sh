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
# `otel` / `stop-otel` deploy / delete the OpenTelemetry collector that the
# nodes export metrics and lineage event logs to when `otel_config.enable=true`.
# Consumes the cluster/deployment environment variables exported by
# `resolve_config.py` and forwards YAML + `key=value` arguments directly to
# the `tunix.dist_rl.launch.nodes` trainer, rollout, and orchestrator mains via
# `tunix.dist_rl.distributed.runtime.main`.

set -Eeuo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
require_cluster_target k8s

YAML_GEN="${YAML_GEN:-${REPO_ROOT}/tunix/dist_rl/distributed/deployment/yaml_generator.py}"
YAML_DIR="${YAML_DIR:-${REPO_ROOT}/tunix/dist_rl/distributed/deployment/yamls}"
OTEL_DASHBOARD="${OTEL_DASHBOARD:-${REPO_ROOT}/tunix/dist_rl/distributed/deployment/otel_dashboard.py}"

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

run_otel_dashboard() {
  if [[ -n "${OTEL_DASHBOARD_BIN:-}" ]]; then
    "${OTEL_DASHBOARD_BIN}" "$@"
  else
    "$PYTHON_BIN" "$OTEL_DASHBOARD" "$@"
  fi
}

# Ensures the shared Cloud Monitoring template dashboard exists in the caller's
# GCP project and logs its URL to stderr (filtered to `$1` when non-empty).
# Runs only on the launcher host without mutating `CONTAINER_CONFIG_ARGS`, so
# every pod keeps the exact same `config_sha256` even across worker restarts.
ensure_otel_dashboard() {
  local filter_run_id="$1"
  if is_true "${DRY_RUN}" || ! is_true "${OTEL_GOOGLECLOUD}"; then
    return 0
  fi
  local url
  if url="$(run_otel_dashboard --ensure \
      --dashboard_id="${OTEL_DASHBOARD_ID}" \
      --project="${OTEL_DASHBOARD_PROJECT}" \
      --run_id="${filter_run_id}")"; then
    if [[ -n "${url}" ]]; then
      echo "[launcher] OTel dashboard: ${url}" >&2
    fi
  else
    echo "[launcher] Warning: could not ensure Cloud Monitoring dashboard; continuing without dashboard URL." >&2
  fi
}

# Normalize host-absolute `override_config_file` paths under REPO_ROOT to
# repo-relative paths (`tunix/...`) so container startup commands resolve them
# inside the GKE image. `cluster_config.dry_run` only affects this launcher and
# is not forwarded to the containers.
#
# Every pod gets these same arguments, so every pod resolves the same config
# (and logs the same `config_sha256`): the run id and the resource ids are
# pinned to the launcher's values, where each pod would otherwise derive the ids
# from the container's `$USER`. A rollout replica gets its index from the
# environment instead (`start_rollout_instance`).
CONTAINER_CONFIG_ARGS=(
  "cluster_config.run_id=${RUN_ID}"
  "cluster_config.k8s.orchestrator_id=${K8S_ORCHESTRATOR_ID}"
  "cluster_config.k8s.trainer_id=${K8S_TRAINER_ID}"
  "cluster_config.k8s.rollout_id=${K8S_ROLLOUT_ID}"
  "cluster_config.k8s.inference_id=${K8S_INFERENCE_ID}"
  "cluster_config.k8s.otel_id=${K8S_OTEL_ID}"
)
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

# Prints generator flag `$2` when the boolean config value `$1` is true. Used
# unquoted, so a false value adds no argument.
bool_flag() {
  if is_true "$1"; then
    printf '%s\n' "$2"
  fi
}

CONTAINER_ARGS_STR="$(format_args "${CONTAINER_CONFIG_ARGS[@]}")"

# Sandbox environment of the orchestrator and rollout processes: the
# `kubernetes_config` placement and `ORCHESTRATOR_ID`, which keys the job-scoped
# `SandboxFleet` they share. Placed before `ROLE_ENV_*` so role overrides take
# precedence.
SANDBOX_ENV_PREFIX="ORCHESTRATOR_ID=$(printf '%q' "${K8S_ORCHESTRATOR_ID}") "
if [[ -n "${SANDBOX_NAMESPACE}" ]]; then
  SANDBOX_ENV_PREFIX+="NAMESPACE=$(printf '%q' "${SANDBOX_NAMESPACE}") "
  if [[ -n "${SANDBOX_NODE_SELECTOR_KEY}" && -n "${SANDBOX_NODE_SELECTOR_VAL}" ]]; then
    SANDBOX_ENV_PREFIX+="NODE_SELECTOR_KEY=$(printf '%q' "${SANDBOX_NODE_SELECTOR_KEY}") NODE_SELECTOR_VAL=$(printf '%q' "${SANDBOX_NODE_SELECTOR_VAL}") "
  fi
fi

# Manifest generator flags shared by the trainer and rollout JobSets: the
# Raiden variables of the TPU containers, the TPU reservation, and the Pathways
# worker, proxy and sidecar settings (`weight_sync_config.raiden`,
# `cluster_config.k8s.tpu_reservation`, `cluster_config.pathways`).
TPU_JOBSET_GEN_ARGS=(
  "--raiden_data_nics=${WEIGHT_SYNC_RAIDEN_DATA_NICS}"
  "--raiden_broadcast_k=${WEIGHT_SYNC_RAIDEN_BROADCAST_K}"
  "--tpu_reservation=${K8S_TPU_RESERVATION}"
  "--pathways_worker_env_json=${K8S_PATHWAYS_WORKER_ENV_JSON}"
  "--pathways_proxy_extra_args=${K8S_PATHWAYS_PROXY_EXTRA_ARGS}"
  "--colocated_python_sidecar_image=${K8S_PATHWAYS_SIDECAR_IMAGE}"
  "--colocated_python_sidecar_memory=${K8S_PATHWAYS_SIDECAR_MEMORY}"
  "--colocated_python_sidecar_memory_limit=${K8S_PATHWAYS_SIDECAR_MEMORY_LIMIT}"
)
if is_true "${WEIGHT_SYNC_RAIDEN_MULTI_NUMA}"; then
  TPU_JOBSET_GEN_ARGS+=(--enable_multi_numa)
fi
if is_true "${K8S_PATHWAYS_SIDECAR_SHM}"; then
  TPU_JOBSET_GEN_ARGS+=(--colocated_python_sidecar_shm)
else
  TPU_JOBSET_GEN_ARGS+=(--no-colocated_python_sidecar_shm)
fi

is_jax_cache_disabled() {
  is_true "${K8S_JAX_CACHE_DISABLE:-false}"
}

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

# TODO(noghabi): Pass HF_TOKEN and WANDB_API_KEY to the pods through a
# Kubernetes Secret. `start_orchestrator`, `start_trainer`, and
# `start_rollout_instance` write them into the container commands, so they
# show up in the JobSet / LeaderWorkerSet specs and in `--dry-run` output.
start_orchestrator() {
  local env_prefix=""
  if [[ -n "${HF_TOKEN:-}" ]]; then
    env_prefix+="HF_TOKEN=$(printf '%q' "${HF_TOKEN}") "
  fi
  if [[ -n "${WANDB_API_KEY:-}" ]]; then
    env_prefix+="WANDB_API_KEY=$(printf '%q' "${WANDB_API_KEY}") "
  fi
  env_prefix+="${SANDBOX_ENV_PREFIX}"
  if [[ -n "${ROLE_ENV_ORCHESTRATOR}" ]]; then
    env_prefix+="${ROLE_ENV_ORCHESTRATOR} "
  fi

  local startup_cmd="${env_prefix}python -m tunix.dist_rl.distributed.runtime.main --discovery_id=${K8S_ORCHESTRATOR_ID} --discovery_port=${ORCHESTRATOR_PORT} --process_main=${ORCHESTRATOR_MAIN} ${CONTAINER_ARGS_STR}"

  run_yaml_gen \
    "${YAML_DIR}/jobset.cpu.yaml" \
    --jobset_name="${K8S_ORCHESTRATOR_ID}" \
    --namespace="${K8S_NAMESPACE}" \
    --queue_name="${K8S_KUEUE_QUEUE_NAME}" \
    $(bool_flag "${K8S_PREEMPTIBLE}" --preemptible) \
    ${K8S_PRIORITY_CLASS:+--priority_class="${K8S_PRIORITY_CLASS}"} \
    ${K8S_SERVICE_ACCOUNT:+--service_account="${K8S_SERVICE_ACCOUNT}"} \
    ${K8S_TERMINATION_GRACE_SECONDS:+--termination_grace_seconds="${K8S_TERMINATION_GRACE_SECONDS}"} \
    $(bool_flag "${K8S_TEARDOWN_GANG_ON_EXIT}" "--gang_id=${RUN_ID}") \
    $(bool_flag "${K8S_TEARDOWN_GANG_ON_EXIT}" --teardown_gang_on_exit) \
    --cpu_machine="${K8S_CPU_MACHINE}" \
    ${K8S_CPU_NODEPOOL:+--cpu_nodepool="${K8S_CPU_NODEPOOL}"} \
    ${K8S_CPU_MEMORY:+--cpu_memory="${K8S_CPU_MEMORY}"} \
    ${K8S_CPU_MEMORY_LIMIT:+--cpu_memory_limit="${K8S_CPU_MEMORY_LIMIT}"} \
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

  local enable_pathways_persistence=0
  if is_true "${K8S_ENABLE_PATHWAYS_PERSISTENCE}"; then
    enable_pathways_persistence=1
  fi
  local env_prefix="ENABLE_PATHWAYS_PERSISTENCE=${enable_pathways_persistence}"
  if [[ -n "${HF_TOKEN:-}" ]]; then
    env_prefix="HF_TOKEN=$(printf '%q' "${HF_TOKEN}") ${env_prefix}"
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
  if [[ -n "${ROLE_ENV_TRAINER}" ]]; then
    env_prefix+=" ${ROLE_ENV_TRAINER}"
  fi

  local startup_cmd="${env_prefix} python -m tunix.dist_rl.distributed.runtime.main --discovery_addrs=${K8S_ORCHESTRATOR_ID}:${ORCHESTRATOR_PORT} --process_executor=tunix.dist_rl.distributed.runtime.executor.K8sExecutor --process_main=${TRAINER_MAIN} ${CONTAINER_ARGS_STR}"

  run_yaml_gen \
    "${YAML_DIR}/${K8S_TRAINER_JOBSET_YAML}" \
    --jobset_name="${K8S_TRAINER_ID}" \
    --namespace="${K8S_NAMESPACE}" \
    --queue_name="${K8S_KUEUE_QUEUE_NAME}" \
    $(bool_flag "${K8S_PREEMPTIBLE}" --preemptible) \
    ${K8S_PRIORITY_CLASS:+--priority_class="${K8S_PRIORITY_CLASS}"} \
    ${K8S_SERVICE_ACCOUNT:+--service_account="${K8S_SERVICE_ACCOUNT}"} \
    ${K8S_TERMINATION_GRACE_SECONDS:+--termination_grace_seconds="${K8S_TERMINATION_GRACE_SECONDS}"} \
    $(bool_flag "${K8S_TEARDOWN_GANG_ON_EXIT}" "--gang_id=${RUN_ID}") \
    $(bool_flag "${K8S_FAIL_FAST}" --fail_fast) \
    --startup_retries="${K8S_STARTUP_RETRIES}" \
    $(bool_flag "${K8S_USE_DYNAMIC_SLICING}" --use_dynamic_slicing) \
    ${K8S_HEAD_NODEPOOL:+--head_nodepool="${K8S_HEAD_NODEPOOL}"} \
    ${K8S_HEAD_EXCLUDED_NODEPOOLS:+--head_excluded_nodepools="${K8S_HEAD_EXCLUDED_NODEPOOLS}"} \
    --tpu_slice="${K8S_TRAINER_TPU_SLICE}" \
    --cpu_machine="${K8S_CPU_MACHINE}" \
    --pathways_server_image="${K8S_PATHWAYS_SERVER_IMAGE}" \
    --pathways_proxy_server_image="${K8S_PATHWAYS_PROXY_IMAGE}" \
    --pathways_proxy_memory_limit="${K8S_PATHWAYS_PROXY_MEMORY_LIMIT}" \
    --pathways_proxy_memory="${K8S_PATHWAYS_PROXY_MEMORY}" \
    --pathways_rm_memory="${K8S_PATHWAYS_RM_MEMORY}" \
    --pathways_rm_memory_limit="${K8S_PATHWAYS_RM_MEMORY_LIMIT}" \
    --user_container_memory="${K8S_TRAINER_MEMORY}" \
    --user_container_memory_limit="${K8S_TRAINER_MEMORY_LIMIT}" \
    --pathways_worker_memory="${K8S_PATHWAYS_SERVER_MEMORY}" \
    --pathways_worker_memory_limit="${K8S_PATHWAYS_SERVER_MEMORY_LIMIT}" \
    ${K8S_PATHWAYS_IFRT_THRESHOLD_BYTES:+--ifrt_proxy_large_transfer_threshold_bytes="${K8S_PATHWAYS_IFRT_THRESHOLD_BYTES}"} \
    --ckpt_gcsfuse_bucket="${K8S_PATHWAYS_CKPT_GCSFUSE_BUCKET}" \
    --ckpt_gcsfuse_mount_path="${K8S_PATHWAYS_CKPT_GCSFUSE_MOUNT_PATH}" \
    --ckpt_gcsfuse_mount_options="${K8S_PATHWAYS_CKPT_GCSFUSE_MOUNT_OPTIONS}" \
    --pathways_gcs_scratch_location="${K8S_GCS_SCRATCH_LOCATION}" \
    "${TPU_JOBSET_GEN_ARGS[@]}" \
    --worker_container_image="${K8S_TUNIX_IMAGE}" \
    --worker_container_port="${TRAINER_PORT}" \
    --worker_startup_command="${startup_cmd}" \
    | apply_manifest
}

stop_rollout() {
  local replicas=${ROLLOUT_REPLICAS:-1}
  if (( replicas <= 0 )); then
    return 0
  fi
  local target_ids=()
  local i
  for ((i = 0; i < replicas; i++)); do
    if (( replicas > 1 )); then
      target_ids+=("${K8S_ROLLOUT_ID}-${i}")
    else
      target_ids+=("${K8S_ROLLOUT_ID}")
    fi
  done
  local resource_kind="jobset"
  if [[ "${K8S_ROLLOUT_JOBSET_YAML}" =~ ^leaderworkerset ]]; then
    resource_kind="leaderworkerset"
  fi
  if is_true "${DRY_RUN}"; then
    echo "kubectl delete ${resource_kind} ${target_ids[*]} -n ${K8S_NAMESPACE}"
  else
    kubectl delete "${resource_kind}" "${target_ids[@]}" -n "${K8S_NAMESPACE}" --ignore-not-found --wait=true
  fi
}

render_rollout_yaml() {
  local target_id="$1"
  local replica="$2"

  local env_prefix=""
  if (( ROLLOUT_REPLICAS > 1 )); then
    # The node appends the index to `cluster_config.k8s.rollout_id`
    # (`node_utils.worker_id`), so it registers as `target_id`.
    env_prefix+=" DIST_RL_ROLLOUT_REPLICA=${replica}"
  fi
  if [[ -n "${HF_TOKEN:-}" ]]; then
    env_prefix="HF_TOKEN=$(printf '%q' "${HF_TOKEN}") ${env_prefix}"
  fi
  if [[ "${WEIGHT_SYNC_MODE}" == "raiden" ]]; then
    env_prefix+=" RAIDEN_USE_FFI=0"
  fi
  env_prefix+=" ${SANDBOX_ENV_PREFIX% }"

  if is_jax_cache_disabled; then
    env_prefix+=" DISABLE_JAX_CACHE=1 VLLM_DISABLE_COMPILE_CACHE=1"
  else
    env_prefix+=" LOCAL_JAX_CACHE_DIR=$(printf '%q' "${K8S_JAX_CACHE_LOCAL_DIR}")"
    # `resolve_config` derives the URI from `cluster_config.k8s.jax_cache` and
    # the rollout config; it is empty without a GCS cache.
    if [[ -n "${K8S_JAX_CACHE_ROLLOUT_GCS_DIR}" ]]; then
      env_prefix+=" ROLLOUT_JAX_CACHE_GCS_DIR=$(printf '%q' "${K8S_JAX_CACHE_ROLLOUT_GCS_DIR}")"
    fi
    if [[ -n "${K8S_JAX_CACHE_SAVE}" ]]; then
      env_prefix+=" SAVE_JAX_CACHE=$(printf '%q' "${K8S_JAX_CACHE_SAVE}")"
    fi
  fi
  if [[ -n "${ROLE_ENV_ROLLOUT}" ]]; then
    env_prefix+=" ${ROLE_ENV_ROLLOUT}"
  fi

  local ray_head_pre_init_cmd=""
  local ray_worker_pre_init_cmd=""
  if is_jax_cache_disabled; then
    ray_head_pre_init_cmd="export VLLM_DISABLE_COMPILE_CACHE=1"
    ray_worker_pre_init_cmd="${ray_head_pre_init_cmd}"
  else
    local local_dir
    local_dir=$(printf '%q' "${K8S_JAX_CACHE_LOCAL_DIR}")
    ray_head_pre_init_cmd="export JAX_COMPILATION_CACHE_DIR=${local_dir} VLLM_XLA_CACHE_PATH=${local_dir} && mkdir -p ${local_dir}"
    ray_worker_pre_init_cmd="${ray_head_pre_init_cmd}"
    if [[ -n "${K8S_JAX_CACHE_ROLLOUT_GCS_DIR}" ]]; then
      ray_worker_pre_init_cmd+=" && (python3 -m tunix.dist_rl.common.gcs_cache download ${local_dir} $(printf '%q' "${K8S_JAX_CACHE_ROLLOUT_GCS_DIR}") || true)"
    fi
  fi

  # Multi-host Ray rollouts start Ray on every host with these commands before
  # `startup_cmd`, so the Ray actors inherit the role env from them. It comes
  # last, as in `startup_cmd`, so it overrides the JAX cache settings above.
  if [[ -n "${ROLE_ENV_ROLLOUT}" ]]; then
    ray_head_pre_init_cmd+=" && export ${ROLE_ENV_ROLLOUT}"
    ray_worker_pre_init_cmd+=" && export ${ROLE_ENV_ROLLOUT}"
  fi

  env_prefix="${env_prefix# }"
  local startup_cmd="${env_prefix} python -m tunix.dist_rl.distributed.runtime.main --discovery_addrs=${K8S_ORCHESTRATOR_ID}:${ORCHESTRATOR_PORT} --process_executor=tunix.dist_rl.distributed.runtime.executor.K8sExecutor --process_main=${ROLLOUT_MAIN} ${CONTAINER_ARGS_STR}"

  run_yaml_gen \
    "${YAML_DIR}/${K8S_ROLLOUT_JOBSET_YAML}" \
    --jobset_name="${target_id}" \
    --namespace="${K8S_NAMESPACE}" \
    --queue_name="${K8S_KUEUE_QUEUE_NAME}" \
    $(bool_flag "${K8S_PREEMPTIBLE}" --preemptible) \
    ${K8S_PRIORITY_CLASS:+--priority_class="${K8S_PRIORITY_CLASS}"} \
    ${K8S_SERVICE_ACCOUNT:+--service_account="${K8S_SERVICE_ACCOUNT}"} \
    ${K8S_TERMINATION_GRACE_SECONDS:+--termination_grace_seconds="${K8S_TERMINATION_GRACE_SECONDS}"} \
    $(bool_flag "${K8S_TEARDOWN_GANG_ON_EXIT}" "--gang_id=${RUN_ID}") \
    $(bool_flag "${K8S_FAIL_FAST}" --fail_fast) \
    --startup_retries="${K8S_STARTUP_RETRIES}" \
    $(bool_flag "${K8S_USE_DYNAMIC_SLICING}" --use_dynamic_slicing) \
    ${K8S_HEAD_NODEPOOL:+--head_nodepool="${K8S_HEAD_NODEPOOL}"} \
    ${K8S_HEAD_EXCLUDED_NODEPOOLS:+--head_excluded_nodepools="${K8S_HEAD_EXCLUDED_NODEPOOLS}"} \
    --tpu_slice="${K8S_ROLLOUT_TPU_SLICE}" \
    --cpu_machine="${K8S_CPU_MACHINE}" \
    --pathways_server_image="${K8S_PATHWAYS_SERVER_IMAGE}" \
    --pathways_proxy_server_image="${K8S_PATHWAYS_PROXY_IMAGE}" \
    --pathways_proxy_memory_limit="${K8S_PATHWAYS_PROXY_MEMORY_LIMIT}" \
    --pathways_proxy_memory="${K8S_PATHWAYS_PROXY_MEMORY}" \
    --pathways_rm_memory="${K8S_PATHWAYS_RM_MEMORY}" \
    --pathways_rm_memory_limit="${K8S_PATHWAYS_RM_MEMORY_LIMIT}" \
    --user_container_memory="${K8S_ROLLOUT_MEMORY}" \
    --user_container_memory_limit="${K8S_ROLLOUT_MEMORY_LIMIT}" \
    --pathways_worker_memory="${K8S_PATHWAYS_SERVER_MEMORY}" \
    --pathways_worker_memory_limit="${K8S_PATHWAYS_SERVER_MEMORY_LIMIT}" \
    --pathways_gcs_scratch_location="${K8S_GCS_SCRATCH_LOCATION}" \
    "${TPU_JOBSET_GEN_ARGS[@]}" \
    --worker_container_image="${K8S_TUNIX_IMAGE}" \
    --worker_container_port="${ROLLOUT_PORT}" \
    --ray_head_pre_init_command="${ray_head_pre_init_cmd}" \
    --ray_worker_pre_init_command="${ray_worker_pre_init_cmd}" \
    --worker_startup_command="${startup_cmd}"
}

start_rollout() {
  local replicas=${ROLLOUT_REPLICAS:-1}
  if (( replicas <= 0 )); then
    return 0
  fi
  if [[ "${K8S_ROLLOUT_JOBSET_YAML}" == "jobset.pathways.yaml" ]]; then
    echo "Rollout Pathways images: server=${K8S_PATHWAYS_SERVER_IMAGE} proxy=${K8S_PATHWAYS_PROXY_IMAGE}" >&2
  fi
  if ! is_jax_cache_disabled && [[ -n "${K8S_JAX_CACHE_ROLLOUT_GCS_DIR}" ]]; then
    echo "[launcher] Rollout JAX cache GCS: ${K8S_JAX_CACHE_ROLLOUT_GCS_DIR} (save=${K8S_JAX_CACHE_SAVE:-true})" >&2
  fi

  local placeholder_id="__TUNIX_ROLLOUT_ID__"
  local placeholder_replica="__TUNIX_ROLLOUT_REPLICA__"
  local manifest_template
  manifest_template="$(render_rollout_yaml "${placeholder_id}" "${placeholder_replica}")" || return $?

  local i
  for ((i = 0; i < replicas; i++)); do
    local target_id="${K8S_ROLLOUT_ID}"
    if (( replicas > 1 )); then
      target_id="${K8S_ROLLOUT_ID}-${i}"
    fi
    if (( i > 0 )); then
      echo "---"
    fi
    local rendered="${manifest_template//${placeholder_id}/${target_id}}"
    printf '%s\n' "${rendered//${placeholder_replica}/${i}}"
  done | apply_manifest
}

stop_otel() {
  if is_true "${DRY_RUN}"; then
    echo "kubectl delete deployment ${K8S_OTEL_ID} -n ${K8S_NAMESPACE}"
    echo "kubectl delete service ${K8S_OTEL_ID} -n ${K8S_NAMESPACE}"
    echo "kubectl delete configmap ${K8S_OTEL_ID}-config -n ${K8S_NAMESPACE}"
  else
    kubectl delete deployment "${K8S_OTEL_ID}" -n "${K8S_NAMESPACE}" --ignore-not-found --wait=true
    kubectl delete service "${K8S_OTEL_ID}" -n "${K8S_NAMESPACE}" --ignore-not-found
    kubectl delete configmap "${K8S_OTEL_ID}-config" -n "${K8S_NAMESPACE}" --ignore-not-found
  fi
}

# Deploys the OpenTelemetry collector: the Deployment and Service in
# `jobset.otel.yaml`, configured by `otel-collector.yaml`. The nodes export to
# it when launched with `otel_config.enable=true`.
start_otel() {
  local otel_gc_flag="--no-otel_googlecloud"
  if is_true "${OTEL_GOOGLECLOUD}"; then
    otel_gc_flag="--otel_googlecloud"
  fi
  if is_true "${DRY_RUN}"; then
    echo "kubectl create configmap ${K8S_OTEL_ID}-config -n ${K8S_NAMESPACE} --from-file=otel-collector.yaml=${YAML_DIR}/otel-collector.yaml"
  else
    kubectl create configmap "${K8S_OTEL_ID}-config" -n "${K8S_NAMESPACE}" \
      --from-file=otel-collector.yaml="${YAML_DIR}/otel-collector.yaml" \
      --dry-run=client -o yaml | apply_manifest
  fi
  run_yaml_gen \
    "${YAML_DIR}/jobset.otel.yaml" \
    --jobset_name="${K8S_OTEL_ID}" \
    --namespace="${K8S_NAMESPACE}" \
    ${K8S_SERVICE_ACCOUNT:+--service_account="${K8S_SERVICE_ACCOUNT}"} \
    --worker_container_port="${OTEL_PORT}" \
    "${otel_gc_flag}" \
    | apply_manifest
  ensure_otel_dashboard ""
}

if ! is_true "${DRY_RUN}"; then
  ENTER_KUBE_CONTEXT="${ENTER_KUBE_CONTEXT:-${LAUNCH_DIR}/enter_kube_context.sh}"
  if [[ -f "${ENTER_KUBE_CONTEXT}" ]]; then
    source "${ENTER_KUBE_CONTEXT}"
  fi
fi

# Checks for the commands that start nodes: `stop` and `stop-otel` only delete
# resources, and the collector (`otel`) runs its own image.
#
# TODO(noghabi): Deploy a reference inference worker from `k8s.sh` (a JobSet
# on its own TPU slice, like the trainer's) and remove the
# `RUN_INFERENCE_NODE` check, so k8s runs can use the KL penalty.
# `nodes/inference.py` and `cluster_config.k8s.inference_id` already support
# k8s.
case "${COMMAND}" in
  stop|otel|stop-otel)
    ;;
  *)
    if [[ -z "${K8S_TUNIX_IMAGE}" ]]; then
      echo "Error: cluster_config.k8s.tunix_image is empty." >&2
      exit 1
    fi
    # The orchestrator would wait for a reference worker that never registers.
    if is_true "${RUN_INFERENCE_NODE}"; then
      echo "Error: cluster_config.target=k8s does not deploy a reference inference worker yet (k8s.sh starts the orchestrator, trainer, and rollout only), but cluster_config.run_inference_node is true (it defaults to agentic_grpo_config.beta > 0). Set agentic_grpo_config.beta=0.0 to train without the KL penalty, or set cluster_config.run_inference_node=false and start a reference worker that registers with the orchestrator through discovery." >&2
      exit 1
    fi
    ;;
esac

# The rollout pods take their JAX compilation cache settings only from
# `cluster_config.k8s.jax_cache`: unlike `local.sh`'s nodes, they don't inherit
# the caller's environment, which would silently drop these variables.
if [[ "${COMMAND}" == "start" || "${COMMAND}" == "rollout" ]]; then
  cache_env_set=()
  for cache_var in DISABLE_JAX_CACHE JAX_CACHE_BUCKET JAX_CACHE_DIR \
    JAX_CACHE_GCS_DIR JAX_CACHE_SYNC_TIMEOUT_S LOCAL_JAX_CACHE_DIR \
    ROLLOUT_JAX_CACHE_GCS_DIR SAVE_JAX_CACHE VLLM_DISABLE_COMPILE_CACHE \
    VLLM_JAX_CACHE_GCS_DIR VLLM_LOCAL_JAX_CACHE_DIR VLLM_SAVE_JAX_CACHE; do
    if [[ -n "${!cache_var:-}" ]]; then
      cache_env_set+=("${cache_var}")
    fi
  done
  if (( ${#cache_env_set[@]} > 0 )); then
    echo "Error: ${cache_env_set[*]} set in the environment, but the k8s" \
      "rollout pods take their JAX cache settings from" \
      "cluster_config.k8s.jax_cache (see base_dist_config.yaml)." >&2
    exit 1
  fi
  if [[ -n "${BUCKET:-}" ]]; then
    echo "Warning: the k8s rollout pods don't read BUCKET; set" \
      "cluster_config.k8s.jax_cache.bucket for a GCS compilation cache." >&2
  fi
fi

# The trainer and rollout nodes set these library variables from the config
# (`node_utils.apply_library_env`). Unlike `local.sh`'s nodes, the pods don't
# inherit the caller's environment, which would silently drop them.
if [[ "${COMMAND}" == "start" || "${COMMAND}" == "trainer" || "${COMMAND}" == "rollout" ]]; then
  library_env_set=()
  for library_var in SKIP_JAX_PRECOMPILE TPU_RAIDEN_DATA_NICS VERIFY_WEIGHTS; do
    if [[ -n "${!library_var:-}" ]]; then
      library_env_set+=("${library_var}")
    fi
  done
  if (( ${#library_env_set[@]} > 0 )); then
    echo "Error: ${library_env_set[*]} set in the environment, but the k8s" \
      "nodes take them from the config: vllm_config.skip_jax_precompile," \
      "weight_sync_config.raiden.data_nics and" \
      "weight_sync_config.verify_weights." >&2
    exit 1
  fi
fi

case "${COMMAND}" in
  start)
    stop_orchestrator
    stop_trainer
    stop_rollout
    start_orchestrator
    start_trainer
    start_rollout
    if is_true "${OTEL_ENABLE}"; then
      ensure_otel_dashboard "${RUN_ID}"
    fi
    ;;
  stop)
    stop_orchestrator
    stop_trainer
    stop_rollout
    ;;
  orchestrator)
    stop_orchestrator
    start_orchestrator
    if is_true "${OTEL_ENABLE}"; then
      ensure_otel_dashboard "${RUN_ID}"
    fi
    ;;
  trainer)
    stop_trainer
    start_trainer
    ;;
  rollout)
    stop_rollout
    start_rollout
    ;;
  otel)
    stop_otel
    start_otel
    ;;
  stop-otel)
    stop_otel
    ;;
  *)
    echo "Error: Invalid command '${COMMAND}'. Available commands: 'start', 'stop', 'orchestrator', 'trainer', 'rollout', 'otel', 'stop-otel'." >&2
    exit 1
    ;;
esac
