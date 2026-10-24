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

COMMAND=""
_ENV_TUNIX_IMAGE="${TUNIX_IMAGE:-}"
TUNIX_IMAGE="us-central1-docker.pkg.dev/cloud-tpu-multipod-dev/yangmu/tunix/tunix_base_image:trellis-demo-0813"
TUNIX_IMAGE="${_ENV_TUNIX_IMAGE:-${TUNIX_IMAGE}}"

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TUNIX_ROOT="$(cd "${DIR}/../../.." && pwd)"
PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" &>/dev/null; then
  PYTHON="python"
fi
YAML_GENERATOR="${YAML_GENERATOR:-${TUNIX_ROOT}/dist_rl/distributed/deployment/yaml_generator.py}"
YAML_DIR="${YAML_DIR:-${TUNIX_ROOT}/dist_rl/distributed/deployment/yamls}"

export JOB_PREFIX=${JOB_PREFIX:-$USER}
export MODEL_NAME=${MODEL_NAME:-Qwen3-1.7B}
export MODEL_ID=${MODEL_ID:-Qwen/Qwen3-1.7B}
export MODEL_DIR=${MODEL_DIR:-artifacts/qwen3_dist_deepswe/models/${MODEL_NAME}}
export TOKENIZER_PATH=${TOKENIZER_PATH:-${MODEL_ID}}

export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1024}
export BATCH_SIZE=${BATCH_SIZE:-1}
export NUM_GENERATIONS=${NUM_GENERATIONS:-2}
export MAX_STEPS=${MAX_STEPS:-1}
export MAX_TURNS=${MAX_TURNS:-3}
export TRAIN_MICRO_BATCH_SIZE=${TRAIN_MICRO_BATCH_SIZE:-1}
export COMPUTE_LOGPS_CHUNK_SIZE=${COMPUTE_LOGPS_CHUNK_SIZE:-0}
export MAX_SEQ_TOKEN_PER_TPU=${MAX_SEQ_TOKEN_PER_TPU:-}
export MAX_SEGMENTS_PER_PACKED_ROW=${MAX_SEGMENTS_PER_PACKED_ROW:-}
export SEGMENT_ALIGNMENT_BOUNDARY=${SEGMENT_ALIGNMENT_BOUNDARY:-}

export TRAINER_BACKEND=${TRAINER_BACKEND:-tunix}
export TRAINER_PRECOMPILE=${TRAINER_PRECOMPILE:-true}
export MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-${BATCH_SIZE}}
export EVAL_EVERY_N_STEPS=${EVAL_EVERY_N_STEPS:-1000000}
export LEARNING_RATE=${LEARNING_RATE:-1e-6}
export OPT_CHAIN_TYPE=${OPT_CHAIN_TYPE-clip_by_global_norm}
export MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}
export ADAM_B1=${ADAM_B1:-0.9}
export ADAM_B2=${ADAM_B2:-0.999}
export ADAM_EPS=${ADAM_EPS:-1.0e-8}
export WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}
export WARMUP_STEPS_FRACTION=${WARMUP_STEPS_FRACTION:-0.0}
export BETA=${BETA:-0.0}
export EPSILON=${EPSILON:-0.2}
export MAX_STALENESS=${MAX_STALENESS:-0}
export TRAJECTORY_GROUP_ORDER=${TRAJECTORY_GROUP_ORDER:-trajectory_completion}
export TEMPERATURE=${TEMPERATURE:-1.0}
export TOP_P=${TOP_P:-1.0}
export TOP_K=${TOP_K:--1}
export LORA_RANK=${LORA_RANK:-64}
export LORA_ALPHA=${LORA_ALPHA:-64.0}
export MODEL_DTYPE=${MODEL_DTYPE:-float32}
export USE_LORA=${USE_LORA:-0}
export DEBUG=${DEBUG:-0}
export USE_ROLLOUT_LOGPS=${USE_ROLLOUT_LOGPS:-true}
export OVERLONG_FILTER=${OVERLONG_FILTER:-false}
export SAMPLER=${SAMPLER:-inprocess_vllm}
export WEIGHT_SYNC_MODE=${WEIGHT_SYNC_MODE:-none}
export USE_WEIGHT_CONVERTER=${USE_WEIGHT_CONVERTER:-true}
export PREFUSE_MOE_WEIGHTS=${PREFUSE_MOE_WEIGHTS:-true}
export IN_FLIGHT_WEIGHT_UPDATES=${IN_FLIGHT_WEIGHT_UPDATES:-false}
export RETURN_ROUTED_EXPERTS=${RETURN_ROUTED_EXPERTS:-false}
export ROUTED_EXPERTS_ENCODE_MODE=${ROUTED_EXPERTS_ENCODE_MODE:-auto}
export ROUTED_EXPERTS_NUM_EXPERTS=${ROUTED_EXPERTS_NUM_EXPERTS:-}
export EOS_TOKENS=${EOS_TOKENS:-}

# DeepSWE dataset and environment configuration
export DATASET_NAME=${DATASET_NAME:-R2E-Gym/R2E-Gym-Subset}
export DATASET_PATH=${DATASET_PATH:-}
export DATASET_SPLIT=${DATASET_SPLIT:-train}
export DATASET_CACHE_DIR=${DATASET_CACHE_DIR:-artifacts/qwen3_dist_deepswe/dataset_cache}
export SHUFFLE=${SHUFFLE:-true}
export SEED=${SEED:-42}
export ENV_BACKEND=${ENV_BACKEND:-kubernetes}
export SCAFFOLD=${SCAFFOLD:-r2egym}
export K8S_NAMESPACE=${K8S_NAMESPACE:-${NAMESPACE:-default}}
export KUEUE_QUEUE_NAME=${KUEUE_QUEUE_NAME:-${KUEUE_QUEUE:-}}
export SERVICE_ACCOUNT=${SERVICE_ACCOUNT:-xpk-sa}
export PRIORITY_CLASS=${PRIORITY_CLASS:-medium}
export USE_AGENT_SANDBOX=${USE_AGENT_SANDBOX:-0}
export SANDBOX_NAMESPACE=${SANDBOX_NAMESPACE:-${K8S_NAMESPACE}}
export SANDBOX_NODE_SELECTOR_KEY=${SANDBOX_NODE_SELECTOR_KEY:-${NODE_SELECTOR_KEY:-}}
export SANDBOX_NODE_SELECTOR_VAL=${SANDBOX_NODE_SELECTOR_VAL:-${NODE_SELECTOR_VAL:-}}
export MAX_WARMPOOL_REPLICAS=${MAX_WARMPOOL_REPLICAS:-}
export IMAGE_REWRITE_PREFIX=${IMAGE_REWRITE_PREFIX:-}
export STEP_TIMEOUT_SECS=${STEP_TIMEOUT_SECS:-1800}
export REWARD_TIMEOUT_SECS=${REWARD_TIMEOUT_SECS:-1800}
export EPISODE_TIMEOUT_SECS=${EPISODE_TIMEOUT_SECS:-1800}
export RPC_TIMEOUT_S=${RPC_TIMEOUT_S:-}
export INIT_TIMEOUT_S=${INIT_TIMEOUT_S:-}
export MAX_CONCURRENCY=${MAX_CONCURRENCY:-128}
export ROLLOUT_MAX_CONCURRENCY=${ROLLOUT_MAX_CONCURRENCY:-64}
export FLUSH_EVERY_N_STEPS=${FLUSH_EVERY_N_STEPS:-1}
export ENABLE_PATHWAYS_PERSISTENCE=${ENABLE_PATHWAYS_PERSISTENCE:-0}
export CKPT_D2H_CONCURRENT_GB=${CKPT_D2H_CONCURRENT_GB:-8}
export CHECKPOINT_SAVE_INTERVAL_STEPS=${CHECKPOINT_SAVE_INTERVAL_STEPS:-0}
export CHECKPOINT_MAX_TO_KEEP=${CHECKPOINT_MAX_TO_KEEP:-10}
export CHECKPOINT_ROOT_DIRECTORY=${CHECKPOINT_ROOT_DIRECTORY:-checkpoints}

# MaxText trainer configuration: only consulted when TRAINER_BACKEND=maxtext
export MAXTEXT_MODEL_NAME=${MAXTEXT_MODEL_NAME:-qwen3-1.7b}
export MAXTEXT_CKPT=${MAXTEXT_CKPT:-}
if [[ "$TRAINER_BACKEND" == "maxtext" && -z "$MAXTEXT_CKPT" ]]; then
  echo "Error: TRAINER_BACKEND=maxtext requires MAXTEXT_CKPT (Orbax params-only checkpoint)."
  exit 1
fi
export MAXTEXT_OUTPUT_DIR=${MAXTEXT_OUTPUT_DIR:-artifacts/deepswe_dist/maxtext}
export MAXTEXT_EXTRA_FLAGS=${MAXTEXT_EXTRA_FLAGS:-}
export REMAT_POLICY=${REMAT_POLICY:-}
export FLOAT32_GATE_LOGITS=${FLOAT32_GATE_LOGITS:-}
export FLOAT32_LOGITS=${FLOAT32_LOGITS:-}
export TRAINER_MAXTEXT_ATTENTION=${TRAINER_MAXTEXT_ATTENTION:-}
export TRAINER_BASE_NUM_KV_HEADS=${TRAINER_BASE_NUM_KV_HEADS:-}
export PROFILER_STEPS=${PROFILER_STEPS:-0}
export SKIP_FIRST_N_PROFILER_STEPS=${SKIP_FIRST_N_PROFILER_STEPS:-}
export PROFILER_PERIOD=${PROFILER_PERIOD:-}
export ENABLE_TPU_PROFILING_OPTIONS=${ENABLE_TPU_PROFILING_OPTIONS:-true}
export TPU_PROFILE_CHIPS_PER_TASK=${TPU_PROFILE_CHIPS_PER_TASK:-1}
export TPU_PROFILE_SPARSE_CORES=${TPU_PROFILE_SPARSE_CORES:-1}
export TPU_PROFILE_SPARSE_CORE_TILES=${TPU_PROFILE_SPARSE_CORE_TILES:-1}
export UPLOAD_ALL_PROFILER_RESULTS=${UPLOAD_ALL_PROFILER_RESULTS:-false}
export TRAINER_MESH_TP=${TRAINER_MESH_TP:-1}
export TRAINER_MESH_EXPERT=${TRAINER_MESH_EXPERT:-1}
export TRAINER_MESH_CONTEXT=${TRAINER_MESH_CONTEXT:-1}
export TRAINER_PADDED_MOE_MLP_DIM=${TRAINER_PADDED_MOE_MLP_DIM:-}
export ROLLOUT_MESH_TP=${ROLLOUT_MESH_TP:-2}
export ROLLOUT_MESH_FSDP=${ROLLOUT_MESH_FSDP:-1}
export ROLLOUT_REPLICAS=${ROLLOUT_REPLICAS:-1}
export ROLLOUT_START_INDEX=${ROLLOUT_START_INDEX:-0}
export ROLLOUT_COUNT=${ROLLOUT_COUNT:-}
export MIN_ROLLOUT_WORKERS=${MIN_ROLLOUT_WORKERS:-${ROLLOUT_REPLICAS}}
export ROLLOUT_USE_BATCHED_RPA=${ROLLOUT_USE_BATCHED_RPA:-}
export ROLLOUT_MAXTEXT_ATTENTION=${ROLLOUT_MAXTEXT_ATTENTION:-}
export ENABLE_PREFIX_CACHING=${ENABLE_PREFIX_CACHING:-}
export VLLM_CONFIG_JSON=${VLLM_CONFIG_JSON:-}
export VLLM_HBM_UTILIZATION=${VLLM_HBM_UTILIZATION:-}
export VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-}
export VLLM_MAX_NUM_BATCHED_TOKENS=${VLLM_MAX_NUM_BATCHED_TOKENS:-}
export VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-}
export VLLM_ASYNC_SCHEDULING=${VLLM_ASYNC_SCHEDULING:-}
export VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY=${VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY:-}
export VLLM_RAY_EXTRA_ENV_VARS_TO_COPY=${VLLM_RAY_EXTRA_ENV_VARS_TO_COPY:-}
export LIBTPU_INIT_ARGS=${LIBTPU_INIT_ARGS:-}

export VERIFY_WEIGHTS=${VERIFY_WEIGHTS:-false}

export WANDB_ENTITY=${WANDB_ENTITY:-}
export WANDB_PROJECT=${WANDB_PROJECT:-trellis-deepswe}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-}
export WANDB_API_KEY=${WANDB_API_KEY:-}
export LOG_DIR=${LOG_DIR:-}
export TRAJECTORY_LOG_DIR=${TRAJECTORY_LOG_DIR:-}
export TRAJECTORY_LOGGING=${TRAJECTORY_LOGGING:-true}
export RUN_ID=${RUN_ID:-${JOB_PREFIX}}
export TRAJECTORY_STORE_ENABLED=${TRAJECTORY_STORE_ENABLED:-false}
export TRAJECTORY_STORE_BACKEND=${TRAJECTORY_STORE_BACKEND:-file}
export TRAJECTORY_STORE_DIR=${TRAJECTORY_STORE_DIR:-${TRAJECTORY_STORE_ROOT_DIR:-"${LOG_DIR:-artifacts/deepswe_dist}/trajectory_store"}}

export ORCHESTRATOR_ID=${ORCHESTRATOR_ID:-${JOB_PREFIX}-orch}
export ORCHESTRATOR_PORT=${ORCHESTRATOR_PORT:-20000}

export ROLLOUT_ID=${ROLLOUT_ID:-${JOB_PREFIX}-roll}
export ROLLOUT_PORT=${ROLLOUT_PORT:-20001}

export TRAINER_ID=${TRAINER_ID:-${JOB_PREFIX}-train}
export TRAINER_PORT=${TRAINER_PORT:-20002}

export CPU_MACHINE=${CPU_MACHINE:-n2-standard-64}
export CPU_NODEPOOL=${CPU_NODEPOOL:-}
export CPU_MEMORY=${CPU_MEMORY:-}
export GCS_SCRATCH_LOCATION=${GCS_SCRATCH_LOCATION:-gs://cloud-pathways-staging/tmp}

export TRAINER_JOBSET_YAML=${TRAINER_JOBSET_YAML:-jobset.pathways.yaml}
export TRAINER_TPU_SLICE=${TRAINER_TPU_SLICE:-tpuv5:2x2x2}
export TRAINER_MESH_FSDP=${TRAINER_MESH_FSDP:-8}
export ROLLOUT_TPU_SLICE=${ROLLOUT_TPU_SLICE:-tpuv5:2x2x1}

# JAX compilation cache configuration
export LOCAL_JAX_CACHE_DIR=${LOCAL_JAX_CACHE_DIR:-${JAX_CACHE_DIR:-/tmp/jax_cache}}
export JAX_CACHE_GCS_DIR=${JAX_CACHE_GCS_DIR:-}
export ROLLOUT_JAX_CACHE_GCS_DIR=${ROLLOUT_JAX_CACHE_GCS_DIR:-}
export SAVE_JAX_CACHE=${SAVE_JAX_CACHE:-true}
export SKIP_JAX_PRECOMPILE=${SKIP_JAX_PRECOMPILE:-1}
export ROLLOUT_SKIP_JAX_PRECOMPILE=${ROLLOUT_SKIP_JAX_PRECOMPILE:-${SKIP_JAX_PRECOMPILE}}
is_jax_cache_disabled() {
  case "${DISABLE_JAX_CACHE:-0}" in
    1|[tT][rR][uU][eE]|[yY][eE][sS]|[yY]|[tT]) return 0 ;;
    *) return 1 ;;
  esac
}

if ! is_jax_cache_disabled && [[ -z "${ROLLOUT_JAX_CACHE_GCS_DIR}" ]]; then
  ROLLOUT_JAX_CACHE_GCS_DIR=$("$PYTHON" -m tunix.dist_rl.common.gcs_cache resolve-uri 2>/dev/null || true)
fi
export EVAL_JAX_CACHE_GCS_DIR=${EVAL_JAX_CACHE_GCS_DIR:-${ROLLOUT_JAX_CACHE_GCS_DIR}}

export ORCHESTRATOR_EXTRA_ENV=${ORCHESTRATOR_EXTRA_ENV:-}
export ROLLOUT_EXTRA_ENV=${ROLLOUT_EXTRA_ENV:-}
export TRAINER_EXTRA_ENV=${TRAINER_EXTRA_ENV:-}
export PATHWAYS_WORKER_EXTRA_ENV=${PATHWAYS_WORKER_EXTRA_ENV:-}
export PATHWAYS_PROXY_EXTRA_ARGS=${PATHWAYS_PROXY_EXTRA_ARGS:-}
export TRAINER_IFRT_PROXY_LARGE_TRANSFER_THRESHOLD_BYTES=${TRAINER_IFRT_PROXY_LARGE_TRANSFER_THRESHOLD_BYTES:-}
export DRY_RUN=${DRY_RUN:-false}

enrich_manifest() {
  local role="$1"
  "$PYTHON" -c '
import json, os, sys, yaml

role = sys.argv[1]
docs = list(yaml.safe_load_all(sys.stdin.read()))
sa = os.getenv("SERVICE_ACCOUNT", "").strip()
prio = os.getenv("PRIORITY_CLASS", "").strip()
cpu_np = os.getenv("CPU_NODEPOOL", "").strip()
cpu_mem = os.getenv("CPU_MEMORY", "").strip()
pw_proxy_args = os.getenv("PATHWAYS_PROXY_EXTRA_ARGS", "").split()
pw_worker_env_lines = os.getenv("PATHWAYS_WORKER_EXTRA_ENV", "").strip().splitlines()
enable_multi_numa = os.getenv("ENABLE_MULTI_NUMA", "0").strip()
raiden_nics = os.getenv("TPU_RAIDEN_DATA_NICS", "eth0").strip()

extra_pw_env = {}
for line in pw_worker_env_lines:
  line = line.strip()
  if not line or "=" not in line:
    continue
  k, v = line.split("=", 1)
  v = v.strip()
  if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'\''":
    v = v[1:-1]
  extra_pw_env[k.strip()] = v

def upsert_env(env_list, name, value):
  for entry in env_list:
    if entry.get("name") == name:
      entry["value"] = str(value)
      return
  env_list.append({"name": name, "value": str(value)})

for doc in docs:
  if not isinstance(doc, dict):
    continue
  for rjob in doc.get("spec", {}).get("replicatedJobs", []):
    pod_tmpl = rjob.get("template", {}).get("spec", {}).get("template") or {}
    pod_meta = pod_tmpl.get("metadata") or {}
    pod_tmpl["metadata"] = pod_meta
    ann = pod_meta.get("annotations") or {}
    pod_meta["annotations"] = ann
    ann.setdefault("cluster-autoscaler.kubernetes.io/safe-to-evict", "false")
    pod_spec = pod_tmpl.get("spec") or {}
    pod_tmpl["spec"] = pod_spec
    if sa:
      pod_spec["serviceAccountName"] = sa
    if prio:
      pod_spec["priorityClassName"] = prio
    if role == "orchestrator":
      if cpu_np:
        pod_spec["nodeSelector"] = {"cloud.google.com/gke-nodepool": cpu_np}
      for c in pod_spec.get("containers", []):
        if cpu_mem:
          res = c.setdefault("resources", {})
          req = res.setdefault("requests", {})
          req.setdefault("cpu", "24")
          req["memory"] = cpu_mem
    elif role == "trainer":
      for c in pod_spec.get("containers", []):
        cname = c.get("name", "")
        c_env = c.setdefault("env", [])
        upsert_env(c_env, "ENABLE_MULTI_NUMA", enable_multi_numa)
        upsert_env(c_env, "TPU_RAIDEN_DATA_NICS", raiden_nics)
        if cname == "pathways-proxy" and pw_proxy_args:
          c_args = c.setdefault("args", [])
          for flag in pw_proxy_args:
            if flag not in c_args:
              c_args.append(flag)
        elif cname == "pathways-worker":
          for k, v in extra_pw_env.items():
            upsert_env(c_env, k, v)
    elif role == "rollout":
      vols = pod_spec.setdefault("volumes", [])
      if not any(v.get("name") == "dshm" for v in vols):
        vols.append({"name": "dshm", "emptyDir": {"medium": "Memory"}})
      for c in pod_spec.get("containers", []):
        c_env = c.setdefault("env", [])
        upsert_env(c_env, "ENABLE_MULTI_NUMA", enable_multi_numa)
        upsert_env(c_env, "TPU_RAIDEN_DATA_NICS", raiden_nics)
        vmounts = c.setdefault("volumeMounts", [])
        if not any(m.get("name") == "dshm" for m in vmounts):
          vmounts.append({"name": "dshm", "mountPath": "/dev/shm"})

print(yaml.safe_dump_all(docs, sort_keys=False))
' "$role"
}

apply_manifest() {
  if [[ "$DRY_RUN" == "true" ]]; then
    echo "---"
    cat
  else
    kubectl apply -f -
  fi
}

if [[ "$BETA" != "0" && "$BETA" != "0.0" ]]; then
  echo "Error: this first DeepSWE distributed launcher only wires trainer+rollout."
  echo "Use BETA=0.0 until the reference inference worker is added."
  exit 1
fi

case "${RETURN_ROUTED_EXPERTS}" in
  1|[tT][rR][uU][eE]|[yY][eE][sS]|[yY]|[tT])
    if [[ "${TRAINER_BACKEND}" != "maxtext" ]]; then
      echo "Error: RETURN_ROUTED_EXPERTS needs TRAINER_BACKEND=maxtext; only the MaxText trainer replays routed experts."
      exit 1
    fi
    ;;
esac

stop_sandboxes() {
  if [[ "${USE_AGENT_SANDBOX}" == "1" || "${USE_AGENT_SANDBOX}" == "true" || "${USE_AGENT_SANDBOX}" == "True" ]]; then
    local sandbox_run_id
    sandbox_run_id=$("$PYTHON" -c 'import hashlib, sys; print(hashlib.sha256(sys.argv[1].strip().encode("utf-8")).hexdigest()[:8])' "${ORCHESTRATOR_ID}")
    local selector="agent-sandbox-rl-run=${sandbox_run_id}"
    if [[ "$DRY_RUN" == "true" ]]; then
      echo "kubectl delete sandboxclaims,sandboxes,sandboxwarmpools,sandboxtemplates -n ${SANDBOX_NAMESPACE} -l ${selector} --ignore-not-found=true"
    else
      echo "Cleaning up sandboxes and warmpools for run_id=${sandbox_run_id} (job_id=${ORCHESTRATOR_ID}) in ${SANDBOX_NAMESPACE}..."
      kubectl delete sandboxclaims,sandboxes,sandboxwarmpools,sandboxtemplates -n "${SANDBOX_NAMESPACE}" -l "${selector}" --ignore-not-found=true 2>/dev/null || true
    fi
  fi
}

stop_orchestrator() {
  if [[ "$DRY_RUN" == "true" ]]; then
    echo "kubectl delete jobset ${ORCHESTRATOR_ID} -n ${K8S_NAMESPACE}"
  else
    kubectl delete jobset "${ORCHESTRATOR_ID}" -n "${K8S_NAMESPACE}" --ignore-not-found --wait=true
    kubectl wait --for=delete pod -l "jobset.sigs.k8s.io/jobset-name=${ORCHESTRATOR_ID}" -n "${K8S_NAMESPACE}" --timeout=120s 2>/dev/null || true
  fi
  stop_sandboxes
}

start_orchestrator() {
  local dataset_args=""
  if [[ -n "${DATASET_PATH}" ]]; then
    dataset_args="--dataset_path=${DATASET_PATH}"
  fi
  local shuffle_arg="--shuffle"
  if [[ "${SHUFFLE}" == "0" || "${SHUFFLE}" == "false" || "${SHUFFLE}" == "False" ]]; then
    shuffle_arg="--no-shuffle"
  fi
  local overlong_arg="--no-overlong_filter"
  if [[ "${OVERLONG_FILTER}" == "1" || "${OVERLONG_FILTER}" == "true" || "${OVERLONG_FILTER}" == "True" ]]; then
    overlong_arg="--overlong_filter"
  fi
  local sandbox_env=""
  local sandbox_arg=""
  if [[ "${USE_AGENT_SANDBOX}" == "1" || "${USE_AGENT_SANDBOX}" == "true" || "${USE_AGENT_SANDBOX}" == "True" ]]; then
    sandbox_env="NAMESPACE=\"${SANDBOX_NAMESPACE}\" ORCHESTRATOR_ID=\"${ORCHESTRATOR_ID}\" RUN_ID=\"${RUN_ID}\" ${SANDBOX_NODE_SELECTOR_KEY:+NODE_SELECTOR_KEY=\"${SANDBOX_NODE_SELECTOR_KEY}\"} ${SANDBOX_NODE_SELECTOR_VAL:+NODE_SELECTOR_VAL=\"${SANDBOX_NODE_SELECTOR_VAL}\"} ${IMAGE_REWRITE_PREFIX:+IMAGE_REWRITE_PREFIX=\"${IMAGE_REWRITE_PREFIX}\"}"
    sandbox_arg="--use_agent_sandbox"
  elif [[ -n "${IMAGE_REWRITE_PREFIX}" ]]; then
    sandbox_env="ORCHESTRATOR_ID=\"${ORCHESTRATOR_ID}\" IMAGE_REWRITE_PREFIX=\"${IMAGE_REWRITE_PREFIX}\""
  else
    sandbox_env="ORCHESTRATOR_ID=\"${ORCHESTRATOR_ID}\""
  fi
  local debug_arg=""
  if [[ "${DEBUG}" == "1" || "${DEBUG}" == "true" || "${DEBUG}" == "True" ]]; then
    debug_arg="--debug"
  fi
  "$PYTHON" "$YAML_GENERATOR" \
    "${YAML_DIR}/jobset.cpu.yaml" \
    --jobset_name="${ORCHESTRATOR_ID}" \
    --namespace="${K8S_NAMESPACE}" \
    ${KUEUE_QUEUE_NAME:+--queue_name="${KUEUE_QUEUE_NAME}"} \
    --cpu_machine=${CPU_MACHINE} \
    --worker_container_image="${TUNIX_IMAGE}" \
    --worker_container_port="${ORCHESTRATOR_PORT}" \
    --worker_startup_command=" \
      ${WANDB_API_KEY:+WANDB_API_KEY=\"${WANDB_API_KEY}\"} \
      ${WANDB_ENTITY:+WANDB_ENTITY=\"${WANDB_ENTITY}\"} \
      ${LOG_DIR:+LOG_DIR=\"${LOG_DIR}\"} \
      ${TRAJECTORY_LOG_DIR:+TRAJECTORY_LOG_DIR=\"${TRAJECTORY_LOG_DIR}\"} \
      WANDB_PROJECT=\"${WANDB_PROJECT}\" \
      WANDB_RUN_NAME=\"${WANDB_RUN_NAME}\" \
      ${sandbox_env} \
      ${ORCHESTRATOR_EXTRA_ENV} \
      python -m tunix.dist_rl.distributed.runtime.main \
        --discovery_id=${ORCHESTRATOR_ID} \
        --discovery_port=${ORCHESTRATOR_PORT} \
        --process_main=tunix.experimental.examples.deepswe_dist.run_deepswe_dist.main \
        --model_id=${MODEL_ID} \
        --tokenizer_path=${TOKENIZER_PATH} \
        --batch_size=${BATCH_SIZE} \
        --mini_batch_size=${MINI_BATCH_SIZE} \
        --num_generations=${NUM_GENERATIONS} \
        --rollout_replicas=${ROLLOUT_REPLICAS} \
        --max_concurrency=${MAX_CONCURRENCY} \
        --temperature=${TEMPERATURE} \
        --top_p=${TOP_P} \
        --top_k=${TOP_K} \
        --max_steps=${MAX_STEPS} \
        --max_turns=${MAX_TURNS} \
        --max_prompt_length=${MAX_PROMPT_LENGTH} \
        --max_response_length=${MAX_RESPONSE_LENGTH} \
        --train_micro_batch_size=${TRAIN_MICRO_BATCH_SIZE} \
        --beta=${BETA} \
        --epsilon=${EPSILON} \
        ${EPSILON_HIGH:+--epsilon_high=${EPSILON_HIGH}} \
        ${ADVANTAGE_ESTIMATOR:+--advantage_estimator=${ADVANTAGE_ESTIMATOR}} \
        ${LOSS_AGG_MODE:+--loss_agg_mode=${LOSS_AGG_MODE}} \
        ${SAMPLER_IS:+--sampler_is=${SAMPLER_IS}} \
        ${SAMPLER_IS_THRESHOLD:+--sampler_is_threshold=${SAMPLER_IS_THRESHOLD}} \
        ${SAMPLER_RS:+--sampler_rs=${SAMPLER_RS}} \
        ${SAMPLER_RS_MIN:+--sampler_rs_min=${SAMPLER_RS_MIN}} \
        ${SAMPLER_RS_MAX:+--sampler_rs_max=${SAMPLER_RS_MAX}} \
        ${SEQ_LOGPROB_ERROR_THRESHOLD:+--seq_logprob_error_threshold=${SEQ_LOGPROB_ERROR_THRESHOLD}} \
        --max_staleness=${MAX_STALENESS} \
        --trajectory_group_order=${TRAJECTORY_GROUP_ORDER} \
        --dataset_name=${DATASET_NAME} \
        --dataset_split=${DATASET_SPLIT} \
        ${DATASET_CACHE_DIR:+--dataset_cache_dir=${DATASET_CACHE_DIR}} \
        --seed=${SEED} \
        --env_backend=${ENV_BACKEND} \
        --scaffold=${SCAFFOLD} \
        --step_timeout_secs=${STEP_TIMEOUT_SECS} \
        --reward_timeout_secs=${REWARD_TIMEOUT_SECS} \
        --episode_timeout_secs=${EPISODE_TIMEOUT_SECS} \
        ${RPC_TIMEOUT_S:+--rpc_timeout_s=${RPC_TIMEOUT_S}} \
        ${INIT_TIMEOUT_S:+--init_timeout_s=${INIT_TIMEOUT_S}} \
        --flush_every_n_steps=${FLUSH_EVERY_N_STEPS} \
        ${WANDB_ENTITY:+--wandb_entity=\"${WANDB_ENTITY}\"} \
        --wandb_project=\"${WANDB_PROJECT}\" \
        --wandb_run_name=\"${WANDB_RUN_NAME}\" \
        --weight_sync_mode=${WEIGHT_SYNC_MODE} \
        --trainer_precompile=${TRAINER_PRECOMPILE} \
        --return_routed_experts=${RETURN_ROUTED_EXPERTS} \
        --routed_experts_encode_mode=${ROUTED_EXPERTS_ENCODE_MODE} \
        ${ROUTED_EXPERTS_NUM_EXPERTS:+--routed_experts_num_experts=${ROUTED_EXPERTS_NUM_EXPERTS}} \
        --stop_workers_on_exit \
        --enable_trajectory_store=${TRAJECTORY_STORE_ENABLED} \
        --trajectory_store_backend=${TRAJECTORY_STORE_BACKEND} \
        --trajectory_store_dir=\"${TRAJECTORY_STORE_DIR}\" \
        --run_id=\"${RUN_ID}\" \
        ${MAX_WARMPOOL_REPLICAS:+--max_warmpool_replicas=${MAX_WARMPOOL_REPLICAS}} \
        $([[ "${USE_ROLLOUT_LOGPS}" == "false" || "${USE_ROLLOUT_LOGPS}" == "False" || "${USE_ROLLOUT_LOGPS}" == "0" ]] && echo --no-use_rollout_logps || echo --use_rollout_logps) \
        $([[ "${TRAJECTORY_LOGGING}" == "false" || "${TRAJECTORY_LOGGING}" == "False" || "${TRAJECTORY_LOGGING}" == "0" ]] && echo --no-trajectory_logging || echo --trajectory_logging) \
        ${overlong_arg} \
        ${dataset_args} \
        ${shuffle_arg} \
        ${sandbox_arg} \
        ${LOG_DIR:+--log_dir=\"${LOG_DIR}\"} \
        ${TRAJECTORY_LOG_DIR:+--trajectory_log_dir=\"${TRAJECTORY_LOG_DIR}\"} \
        ${MAX_SEQ_TOKEN_PER_TPU:+--max_seq_token_per_tpu=${MAX_SEQ_TOKEN_PER_TPU}} \
        ${MAX_SEGMENTS_PER_PACKED_ROW:+--max_segments_per_packed_row=${MAX_SEGMENTS_PER_PACKED_ROW}} \
        ${SEGMENT_ALIGNMENT_BOUNDARY:+--segment_alignment_boundary=${SEGMENT_ALIGNMENT_BOUNDARY}} \
        --num_data_shards=$(( ${TRAINER_MESH_FSDP:-1} * ${TRAINER_MESH_EXPERT:-1} )) \
        ${debug_arg} \
    " \
    | enrich_manifest orchestrator \
    | apply_manifest
}

stop_trainer() {
  if [[ "$DRY_RUN" == "true" ]]; then
    echo "kubectl delete jobset ${TRAINER_ID} -n ${K8S_NAMESPACE}"
  else
    kubectl delete jobset "${TRAINER_ID}" -n "${K8S_NAMESPACE}" --ignore-not-found --wait=true
    kubectl wait --for=delete pod -l "jobset.sigs.k8s.io/jobset-name=${TRAINER_ID}" -n "${K8S_NAMESPACE}" --timeout=120s 2>/dev/null || true
  fi
}

start_trainer() {
  local maxtext_args=""
  local profiler_flags=""
  if [[ -n "$PROFILER_STEPS" ]]; then
    profiler_flags+=" --profiler_steps=${PROFILER_STEPS}"
  fi
  if [[ -n "$SKIP_FIRST_N_PROFILER_STEPS" ]]; then
    profiler_flags+=" --skip_first_n_profiler_steps=${SKIP_FIRST_N_PROFILER_STEPS}"
  fi
  if [[ -n "$PROFILER_PERIOD" ]]; then
    profiler_flags+=" --profiler_period=${PROFILER_PERIOD}"
  fi
  if [[ -n "$ENABLE_TPU_PROFILING_OPTIONS" ]]; then
    profiler_flags+=" --enable_tpu_profiling_options=${ENABLE_TPU_PROFILING_OPTIONS}"
  fi
  if [[ -n "$TPU_PROFILE_CHIPS_PER_TASK" ]]; then
    profiler_flags+=" --tpu_num_chips_to_profile_per_task=${TPU_PROFILE_CHIPS_PER_TASK}"
  fi
  if [[ -n "$TPU_PROFILE_SPARSE_CORES" ]]; then
    profiler_flags+=" --tpu_num_sparse_cores_to_trace=${TPU_PROFILE_SPARSE_CORES}"
  fi
  if [[ -n "$TPU_PROFILE_SPARSE_CORE_TILES" ]]; then
    profiler_flags+=" --tpu_num_sparse_core_tiles_to_trace=${TPU_PROFILE_SPARSE_CORE_TILES}"
  fi
  if [[ -n "$UPLOAD_ALL_PROFILER_RESULTS" ]]; then
    profiler_flags+=" --upload_all_profiler_results=${UPLOAD_ALL_PROFILER_RESULTS}"
  fi
  local effective_maxtext_extra_flags="${MAXTEXT_EXTRA_FLAGS}"
  if [[ "${TRAINER_BACKEND}" == "maxtext" ]]; then
    local _mt_extra=(
      ${REMAT_POLICY:+"remat_policy=${REMAT_POLICY}"}
      ${FLOAT32_GATE_LOGITS:+"float32_gate_logits=${FLOAT32_GATE_LOGITS}"}
      ${FLOAT32_LOGITS:+"float32_logits=${FLOAT32_LOGITS}"}
    )
    if [[ ${#_mt_extra[@]} -gt 0 ]]; then
      effective_maxtext_extra_flags="${effective_maxtext_extra_flags:+${effective_maxtext_extra_flags} }${_mt_extra[*]}"
    fi
    maxtext_args=" \
      --maxtext_model_name=${MAXTEXT_MODEL_NAME} \
      ${TRAINER_PADDED_MOE_MLP_DIM:+--maxtext_padded_moe_mlp_dim=${TRAINER_PADDED_MOE_MLP_DIM}} \
      --maxtext_ckpt_path=${MAXTEXT_CKPT} \
      --maxtext_output_directory=${MAXTEXT_OUTPUT_DIR} \
      --maxtext_warmup_steps_fraction=${WARMUP_STEPS_FRACTION} \
      ${TRAINER_MAXTEXT_ATTENTION:+--maxtext_attention=${TRAINER_MAXTEXT_ATTENTION}} \
      ${TRAINER_BASE_NUM_KV_HEADS:+--maxtext_base_num_kv_heads=${TRAINER_BASE_NUM_KV_HEADS}} \
      --prefuse_moe_weights=${PREFUSE_MOE_WEIGHTS} \
      --mesh_tp=${TRAINER_MESH_TP} \
      --mesh_expert=${TRAINER_MESH_EXPERT} \
      --maxtext_mesh_context=${TRAINER_MESH_CONTEXT} \
      ${MAX_SEQ_TOKEN_PER_TPU:+--max_seq_token_per_tpu=${MAX_SEQ_TOKEN_PER_TPU}} \
      ${ROLLOUT_MESH_TP:+--rollout_mesh_tp=${ROLLOUT_MESH_TP}} \
      ${USE_WEIGHT_CONVERTER:+--use_weight_converter=${USE_WEIGHT_CONVERTER}} \
    "
  fi
  local lora_args=""
  if [[ "${USE_LORA}" == "1" || "${USE_LORA}" == "true" || "${USE_LORA}" == "True" ]]; then
    lora_args="--use_lora"
  fi
  local debug_arg=""
  if [[ "${DEBUG}" == "1" || "${DEBUG}" == "true" || "${DEBUG}" == "True" ]]; then
    debug_arg="--debug"
  fi
  local raiden_env=""
  if [[ "${WEIGHT_SYNC_MODE}" == "raiden" ]]; then
    if [[ "${TRAINER_JOBSET_YAML}" == "jobset.pathways.yaml" ]]; then
      raiden_env+=" RAIDEN_USE_FFI=1 RAIDEN_FFI_USE_DIRECT_DEVICE_BUFFER=${RAIDEN_FFI_USE_DIRECT_DEVICE_BUFFER:-0}"
    fi
  fi
  local opt_chain_flags=""
  if [[ -n "${OPT_CHAIN_TYPE}" ]]; then
    opt_chain_flags="--optimizer_opt_chain_type=\"${OPT_CHAIN_TYPE}\" --optimizer_chain_kwargs=\"{'max_norm': ${MAX_GRAD_NORM}}\""
  fi
  "$PYTHON" "$YAML_GENERATOR" \
    "${YAML_DIR}/${TRAINER_JOBSET_YAML}" \
    --jobset_name="${TRAINER_ID}" \
    --namespace="${K8S_NAMESPACE}" \
    ${KUEUE_QUEUE_NAME:+--queue_name="${KUEUE_QUEUE_NAME}"} \
    --tpu_slice=${TRAINER_TPU_SLICE} \
    --cpu_machine=${CPU_MACHINE} \
    ${PATHWAYS_SERVER_IMAGE:+--pathways_server_image="${PATHWAYS_SERVER_IMAGE}"} \
    ${PATHWAYS_PROXY_IMAGE:+--pathways_proxy_server_image="${PATHWAYS_PROXY_IMAGE}"} \
    ${PATHWAYS_PROXY_MEMORY_LIMIT:+--pathways_proxy_memory_limit="${PATHWAYS_PROXY_MEMORY_LIMIT}"} \
    ${PATHWAYS_PROXY_MEMORY:+--pathways_proxy_memory="${PATHWAYS_PROXY_MEMORY}"} \
    ${PATHWAYS_RM_MEMORY:+--pathways_rm_memory="${PATHWAYS_RM_MEMORY}"} \
    ${USER_CONTAINER_MEMORY:+--user_container_memory="${USER_CONTAINER_MEMORY}"} \
    ${USER_CONTAINER_MEMORY_LIMIT:+--user_container_memory_limit="${USER_CONTAINER_MEMORY_LIMIT}"} \
    ${PATHWAYS_WORKER_MEMORY:+--pathways_worker_memory="${PATHWAYS_WORKER_MEMORY}"} \
    ${TRAINER_IFRT_PROXY_LARGE_TRANSFER_THRESHOLD_BYTES:+--ifrt_proxy_large_transfer_threshold_bytes="${TRAINER_IFRT_PROXY_LARGE_TRANSFER_THRESHOLD_BYTES}"} \
    --pathways_gcs_scratch_location=${GCS_SCRATCH_LOCATION} \
    --worker_container_image="${TUNIX_IMAGE}" \
    --worker_container_port="${TRAINER_PORT}" \
    --worker_startup_command=" \
      ${HF_TOKEN:+HF_TOKEN=\"${HF_TOKEN}\"} VERIFY_WEIGHTS=${VERIFY_WEIGHTS} ENABLE_PATHWAYS_PERSISTENCE=${ENABLE_PATHWAYS_PERSISTENCE}${CKPT_D2H_CONCURRENT_GB:+ CKPT_D2H_CONCURRENT_GB=${CKPT_D2H_CONCURRENT_GB}}${effective_maxtext_extra_flags:+ MAXTEXT_EXTRA_FLAGS=\"${effective_maxtext_extra_flags}\"}${raiden_env}${TRAINER_EXTRA_ENV:+ ${TRAINER_EXTRA_ENV}} python -m tunix.dist_rl.distributed.runtime.main \
        --discovery_addrs=${ORCHESTRATOR_ID}:${ORCHESTRATOR_PORT} \
        --process_executor=tunix.dist_rl.distributed.runtime.executor.K8sExecutor \
        --process_main=tunix.dist_rl.examples.common.run_trainer_node.main \
        --worker_id=${TRAINER_ID} \
        --port=${TRAINER_PORT} \
        --mesh_fsdp=${TRAINER_MESH_FSDP} \
        --mesh_tp=${TRAINER_MESH_TP} \
        --trainer_backend=${TRAINER_BACKEND} \
        --model_name=${MODEL_NAME} \
        --model_id=${MODEL_ID} \
        --model_dir=${MODEL_DIR} \
        --tokenizer_path=${TOKENIZER_PATH} \
        --max_prompt_length=${MAX_PROMPT_LENGTH} \
        --max_response_length=${MAX_RESPONSE_LENGTH} \
        --mini_batch_size=${MINI_BATCH_SIZE} \
        --num_generations=${NUM_GENERATIONS} \
        --train_micro_batch_size=${TRAIN_MICRO_BATCH_SIZE} \
        --compute_logps_chunk_size=${COMPUTE_LOGPS_CHUNK_SIZE} \
        --model_dtype=${MODEL_DTYPE} \
        --eval_every_n_steps=${EVAL_EVERY_N_STEPS} \
        --optimizer_b1=${ADAM_B1} \
        --optimizer_b2=${ADAM_B2} \
        --optimizer_eps=${ADAM_EPS} \
        --optimizer_weight_decay=${WEIGHT_DECAY} \
        ${opt_chain_flags} \
        --learning_rate=${LEARNING_RATE} \
        --checkpoint_save_interval_steps=${CHECKPOINT_SAVE_INTERVAL_STEPS} \
        --checkpoint_max_to_keep=${CHECKPOINT_MAX_TO_KEEP} \
        --checkpoint_root_directory=${CHECKPOINT_ROOT_DIRECTORY} \
        --lora_rank=${LORA_RANK} \
        --lora_alpha=${LORA_ALPHA} \
        --sampler_type=${SAMPLER} \
        ${lora_args} \
        ${maxtext_args} \
        ${profiler_flags} \
        ${debug_arg} \
    " \
    | enrich_manifest trainer \
    | apply_manifest
}

stop_rollout() {
  local replicas=${ROLLOUT_REPLICAS:-1}
  if [[ "$DRY_RUN" == "true" ]]; then
    echo "kubectl delete jobset ${ROLLOUT_ID} -n ${K8S_NAMESPACE}"
    if [[ ${replicas} -gt 1 ]]; then
      echo "kubectl delete jobset $(seq -f "${ROLLOUT_ID}-%g" 0 $((replicas - 1))) -n ${K8S_NAMESPACE}"
    fi
  else
    kubectl delete jobset "${ROLLOUT_ID}" -n "${K8S_NAMESPACE}" --ignore-not-found --wait=true
    if [[ ${replicas} -gt 1 ]]; then
      kubectl delete jobset $(seq -f "${ROLLOUT_ID}-%g" 0 $((replicas - 1))) -n "${K8S_NAMESPACE}" --ignore-not-found --wait=true 2>/dev/null || true
    fi
    for ((i = 0; i < replicas; i++)); do
      local rid="${ROLLOUT_ID}"
      if [[ ${replicas} -gt 1 ]]; then rid="${ROLLOUT_ID}-${i}"; fi
      kubectl wait --for=delete pod -l "jobset.sigs.k8s.io/jobset-name=${rid}" -n "${K8S_NAMESPACE}" --timeout=120s 2>/dev/null || true
    done
  fi
}

start_rollout() {
  local maxtext_args=""
  if [[ "${TRAINER_BACKEND}" == "maxtext" ]]; then
    maxtext_args=" \
      --maxtext_model_name=${MAXTEXT_MODEL_NAME} \
      ${ROLLOUT_MAXTEXT_ATTENTION:+--maxtext_attention=${ROLLOUT_MAXTEXT_ATTENTION}} \
      ${FLOAT32_GATE_LOGITS:+--maxtext_float32_gate_logits=${FLOAT32_GATE_LOGITS}} \
      ${FLOAT32_LOGITS:+--maxtext_float32_logits=${FLOAT32_LOGITS}} \
    "
  fi
  local vllm_args=""
  if [[ "$SAMPLER" == "vllm" || "$SAMPLER" == "inprocess_vllm" ]]; then
    local async_sched_flag=""
    if [[ "${VLLM_ASYNC_SCHEDULING}" == "1" || "${VLLM_ASYNC_SCHEDULING}" == "true" || "${VLLM_ASYNC_SCHEDULING}" == "True" ]]; then
      async_sched_flag="--vllm_async_scheduling"
    elif [[ "${VLLM_ASYNC_SCHEDULING}" == "0" || "${VLLM_ASYNC_SCHEDULING}" == "false" || "${VLLM_ASYNC_SCHEDULING}" == "False" ]]; then
      async_sched_flag="--no-vllm_async_scheduling"
    fi
    vllm_args="\
    --tensor_parallel_size=${ROLLOUT_MESH_TP} \
    ${PREFUSE_MOE_WEIGHTS:+--prefuse_moe_weights=${PREFUSE_MOE_WEIGHTS}} \
    ${ENABLE_PREFIX_CACHING:+--enable_prefix_caching=${ENABLE_PREFIX_CACHING}} \
    ${VLLM_HBM_UTILIZATION:+--vllm_hbm_utilization=${VLLM_HBM_UTILIZATION}} \
    ${VLLM_MAX_NUM_SEQS:+--vllm_max_num_seqs=${VLLM_MAX_NUM_SEQS}} \
    ${VLLM_MAX_NUM_BATCHED_TOKENS:+--vllm_max_num_batched_tokens=${VLLM_MAX_NUM_BATCHED_TOKENS}} \
    ${VLLM_MAX_MODEL_LEN:+--vllm_max_model_len=${VLLM_MAX_MODEL_LEN}} \
    ${async_sched_flag} \
    ${VLLM_CONFIG_JSON:+--vllm_config_json='${VLLM_CONFIG_JSON}'} \
    "
  fi
  local lora_args=""
  if [[ "${USE_LORA}" == "1" || "${USE_LORA}" == "true" || "${USE_LORA}" == "True" ]]; then
    lora_args="--use_lora"
  fi
  local debug_arg=""
  if [[ "${DEBUG}" == "1" || "${DEBUG}" == "true" || "${DEBUG}" == "True" ]]; then
    debug_arg="--debug"
  fi
  local raiden_env=""
  if [[ "${WEIGHT_SYNC_MODE}" == "raiden" ]]; then
    raiden_env="RAIDEN_USE_FFI=0"
  fi
  local sandbox_env=""
  if [[ "$USE_AGENT_SANDBOX" == "1" || "$USE_AGENT_SANDBOX" == "true" || "$USE_AGENT_SANDBOX" == "True" ]]; then
    sandbox_env="NAMESPACE=\"${SANDBOX_NAMESPACE}\" ORCHESTRATOR_ID=\"${ORCHESTRATOR_ID}\" RUN_ID=\"${RUN_ID}\" ${SANDBOX_NODE_SELECTOR_KEY:+NODE_SELECTOR_KEY=\"${SANDBOX_NODE_SELECTOR_KEY}\"} ${SANDBOX_NODE_SELECTOR_VAL:+NODE_SELECTOR_VAL=\"${SANDBOX_NODE_SELECTOR_VAL}\"} ${IMAGE_REWRITE_PREFIX:+IMAGE_REWRITE_PREFIX=\"${IMAGE_REWRITE_PREFIX}\"}"
  elif [[ -n "${IMAGE_REWRITE_PREFIX}" ]]; then
    sandbox_env="ORCHESTRATOR_ID=\"${ORCHESTRATOR_ID}\" IMAGE_REWRITE_PREFIX=\"${IMAGE_REWRITE_PREFIX}\""
  else
    sandbox_env="ORCHESTRATOR_ID=\"${ORCHESTRATOR_ID}\""
  fi
  if ! is_jax_cache_disabled && [[ -n "${ROLLOUT_JAX_CACHE_GCS_DIR}" ]]; then
    echo "[launcher] Rollout JAX cache GCS: ${ROLLOUT_JAX_CACHE_GCS_DIR} (save=${SAVE_JAX_CACHE:-true})" >&2
  fi
  local jax_cache_env=" ROLLOUT_TPU_SLICE=\"${ROLLOUT_TPU_SLICE}\""
  if is_jax_cache_disabled; then
    jax_cache_env+=" DISABLE_JAX_CACHE=1"
  else
    if [[ -n "${ROLLOUT_JAX_CACHE_GCS_DIR}" ]]; then
      jax_cache_env+=" ROLLOUT_JAX_CACHE_GCS_DIR=\"${ROLLOUT_JAX_CACHE_GCS_DIR}\""
    fi
    if [[ -n "${JAX_CACHE_GCS_DIR}" ]]; then
      jax_cache_env+=" JAX_CACHE_GCS_DIR=\"${JAX_CACHE_GCS_DIR}\""
    fi
    if [[ -n "${JAX_CACHE_BUCKET:-}" ]]; then
      jax_cache_env+=" JAX_CACHE_BUCKET=\"${JAX_CACHE_BUCKET}\""
    fi
    if [[ -n "${BUCKET:-}" ]]; then
      jax_cache_env+=" BUCKET=\"${BUCKET}\""
    fi
    if [[ -n "${MAXTEXT_OUTPUT_DIR}" ]]; then
      jax_cache_env+=" MAXTEXT_OUTPUT_DIR=\"${MAXTEXT_OUTPUT_DIR}\""
    fi
    if [[ -n "${SAVE_JAX_CACHE}" ]]; then
      jax_cache_env+=" SAVE_JAX_CACHE=\"${SAVE_JAX_CACHE}\""
    fi
  fi
  local ray_head_pre_init_cmd="export JAX_COMPILATION_CACHE_DIR=\"${LOCAL_JAX_CACHE_DIR}\" VLLM_XLA_CACHE_PATH=\"${LOCAL_JAX_CACHE_DIR}\" && mkdir -p \"${LOCAL_JAX_CACHE_DIR}\""
  local ray_worker_pre_init_cmd="${ray_head_pre_init_cmd}"
  if ! is_jax_cache_disabled && [[ -n "${ROLLOUT_JAX_CACHE_GCS_DIR}" ]]; then
    ray_worker_pre_init_cmd+=" && (python3 -m tunix.dist_rl.common.gcs_cache download \"${LOCAL_JAX_CACHE_DIR}\" \"${ROLLOUT_JAX_CACHE_GCS_DIR}\" || true)"
  fi

  local start_index=${ROLLOUT_START_INDEX:-0}
  local replicas=${ROLLOUT_REPLICAS:-1}
  local end_index=${replicas}
  if [[ -n "${ROLLOUT_COUNT}" ]]; then
    end_index=$(( start_index + ROLLOUT_COUNT ))
    if (( end_index > replicas )); then
      end_index=${replicas}
    fi
  fi
  local replica_ids=()
  for ((i = start_index; i < end_index; i++)); do
    local replica_id="${ROLLOUT_ID}"
    if [[ ${replicas} -gt 1 ]]; then
      replica_id="${ROLLOUT_ID}-${i}"
    fi
    replica_ids+=("${replica_id}")
  done
  if [[ ${#replica_ids[@]} -eq 0 ]]; then
    return 0
  fi

  local placeholder="__TUNIX_REPLICA_ID__"
  local manifest_template
  manifest_template=$(
    "$PYTHON" "$YAML_GENERATOR" \
      "${YAML_DIR}/jobset.tpu.yaml" \
      --jobset_name="${placeholder}" \
      --namespace="${K8S_NAMESPACE}" \
      ${KUEUE_QUEUE_NAME:+--queue_name="${KUEUE_QUEUE_NAME}"} \
      --tpu_slice=${ROLLOUT_TPU_SLICE} \
      --worker_container_image="${TUNIX_IMAGE}" \
      --worker_container_port="${ROLLOUT_PORT}" \
      --ray_head_pre_init_command="${ray_head_pre_init_cmd}" \
      --ray_worker_pre_init_command="${ray_worker_pre_init_cmd}" \
      --worker_startup_command=" \
        SKIP_JAX_PRECOMPILE=${ROLLOUT_SKIP_JAX_PRECOMPILE} \
        VERIFY_WEIGHTS=${VERIFY_WEIGHTS} \
        ${raiden_env} \
        ${sandbox_env} \
        ${LIBTPU_INIT_ARGS:+LIBTPU_INIT_ARGS=\"${LIBTPU_INIT_ARGS}\"} \
        ${VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY:+VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY=\"${VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY}\"} \
        ${VLLM_RAY_EXTRA_ENV_VARS_TO_COPY:+VLLM_RAY_EXTRA_ENV_VARS_TO_COPY=\"${VLLM_RAY_EXTRA_ENV_VARS_TO_COPY}\"} \
        ${ROLLOUT_EXTRA_ENV} \
        ${ROLLOUT_USE_BATCHED_RPA:+USE_BATCHED_RPA_KERNEL=1}${jax_cache_env} \
        python -m tunix.dist_rl.distributed.runtime.main \
          --discovery_addrs=${ORCHESTRATOR_ID}:${ORCHESTRATOR_PORT} \
          --process_executor=tunix.dist_rl.distributed.runtime.executor.K8sExecutor \
          --process_main=tunix.dist_rl.examples.common.run_rollout_node.main \
          --worker_id=${placeholder} \
          --port=${ROLLOUT_PORT} \
          --model_id=${MODEL_ID} \
          --model_dir=${MODEL_DIR} \
          --model_name=${MODEL_NAME} \
          --tokenizer_path=${TOKENIZER_PATH} \
          --mesh_fsdp=${ROLLOUT_MESH_FSDP} \
          --mesh_tp=${ROLLOUT_MESH_TP} \
          --max_prompt_length=${MAX_PROMPT_LENGTH} \
          --max_response_length=${MAX_RESPONSE_LENGTH} \
          ${EOS_TOKENS:+--eos_tokens=\"${EOS_TOKENS}\"} \
          --sampler=${SAMPLER} \
          --lora_rank=${LORA_RANK} \
          --lora_alpha=${LORA_ALPHA} \
          --weight_sync_mode=${WEIGHT_SYNC_MODE} \
          --in_flight_weight_updates=${IN_FLIGHT_WEIGHT_UPDATES} \
          --return_routed_experts=${RETURN_ROUTED_EXPERTS} \
          --routed_experts_encode_mode=${ROUTED_EXPERTS_ENCODE_MODE} \
          ${ROUTED_EXPERTS_NUM_EXPERTS:+--routed_experts_num_experts=${ROUTED_EXPERTS_NUM_EXPERTS}} \
          --registry_module=tunix.experimental.examples.deepswe_dist.deepswe \
          --env_name=deepswe_env \
          --agent_name=deepswe_agent \
          --max_concurrency=${ROLLOUT_MAX_CONCURRENCY} \
          --episode_timeout_secs=${EPISODE_TIMEOUT_SECS} \
          --enable_trajectory_store=${TRAJECTORY_STORE_ENABLED} \
          --trajectory_store_backend=${TRAJECTORY_STORE_BACKEND} \
          --trajectory_store_dir=\"${TRAJECTORY_STORE_DIR}\" \
          --run_id=\"${RUN_ID}\" \
          ${lora_args} \
          ${maxtext_args} \
          ${vllm_args} \
          ${debug_arg} \
      " | enrich_manifest rollout
  ) || return $?

  local idx=0
  for replica_id in "${replica_ids[@]}"; do
    if [[ ${idx} -gt 0 ]]; then
      echo "---"
    fi
    printf '%s\n' "${manifest_template//${placeholder}/${replica_id}}"
    ((idx++))
  done | apply_manifest
}

while [[ $# -gt 0 ]]; do
  case "$1" in
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
    --dry-run|--dry_run|--render)
      DRY_RUN=true
      shift
      ;;
    --rollout-start-index|--rollout_start_index)
      ROLLOUT_START_INDEX="$2"
      shift 2
      ;;
    --rollout-start-index=*|--rollout_start_index=*)
      ROLLOUT_START_INDEX="${1#*=}"
      shift
      ;;
    --rollout-count|--rollout_count)
      ROLLOUT_COUNT="$2"
      shift 2
      ;;
    --rollout-count=*|--rollout_count=*)
      ROLLOUT_COUNT="${1#*=}"
      shift
      ;;
    *)
      shift
      ;;
  esac
done

if [[ "$DRY_RUN" != "true" ]]; then
  if [[ -z "${KUBECONFIG:-}" && -f "$HOME/.kube/config" ]]; then
    export KUBECONFIG="$HOME/.kube/config"
  fi
  if ! kubectl config current-context &>/dev/null; then
    export CLUSTER="${CLUSTER:-${CLUSTER_NAME:-}}"
    export LOCATION_NAME="${LOCATION_NAME:-${ZONE:-}}"
    if [[ -f "${TUNIX_ROOT}/dist_rl/examples/common/enter_kube_context.sh" ]]; then
      source "${TUNIX_ROOT}/dist_rl/examples/common/enter_kube_context.sh"
    fi
  fi
  kubectl config set-context --current --namespace="${K8S_NAMESPACE}" >/dev/null 2>&1 || true
fi

if [[ -z "$TUNIX_IMAGE" ]]; then
  echo "Error: no image set. Build one with tunix, maxtext, and" \
       "tpu-inference installed, then pass it via TUNIX_IMAGE=... or" \
       "--image=..."
  exit 1
fi

if [[ "$COMMAND" == "start" ]]; then
  if [[ "$DRY_RUN" != "true" ]] && [[ "${USE_AGENT_SANDBOX}" == "1" || "${USE_AGENT_SANDBOX}" == "true" || "${USE_AGENT_SANDBOX}" == "True" ]]; then
    kubectl create rolebinding xpk-sa-default-pod-exec -n "${SANDBOX_NAMESPACE:-trellis}" --role=pod-exec --serviceaccount=default:xpk-sa --dry-run=client -o yaml | kubectl apply -f - 2>/dev/null || true
    kubectl create rolebinding xpk-sa-default-power-users -n "${SANDBOX_NAMESPACE:-trellis}" --clusterrole=power-users --serviceaccount=default:xpk-sa --dry-run=client -o yaml | kubectl apply -f - 2>/dev/null || true
  fi
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
