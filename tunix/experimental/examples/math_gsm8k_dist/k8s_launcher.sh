#!/usr/bin/bash

ROLE=""
TUNIX_IMAGE="us-central1-docker.pkg.dev/cloud-tpu-multipod-dev/yangmu/tunix/tunix_base_image:latest"

export MODEL_NAME=${MODEL_NAME:-Qwen3-1.7B}
export MODEL_ID=${MODEL_ID:-Qwen/Qwen3-1.7B}
export MODEL_DIR=${MODEL_DIR:-artifacts/qwen3_dist_gsm8k/models}
export TOKENIZER_PATH=${TOKENIZER_PATH:-${MODEL_DIR}}

export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-128}
export BATCH_SIZE=${BATCH_SIZE:-2}
export NUM_GENERATIONS=${NUM_GENERATIONS:-2}
export MAX_STEPS=${MAX_STEPS:-1}
export TRAIN_MICRO_BATCH_SIZE=${TRAIN_MICRO_BATCH_SIZE:-1}
export MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-$((BATCH_SIZE * NUM_GENERATIONS))}
export EVAL_EVERY_N_STEPS=${EVAL_EVERY_N_STEPS:-1000000}
export LORA_RANK=${LORA_RANK:-16}
export LORA_ALPHA=${LORA_ALPHA:-16.0}
export USE_LORA=${USE_LORA:-0}

export ORCHESTRATOR_ID=$USER-orch
export ORCHESTRATOR_PORT=20000

export ROLLOUT_ID=$USER-roll
export ROLLOUT_PORT=20001

export TRAINER_ID=$USER-train
export TRAINER_PORT=20002

stop_orchestrator() {
  kubectl delete jobset "${ORCHESTRATOR_ID}"
}

start_orchestrator() {
  python tunix/experimental/distributed/deployment/yaml_generator.py \
    tunix/experimental/distributed/deployment/yamls/jobset.cpu.yaml \
    --jobset_name="${ORCHESTRATOR_ID}" \
    --cpu_machine=n2-standard-64 \
    --worker_container_image="${TUNIX_IMAGE}" \
    --worker_container_port="${ORCHESTRATOR_PORT}" \
    --worker_startup_command=" \
      python -m tunix.experimental.distributed.runtime.main \
        --discovery_id=${ORCHESTRATOR_ID} \
        --discovery_port=${ORCHESTRATOR_PORT} \
        --process_main=tunix.experimental.examples.math_gsm8k_dist.run_gsm8k_dist_grpo.main \
        --model_id=${MODEL_ID} \
        --tokenizer_path=${TOKENIZER_PATH} \
        --batch_size=${BATCH_SIZE} \
        --num_generations=${NUM_GENERATIONS} \
        --max_steps=${MAX_STEPS} \
        --max_prompt_length=${MAX_PROMPT_LENGTH} \
        --max_response_length=${MAX_RESPONSE_LENGTH} \
        --train_micro_batch_size=${TRAIN_MICRO_BATCH_SIZE} \
        --stop_workers_on_exit \
    " \
    | kubectl apply -f -
}

stop_trainer() {
  kubectl delete jobset "${TRAINER_ID}"
}

start_trainer() {
  python tunix/experimental/distributed/deployment/yaml_generator.py \
    tunix/experimental/distributed/deployment/yamls/jobset.pathways.yaml \
    --jobset_name="${TRAINER_ID}" \
    --tpu_slice=tpuv5:2x2x2 \
    --worker_container_image="${TUNIX_IMAGE}" \
    --worker_container_port="${TRAINER_PORT}" \
    --worker_startup_command=" \
      python -m tunix.experimental.distributed.runtime.main \
        --discovery_addrs=${ORCHESTRATOR_ID}:${ORCHESTRATOR_PORT} \
        --process_executor=tunix.experimental.distributed.runtime.executor.K8sExecutor \
        --process_main=tunix.experimental.examples.math_gsm8k_dist.run_trainer_node.main \
        --worker_id=${TRAINER_ID} \
        --port=${TRAINER_PORT} \
        --mesh_fsdp=8 \
        --model_name=${MODEL_NAME} \
        --model_id=${MODEL_ID} \
        --model_dir=${MODEL_DIR} \
        --tokenizer_path=${TOKENIZER_PATH} \
        --max_prompt_length=${MAX_PROMPT_LENGTH} \
        --max_response_length=${MAX_RESPONSE_LENGTH} \
        --mini_batch_size=${MINI_BATCH_SIZE} \
        --train_micro_batch_size=${TRAIN_MICRO_BATCH_SIZE} \
        --eval_every_n_steps=${EVAL_EVERY_N_STEPS} \
        --lora_rank=${LORA_RANK} \
        --lora_alpha=${LORA_ALPHA} \
    " \
    | kubectl apply -f -

}

stop_rollout() {
  kubectl delete jobset "${ROLLOUT_ID}"
}

start_rollout() {
  python tunix/experimental/distributed/deployment/yaml_generator.py \
    tunix/experimental/distributed/deployment/yamls/jobset.pathways.yaml \
    --jobset_name="${ROLLOUT_ID}" \
    --tpu_slice=tpuv5:2x2x2 \
    --worker_container_image="${TUNIX_IMAGE}" \
    --worker_container_port="${ROLLOUT_PORT}" \
    --worker_startup_command=" \
      SKIP_JAX_PRECOMPILE=1 python -m tunix.experimental.distributed.runtime.main \
        --discovery_addrs=${ORCHESTRATOR_ID}:${ORCHESTRATOR_PORT} \
        --process_executor=tunix.experimental.distributed.runtime.executor.K8sExecutor \
        --process_main=tunix.experimental.examples.math_gsm8k_dist.run_rollout_node.main \
        --worker_id=${ROLLOUT_ID} \
        --port=${ROLLOUT_PORT} \
        --model_id=${MODEL_ID} \
        --model_dir=${MODEL_DIR} \
        --tokenizer_path=${TOKENIZER_PATH} \
        --max_prompt_length=${MAX_PROMPT_LENGTH} \
        --max_response_length=${MAX_RESPONSE_LENGTH} \
        --lora_rank=${LORA_RANK} \
        --lora_alpha=${LORA_ALPHA} \
    " \
    | kubectl apply -f -
}

enter_kube_context() {
  PROJECT="cloud-tpu-multipod-dev"
  REGION="us-central1"
  ZONE="us-central1-a"
  CLUSTER="rl-scaffolding"

  export KUBECONFIG="${KUBECONFIG:-$HOME/.kube/config.$PROJECT.$REGION.$CLUSTER}"
  if ! [ -f "$KUBECONFIG" ] || ! kubectl get namespaces &>/dev/null; then
    gcloud container clusters get-credentials $CLUSTER --region=$REGION --project=$PROJECT --dns-endpoint &>/dev/null || { echo "gcloud get-credentials failed"; exit 1; }
    kubectl config use-context "gke_${PROJECT}_${REGION}_${CLUSTER}" >/dev/null || { echo "kubectl use-context failed"; exit 1; }
  fi
  kubectl config set-context --current --namespace=default >/dev/null || { echo "kubectl set-context failed"; exit 1; }
}

enter_kube_context

while [[ $# -gt 0 ]]; do
  case "$1" in
    --role)
      ROLE="$2"
      shift 2
      ;;
    --role=*)
      ROLE="${1#*=}"
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
    *)
      shift
      ;;
  esac
done

if [[ "$ROLE" == "orchestrator" ]]; then
  stop_orchestrator; start_orchestrator
elif [[ "$ROLE" == "trainer" ]]; then
  stop_trainer; start_trainer
elif [[ "$ROLE" == "rollout" ]]; then
  stop_rollout; start_rollout
elif [[ "$ROLE" == "stop" ]]; then
  stop_orchestrator
  stop_trainer
  stop_rollout
elif [[ "$ROLE" == "start" ]]; then
  stop_orchestrator
  stop_trainer
  stop_rollout
  start_orchestrator
  start_trainer
  start_rollout
else
  echo "Error: Invalid role '$ROLE'. Available roles: 'orchestrator', 'trainer', 'rollout'."
  exit 1
fi
