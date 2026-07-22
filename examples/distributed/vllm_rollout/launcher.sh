#!/usr/bin/bash
#
# Usage:
#   To run interactively:
#     ./launcher.sh
#     ./launcher.sh --local
#
#   To run a role:
#     ./launcher.sh --role=orchestrator
#     ./launcher.sh --role=rollout
#     ./launcher.sh --role=rollout --worker_id=1
#
#   To run a role with specific docker image:
#     ./launcher.sh --role=orchestrator --image=my_awesome_image
#
#   To run a role in local mode:
#     ./launcher.sh --role=orchestrator --local

# Flag set by --local to run processes locally instead of Kubernetes.
LOCAL_MODE=false
# Role to execute ('orchestrator' or 'rollout').
ROLE=""
# For rollout worker. If all, start all workers, otherwise, start just one with this id (e.g. --id=1).
WORKER_ID="all"
# Docker image URI used for worker containers on Kubernetes.
TUNIX_IMAGE="us-central1-docker.pkg.dev/cloud-tpu-multipod-dev/yangmu/tunix/tunix_base_image:latest"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --local)
      LOCAL_MODE=true
      shift
      ;;
    --role)
      ROLE="$2"
      shift 2
      ;;
    --role=*)
      ROLE="${1#*=}"
      shift
      ;;
    --worker_id)
      WORKER_ID="$2"
      shift 2
      ;;
    --worker_id=*)
      WORKER_ID="${1#*=}"
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

# Calculate repo root path (three directories up from examples/distributed/vllm_rollout).
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
REPO_ROOT="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"

# Configures kubectl to connect to the required GKE cluster and namespace.
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

# Launches the orchestrator process locally or on Kubernetes (GKE).
launch_orchestrator() {
  if [[ "$LOCAL_MODE" == "true" ]]; then
    cd "$REPO_ROOT"
    PYTHONPATH=./examples/distributed python -m tunix.experimental.distributed.runtime.main --discovery_id=orchestrator --discovery_port=12345 --process_main=vllm_rollout.orchestrator.main
  else
    cd "$REPO_ROOT/tunix/experimental/distributed/deployment"
    kubectl delete jobset orchestrator
    python yaml_generator.py \
      yamls/jobset.cpu.yaml \
      --jobset_name=orchestrator \
      --cpu_machine=n2-standard-64 \
      --worker_container_image="$TUNIX_IMAGE" \
      --worker_container_port=12345 \
      --worker_startup_command="PYTHONPATH=./examples/distributed python -m tunix.experimental.distributed.runtime.main --discovery_id=orchestrator --discovery_port=12345 --process_executor=tunix.experimental.distributed.runtime.executor.K8sExecutor --process_main=vllm_rollout.orchestrator.main --min_rollout_workers=2" \
      | kubectl apply -f -
  fi
}

# Launches rollout worker processes locally or on Kubernetes (GKE).
launch_rollout() {
  if [[ "$LOCAL_MODE" == "true" ]]; then
    cd "$REPO_ROOT"
    PYTHONPATH=./examples/distributed python -m tunix.experimental.distributed.runtime.main --discovery_addrs=orchestrator:12345 --process_main=vllm_rollout.rollout.main --worker_id=rollout-0 --service_port=11111
  else
    cd "$REPO_ROOT/tunix/experimental/distributed/deployment"
    if [[ "$WORKER_ID" == "all" ]]; then
      for ((i=0; i<=3; i++)); do
        kubectl delete jobset rollout-$i
        python yaml_generator.py \
          yamls/jobset.pathways.yaml \
          --jobset_name=rollout-$i \
          --tpu_slice=tpuv5e:4x4 \
          --pathways_server_image=us-central1-docker.pkg.dev/cloud-tpu-multipod-dev/yangmu/tunix/unsanitized_server:latest \
          --pathways_proxy_server_image=us-central1-docker.pkg.dev/cloud-tpu-multipod-dev/yangmu/tunix/unsanitized_proxy_server:latest \
          --worker_container_image="$TUNIX_IMAGE" \
          --worker_container_port=$((10000+i)) \
          --worker_startup_command="PYTHONPATH=./examples/distributed python -m tunix.experimental.distributed.runtime.main --discovery_addrs=orchestrator:12345 --process_executor=tunix.experimental.distributed.runtime.executor.K8sExecutor --process_main=vllm_rollout.rollout.main --worker_id=rollout-$i --service_port=$((10000+i))" \
          | kubectl apply -f -
      done
    else
      kubectl delete jobset "rollout-${WORKER_ID}"
      python yaml_generator.py \
        yamls/jobset.pathways.yaml \
        --jobset_name="rollout-${WORKER_ID}" \
        --tpu_slice=tpuv5e:4x4 \
        --pathways_server_image=us-central1-docker.pkg.dev/cloud-tpu-multipod-dev/yangmu/tunix/unsanitized_server:latest \
        --pathways_proxy_server_image=us-central1-docker.pkg.dev/cloud-tpu-multipod-dev/yangmu/tunix/unsanitized_proxy_server:latest \
        --worker_container_image="$TUNIX_IMAGE" \
        --worker_container_port=11111 \
        --worker_startup_command="PYTHONPATH=./examples/distributed python -m tunix.experimental.distributed.runtime.main --discovery_addrs=orchestrator:12345 --process_executor=tunix.experimental.distributed.runtime.executor.K8sExecutor --process_main=vllm_rollout.rollout.main --worker_id=rollout-${WORKER_ID} --service_port=11111" \
        | kubectl apply -f -
    fi
  fi
}

if [[ "$LOCAL_MODE" == "false" ]]; then
  enter_kube_context
fi

if [[ -n "$ROLE" ]]; then
  if [[ "$ROLE" == "orchestrator" ]]; then
    launch_orchestrator
  elif [[ "$ROLE" == "rollout" ]]; then
    launch_rollout
  else
    echo "Error: Invalid role '$ROLE'. Available roles: 'orchestrator', 'rollout'."
    exit 1
  fi
else
  echo "Available roles:"
  echo "  [0] orchestrator"
  echo "  [1] rollout"
  while true; do
    echo -n "Select role to launch [0]: "
    read role_idx
    role_idx=${role_idx:-0}
    if [[ "$role_idx" == "0" ]]; then launch_orchestrator; break
    elif [[ "$role_idx" == "1" ]]; then launch_rollout; break
    else echo "Invalid selection. Please try again."
    fi
  done
fi
