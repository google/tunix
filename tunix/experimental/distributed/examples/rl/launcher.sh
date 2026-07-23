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
#     ./launcher.sh --role=trainer
#
#   To run a role with specific docker image:
#     ./launcher.sh --role=orchestrator --image=my_awesome_image
#
#   To run a role in local mode:
#     ./launcher.sh --role=orchestrator --local

# set by --local
LOCAL_MODE=false
# set by --role
ROLE=""
# set by --image
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

# Calculate repo root path (four directories up from experimental/distributed/examples/rl)
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
REPO_ROOT="$(dirname "$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")")"

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

launch_orchestrator() {
  if [[ "$LOCAL_MODE" == "true" ]]; then
    cd "$REPO_ROOT"
    python -m tunix.experimental.distributed.runtime.main \
      --discovery_id=orchestrator \
      --discovery_port=12345 \
      --process_main=tunix.experimental.distributed.examples.rl.orchestrator.main \
      --max_train_step=1000
  else
    cd "$REPO_ROOT"
    kubectl delete jobset orchestrator
    python tunix/experimental/distributed/deployment/yaml_generator.py \
      tunix/experimental/distributed/deployment/yamls/jobset.cpu.yaml \
      --jobset_name=orchestrator \
      --cpu_machine=n2-standard-64 \
      --worker_container_image="$TUNIX_IMAGE" \
      --worker_container_port=12345 \
      --worker_startup_command="python -m tunix.experimental.distributed.runtime.main \
        --discovery_id=orchestrator \
        --discovery_port=12345 \
        --process_executor=tunix.experimental.distributed.runtime.executor.K8sExecutor \
        --process_main=tunix.experimental.distributed.examples.rl.orchestrator.main \
        --max_train_step=1000" \
      | kubectl apply -f -
  fi
}

launch_rollout() {
  if [[ "$LOCAL_MODE" == "true" ]]; then
    cd "$REPO_ROOT"
    for ((i=0; i<=3; i++)); do
      python -m tunix.experimental.distributed.runtime.main \
        --discovery_addrs=orchestrator:12345 \
        --process_main=tunix.experimental.distributed.examples.rl.rollout.main \
        --server_id=rollout-$i \
        --server_port=$((10000+i)) &
    done
    wait
  else
    cd "$REPO_ROOT"
    for ((i=0; i<=3; i++)); do
      kubectl delete jobset rollout-$i
      python tunix/experimental/distributed/deployment/yaml_generator.py \
        tunix/experimental/distributed/deployment/yamls/jobset.pathways.yaml \
        --jobset_name=rollout-$i \
        --tpu_slice=tpuv5e:4x4 \
        --worker_container_image="$TUNIX_IMAGE" \
        --worker_container_port=$((10000+i)) \
        --worker_startup_command="python -m tunix.experimental.distributed.runtime.main \
          --discovery_addrs=orchestrator:12345 \
          --process_executor=tunix.experimental.distributed.runtime.executor.K8sExecutor \
          --process_main=tunix.experimental.distributed.examples.rl.rollout.main \
          --server_id=rollout-$i \
          --server_port=$((10000+i))" \
        | kubectl apply -f -
    done
  fi
}

launch_trainer() {
  if [[ "$LOCAL_MODE" == "true" ]]; then
    cd "$REPO_ROOT"
    python -m tunix.experimental.distributed.runtime.main \
      --discovery_addrs=orchestrator:12345 \
      --process_main=tunix.experimental.distributed.examples.rl.trainer.main \
      --server_id=trainer \
      --server_port=20000
  else
    cd "$REPO_ROOT"
    kubectl delete jobset trainer
    python tunix/experimental/distributed/deployment/yaml_generator.py \
      tunix/experimental/distributed/deployment/yamls/jobset.pathways.yaml \
      --jobset_name=trainer \
      --tpu_slice=tpuv5e:4x4 \
      --worker_container_image="$TUNIX_IMAGE" \
      --worker_container_port=20000 \
      --worker_startup_command="python -m tunix.experimental.distributed.runtime.main \
        --discovery_addrs=orchestrator:12345 \
        --process_executor=tunix.experimental.distributed.runtime.executor.K8sExecutor \
        --process_main=tunix.experimental.distributed.examples.rl.trainer.main \
        --server_id=trainer \
        --server_port=20000" \
      | kubectl apply -f -
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
  elif [[ "$ROLE" == "trainer" ]]; then
    launch_trainer
  else
    echo "Error: Invalid role '$ROLE'. Available roles: 'orchestrator', 'rollout', 'trainer'."
    exit 1
  fi
else
  echo "Available roles:"
  echo "  [0] orchestrator"
  echo "  [1] rollout"
  echo "  [2] trainer"
  while true; do
    echo -n "Select role to launch [0]: "
    read role_idx
    role_idx=${role_idx:-0}
    if [[ "$role_idx" == "0" ]]; then launch_orchestrator; break
    elif [[ "$role_idx" == "1" ]]; then launch_rollout; break
    elif [[ "$role_idx" == "2" ]]; then launch_trainer; break
    else echo "Invalid selection. Please try again."
    fi
  done
fi
