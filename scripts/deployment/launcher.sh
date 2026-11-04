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
#
#   To run a role with specific docker image:
#     ./launcher.sh --role=orchestrator --image=my_awesome_image
#
#   To run a role in local mode:
#     ./launcher.sh --role=orchestrator --local

LOCAL_MODE=false
ROLE=""
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

# Calculate repo root path (two directories up from scripts/deployment)
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

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
    # python -m tunix.distributed.worker.orchestrator --name=orchestrator --registration_service_port=12345
    python -c "import logging, sys, runpy; \
      logging.basicConfig(level=logging.INFO); \
      sys.argv = ['python -m tunix.distributed.worker.orchestrator', '--name=orchestrator', '--registration_service_port=12345']; \
      runpy.run_module('tunix.distributed.worker.orchestrator', run_name='__main__')"
  else
    cd "$SCRIPT_DIR"
    kubectl delete jobset orchestrator
    python yaml_generator.py \
      yamls/jobset.cpu.yaml \
      --jobset_name=orchestrator \
      --cpu_machine=n2-standard-64 \
      --worker_container_image="$TUNIX_IMAGE" \
      --worker_container_port=12345 \
      --worker_startup_command="python -m tunix.distributed.worker.orchestrator --name=orchestrator --registration_service_port=12345" \
      | kubectl apply -f -
  fi
}

launch_rollout() {
  if [[ "$LOCAL_MODE" == "true" ]]; then
    cd "$REPO_ROOT"
    # python -m tunix.distributed.worker.rollout --name=rollout --registration_service_address=localhost:12345 --rollout_service_port=11111
    python -c "import logging, sys, runpy; \
      logging.basicConfig(level=logging.INFO); \
      sys.argv = ['python -m tunix.distributed.worker.rollout', '--name=rollout', '--registration_service_address=localhost:12345', '--rollout_service_port=11111']; \
      runpy.run_module('tunix.distributed.worker.rollout', run_name='__main__')"
  else
    cd "$SCRIPT_DIR"
    kubectl delete jobset rollout
    python yaml_generator.py \
      yamls/jobset.pathways.yaml \
      --jobset_name=rollout \
      --tpu_slice=tpuv5e:4x4 \
      --pathways_server_image=us-central1-docker.pkg.dev/cloud-tpu-multipod-dev/yangmu/tunix/unsanitized_server:latest \
      --pathways_proxy_server_image=us-central1-docker.pkg.dev/cloud-tpu-multipod-dev/yangmu/tunix/unsanitized_proxy_server:latest \
      --worker_container_image="$TUNIX_IMAGE" \
      --worker_container_port=11111 \
      --worker_startup_command="python -m tunix.distributed.worker.rollout --name=rollout --registration_service_address=orchestrator-proc-0-0.orchestrator:12345 --rollout_service_port=11111" \
      | kubectl apply -f -
  fi
}

enter_kube_context

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
