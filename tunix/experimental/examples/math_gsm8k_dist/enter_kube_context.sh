#!/bin/bash
# Usage:
#   source tunix/experimental/examples/math_gsm8k_dist/enter_kube_context.sh
#   bash tunix/experimental/examples/math_gsm8k_dist/enter_kube_context.sh
#   bash tunix/experimental/examples/math_gsm8k_dist/enter_kube_context.sh k9s

is_sourced() {
  [[ "${BASH_SOURCE[0]}" != "$0" ]]
}

finish() {
  local code="$1"
  if is_sourced; then
    return "$code"
  fi
  exit "$code"
}

PROJECT=${PROJECT:-cloud-tpu-multipod-dev}
REGION=${REGION:-us-central1}
ZONE=${ZONE:-us-central1-a}
CLUSTER=${CLUSTER:-trellis-demo-0810}
CLUSTER_LOCATION=${CLUSTER_LOCATION:-${REGION}}
KUBECTL_ENDPOINT_MODE=${KUBECTL_ENDPOINT_MODE:-standard}

export KUBECONFIG="$HOME/.kube/config.$PROJECT.$CLUSTER_LOCATION.$CLUSTER"
CONTEXT_NAME="gke_${PROJECT}_${CLUSTER_LOCATION}_${CLUSTER}"

fetch_credentials() {
  local -a args
  args=(container clusters get-credentials "$CLUSTER" --location="$CLUSTER_LOCATION" --project="$PROJECT")

  case "$KUBECTL_ENDPOINT_MODE" in
    standard)
      ;;
    dns)
      args+=(--dns-endpoint)
      ;;
    internal)
      args+=(--internal-ip)
      ;;
    *)
      echo "Unsupported KUBECTL_ENDPOINT_MODE '$KUBECTL_ENDPOINT_MODE'. Use one of: standard, dns, internal." >&2
      finish 1
      ;;
  esac

  gcloud "${args[@]}"
}

fetch_credentials || { echo "gcloud get-credentials failed" >&2; finish 1; }
kubectl config use-context "$CONTEXT_NAME" >/dev/null || { echo "kubectl use-context failed" >&2; finish 1; }
kubectl config set-context --current --namespace=default >/dev/null || true

if ! is_sourced; then
  if [[ $# -gt 0 ]]; then
    exec "$@"
  fi

  echo "Entered Kubernetes context ${CONTEXT_NAME} with KUBECONFIG=${KUBECONFIG}."
  echo "Starting an interactive shell so tools like k9s inherit this context."
  exec "${SHELL:-/bin/bash}" -i
fi
