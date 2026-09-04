#!/bin/bash
# Usage:
#   source tunix/experimental/examples/math_gsm8k_dist/enter_kube_context.sh

PROJECT=${PROJECT:-cloud-tpu-multipod-dev}
REGION=${REGION:-us-central1}
ZONE=${ZONE:-us-central1-a}
CLUSTER=${CLUSTER:-trellis-demo-0810}

LOCATION_NAME=$(gcloud container clusters list \
  --project="$PROJECT" \
  --filter="name=$CLUSTER" \
  --format='value(location)' \
  | head -n 1)

if [[ -z "$LOCATION_NAME" ]]; then
  echo "Could not determine location for cluster '$CLUSTER' in project '$PROJECT'." >&2
  return 1 2>/dev/null || exit 1
fi

LOCATION_FLAG="--region=$LOCATION_NAME"
if [[ "$LOCATION_NAME" =~ -[a-z]$ ]]; then
  LOCATION_FLAG="--zone=$LOCATION_NAME"
fi

export KUBECONFIG="${KUBECONFIG:-$HOME/.kube/config.$PROJECT.$LOCATION_NAME.$CLUSTER}"
CONTEXT_NAME="gke_${PROJECT}_${LOCATION_NAME}_${CLUSTER}"

if kubectl config get-contexts "$CONTEXT_NAME" &>/dev/null; then
  kubectl config use-context "$CONTEXT_NAME" >/dev/null || true
else
  gcloud container clusters get-credentials "$CLUSTER" "$LOCATION_FLAG" --project="$PROJECT" --dns-endpoint || { echo "gcloud get-credentials failed" >&2; return 1 2>/dev/null || exit 1; }
  kubectl config use-context "$CONTEXT_NAME" >/dev/null || { echo "kubectl use-context failed" >&2; return 1 2>/dev/null || exit 1; }
fi
kubectl config set-context --current --namespace=default >/dev/null || true
