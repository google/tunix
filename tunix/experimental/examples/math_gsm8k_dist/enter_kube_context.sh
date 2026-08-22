#!/bin/bash

PROJECT=${PROJECT:-cloud-tpu-multipod-dev}
REGION=${REGION:-us-central1}
ZONE=${ZONE:-us-central1-a}
CLUSTER=${CLUSTER:-trellis-demo-0810}

CONTEXT_NAME="gke_${PROJECT}_${REGION}_${CLUSTER}"
if kubectl config get-contexts "$CONTEXT_NAME" &>/dev/null; then
  kubectl config use-context "$CONTEXT_NAME" >/dev/null || true
else
  export KUBECONFIG="${KUBECONFIG:-$HOME/.kube/config.$PROJECT.$REGION.$CLUSTER}"
  gcloud container clusters get-credentials $CLUSTER --region=$REGION --project=$PROJECT --dns-endpoint &>/dev/null || { echo "gcloud get-credentials failed"; exit 1; }
  kubectl config use-context "$CONTEXT_NAME" >/dev/null || { echo "kubectl use-context failed"; exit 1; }
fi
kubectl config set-context --current --namespace=default >/dev/null || true
