#!/bin/bash

PROJECT=${PROJECT:-cloud-tpu-multipod-dev}
REGION=${REGION:-europe-west4}
ZONE=${ZONE:-europe-west4-a}
CLUSTER=${CLUSTER:-auto-v5p-8-bodaborg}

export KUBECONFIG="${KUBECONFIG:-$HOME/.kube/config.$PROJECT.$REGION.$CLUSTER}"
gcloud container clusters get-credentials $CLUSTER --region=$REGION --project=$PROJECT --dns-endpoint &>/dev/null || { echo "gcloud get-credentials failed"; exit 1; }
kubectl config use-context "gke_${PROJECT}_${REGION}_${CLUSTER}" >/dev/null || { echo "kubectl use-context failed"; exit 1; }
kubectl config set-context --current --namespace=default >/dev/null || { echo "kubectl set-context failed"; exit 1; }
