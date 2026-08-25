#!/bin/bash
# Usage:
#   source tunix/experimental/examples/math_gsm8k_dist/enter_kube_context.sh

PROJECT=${PROJECT:-cloud-tpu-multipod-dev}
REGION=${REGION:-us-central1}
ZONE=${ZONE:-us-central1-a}
CLUSTER=${CLUSTER:-trellis-demo-0810}

export KUBECONFIG="${KUBECONFIG:-$HOME/.kube/config.$PROJECT.$REGION.$CLUSTER}"
gcloud container clusters get-credentials $CLUSTER --region=$REGION --project=$PROJECT --dns-endpoint || { echo "gcloud get-credentials failed"; return; }
kubectl config use-context "gke_${PROJECT}_${REGION}_${CLUSTER}" || { echo "kubectl use-context failed"; return; }
kubectl config set-context --current --namespace=default || { echo "kubectl set-context failed"; return; }
