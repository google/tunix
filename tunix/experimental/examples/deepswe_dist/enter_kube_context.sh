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
#
# Usage:
#   source tunix/experimental/examples/deepswe_dist/enter_kube_context.sh

PROJECT=${PROJECT:-cloud-tpu-multipod-dev}
REGION=${REGION:-us-central1}
ZONE=${ZONE:-us-central1-a}
CLUSTER=${CLUSTER:-trellis-demo-0810}

export KUBECONFIG="${KUBECONFIG:-$HOME/.kube/config.$PROJECT.$REGION.$CLUSTER}"
CONTEXT_NAME="gke_${PROJECT}_${REGION}_${CLUSTER}"

if kubectl config get-contexts "$CONTEXT_NAME" &>/dev/null; then
  kubectl config use-context "$CONTEXT_NAME" >/dev/null || true
else
  gcloud container clusters get-credentials $CLUSTER --region=$REGION --project=$PROJECT --dns-endpoint || { echo "gcloud get-credentials failed" >&2; return 1 2>/dev/null || exit 1; }
  kubectl config use-context "$CONTEXT_NAME" >/dev/null || { echo "kubectl use-context failed" >&2; return 1 2>/dev/null || exit 1; }
fi
kubectl config set-context --current --namespace=default >/dev/null || true
