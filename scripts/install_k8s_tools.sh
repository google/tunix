#!/bin/bash
# Copyright 2025 Google LLC
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

# Installs Google Cloud SDK, GKE auth plugin, kubectl, k9s, and debugging
# utilities inside the container when `INSTALL_K8S_TOOLS=true`.

set -euo pipefail

apt-get update
apt-get install -y --no-install-recommends \
  apt-transport-https \
  ca-certificates \
  gnupg \
  lsof \
  procps \
  vim

echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
  | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
  | gpg --batch --yes --no-tty --dearmor -o /usr/share/keyrings/cloud.google.gpg

apt-get update
apt-get install -y --no-install-recommends \
  google-cloud-cli \
  google-cloud-cli-gke-gcloud-auth-plugin \
  kubectl

curl -sS https://webinstall.dev/k9s | bash
rm -rf /var/lib/apt/lists/*
