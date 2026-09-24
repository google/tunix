#!/bin/bash

# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Installs Raiden / TPU Sync, the native TPU weight-synchronization library.
#
# Resolution order:
#   1. A locally built wheel in $RAIDEN_WHEEL_DIR (see build_raiden_wheel.sh).
#   2. The wheel pinned below.
#
# The pin lives here rather than in pyproject.toml because pip cannot install
# it directly: the bucket enforces public access prevention, so the fetch needs
# Google Cloud credentials that pip has no way to present.
#
# Pinning a URL rather than a package name is also deliberate. A package named
# `tpu-raiden-jax` exists on public PyPI and is malicious: a dependency
# confusion implant that exfiltrates environment variables on import. Resolving
# by name with `--extra-index-url` lets pip prefer that copy whenever it
# advertises a higher version. A direct URL never consults an index at all.
#
# Upstream renamed the project: the wheel is `tpu_sync_jax` (imported as
# `tpu_sync`), formerly `tpu_raiden_jax`. See https://github.com/google/tpu-sync
# The wheel is cp312-only and therefore requires Python 3.12.

set -euo pipefail

RAIDEN_WHEEL_URL=${RAIDEN_WHEEL_URL:-"https://storage.googleapis.com/tunix-ci-artifacts/raiden/tpu_sync_jax-0.0.1.dev20260914193202-cp312-cp312-manylinux_2_31_x86_64.whl"}
RAIDEN_WHEEL_SHA256=${RAIDEN_WHEEL_SHA256:-"a442ac543f54d8ff11d22dbd009671890f2fcf42446572ddb59a6eaabdee8f94"}

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RAIDEN_WHEEL_DIR=${RAIDEN_WHEEL_DIR:-"${ROOT_DIR}/raiden_wheels"}
if command -v uv >/dev/null 2>&1; then
  PIP_INSTALL=(uv pip install --reinstall --no-deps)
else
  PIP_INSTALL=(python3 -m pip install --force-reinstall --no-deps)
fi

# Raiden is imported at module scope by the trainer, the rollout, and the
# orchestrator. If it is missing, weight sync silently no-ops and only surfaces
# much later as a confusing transfer failure, so assert the import every time.
verify_install() {
  python3 -c "import tpu_sync"
  echo "Raiden (tpu_sync) installed."
}

compile_protos() {
  local proto_dir="${ROOT_DIR}/tunix/experimental/distributed"
  if [[ ! -d "${proto_dir}" ]]; then
    proto_dir="${ROOT_DIR}/../tunix/experimental/distributed"
  fi
  local base_dir
  base_dir=$(cd "${proto_dir}/../../.." && pwd)
  local discovery_proto="${base_dir}/tunix/experimental/distributed/runtime/discovery/discovery_service.proto"
  local discovery_pb2="${base_dir}/tunix/experimental/distributed/runtime/discovery/discovery_service_pb2.py"
  local discovery_pb2_grpc="${base_dir}/tunix/experimental/distributed/runtime/discovery/discovery_service_pb2_grpc.py"
  if [[ ! -f "${discovery_proto}" ]]; then
    echo "ERROR: Expected distributed orchestrator proto not found at ${discovery_proto}" >&2
    exit 1
  fi

  if [[ -f "${discovery_pb2}" && -f "${discovery_pb2_grpc}" && "${discovery_pb2}" -nt "${discovery_proto}" ]]; then
    echo "Distributed protobuf definitions already up to date."
    return 0
  fi

  echo "Compiling distributed runtime gRPC protobuf definitions..."
  python3 -c "import grpc_tools.protoc" 2>/dev/null || python3 -m pip install grpcio-tools

  python3 -m grpc_tools.protoc \
    -I"${base_dir}" \
    --python_out="${base_dir}" \
    --grpc_python_out="${base_dir}" \
    "${discovery_proto}"
  python3 -c "import tunix.experimental.distributed.runtime.discovery.discovery_service_pb2"
  echo "Distributed protobuf definitions compiled and verified."
}

# A locally built wheel wins, so the developer workflow never touches the
# network. build_raiden_wheel.sh produces these.
if compgen -G "${RAIDEN_WHEEL_DIR}/*.whl" >/dev/null; then
  echo "Installing locally built Raiden wheel(s) from ${RAIDEN_WHEEL_DIR}:"
  ls -1 "${RAIDEN_WHEEL_DIR}"/*.whl
  "${PIP_INSTALL[@]}" "${RAIDEN_WHEEL_DIR}"/*.whl
  verify_install
  compile_protos
  exit 0
fi

WHEEL_NAME=$(basename "${RAIDEN_WHEEL_URL}")
DEST_DIR=$(mktemp -d)
trap 'rm -rf "${DEST_DIR}"' EXIT
DEST="${DEST_DIR}/${WHEEL_NAME}"

echo "Installing pinned Raiden wheel:"
echo "  wheel:  ${WHEEL_NAME}"
echo "  sha256: ${RAIDEN_WHEEL_SHA256}"

fetch_wheel() {
  local gcs_uri="gs://${RAIDEN_WHEEL_URL#https://storage.googleapis.com/}"

  if command -v gcloud >/dev/null 2>&1; then
    echo "Fetching with gcloud from ${gcs_uri}..."
    gcloud storage cp "${gcs_uri}" "${DEST}" && return 0
  fi

  if command -v gsutil >/dev/null 2>&1; then
    echo "Fetching with gsutil from ${gcs_uri}..."
    gsutil cp "${gcs_uri}" "${DEST}" && return 0
  fi

  local token sa_email
  sa_email=$(curl -fsS -H "Metadata-Flavor: Google" --max-time 5 \
    "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/email" \
    2>/dev/null) || sa_email="unknown"

  token=$(curl -fsS -H "Metadata-Flavor: Google" --max-time 5 \
    "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token" \
    2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin)['access_token'])" \
    2>/dev/null) || token=""

  if [[ -n "${token}" ]]; then
    echo "Fetching with an ambient GCE service-account token (${sa_email})..."
    curl -fsSL -H "Authorization: Bearer ${token}" "${RAIDEN_WHEEL_URL}" -o "${DEST}" && return 0
  fi

  echo "Fetching unauthenticated..."
  curl -fsSL "${RAIDEN_WHEEL_URL}" -o "${DEST}"
}

if ! fetch_wheel; then
  cat >&2 <<EOF

Error: could not download the pinned Raiden wheel.
  ${RAIDEN_WHEEL_URL}

That bucket enforces public access prevention, so the download needs Google
Cloud credentials with read access to it. Either:

  * run 'gcloud auth application-default login' (or use a machine whose
    service account has roles/storage.objectViewer on the bucket), or
  * build the wheel from source and drop it in ${RAIDEN_WHEEL_DIR}:
        bash scripts/build_raiden_wheel.sh
EOF
  exit 1
fi

echo "${RAIDEN_WHEEL_SHA256}  ${DEST}" | sha256sum -c -

"${PIP_INSTALL[@]}" "${DEST}"
verify_install
compile_protos
