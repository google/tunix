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

# Starts a single-host Pathways resource manager, worker and proxy so a trainer
# can run as an IFRT proxy client (JAX_PLATFORMS=proxy) instead of mcJAX. This
# is what `TRAINER_PATHWAYS=1` in the MaxText smoke recipe needs.
#
# Production runs the trainer on Pathways, so without this the CI smoke test
# covers MaxText, vLLM and Raiden but not the execution mode all three run under.
#
# Adapted from MaxText's .github/workflows/run_pathways_tests.yml, with two
# deliberate differences:
#
#   1. No --gcs_scratch_location. That flag only names a persistent XLA
#      compilation-cache root; the daemons start clean without it. Dropping it
#      keeps this hermetic -- no bucket, no credentials, nothing to clean up.
#      (The flag also rejects non-GCS paths outright, so there is no local
#      equivalent to point it at.)
#   2. The worker is confined to a chip subset, because the rollout holds the
#      rest of the host. MaxText's Pathways CI instead gives the worker a whole
#      v6e-4 host, so the split is specific to this test. The resource manager
#      validates the detected topology against --instance_type strictly; a
#      rejected split shows up as "The newly added instance does not match with
#      the expected instances" in the worker log. Note that the same message
#      appears for a generation mismatch, which is why the instance type is
#      autodetected below rather than hardcoded to the CI runner's value.
#
# The images are pulled by digest over the registry HTTP API rather than with
# docker, since this runs inside a container with no docker socket. Only
# usr/grte and usr/pathways are extracted, so nothing in the container's own
# /usr/bin or /usr/lib is overwritten.
#
# The client side (the trainer process) additionally needs:
#   JAX_PLATFORMS=proxy,cpu
#   JAX_BACKEND_TARGET=grpc://127.0.0.1:29000
#   IFRT_PROXY_USE_INSECURE_GRPC_CREDENTIALS=true
#   PATHWAYS_UNSAFE_UNSAFE_OVERRIDE_GRPC_CREDENTIALS=grpc_insecure_override
# The first two are set by launcher.sh under TRAINER_PATHWAYS=1; the recipe sets
# the credential overrides.

set -euo pipefail

# Chips the Pathways worker owns. Must not overlap ROLLOUT_TPU_CHIPS.
PATHWAYS_WORKER_TPU_CHIPS=${PATHWAYS_WORKER_TPU_CHIPS:-${TRAINER_TPU_CHIPS:-0,1,2,3}}

# Topology the resource manager expects the worker to report, as
# "<generation>:<XxY>". It must agree with the bounds below and with the chip
# count, and the generation token must match the hardware: passing tpuv6e on a
# v7 host fails with "The newly added instance does not match with the expected
# instances", which reads like a topology error but is a generation mismatch.
#
# Autodetected rather than defaulted to the CI runner's value, so the same
# script works on a developer's TPU VM. Neither source is jax.devices(), because
# initializing JAX here would claim the chips the worker is about to take.
# Override PATHWAYS_INSTANCE_TYPE to skip all this.
#
# CLOUD_TPU_ACCELERATOR is tried before the metadata server: on a TPU VM the
# metadata carries instance/attributes/accelerator-type, but the CI runners are
# GKE pods, where that attribute does not exist and the lookup comes back empty.
# There the container declares its hardware in this variable instead.
detect_accelerator_type() {
  if [[ -n "${CLOUD_TPU_ACCELERATOR:-}" ]]; then
    echo "${CLOUD_TPU_ACCELERATOR}"
    return
  fi
  curl -s -m 5 -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/accelerator-type" \
    2>/dev/null || true
}

detect_pathways_generation() {
  local accel
  accel=$(detect_accelerator_type)
  case "${accel}" in
    tpu7*)      echo "tpu7x" ;;      # Note: tpu7x, not tpuv7.
    v6e*)       echo "tpuv6e" ;;
    v6*)        echo "tpuv6lite" ;;
    v5litepod*) echo "tpuv5lite" ;;
    v5p*)       echo "tpuv5" ;;
    v4*)        echo "tpuv4" ;;
    *)          echo "" ;;
  esac
}

if [[ -z "${PATHWAYS_INSTANCE_TYPE:-}" ]]; then
  PW_GENERATION=$(detect_pathways_generation)
  if [[ -z "${PW_GENERATION}" ]]; then
    echo "Error: could not detect the TPU generation." >&2
    echo "Tried CLOUD_TPU_ACCELERATOR (='${CLOUD_TPU_ACCELERATOR:-unset}') and the" >&2
    echo "GCE metadata accelerator-type, which yielded '$(detect_accelerator_type)'." >&2
    echo "Set PATHWAYS_INSTANCE_TYPE explicitly, e.g. PATHWAYS_INSTANCE_TYPE=tpuv6e:2x2." >&2
    exit 1
  fi
  PW_NUM_CHIPS=$(awk -F, '{print NF}' <<< "${PATHWAYS_WORKER_TPU_CHIPS}")
  case "${PW_NUM_CHIPS}" in
    1) PW_TOPOLOGY="1x1" ;;
    2) PW_TOPOLOGY="1x2" ;;
    4) PW_TOPOLOGY="2x2" ;;
    8) PW_TOPOLOGY="2x4" ;;
    *)
      echo "Error: no known Pathways topology for ${PW_NUM_CHIPS} chips." >&2
      echo "Set PATHWAYS_INSTANCE_TYPE explicitly." >&2
      exit 1
      ;;
  esac
  PATHWAYS_INSTANCE_TYPE="${PW_GENERATION}:${PW_TOPOLOGY}"
  echo "Detected Pathways instance type: ${PATHWAYS_INSTANCE_TYPE} (accelerator=${PW_GENERATION}, chips=${PW_NUM_CHIPS})"
fi
PATHWAYS_CHIPS_PER_HOST_BOUNDS=${PATHWAYS_CHIPS_PER_HOST_BOUNDS:-${TPU_CHIPS_PER_HOST_BOUNDS:-2,2,1}}
PATHWAYS_HOST_BOUNDS=${PATHWAYS_HOST_BOUNDS:-${TPU_HOST_BOUNDS:-1,1,1}}

PATHWAYS_RM_PORT=${PATHWAYS_RM_PORT:-29001}
PATHWAYS_WORKER_PORT=${PATHWAYS_WORKER_PORT:-29005}
PATHWAYS_PROXY_PORT=${PATHWAYS_PROXY_PORT:-29000}

# Pinned by digest. Tags including ":latest" are mutable and rebuilt daily; the
# 2026-09-12 build regressed the XLA TPU compiler, which is why MaxText's CI
# pins these exact two digests.
PATHWAYS_SERVER_DIGEST=${PATHWAYS_SERVER_DIGEST:-sha256:6fb1399d265fc4532805bfb04d6f44a596b3111e30c7f2e4d9fafe98adf7456b}
PATHWAYS_PROXY_DIGEST=${PATHWAYS_PROXY_DIGEST:-sha256:57781f09427eb2f1863672c04b239ed5fce41e05e598da4c071880ce3eef51d5}

export PATHWAYS_UNSAFE_UNSAFE_OVERRIDE_GRPC_CREDENTIALS=grpc_insecure_override
export IFRT_PROXY_USE_INSECURE_GRPC_CREDENTIALS=true

# cloud_proxy_server_sanitized drops privileges to `nobody`, which some slim
# base images do not define.
grep -q '^nobody:' /etc/group || echo "nobody:x:65534:" >> /etc/group
grep -q '^nobody:' /etc/passwd || echo "nobody:x:65534:65534:nobody:/nonexistent:/usr/sbin/nologin" >> /etc/passwd

if [[ ! -x /usr/pathways/run/cloud_pathways_server_sanitized ]]; then
  echo "Extracting Pathways binaries..."
  PATHWAYS_SERVER_DIGEST="${PATHWAYS_SERVER_DIGEST}" \
  PATHWAYS_PROXY_DIGEST="${PATHWAYS_PROXY_DIGEST}" \
  python3 - <<'PY'
import io
import json
import os
import tarfile
import urllib.request

IMAGES = {
    "cloud-tpu-v2-images/pathways/server": os.environ["PATHWAYS_SERVER_DIGEST"],
    "cloud-tpu-v2-images/pathways/proxy_server": os.environ["PATHWAYS_PROXY_DIGEST"],
}


def get_token(repo):
  url = (
      "https://us-docker.pkg.dev/v2/token?service=us-docker.pkg.dev"
      f"&scope=repository:{repo}:pull"
  )
  with urllib.request.urlopen(url) as r:
    return json.load(r)["token"]


def extract(repo, ref):
  headers = {
      "Authorization": f"Bearer {get_token(repo)}",
      "Accept": "application/vnd.docker.distribution.manifest.v2+json",
  }
  req = urllib.request.Request(
      f"https://us-docker.pkg.dev/v2/{repo}/manifests/{ref}", headers=headers
  )
  with urllib.request.urlopen(req) as r:
    manifest = json.load(r)
    # Proof the pin took effect, and the only record of which build a run used.
    print(f"  {repo} @ {ref} -> {r.headers.get('Docker-Content-Digest')}", flush=True)

  for layer in manifest.get("layers", []):
    req = urllib.request.Request(
        f"https://us-docker.pkg.dev/v2/{repo}/blobs/{layer['digest']}",
        headers=headers,
    )
    with urllib.request.urlopen(req) as r:
      blob = r.read()
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:*") as tar:
      for member in tar:
        if member.name.startswith("/") or ".." in member.name:
          continue
        norm = os.path.normpath(member.name).lstrip("./")
        # Only these two trees. The layers also carry a full distro root, and
        # unpacking that over the running container would replace /bin/sh.
        if not (norm.startswith("usr/grte") or norm.startswith("usr/pathways")):
          continue
        target = os.path.join("/", norm)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        if member.isdir():
          os.makedirs(target, exist_ok=True)
          continue
        if member.issym() or member.islnk():
          if not os.path.lexists(target):
            os.symlink(member.linkname, target)
          continue
        src = tar.extractfile(member)
        if src is None:
          continue
        if os.path.lexists(target):
          os.remove(target)
        with open(target, "wb") as out:
          out.write(src.read())
        os.chmod(target, (member.mode or 0o755) | 0o755)


for repo, ref in IMAGES.items():
  print(f"Extracting {repo}...")
  extract(repo, ref)
print("Extraction complete.")
PY
fi

mkdir -p /tmp/pathways-logs

echo "Starting Pathways resource manager (${PATHWAYS_INSTANCE_TYPE}) on :${PATHWAYS_RM_PORT}..."
TPU_SKIP_MDS_QUERY=true /usr/pathways/run/cloud_pathways_server_sanitized \
  --server_port="${PATHWAYS_RM_PORT}" \
  --node_type=resource_manager \
  --enforce_kernel_ipv6_support=false \
  --instance_count=1 \
  --instance_type="${PATHWAYS_INSTANCE_TYPE}" \
  > /tmp/pathways-logs/pathways_rm.log 2>&1 &

echo "Starting Pathways worker on chips ${PATHWAYS_WORKER_TPU_CHIPS}..."
(
  export TPU_VISIBLE_CHIPS="${PATHWAYS_WORKER_TPU_CHIPS}"
  export TPU_VISIBLE_DEVICES="${PATHWAYS_WORKER_TPU_CHIPS}"
  export TPU_CHIPS_PER_HOST_BOUNDS="${PATHWAYS_CHIPS_PER_HOST_BOUNDS}"
  export TPU_HOST_BOUNDS="${PATHWAYS_HOST_BOUNDS}"
  export LIBTPU_INIT_ARGS="--deepsea_chips_per_host_bounds=${PATHWAYS_CHIPS_PER_HOST_BOUNDS} --deepsea_host_bounds=${PATHWAYS_HOST_BOUNDS}"
  export ALLOW_MULTIPLE_LIBTPU_LOAD=1
  exec /usr/pathways/run/cloud_pathways_server_sanitized \
    --server_port="${PATHWAYS_WORKER_PORT}" \
    --resource_manager_address="127.0.0.1:${PATHWAYS_RM_PORT}" \
    --enforce_kernel_ipv6_support=false \
    > /tmp/pathways-logs/pathways_worker.log 2>&1
) &

echo "Starting Pathways proxy on :${PATHWAYS_PROXY_PORT}..."
/usr/pathways/run/cloud_proxy_server_sanitized \
  --server_port="${PATHWAYS_PROXY_PORT}" \
  --resource_manager_address="127.0.0.1:${PATHWAYS_RM_PORT}" \
  --xla_tpu_scoped_vmem_limit_kib=65536 \
  --xla_tpu_spmd_rng_bit_generator_unsafe=true \
  > /tmp/pathways-logs/pathways_proxy.log 2>&1 &

echo "Waiting for the proxy to listen on 127.0.0.1:${PATHWAYS_PROXY_PORT}..."
for _ in {1..45}; do
  if python3 -c "import socket,sys; socket.create_connection(('127.0.0.1', int(sys.argv[1])), timeout=1)" \
      "${PATHWAYS_PROXY_PORT}" 2>/dev/null; then
    sleep 2  # let the gRPC server finish its handshakes
    echo "Pathways daemons are up."
    exit 0
  fi
  sleep 2
done

echo "ERROR: the Pathways proxy did not come up within 90s." >&2
for log in /tmp/pathways-logs/pathways_{rm,worker,proxy}.log; do
  echo "=== ${log} ==="
  tail -n 100 "${log}" 2>/dev/null || true
done
exit 1
