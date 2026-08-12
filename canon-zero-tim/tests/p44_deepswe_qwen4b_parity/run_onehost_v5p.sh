#!/usr/bin/env bash
# Real direct-attached v5p lowering gate for the P44 Qwen3-4B matmul repair.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
IMAGE="${1:?usage: run_onehost_v5p.sh <image@sha256:digest|sha256:image-id>}"
DOCKER="${DOCKER:-sudo docker}"

if [[ ! "$IMAGE" =~ @sha256:[0-9a-f]{64}$ ]] && \
   [[ ! "$IMAGE" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  echo "P44 one-host v5p gate requires a sha256-pinned image" >&2
  exit 2
fi

$DOCKER image inspect "$IMAGE" \
  --format 'P44_ONEHOST_V5P image_id={{.Id}}' >/dev/null
$DOCKER run --rm --privileged --network=host --ipc=host \
  -v "$ROOT:/workspace:ro" \
  -w /workspace \
  "$IMAGE" \
  bash -euo pipefail -c '
    rm -rf /tmp/p44-onehost-overlay
    bash canon-zero-tim/install.sh /tmp/p44-onehost-overlay \
      --from-path /usr/local/lib/python3.12/site-packages/tpu_inference \
      --model qwen4b
    env -u JAX_BACKEND_TARGET -u PATHWAYS_HEAD \
      JAX_PLATFORMS=tpu PYTHONPATH=/tmp/p44-onehost-overlay \
      python3 canon-zero-tim/tests/p44_deepswe_qwen4b_parity/probe_matmul_dim_padding.py \
      --mode tpu
    echo "P44_ONEHOST_V5P_MATMUL_PASS model=qwen4b devices=4"
  '
