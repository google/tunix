#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
IMAGE="${1:?usage: run_exact_image.sh <image@sha256:digest|sha256:image-id>}"
DOCKER="${DOCKER:-sudo docker}"

if [[ ! "$IMAGE" =~ @sha256:[0-9a-f]{64}$ ]] && \
   [[ ! "$IMAGE" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  echo "P44 exact-image gate requires a sha256-pinned image" >&2
  exit 2
fi

$DOCKER image inspect "$IMAGE" \
  --format 'P44_EXACT_IMAGE image_id={{.Id}}' >/dev/null
$DOCKER run --rm \
  -v "$ROOT:/workspace:ro" \
  -w /workspace \
  -e JAX_PLATFORMS=cpu \
  "$IMAGE" \
  bash -euo pipefail -c '
    rm -rf /tmp/p44-overlay
    bash canon-zero-tim/install.sh /tmp/p44-overlay \
      --from-path /usr/local/lib/python3.12/site-packages/tpu_inference \
      --model qwen4b
    bash canon-zero-tim/tests/p44_deepswe_qwen4b_parity/run_cpu.sh
    PYTHONPATH=/workspace/canon-zero-tim/src/engine_shims/models/qwen4b \
      python3 canon-zero-tim/src/engine_shims/models/qwen4b/p22xf_contract.py
    echo "P44_EXACT_IMAGE_CPU_PASS overlay=qwen4b"
  '
