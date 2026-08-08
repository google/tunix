#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
IMAGE="${1:-tunix_frozenlake_image:vllm-tpu0.25.0}"
DOCKER="${DOCKER:-sudo docker}"

$DOCKER image inspect "$IMAGE" --format 'P33_EXACT_IMAGE image_id={{.Id}}' >/dev/null
$DOCKER run --rm \
  -v "$ROOT:/workspace:ro" \
  -w /workspace \
  -e JAX_PLATFORMS=cpu \
  "$IMAGE" \
  bash -euo pipefail -c '
    qwen1p7b_overlay="$(mktemp -d /tmp/p33-qwen1p7b.XXXXXX)"
    qwen8b_overlay="$(mktemp -d /tmp/p33-qwen8b.XXXXXX)"
    trap '\''rm -r "$qwen1p7b_overlay" "$qwen8b_overlay"'\'' EXIT
    bash canon-zero-tim/install.sh "$qwen1p7b_overlay" \
      --from-path /usr/local/lib/python3.12/site-packages/tpu_inference \
      --model qwen1p7b
    python3 canon-zero-tim/tests/p33_workloads/test_decode_logprob_chunking.py \
      --overlay "$qwen1p7b_overlay"
    bash canon-zero-tim/install.sh "$qwen8b_overlay" \
      --from-path /usr/local/lib/python3.12/site-packages/tpu_inference \
      --model qwen8b
    echo "P33_EXACT_IMAGE_PASS decode_chunk_cases=5 overlays=2"
  '
