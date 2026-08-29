#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
image_ref="${1:-tunix_frozenlake_image:vllm-tpu0.25.0}"
docker="${DOCKER:-sudo docker}"

image_id="$($docker image inspect "$image_ref" --format '{{.Id}}')"
if [[ ! "$image_id" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  echo "Native exact-image gate could not resolve immutable image ID: $image_id" >&2
  exit 2
fi
echo "V1_GSM8K_NATIVE_FULL_EXACT_IMAGE image_ref=$image_ref image_id=$image_id"

$docker run --rm \
  -v "$root:/workspace:ro" \
  -w /workspace \
  -e JAX_PLATFORMS=cpu \
  -e PYTHONPATH=/workspace \
  -e TEST_GSM8K_NATIVE_STOCK_PREFLIGHT=1 \
  "$image_id" \
  bash -euo pipefail -c '
    python3 canon-zero-tim/tests/v1_gsm8k_native_full/test_renderer.py
    python3 canon-zero-tim/tests/v1_phase4/test_three_full_renderer.py \
      ThreeFullRendererTest.test_gsm8k_p74_wrapper_renders_one_job_and_never_launches
    echo "V1_GSM8K_NATIVE_FULL_EXACT_IMAGE_PASS native_contract=10 zero_neighbor=1"
  '
