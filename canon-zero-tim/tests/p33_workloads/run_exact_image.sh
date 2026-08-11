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
    grep -Fq 'CANON_KV_UNIFIED' "$qwen1p7b_overlay/attn_iface_patched.py"
    grep -Fq 'KV_UNIFIED_two_pass' "$qwen1p7b_overlay/attn_iface_patched.py"
    grep -Fq 'CANON_P38_SERVING_CAPTURE_DIR' "$qwen1p7b_overlay/tpu_runner_p21_l30.py"
    grep -Fq 'tokens_indices_selector' "$qwen1p7b_overlay/tpu_runner_p21_l30.py"
    python3 -m py_compile "$qwen1p7b_overlay/tpu_runner_p21_l30.py"
    python3 canon-zero-tim/tests/p33_workloads/test_decode_logprob_chunking.py \
      --overlay "$qwen1p7b_overlay"
    bash canon-zero-tim/install.sh "$qwen8b_overlay" \
      --from-path /usr/local/lib/python3.12/site-packages/tpu_inference \
      --model qwen8b
    grep -Fq 'CANON_KV_UNIFIED' "$qwen8b_overlay/attn_iface_patched.py"
    grep -Fq 'KV_UNIFIED_two_pass' "$qwen8b_overlay/attn_iface_patched.py"
    grep -Fq 'CANON_P38_SERVING_CAPTURE_DIR' "$qwen8b_overlay/tpu_runner_p21_l30.py"
    grep -Fq 'tokens_indices_selector' "$qwen8b_overlay/tpu_runner_p21_l30.py"
    python3 -m py_compile "$qwen8b_overlay/tpu_runner_p21_l30.py"
    python3 canon-zero-tim/tests/p33_workloads/test_decode_logprob_chunking.py \
      --overlay "$qwen8b_overlay"
    echo "P33_EXACT_IMAGE_PASS decode_chunk_cases=5 prompt_chunk_cases=5 overlays=2"
  '
