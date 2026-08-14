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
    export CANON_P38_SERVING_CAPTURE_DIR=/tmp/p38-exact-image-capture
    export CANON_P38_REQUEST_JOURNAL=/tmp/p38-exact-image-capture/p38_request_journal.jsonl
    export CANON_P38_INCIDENT_LEDGER=/tmp/p38-exact-image-capture/p38_incident_ledger.jsonl
    export CANON_P38_INCIDENT_MIN_PREFIX=1400
    export CANON_P38_INCIDENT_MAX_PREFIX=3072
    export CANON_P38_INCIDENT_MAX_BYTES=134217728
    export CANON_P38_DIAGNOSTIC_ROUND_FILE=/tmp/p38-exact-image-round
    printf "0\n" > "$CANON_P38_DIAGNOSTIC_ROUND_FILE"
    export CANON_P38_SERVING_CAPTURE_MAX_CALLS=4
    export CANON_P38_SERVING_CAPTURE_MIN_PREFIX=1536
    export CANON_P38_SERVING_CAPTURE_PREFIX_BOUNDS=1536,1664,1792,1920,2048
    export CANON_P38_SERVING_CAPTURE_FREE_SPACE_MULTIPLIER=5
    export CANON_P38_SERVING_CAPTURE_EXPECTED_PATH=standard
    bash canon-zero-tim/install.sh "$qwen1p7b_overlay" \
      --from-path /usr/local/lib/python3.12/site-packages/tpu_inference \
      --model qwen1p7b
    grep -Fq 'CANON_KV_UNIFIED' "$qwen1p7b_overlay/attn_iface_patched.py"
    grep -Fq 'KV_UNIFIED_two_pass' "$qwen1p7b_overlay/attn_iface_patched.py"
    grep -Fq 'CANON_P38_SERVING_CAPTURE_DIR' "$qwen1p7b_overlay/tpu_runner_p21_l30.py"
    grep -Fq '_p38_capture_leaf' "$qwen1p7b_overlay/tpu_runner_p21_l30.py"
    grep -Fq 'tokens_indices_selector' "$qwen1p7b_overlay/tpu_runner_p21_l30.py"
    grep -Fq 'implementation_identity' "$qwen1p7b_overlay/tpu_runner_p21_l30.py"
    grep -Fq '_p38_request_journal' "$qwen1p7b_overlay/tpu_runner_p21_l30.py"
    grep -Fq '_p38_incident_ledger' "$qwen1p7b_overlay/tpu_runner_p21_l30.py"
    grep -Eq program_path=.standard. "$qwen1p7b_overlay/tpu_runner_p21_l30.py"
    python3 -m py_compile "$qwen1p7b_overlay/tpu_runner_p21_l30.py"
    python3 canon-zero-tim/tests/p33_workloads/test_decode_logprob_chunking.py \
      --overlay "$qwen1p7b_overlay"
    bash canon-zero-tim/install.sh "$qwen8b_overlay" \
      --from-path /usr/local/lib/python3.12/site-packages/tpu_inference \
      --model qwen8b
    grep -Fq 'CANON_KV_UNIFIED' "$qwen8b_overlay/attn_iface_patched.py"
    grep -Fq 'KV_UNIFIED_two_pass' "$qwen8b_overlay/attn_iface_patched.py"
    grep -Fq 'CANON_P38_SERVING_CAPTURE_DIR' "$qwen8b_overlay/tpu_runner_p21_l30.py"
    grep -Fq '_p38_capture_leaf' "$qwen8b_overlay/tpu_runner_p21_l30.py"
    grep -Fq 'tokens_indices_selector' "$qwen8b_overlay/tpu_runner_p21_l30.py"
    grep -Fq 'implementation_identity' "$qwen8b_overlay/tpu_runner_p21_l30.py"
    grep -Fq '_p38_request_journal' "$qwen8b_overlay/tpu_runner_p21_l30.py"
    grep -Fq '_p38_incident_ledger' "$qwen8b_overlay/tpu_runner_p21_l30.py"
    grep -Eq program_path=.standard. "$qwen8b_overlay/tpu_runner_p21_l30.py"
    python3 -m py_compile "$qwen8b_overlay/tpu_runner_p21_l30.py"
    python3 canon-zero-tim/tests/p33_workloads/test_decode_logprob_chunking.py \
      --overlay "$qwen8b_overlay"
    echo "P33_EXACT_IMAGE_PASS decode_chunk_cases=5 prompt_chunk_cases=5 serving_capture_cases=13 overlays=2"
  '
