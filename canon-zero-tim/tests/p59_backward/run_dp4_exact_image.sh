#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
image_ref="${1:-tunix_frozenlake_image:vllm-tpu0.25.0}"
docker="${DOCKER:-sudo docker}"

image_id="$($docker image inspect "$image_ref" --format '{{.Id}}')"
if [[ ! "$image_id" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  echo "P59 exact-image gate could not resolve immutable image ID: $image_id" >&2
  exit 2
fi
echo "P59_DP4_EXACT_IMAGE image_ref=$image_ref image_id=$image_id"

$docker run --rm \
  -v "$root:/workspace:ro" \
  -w /workspace \
  -e JAX_PLATFORMS=cpu \
  "$image_id" \
  bash -euo pipefail -c '
    overlay="$(mktemp -d /tmp/p59-qwen1p7b-tp1.XXXXXX)"
    trap '\''rm -r "$overlay"'\'' EXIT
    export CANON_P38_SERVING_CAPTURE_DIR=/tmp/p59-p38-exact-image-capture
    export CANON_P38_REQUEST_JOURNAL=/tmp/p59-p38-exact-image-capture/p38_request_journal.jsonl
    export CANON_P38_INCIDENT_LEDGER=/tmp/p59-p38-exact-image-capture/p38_incident_ledger.jsonl
    export CANON_P38_INCIDENT_MIN_PREFIX=1400
    export CANON_P38_INCIDENT_MAX_PREFIX=3072
    export CANON_P38_INCIDENT_MAX_BYTES=134217728
    export CANON_LOGPROB_M=256
    export CANON_P38_DIAGNOSTIC_ROUND_FILE=/tmp/p59-p38-exact-image-round
    printf "0\n" > "$CANON_P38_DIAGNOSTIC_ROUND_FILE"
    export CANON_P38_SERVING_CAPTURE_MAX_CALLS=4
    export CANON_P38_SERVING_CAPTURE_MIN_PREFIX=1536
    export CANON_P38_SERVING_CAPTURE_PREFIX_BOUNDS=1536,1664,1792,1920,2048
    export CANON_P38_SERVING_CAPTURE_FREE_SPACE_MULTIPLIER=5
    export CANON_P38_SERVING_CAPTURE_EXPECTED_PATH=standard
    bash canon-zero-tim/install.sh "$overlay" \
      --from-path /usr/local/lib/python3.12/site-packages/tpu_inference \
      --model qwen1p7b_tp1
    PYTHONPATH="$overlay" python3 "$overlay/p22xf_contract.py"
    CANON_SHIM_ROOT="$overlay" PYTHONPATH="$overlay" python3 \
      canon-zero-tim/tests/p59_backward/probe_dp4_tp1_overlay.py
    python3 canon-zero-tim/tests/p33_workloads/test_decode_logprob_chunking.py \
      --overlay "$overlay"
    echo "P59_DP4_EXACT_IMAGE_PASS overlay=qwen1p7b_tp1 manifest=36/36 runner_tests=34"
  '
