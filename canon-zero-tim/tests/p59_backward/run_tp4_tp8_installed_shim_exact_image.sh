#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
image_ref="${1:?usage: run_tp4_tp8_installed_shim_exact_image.sh IMAGE}"
docker="${DOCKER:-sudo docker}"

image_id="$($docker image inspect "$image_ref" --format '{{.Id}}')"
if [[ ! "$image_id" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  echo "P59 TP shim gate could not resolve immutable image ID: $image_id" >&2
  exit 2
fi
echo "P59_TP_SHIM_EXACT_IMAGE image_ref=$image_ref image_id=$image_id"

$docker run --rm \
  -v "$root:/workspace:ro" \
  -w /workspace \
  -e JAX_PLATFORMS=cpu \
  "$image_id" \
  bash -euo pipefail -c '
    qwen1p7b_overlay="$(mktemp -d /tmp/p59-qwen1p7b-tp4.XXXXXX)"
    qwen8b_overlay="$(mktemp -d /tmp/p59-qwen8b-tp8.XXXXXX)"
    trap '\''rm -r "$qwen1p7b_overlay" "$qwen8b_overlay"'\'' EXIT

    export CANON_P38_SERVING_CAPTURE_DIR=/tmp/p59-p38-exact-image-capture
    export CANON_P38_REQUEST_JOURNAL="$CANON_P38_SERVING_CAPTURE_DIR/p38_request_journal.jsonl"
    export CANON_P38_INCIDENT_LEDGER="$CANON_P38_SERVING_CAPTURE_DIR/p38_incident_ledger.jsonl"
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

    XLA_FLAGS=--xla_force_host_platform_device_count=8 \
      PYTHONPATH=/workspace python3 tests/rl/canonical_qwen3_adapter_test.py \
        CanonicalQwen3AdapterTest.test_p59_tpx_executes_fixed_head_local_vjp_and_global_negative \
        CanonicalQwen3AdapterTest.test_p59_restores_only_physically_equal_tpx_staged_specs
    P59_TEST_TP_SIZE=8 \
      XLA_FLAGS=--xla_force_host_platform_device_count=16 \
      PYTHONPATH=/workspace python3 tests/rl/canonical_qwen3_adapter_test.py \
        CanonicalQwen3AdapterTest.test_p59_tpx_executes_fixed_head_local_vjp_and_global_negative \
        CanonicalQwen3AdapterTest.test_p59_restores_only_physically_equal_tpx_staged_specs

    bash canon-zero-tim/install.sh "$qwen1p7b_overlay" \
      --from-path /usr/local/lib/python3.12/site-packages/tpu_inference \
      --model qwen1p7b
    XLA_FLAGS=--xla_force_host_platform_device_count=8 \
      PYTHONPATH="$qwen1p7b_overlay:/workspace" python3 \
        canon-zero-tim/tests/p59_backward/probe_tp4_installed_shim_composition.py
    XLA_FLAGS=--xla_force_host_platform_device_count=8 \
      PYTHONPATH="$qwen1p7b_overlay:/workspace" python3 \
        canon-zero-tim/tests/p59_backward/probe_tp4_installed_attention_composition.py

    bash canon-zero-tim/install.sh "$qwen8b_overlay" \
      --from-path /usr/local/lib/python3.12/site-packages/tpu_inference \
      --model qwen8b_tp8
    P59_TEST_TP_SIZE=8 \
      XLA_FLAGS=--xla_force_host_platform_device_count=16 \
      PYTHONPATH="$qwen8b_overlay:/workspace" python3 \
        canon-zero-tim/tests/p59_backward/probe_tp4_installed_shim_composition.py
    P59_TEST_TP_SIZE=8 \
      XLA_FLAGS=--xla_force_host_platform_device_count=16 \
      PYTHONPATH="$qwen8b_overlay:/workspace" python3 \
        canon-zero-tim/tests/p59_backward/probe_tp4_installed_attention_composition.py

    echo "P59_TP_SHIM_EXACT_IMAGE_PASS fixed_head=2 installed_projection=2 installed_attention=2 report_adjoint=2 staged_spec_restore=2 fixed_reducer=2 topologies=DP2xTP4,DP2xTP8 optimizer_commits=0 manifests=2x36/36"
  '
