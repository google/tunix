#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
image_ref="${1:-tunix_frozenlake_image:vllm-tpu0.25.0}"
docker="${DOCKER:-sudo docker}"

bash "$root/canon-zero-tim/tests/v1_phase4/run_exact_image.sh" "$image_ref"
image_id="$($docker image inspect "$image_ref" --format '{{.Id}}')"
if [[ ! "$image_id" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  echo "P60-2 exact-image gate could not resolve immutable image ID" >&2
  exit 2
fi
echo "P60_2B_EXACT_IMAGE image_ref=$image_ref image_id=$image_id"

$docker run --rm \
  -v "$root:/workspace:ro" \
  -w /workspace \
  -e JAX_PLATFORMS=cpu \
  -e PYTHONPATH=/workspace \
  "$image_id" \
  bash -euo pipefail -c '
    bash canon-zero-tim/tests/v1_gsm8k_xprof_pair/run_cpu.sh
    python3 tests/rl/canonical_qwen3_adapter_test.py \
      CanonicalQwen3AdapterTest.test_xprof_jit_is_exactly_plain_jit_when_disabled \
      CanonicalQwen3AdapterTest.test_xprof_jit_labels_module_and_operation_stack \
      CanonicalQwen3AdapterTest.test_xprof_jit_rejects_unknown_mode
    python3 tests/rl/agentic/agentic_rl_learner_test.py \
      AgenticRLLearnerTest.test_p60_xprof_compute_mode_is_forwarded \
      AgenticRLLearnerTest.test_p60_xprof_rejects_unknown_tpu_trace_mode
    for labels in 0 1; do
      CANON_XPROF_LABELS="$labels" \
        XLA_FLAGS=--xla_force_host_platform_device_count=4 \
        python3 tests/rl/canonical_qwen3_adapter_test.py \
          CanonicalQwen3AdapterTest.test_p59_rank_parallel_nonhead_pullbacks_match_serial_dp2_tp2
      CANON_XPROF_LABELS="$labels" \
        XLA_FLAGS=--xla_force_host_platform_device_count=16 \
        python3 tests/rl/canonical_qwen3_adapter_test.py \
          CanonicalQwen3AdapterTest.test_p59_tp4_tp8_localizes_nested_engine_maps_and_collectives
    done
    python3 tests/rl/alignment_test.py \
      AlignmentTest.test_one_ulp_drift_fails_closed
    python3 canon-zero-tim/tests/v1_gsm8k_xprof_pair/probe_xprof_annotations.py
  '

echo "P60_2B_EXACT_IMAGE_PASS hierarchy_api=1 labels_off_on=1 one_ulp=1 p59_v1=1 tpu_devices=0"
