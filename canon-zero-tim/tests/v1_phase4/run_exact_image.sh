#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
image_ref="${1:-tunix_frozenlake_image:vllm-tpu0.25.0}"
docker="${DOCKER:-sudo docker}"

bash "$root/canon-zero-tim/tests/p33_workloads/run_exact_image.sh" "$image_ref"
bash "$root/canon-zero-tim/tests/p45_frozenlake_dp8_tp8/run_exact_image.sh" "$image_ref"

image_id="$($docker image inspect "$image_ref" --format '{{.Id}}')"
if [[ ! "$image_id" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  echo "V1 exact-image gate could not resolve immutable image ID: $image_id" >&2
  exit 2
fi
echo "V1_HP_EXACT_IMAGE image_ref=$image_ref image_id=$image_id"

$docker run --rm \
  -v "$root:/workspace:ro" \
  -w /workspace \
  -e JAX_PLATFORMS=cpu \
  -e PYTHONPATH=/workspace \
  "$image_id" \
  bash -euo pipefail -c '
    XLA_FLAGS=--xla_force_host_platform_device_count=64 \
      python3 tests/rl/canonical_qwen3_adapter_test.py \
        CanonicalQwen3AdapterTest.test_dp16_gathered_logprobs_pads_and_slices_each_data_rank
    XLA_FLAGS=--xla_force_host_platform_device_count=4 \
      python3 tests/rl/canonical_qwen3_adapter_test.py \
        CanonicalQwen3AdapterTest.test_p59_rank_parallel_endpoint_pullbacks_match_serial_dp2_tp2
    XLA_FLAGS=--xla_force_host_platform_device_count=4 \
      python3 tests/rl/dp_training_test.py \
        DPTrainingTest.test_dp2_tp2_rank_parallel_vjp_matches_serial_rank_isolation
    python3 tests/perf/profile_window_test.py
    bash canon-zero-tim/tests/v1_phase4/run_cpu.sh
    echo "V1_HP_EXACT_IMAGE_PASS dp16_gathered=1 dp2tp2_parallel=2 perfetto_window=1 manifests=3"
  '
