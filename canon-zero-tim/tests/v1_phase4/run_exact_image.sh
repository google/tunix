#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
image_ref="${1:-tunix_frozenlake_image:vllm-tpu0.25.0}"
docker="${DOCKER:-sudo docker}"
git_common_dir="$(git -C "$root" rev-parse --git-common-dir)"
case "$git_common_dir" in
  /*) ;;
  *) git_common_dir="$root/$git_common_dir" ;;
esac
if [ ! -d "$git_common_dir" ]; then
  echo "V1 exact-image gate could not resolve Git common directory" >&2
  exit 2
fi

bash "$root/canon-zero-tim/tests/p33_workloads/run_exact_image.sh" "$image_ref"
bash "$root/canon-zero-tim/tests/p45_frozenlake_dp8_tp8/run_exact_image.sh" "$image_ref"
CANON_P66_P59_CHECK_VMA=1 \
  CANON_P67_P66_VMA_P59_ONLY=1 \
  bash "$root/canon-zero-tim/tests/p59_backward/run_tp4_tp8_installed_shim_exact_image.sh" \
  "$image_ref"

image_id="$($docker image inspect "$image_ref" --format '{{.Id}}')"
if [[ ! "$image_id" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  echo "V1 exact-image gate could not resolve immutable image ID: $image_id" >&2
  exit 2
fi
echo "V1_HP_EXACT_IMAGE image_ref=$image_ref image_id=$image_id"

$docker run --rm \
  -v "$root:/workspace:ro" \
  -v "$git_common_dir:$git_common_dir:ro" \
  -w /workspace \
  -e JAX_PLATFORMS=cpu \
  -e PYTHONPATH=/workspace \
  -e GIT_CONFIG_COUNT=1 \
  -e GIT_CONFIG_KEY_0=safe.directory \
  -e GIT_CONFIG_VALUE_0=/workspace \
  "$image_id" \
  bash -euo pipefail -c '
    XLA_FLAGS=--xla_force_host_platform_device_count=64 \
      python3 tests/rl/canonical_qwen3_adapter_test.py \
        CanonicalQwen3AdapterTest.test_dp16_gathered_logprobs_pads_and_slices_each_data_rank
    XLA_FLAGS=--xla_force_host_platform_device_count=4 \
      python3 tests/rl/canonical_qwen3_adapter_test.py \
        CanonicalQwen3AdapterTest.test_p59_rank_parallel_nonhead_pullbacks_match_serial_dp2_tp2
    XLA_FLAGS=--xla_force_host_platform_device_count=16 \
      python3 tests/rl/canonical_qwen3_adapter_test.py \
        CanonicalQwen3AdapterTest.test_p59_tp4_tp8_localizes_nested_engine_maps_and_collectives
    XLA_FLAGS=--xla_force_host_platform_device_count=4 \
      python3 tests/rl/dp_training_test.py \
        DPTrainingTest.test_dp2_tp2_rank_parallel_vjp_matches_serial_rank_isolation
    XLA_FLAGS=--xla_force_host_platform_device_count=64 \
      python3 tests/rl/dp_training_test.py \
        DPTrainingTest.test_dp8_tp8_rank_gradient_reducer_consumes_finite_staged_table \
        DPTrainingTest.test_parallel_staged_common_nan_is_nonfinite_not_replica_mismatch \
        DPTrainingTest.test_finite_replica_mismatch_remains_fatal
    python3 tests/sft/sft_utils_test.py StableGlobalNormTest
    python3 tests/sft/peft_trainer_test.py \
      PeftTrainerTest.test_p62_precomputed_diagnostic_discard_resets_without_commit \
      PeftTrainerTest.test_p64_precomputed_diagnostic_discard_resets_without_commit \
      PeftTrainerTest.test_p63_finite_overflow_commits_nonzero_clipped_update
    python3 tests/rl/canonical_qwen3_adapter_test.py \
      CanonicalQwen3AdapterTest.test_p62_tree_receipt_catches_first_nonfinite_boundary \
      CanonicalQwen3AdapterTest.test_p62_loss_receipt_locks_denominator_and_scale \
      CanonicalQwen3AdapterTest.test_p62_flag_parser_is_exact_boolean \
      CanonicalQwen3AdapterTest.test_p64_receipts_and_flag_parser_are_isolated_from_p62
    python3 canon-zero-tim/tests/v1_phase4/test_p64_numeric_debug.py \
      P64NumericDebugTest.test_classifier_accepts_group0_only_diagnostic_replay \
      P64NumericDebugTest.test_capsule_gcs_transport_round_trip_and_hash_negative \
      P64NumericDebugTest.test_capsule_round_trip_and_model_binding_are_fail_closed
    python3 canon-zero-tim/tests/p33_workloads/test_dp_workloads.py \
      DPWorkloadsTest.test_p62_numeric_debug_is_exact_no_commit_geometry \
      DPWorkloadsTest.test_p64_numeric_debug_is_exact_p45_no_commit_geometry \
      DPWorkloadsTest.test_p57_zero_full_admits_its_signed_wandb_project_only \
      DPWorkloadsTest.test_p57_m15_uses_its_signed_wide_token_contract \
      DPWorkloadsTest.test_p57_token_contract_rejects_partial_or_foreign_pairs
    XLA_FLAGS=--xla_force_host_platform_device_count=64 \
      python3 canon-zero-tim/tasks/v1-phase4-three-full-recipes/scripts/probe_gsm_fixed_replay_scale.py
    python3 tests/perf/profile_window_test.py
    bash canon-zero-tim/tests/v1_phase4/run_cpu.sh
    python3 canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/test_analyze_m15i_evidence.py
    python3 canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/test_classify_m15_apc_target_run.py
    python3 canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/test_package_first_red_replay.py
    python3 canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/test_package_full_replay_carrier.py
    python3 canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/test_target_carrier.py
    python3 canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/test_resolved_env.py
    python3 canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/test_classify_m15_apc_wide_seam.py
    python3 canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/test_m15_wide_durability.py
    bash canon-zero-tim/tests/p38_serving/test_gcs_persistence.sh
    echo "V1_HP_EXACT_IMAGE_PASS dp16_gathered=1 dp2tp2_parallel=2 p59_tp4_tp8=2 p59_checked_vma_real_shim=4 p59_rpa=2 p59_fused_linear=2 p62_numeric=6 p64_numeric=4 p64_capsule=3 p63_clip=1 first_update_gate=4 gsm_scale_replay=1 p57_wandb=1 m15_token=1 apc_m15_carrier=66 m15_durability=1 perfetto_window=1 manifests=3"
  '
