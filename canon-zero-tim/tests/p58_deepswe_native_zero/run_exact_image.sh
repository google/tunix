#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
IMAGE="${1:?usage: run_exact_image.sh sha256:image-id}"
DOCKER="${DOCKER:-sudo docker}"

if [[ ! "$IMAGE" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  echo "P58 exact-image gate requires a sha256 image id" >&2
  exit 2
fi

bash "$ROOT/canon-zero-tim/tests/p44_deepswe_qwen4b_parity/run_exact_image.sh" \
  "$IMAGE"
bash "$ROOT/canon-zero-tim/tests/p59_backward/run_tp4_tp8_installed_shim_exact_image.sh" \
  "$IMAGE"

$DOCKER image inspect "$IMAGE" \
  --format 'P58_EXACT_IMAGE image_id={{.Id}}' >/dev/null
$DOCKER run --rm \
  -v "$ROOT:/workspace:ro" \
  -w /workspace \
  -e JAX_PLATFORMS=cpu \
  "$IMAGE" \
  bash -euo pipefail -c '
    bash -n \
      canon-zero-tim/cluster/steps/00_env.sh \
      canon-zero-tim/cluster/steps/90_run.sh \
      canon-zero-tim/tests/p58_deepswe_native_zero/run_onehost_alignment_v5p.sh \
      canon-zero-tim/cluster/steps/p58_verify_sandbox_capacity.sh \
      canon-zero-tim/tasks/p58-deepswe-native-zero-comparison/scripts/run_onehost_deepswe_xprof_common.sh \
      canon-zero-tim/tasks/p58-deepswe-native-zero-comparison/scripts/run_onehost_deepswe_xprof_native.sh \
      canon-zero-tim/tasks/p58-deepswe-native-zero-comparison/scripts/run_onehost_deepswe_xprof_zero_hp.sh \
      canon-zero-tim/tasks/p58-deepswe-native-zero-comparison/scripts/run_onehost_deepswe_seam_probe.sh \
      canon-zero-tim/tasks/p58-deepswe-native-zero-comparison/scripts/run_onehost_deepswe_seam_probe_docker.sh \
      canon-zero-tim/tasks/p58-deepswe-native-zero-comparison/scripts/run_onehost_deepswe_zero_admission.sh \
      canon-zero-tim/tasks/p58-deepswe-native-zero-comparison/scripts/run_onehost_deepswe_zero_admission_docker.sh \
      canon-zero-tim/tasks/p58-deepswe-native-zero-comparison/scripts/run_onehost_deepswe_zero_standard_decode.sh \
      canon-zero-tim/tasks/p58-deepswe-native-zero-comparison/scripts/run_onehost_deepswe_zero_standard_decode_docker.sh \
      canon-zero-tim/tasks/p58-deepswe-native-zero-comparison/scripts/run_onehost_deepswe_zero_continue_kv.sh \
      canon-zero-tim/tasks/p58-deepswe-native-zero-comparison/scripts/run_onehost_deepswe_zero_continue_kv_docker.sh \
      canon-zero-tim/tasks/p58-deepswe-native-zero-comparison/scripts/run_onehost_deepswe_zero_short_backward.sh \
      canon-zero-tim/tasks/p58-deepswe-native-zero-comparison/scripts/run_onehost_deepswe_zero_short_backward_docker.sh \
      canon-zero-tim/tasks/p58-deepswe-native-zero-comparison/scripts/run_onehost_deepswe_zero_carrier_screen.sh \
      canon-zero-tim/tasks/p58-deepswe-native-zero-comparison/scripts/run_onehost_deepswe_zero_carrier_screen_docker.sh \
      canon-zero-tim/tasks/p58-deepswe-native-zero-comparison/scripts/run_onehost_deepswe_zero_trajectory_replay.sh \
      canon-zero-tim/tasks/p58-deepswe-native-zero-comparison/scripts/run_onehost_deepswe_zero_trajectory_replay_docker.sh \
      canon-zero-tim/tasks/p58-deepswe-native-zero-comparison/scripts/prepare_p58_checked_vma_off_diagnostic.sh \
      canon-zero-tim/tasks/p58-deepswe-native-zero-comparison/scripts/prepare_p58_checked_vma_aba_wave.sh \
      canon-zero-tim/tasks/p58-deepswe-native-zero-comparison/scripts/prepare_p58_coarse_seam_localization.sh \
      canon-zero-tim/tasks/v1-system-optimization-workload-rollout/prepare_deepswe_zero_hp_full.sh \
      canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/persist_p38_gcs.sh \
      canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/p38_live_snapshot_worker.sh \
      canon-zero-tim/cluster/profiles/qwen3-4b-dp8-tp8-deepswe-v1-hp.env \
      canon-zero-tim/cluster/profiles/qwen3-4b-dp1-tp4-deepswe-zero.env
    PYTHONPATH=/workspace python3 \
      canon-zero-tim/tests/p58_deepswe_native_zero/test_loss_contract.py
    PYTHONPATH=/workspace python3 \
      canon-zero-tim/tests/p58_deepswe_native_zero/test_renderer.py
    PYTHONPATH=/workspace python3 \
      canon-zero-tim/tests/p58_deepswe_native_zero/test_profile.py
    PYTHONPATH=/workspace python3 \
      canon-zero-tim/tests/p58_deepswe_native_zero/test_sampler_recipe.py
    PYTHONPATH=/workspace python3 \
      canon-zero-tim/tests/p58_deepswe_native_zero/test_alignment_policy.py
    PYTHONPATH=/workspace python3 \
      canon-zero-tim/tests/p58_deepswe_native_zero/test_environment_contract.py
    PYTHONPATH=/workspace python3 \
      canon-zero-tim/tests/v1_system_optimization/test_workload_rollout.py
    PYTHONPATH=/workspace python3 \
      canon-zero-tim/tests/p58_deepswe_native_zero/test_sandbox_capacity_probe.py
    PYTHONPATH=/workspace python3 \
      canon-zero-tim/tests/p58_deepswe_native_zero/test_artifacts.py
    PYTHONPATH=/workspace python3 \
      canon-zero-tim/tests/p58_deepswe_native_zero/test_classifier.py
    PYTHONPATH=/workspace python3 \
      canon-zero-tim/tests/p58_deepswe_native_zero/test_stock_prompt_observer.py
    PYTHONPATH=/workspace python3 \
      canon-zero-tim/tests/p58_deepswe_native_zero/test_onehost_xprof.py
    PYTHONPATH=/workspace python3 \
      canon-zero-tim/tests/p58_deepswe_native_zero/test_onehost_xprof_pair.py
    PYTHONPATH=/workspace python3 \
      canon-zero-tim/tests/p58_deepswe_native_zero/test_decode_prefill_probe.py
    PYTHONPATH=/workspace python3 \
      canon-zero-tim/tests/p58_deepswe_native_zero/test_qwen4b_tp4_zero_contract.py
    PYTHONPATH=/workspace python3 \
      canon-zero-tim/tests/p58_deepswe_native_zero/test_continue_kv_probe.py
    PYTHONPATH=/workspace python3 \
      canon-zero-tim/tests/p58_deepswe_native_zero/test_short_carrier_screen.py
    PYTHONPATH=/workspace python3 \
      canon-zero-tim/tests/p58_deepswe_native_zero/test_trajectory_replay.py
    PYTHONPATH=/workspace python3 \
      canon-zero-tim/tests/p58_deepswe_native_zero/test_checked_vma_diagnostic.py
    PYTHONPATH=/workspace python3 \
      canon-zero-tim/tests/p58_deepswe_native_zero/test_checked_vma_aba_wave.py
    PYTHONPATH=/workspace python3 \
      canon-zero-tim/tests/p58_deepswe_native_zero/test_zero_hp_full_classifier.py
    PYTHONPATH=/workspace python3 \
      canon-zero-tim/tests/p58_deepswe_native_zero/test_coarse_seam_classifier.py
    PYTHONPATH=/workspace python3 \
      canon-zero-tim/tests/v1_phase4/test_first_update_gate.py
    PYTHONPATH=/workspace python3 \
      canon-zero-tim/tests/v1_phase4/test_stable_clip_contract.py
    PYTHONPATH=/workspace python3 -m unittest discover \
      -s canon-zero-tim/tests/p3_prefix_cache \
      -p "test_*.py"
    (
      cd tests/rl
      PYTHONPATH=/workspace python3 -m unittest alignment_test
    )
    (
      cd tests/rl/rollout
      PYTHONPATH=/workspace python3 -m unittest \
        vllm_rollout_canonical_test.VllmRolloutCanonicalTest.test_sampler_preserves_pre_tokenized_prompt_without_reencoding \
        vllm_rollout_canonical_test.VllmRolloutCanonicalTest.test_sampler_rejects_malformed_pre_tokenized_prompt \
        vllm_rollout_canonical_test.VllmRolloutCanonicalTest.test_selected_engine_weight_attestation_uses_registered_adapter \
        vllm_rollout_canonical_test.VllmRolloutCanonicalTest.test_p58_native_uses_observer_without_registering_adapter \
        vllm_rollout_canonical_test.VllmRolloutCanonicalTest.test_stock_weight_observer_rejects_unsigned_or_zero_arm \
        vllm_rollout_canonical_test.VllmRolloutCanonicalTest.test_p58_native_rejects_registered_canonical_adapter \
        vllm_rollout_canonical_test.VllmRolloutCanonicalTest.test_p58_native_processed_rescore_uses_only_signed_stock_observer \
        vllm_rollout_canonical_test.VllmRolloutCanonicalTest.test_p58_native_processed_rescore_rejects_missing_observer \
        vllm_rollout_canonical_test.VllmRolloutCanonicalTest.test_p58_zero_processed_rescore_rejects_native_observer \
        vllm_rollout_canonical_test.VllmRolloutCanonicalTest.test_p58_zero_processed_rescore_keeps_canonical_processor \
        vllm_rollout_canonical_test.VllmRolloutCanonicalTest.test_processed_rescore_skips_engine_for_empty_completion_batch \
        vllm_rollout_canonical_test.VllmRolloutCanonicalTest.test_processed_rescore_still_requires_provenance_for_any_target \
        vllm_rollout_canonical_test.VllmRolloutCanonicalTest.test_p58_replay_primes_exact_recorded_sampling_provenance \
        vllm_rollout_canonical_test.VllmRolloutCanonicalTest.test_p58_replay_sampling_provenance_is_fail_closed \
        vllm_rollout_canonical_test.VllmRolloutCanonicalTest.test_canonical_adapter_registration_passes_live_trainer_state \
        vllm_rollout_canonical_test.VllmRolloutCanonicalTest.test_jax_seed_route_uses_engine_global_and_rejects_per_request
    )
    (
      cd tests/rl
      PYTHONPATH=/workspace python3 -m unittest \
        canonical_qwen3_adapter_test.CanonicalQwen3AdapterTest.test_observer_only_attestation_compares_stock_live_state_exactly \
        canonical_qwen3_adapter_test.CanonicalQwen3AdapterTest.test_deepswe_weight_report_normalizes_and_validates_logical_mesh \
        canonical_qwen3_adapter_test.CanonicalQwen3AdapterTest.test_p58_replay_segmented_loss_geometry_is_b2g2
    )
    (
      cd tests/rl
      XLA_FLAGS=--xla_force_host_platform_device_count=16 \
        PYTHONPATH=/workspace python3 -m unittest \
        canonical_qwen3_adapter_test.CanonicalQwen3AdapterTest.test_p59_tp4_tp8_localizes_nested_engine_maps_and_collectives \
        canonical_qwen3_adapter_test.CanonicalQwen3AdapterTest.test_p34_group_spec_admits_empty_completion_as_zero_cotangent \
        canonical_qwen3_adapter_test.CanonicalQwen3AdapterTest.test_p34_k11_group_spec_preserves_prompt_only_dp8_rows \
        canonical_qwen3_adapter_test.CanonicalQwen3AdapterTest.test_p32_m15_step61_group_spec_preserves_prompt_only_dp8_row \
        canonical_qwen3_adapter_test.CanonicalQwen3AdapterTest.test_p32_nonempty_group_is_identical_when_empty_rows_are_admitted
    )
    (
      cd tests/rl
      XLA_FLAGS=--xla_force_host_platform_device_count=4 \
        PYTHONPATH=/workspace python3 -m unittest \
        canonical_qwen3_adapter_test.CanonicalQwen3AdapterTest.test_disaggregated_canonical_forward_executes_on_trainer_devices \
        canonical_qwen3_adapter_test.CanonicalQwen3AdapterTest.test_disaggregated_segmented_backward_uses_trainer_graph \
        canonical_qwen3_adapter_test.CanonicalQwen3AdapterTest.test_disaggregated_segmented_scans_bind_trainer_execution_mesh \
        canonical_qwen3_adapter_test.CanonicalQwen3AdapterTest.test_p32_grouped_trainer_axis_uses_dp_tp_state_identity \
        canonical_qwen3_adapter_test.CanonicalQwen3AdapterTest.test_p32_grouped_trainer_axis_keeps_data_model_identity \
        canonical_qwen3_adapter_test.CanonicalQwen3AdapterTest.test_p32_grouped_trainer_axis_rejects_shape_based_identity \
        canonical_qwen3_adapter_test.CanonicalQwen3AdapterTest.test_colocated_segmented_execution_mesh_binding_is_identity \
        canonical_qwen3_adapter_test.CanonicalQwen3AdapterTest.test_canonical_trainer_execution_mesh_rejects_partial_overlap
    )
    PYTHONPATH=/workspace python3 \
      canon-zero-tim/tests/p33_workloads/test_dp_workloads.py \
      DPWorkloadsTest.test_p57_zero_full_admits_its_signed_wandb_project_only \
      DPWorkloadsTest.test_p57_m15_uses_its_signed_wide_token_contract \
      DPWorkloadsTest.test_p57_token_contract_rejects_partial_or_foreign_pairs \
      DPWorkloadsTest.test_deepswe_p58_satisfies_the_shared_token_width_interface
    PYTHONPATH=/workspace python3 \
      canon-zero-tim/tests/p34_deepswe/test_contract.py
    PYTHONPATH=/workspace python3 \
      canon-zero-tim/tests/p34_deepswe/test_env_contract.py
    PYTHONPATH=/workspace python3 \
      canon-zero-tim/tests/p34_deepswe/test_render_p34_jobset.py
    PYTHONPATH=/workspace python3 \
      canon-zero-tim/tests/p34_deepswe/test_script_contract.py
    PYTHONPATH=/workspace python3 \
      canon-zero-tim/tests/p34_deepswe/test_r2egym_optional.py
    PYTHONPATH=/workspace python3 \
      canon-zero-tim/tests/p44_deepswe_qwen4b_parity/test_renderer.py
    (
      cd tests/rl
      PYTHONPATH=/workspace python3 -m unittest \
        common_test.CommonTest
    )
    (
      cd tests/rl
      PYTHONPATH=/workspace python3 -m unittest \
        rl_cluster_test.RlClusterTest.test_batch_size_config
    )
    (
      cd tests/sft
      PYTHONPATH=/workspace python3 -m unittest \
        sft_utils_test.StableGlobalNormTest \
        peft_trainer_test.PeftTrainerTest.test_p58_precomputed_transaction_requires_sixteen_gradient_groups \
        peft_trainer_test.PeftTrainerTest.test_effective_learning_rate_skips_empty_chain_state \
        peft_trainer_test.PeftTrainerTest.test_effective_learning_rate_fails_closed_without_hyperparams \
        peft_trainer_test.PeftTrainerTest.test_p63_finite_overflow_commits_nonzero_clipped_update \
        peft_trainer_test.PeftTrainerTest.test_denominator_weighted_accumulation_matches_concatenated_batch \
        peft_trainer_test.PeftTrainerTest.test_denominator_weighted_all_empty_skips_optimizer \
        peft_trainer_test.PeftTrainerTest.test_p58_precomputed_all_filtered_discard_resets_without_commit
    )
    (
      cd tests/rl/agentic
      PYTHONPATH=/workspace python3 -m unittest \
        trajectory.trajectory_collect_engine_test.TrajectoryCollectEngineTest.test_deepswe_continuation_reuses_exact_sampled_and_environment_tokens \
        trajectory.trajectory_collect_engine_test.TrajectoryCollectEngineTest.test_deepswe_continuation_rejects_missing_environment_tokens \
        agentic_rl_learner_test.AgenticRLLearnerTest.test_model_call_routes_signed_deepswe_pre_tokenized_prompt_exactly \
        agentic_rl_learner_test.AgenticRLLearnerTest.test_model_call_rejects_unsigned_pre_tokenized_prompt \
        agentic_rl_learner_test.AgenticRLLearnerTest.test_p38_diagnostic_consumer_admits_p58_seam_localization \
        agentic_rl_learner_test.AgenticRLLearnerTest.test_p38_diagnostic_consumer_admits_p58_q4_continue_kv \
        agentic_rl_learner_test.AgenticRLLearnerTest.test_p58_partial_consumer_propagates_producer_timeout \
        agentic_rl_learner_test.AgenticRLLearnerTest.test_p58_full_batch_group_contract_rejects_missing_generation \
        agentic_grpo_learner_test.AgenticGrpoLearnerTest.test_environment_is_seeded_with_policy_version_before_reset \
        agentic_grpo_learner_test.AgenticGrpoLearnerTest.test_p58_replay_segmented_geometry_is_b2g2_not_batch_one \
        agentic_grpo_learner_test.AgenticGrpoLearnerTest.test_p58_full_segmented_geometry_is_dp8_by_sixteen_groups \
        agentic_grpo_learner_test.AgenticGrpoLearnerTest.test_p58_all_sandbox_timeout_blocks_after_durable_journal \
        agentic_grpo_learner_test.AgenticGrpoLearnerTest.test_non_infrastructure_all_filtered_batch_does_not_capacity_block \
        agentic_grpo_learner_test.AgenticGrpoLearnerTest.test_p58_sandbox_capacity_evidence_is_fail_closed \
        agentic_grpo_learner_test.AgenticGrpoLearnerTest.test_p58_all_filtered_no_commit_suppresses_step_advance \
        agentic_grpo_learner_test.AgenticGrpoLearnerTest.test_p58_all_filtered_no_commit_rejects_optimizer_advance \
        agentic_grpo_learner_test.AgenticGrpoLearnerTest.test_p58_checked_vma_first_update_commits_sixteen_groups \
        trajectory.trajectory_collect_engine_test.TrajectoryCollectEngineTest.test_overlong_filter_masks_out_and_skips_reward \
        trajectory.trajectory_collect_engine_test.TrajectoryCollectEngineTest.test_reset_raised_timeout_is_env_timeout \
        trajectory.trajectory_collect_engine_test.TrajectoryCollectEngineTest.test_reset_scheduling_gate_is_distinct_env_timeout \
        trajectory.trajectory_collect_engine_test.TrajectoryCollectEngineTest.test_reset_timeout_token_preserves_environment_task \
        trajectory.trajectory_collect_engine_test.TrajectoryCollectEngineTest.test_shared_batch_deadline_reduces_late_collector_budget \
        trajectory.trajectory_collect_engine_test.TrajectoryCollectEngineTest.test_token_prefers_policy_seeded_environment_task \
        trajectory.trajectory_collect_engine_test.TrajectoryCollectEngineTest.test_policy_seeded_original_input_missing_prompt_fails_closed \
        trajectory.trajectory_collect_engine_test.TrajectoryCollectEngineTest.test_token_missing_original_input_fails_closed
    )
    bash canon-zero-tim/install.sh /tmp/p58-continue-overlay \
      --from-path /usr/local/lib/python3.12/site-packages/tpu_inference \
      --model qwen4b
    bash canon-zero-tim/install.sh /tmp/p58-qwen4b-tp4-overlay \
      --from-path /usr/local/lib/python3.12/site-packages/tpu_inference \
      --model qwen4b_tp4
    CANON_QWEN3_HIDDEN_SIZE=2560 \
      CANON_QWEN3_INTERMEDIATE_SIZE=9728 \
      CANON_QWEN3_NUM_ATTENTION_HEADS=32 \
      CANON_QWEN3_NUM_KV_HEADS=8 \
      CANON_QWEN3_HEAD_DIM=128 \
      CANON_QWEN3_TP_SIZE=4 \
      CANON_FIXED_AR=1 \
      CANON_PALLAS_ALL_PROJ=1 \
      PYTHONPATH=/tmp/p58-qwen4b-tp4-overlay \
      python3 /tmp/p58-qwen4b-tp4-overlay/p22xf_contract.py
    PYTHONPATH=/workspace python3 \
      canon-zero-tim/tests/p58_deepswe_native_zero/probe_continue_decode_observer_overlay.py \
      --runner /tmp/p58-continue-overlay/tpu_runner_p21_l30.py
    CANON_P58_Q4_TP4_CONTINUE_KV_DIAGNOSTIC=1 \
      CANON_P38_KV_OBSERVER_DIR=/tmp/p58-continue-kv-contract \
      CANON_P38_KV_OBSERVER_MAX_CANDIDATES=1 \
      CANON_P38_KV_OBSERVER_MAX_PAGES=192 \
      CANON_P38_KV_OBSERVER_MAX_BYTES=134217728 \
      CANON_P38_KV_OBSERVER_MAX_READ_BYTES=671088640 \
      CANON_P38_SERVING_CAPTURE_EXPECTED_PATH=standard \
      CANON_P58_Q4_TP4_CONTINUE_KV_MIN_PREFIX=2280 \
      CANON_P58_Q4_TP4_CONTINUE_KV_MAX_PREFIX=3072 \
      PYTHONPATH=/workspace python3 \
      canon-zero-tim/tests/p58_deepswe_native_zero/probe_continue_kv_overlay.py \
      --runner /tmp/p58-qwen4b-tp4-overlay/tpu_runner_p21_l30.py
    observer_state="$(mktemp -d /tmp/p58-stock-observer-state.XXXXXX)"
    printf "%s\n" \
      "export CANON_PROFILE_FILE=cluster/profiles/qwen3-4b-dp8-tp8-deepswe-tim.env" \
      "export CANON_P34_DEEPSWE=1" \
      "export CANON_P58_DEEPSWE_TIM=1" \
      "export CANON_P58_TIM_ADMITTED=1" \
      "export CANON_P58_TIM_ARM=native" \
      "export CANON_P58_NATIVE_STOCK_PROMPT_OBSERVER=1" \
      "export CANON_PROMPT_PROCESSED_LOGPROBS=0" \
      "export CANON_ENGINE_MODULE_C=0" \
      > "$observer_state/env.sh"
    printf "%s\n" /usr/local/lib/python3.12/site-packages/tpu_inference \
      > "$observer_state/tpu_inference_path"
    CANON_STATE="$observer_state" CANON_PKG=/workspace/canon-zero-tim \
      bash canon-zero-tim/cluster/steps/p58_install_stock_prompt_observer.sh
    env PATHWAYS_HEAD="" JAX_BACKEND_TARGET="" JAX_PLATFORMS=cpu \
      python3 \
      canon-zero-tim/tests/p58_deepswe_native_zero/probe_stock_prompt_observer.py
    rm -r "$observer_state"
    echo "P58_EXACT_IMAGE_CPU_PASS loss_oracle=1 weighted_accumulation=1 compact_filter=1 durable_journal=1 paired_renderer=1 alignment_policy=1 stock_observer=1 continue_decode_observer=1 continue_kv_observer=1 onehost_xprof=1 zero_hp_full=1 system_optimization=1 checked_vma_diagnostic=1 checked_vma_aba=1 coarse_seam=1 qwen4b_fixed_head=1 qwen4b_tp4=1 trajectory_replay_b2g2=1 checked_vma=1 vma_p59_only=1 first_update=1 stable_clip=1 apc=1 p59_tp4_tp8=2 p59_real_shim=4 p59_rpa=2 p59_fused_linear=2 disaggregated_trainer_mesh=4 disaggregated_scan_mesh=2 grouped_trainer_axis=3 p57_wandb=1 m15_token=1 deepswe_workload_identity=1 p32_empty_completion=4 regressions=1"
  '
