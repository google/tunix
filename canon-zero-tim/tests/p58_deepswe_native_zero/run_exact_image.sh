#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
IMAGE="${1:?usage: run_exact_image.sh sha256:image-id}"
DOCKER="${DOCKER:-sudo docker}"

if [[ ! "$IMAGE" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  echo "P58 exact-image gate requires a sha256 image id" >&2
  exit 2
fi

$DOCKER image inspect "$IMAGE" \
  --format 'P58_EXACT_IMAGE image_id={{.Id}}' >/dev/null
$DOCKER run --rm \
  -v "$ROOT:/workspace:ro" \
  -w /workspace \
  -e JAX_PLATFORMS=cpu \
  "$IMAGE" \
  bash -euo pipefail -c '
    PYTHONPATH=/workspace python3 \
      canon-zero-tim/tests/p58_deepswe_native_zero/test_loss_contract.py
    PYTHONPATH=/workspace python3 \
      canon-zero-tim/tests/p58_deepswe_native_zero/test_renderer.py
    PYTHONPATH=/workspace python3 \
      canon-zero-tim/tests/p58_deepswe_native_zero/test_profile.py
    PYTHONPATH=/workspace python3 \
      canon-zero-tim/tests/p58_deepswe_native_zero/test_alignment_policy.py
    PYTHONPATH=/workspace python3 \
      canon-zero-tim/tests/p58_deepswe_native_zero/test_environment_contract.py
    PYTHONPATH=/workspace python3 \
      canon-zero-tim/tests/p58_deepswe_native_zero/test_artifacts.py
    PYTHONPATH=/workspace python3 \
      canon-zero-tim/tests/p58_deepswe_native_zero/test_classifier.py
    PYTHONPATH=/workspace python3 \
      canon-zero-tim/tests/p58_deepswe_native_zero/test_stock_prompt_observer.py
    (
      cd tests/rl
      PYTHONPATH=/workspace python3 -m unittest alignment_test
    )
    (
      cd tests/rl/rollout
      PYTHONPATH=/workspace python3 -m unittest \
        vllm_rollout_canonical_test.VllmRolloutCanonicalTest.test_selected_engine_weight_attestation_uses_registered_adapter \
        vllm_rollout_canonical_test.VllmRolloutCanonicalTest.test_p58_native_uses_observer_without_registering_adapter \
        vllm_rollout_canonical_test.VllmRolloutCanonicalTest.test_stock_weight_observer_rejects_unsigned_or_zero_arm \
        vllm_rollout_canonical_test.VllmRolloutCanonicalTest.test_p58_native_rejects_registered_canonical_adapter \
        vllm_rollout_canonical_test.VllmRolloutCanonicalTest.test_p58_native_processed_rescore_uses_only_signed_stock_observer \
        vllm_rollout_canonical_test.VllmRolloutCanonicalTest.test_p58_native_processed_rescore_rejects_missing_observer \
        vllm_rollout_canonical_test.VllmRolloutCanonicalTest.test_p58_zero_processed_rescore_rejects_native_observer \
        vllm_rollout_canonical_test.VllmRolloutCanonicalTest.test_p58_zero_processed_rescore_keeps_canonical_processor
    )
    (
      cd tests/rl
      PYTHONPATH=/workspace python3 -m unittest \
        canonical_qwen3_adapter_test.CanonicalQwen3AdapterTest.test_observer_only_attestation_compares_stock_live_state_exactly \
        canonical_qwen3_adapter_test.CanonicalQwen3AdapterTest.test_deepswe_weight_report_normalizes_and_validates_logical_mesh
    )
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
      cd tests/sft
      PYTHONPATH=/workspace python3 -m unittest \
        peft_trainer_test.PeftTrainerTest.test_denominator_weighted_accumulation_matches_concatenated_batch \
        peft_trainer_test.PeftTrainerTest.test_denominator_weighted_all_empty_skips_optimizer \
        peft_trainer_test.PeftTrainerTest.test_p58_precomputed_all_filtered_discard_resets_without_commit
    )
    (
      cd tests/rl/agentic
      PYTHONPATH=/workspace python3 -m unittest \
        agentic_grpo_learner_test.AgenticGrpoLearnerTest.test_environment_is_seeded_with_policy_version_before_reset \
        trajectory.trajectory_collect_engine_test.TrajectoryCollectEngineTest.test_overlong_filter_masks_out_and_skips_reward \
        trajectory.trajectory_collect_engine_test.TrajectoryCollectEngineTest.test_reset_raised_timeout_is_env_timeout \
        trajectory.trajectory_collect_engine_test.TrajectoryCollectEngineTest.test_reset_scheduling_gate_is_distinct_env_timeout
    )
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
    echo "P58_EXACT_IMAGE_CPU_PASS loss_oracle=1 weighted_accumulation=1 compact_filter=1 durable_journal=1 paired_renderer=1 alignment_policy=1 stock_observer=1 regressions=1"
  '
