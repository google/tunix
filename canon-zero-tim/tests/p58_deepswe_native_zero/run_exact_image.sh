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
    (
      cd tests/rl
      PYTHONPATH=/workspace python3 -m unittest alignment_test
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
    echo "P58_EXACT_IMAGE_CPU_PASS loss_oracle=1 weighted_accumulation=1 compact_filter=1 durable_journal=1 paired_renderer=1 alignment_policy=1 regressions=1"
  '
