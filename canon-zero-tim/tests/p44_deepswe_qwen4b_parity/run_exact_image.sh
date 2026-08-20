#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
IMAGE="${1:?usage: run_exact_image.sh <image@sha256:digest|sha256:image-id>}"
DOCKER="${DOCKER:-sudo docker}"

if [[ ! "$IMAGE" =~ @sha256:[0-9a-f]{64}$ ]] && \
   [[ ! "$IMAGE" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  echo "P44 exact-image gate requires a sha256-pinned image" >&2
  exit 2
fi

$DOCKER image inspect "$IMAGE" \
  --format 'P44_EXACT_IMAGE image_id={{.Id}}' >/dev/null
$DOCKER run --rm \
  -v "$ROOT:/workspace:ro" \
  -w /workspace \
  -e JAX_PLATFORMS=cpu \
  "$IMAGE" \
  bash -euo pipefail -c '
    rm -rf /tmp/p44-overlay
    bash canon-zero-tim/install.sh /tmp/p44-overlay \
      --from-path /usr/local/lib/python3.12/site-packages/tpu_inference \
      --model qwen4b
    bash canon-zero-tim/tests/p44_deepswe_qwen4b_parity/run_cpu.sh
    PYTHONPATH=/workspace/canon-zero-tim/src/engine_shims/models/qwen4b \
      python3 canon-zero-tim/src/engine_shims/models/qwen4b/p22xf_contract.py
    PYTHONPATH=/tmp/p44-overlay python3 \
      canon-zero-tim/tests/p38_serving/probe_fixed_lm_head_overlay.py \
      --hidden-size 2560
    PYTHONPATH=/tmp/p44-overlay python3 \
      canon-zero-tim/tests/p44_deepswe_qwen4b_parity/probe_swiglu_feature_padding.py \
      --feature 1216 --padded-feature 1280 --model qwen3-4b-tp8
    PYTHONPATH=/tmp/p44-overlay python3 \
      canon-zero-tim/tests/p44_deepswe_qwen4b_parity/probe_matmul_dim_padding.py \
      --mode interpret
    (
      cd tests/rl/agentic
      PYTHONPATH=/workspace python3 -m unittest \
        agentic_rl_learner_test.AgenticRLLearnerTest.test_model_call_wraps_one_conversation_as_a_prompt_batch \
        agentic_rl_learner_test.AgenticRLLearnerTest.test_rollout_batch_watchdog_fails_waiting_for_first_group \
        agentic_grpo_learner_test.AgenticGrpoLearnerTest.test_compute_logps_micro_batch_size \
        trajectory.trajectory_collect_engine_test.TrajectoryCollectEngineTest.test_model_timeout_aborts_turn_and_always_closes \
        trajectory.trajectory_collect_engine_test.TrajectoryCollectEngineTest.test_reset_timeout_still_closes_environment \
        trajectory.trajectory_collect_engine_test.TrajectoryCollectEngineTest.test_final_reward_timeout_is_recorded_and_closes \
        trajectory.trajectory_collect_engine_test.TrajectoryCollectEngineTest.test_cleanup_timeout_is_a_hard_error
    )
    (
      cd tests/rl/rollout
      PYTHONPATH=/workspace python3 -m unittest \
        vllm_rollout_canonical_test.VllmRolloutCanonicalTest.test_server_mode_deadline_aborts_unfinished_request
    )
    echo "P44_EXACT_IMAGE_CPU_PASS overlay=qwen4b"
  '
