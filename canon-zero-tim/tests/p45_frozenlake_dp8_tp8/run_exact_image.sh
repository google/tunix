#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
IMAGE_REF="${1:-tunix_frozenlake_image:vllm-tpu0.25.0}"
DOCKER="${DOCKER:-sudo docker}"

IMAGE_ID="$($DOCKER image inspect "$IMAGE_REF" --format '{{.Id}}')"
if [[ ! "$IMAGE_ID" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  echo "P45 exact-image gate could not resolve immutable image ID: $IMAGE_ID" >&2
  exit 2
fi
echo "P45_EXACT_IMAGE image_ref=$IMAGE_REF image_id=$IMAGE_ID"

$DOCKER run --rm \
  -v "$ROOT:/workspace:ro" \
  -w /workspace \
  -e JAX_PLATFORMS=cpu \
  "$IMAGE_ID" \
  bash -euo pipefail -c '
    overlay="$(mktemp -d /tmp/p45-qwen8b-tp8.XXXXXX)"
    trap '\''rm -r "$overlay"'\'' EXIT
    bash canon-zero-tim/install.sh "$overlay" \
      --from-path /usr/local/lib/python3.12/site-packages/tpu_inference \
      --model qwen8b_tp8
    bash canon-zero-tim/tests/p45_frozenlake_dp8_tp8/run_cpu.sh
    PYTHONPATH=/workspace python3 tests/sft/peft_trainer_test.py \
      PeftTrainerTest.test_p28_g6_checkpointing_is_isolated_to_signed_p45 \
      PeftTrainerTest.test_p28_g6_precomputed_four_microstep_update
    PYTHONPATH=/workspace python3 \
      tests/rl/agentic/agentic_rl_learner_test.py \
      AgenticRLLearnerTest.test_nonpositive_eval_cadence_disables_evaluation \
      AgenticRLLearnerTest.test_p31_segmented_eval_uses_preupdate_step_exactly_once \
      AgenticRLLearnerTest.test_p57_evaluate_only_covers_dataset_without_train_update \
      AgenticRLLearnerTest.test_p57_rollout_only_evaluate_skips_trainer_recompute
    PYTHONPATH=/workspace python3 \
      canon-zero-tim/tests/p57_frozenlake_tim/test_stock_fast_contract.py
    PYTHONPATH="$overlay" python3 \
      canon-zero-tim/src/engine_shims/models/qwen8b_tp8/p22xf_contract.py
    CANON_SHIM_ROOT="$overlay" PYTHONPATH="$overlay" python3 \
      canon-zero-tim/tests/p45_frozenlake_dp8_tp8/probe_overlay_import.py
    PYTHONPATH="$overlay" python3 \
      canon-zero-tim/tests/p38_serving/probe_fixed_lm_head_overlay.py \
      --hidden-size 4096 --tp-size 8
    PYTHONPATH="$overlay" python3 \
      canon-zero-tim/tests/p45_frozenlake_dp8_tp8/probe_qwen8b_tp8.py
    echo "P45_EXACT_IMAGE_CPU_PASS overlay=qwen8b_tp8"
  '
