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
    stock_state="$(mktemp -d /tmp/p57-stock-state.XXXXXX)"
    printf "%s\n" /usr/local/lib/python3.12/site-packages/tpu_inference \
      > "$stock_state/tpu_inference_path"
    for stock_kind in calibration train eval; do
      cat > "$stock_state/env.sh" <<EOF
export CANON_PROFILE_FILE=cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-tim.env
export CANON_P57_RUN_KIND=$stock_kind
export CANON_P57_TIM_ARM=mismatch
export CANON_P57_INFERENCE_REGIME=stock-fast
EOF
      CANON_STATE="$stock_state" CANON_PKG=/workspace/canon-zero-tim \
        bash canon-zero-tim/cluster/steps/37_install_stock_runtime.sh
      CANON_STATE="$stock_state" CANON_PKG=/workspace/canon-zero-tim \
        bash canon-zero-tim/cluster/steps/38_verify_stock_engine.sh
      echo "P57_STOCK_RUNTIME_MODE_PASS run_kind=$stock_kind"
    done
    cat > "$stock_state/env.sh" <<EOF
export CANON_PROFILE_FILE=cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-tim.env
export CANON_P57_RUN_KIND=train
export CANON_P57_TIM_ARM=is
export CANON_P57_INFERENCE_REGIME=stock-fast
EOF
    CANON_STATE="$stock_state" CANON_PKG=/workspace/canon-zero-tim \
      bash canon-zero-tim/cluster/steps/37_install_stock_runtime.sh
    CANON_STATE="$stock_state" CANON_PKG=/workspace/canon-zero-tim \
      bash canon-zero-tim/cluster/steps/38_verify_stock_engine.sh
    echo "P57_STOCK_RUNTIME_MODE_PASS run_kind=train arm=is"
    cat > "$stock_state/env.sh" <<EOF
export CANON_PROFILE_FILE=cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-tim.env
export CANON_P57_RUN_KIND=train
export CANON_P57_TIM_ARM=zero
export CANON_P57_INFERENCE_REGIME=stock-fast
EOF
    if CANON_STATE="$stock_state" CANON_PKG=/workspace/canon-zero-tim \
        bash canon-zero-tim/cluster/steps/37_install_stock_runtime.sh; then
      echo "P57 stock-runtime negative admitted the zero arm" >&2
      exit 1
    fi
    echo "P57_STOCK_RUNTIME_NEGATIVE_PASS arm=zero rejected=1"
    cat > "$stock_state/env.sh" <<EOF
export CANON_PROFILE_FILE=cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-tim.env
export CANON_P57_RUN_KIND=eval
export CANON_P57_TIM_ARM=mismatch
export CANON_P57_INFERENCE_REGIME=stock-fast
EOF
    bad_package="$(mktemp -d /tmp/p57-stock-negative.XXXXXX)/tpu_inference"
    while read -r _ relative; do
      mkdir -p "$bad_package/$(dirname "$relative")"
      cp "/usr/local/lib/python3.12/site-packages/tpu_inference/$relative" \
        "$bad_package/$relative"
    done < canon-zero-tim/STOCK_MANIFEST.sha256
    printf "\n# deliberate stock drift\n" >> "$bad_package/layers/jax/linear.py"
    printf "%s\n" "$bad_package" > "$stock_state/tpu_inference_path"
    if CANON_STATE="$stock_state" CANON_PKG=/workspace/canon-zero-tim \
        bash canon-zero-tim/cluster/steps/38_verify_stock_engine.sh; then
      echo "P57 stock-engine negative control failed to reject drift" >&2
      exit 1
    fi
    echo "P57_STOCK_ENGINE_NEGATIVE_PASS drift=rejected"
    rm -r "$(dirname "$bad_package")"
    rm -r "$stock_state"
    script_log="$(mktemp /tmp/p57-script-mode-negative.XXXXXX)"
    if env -u PYTHONPATH JAX_PLATFORMS=cpu python3 -u \
        examples/frozenlake/train_frozenlake_qwen3.py --helpshort \
        > "$script_log" 2>&1; then
      echo "P57 file-path entrypoint negative control unexpectedly passed" >&2
      exit 1
    fi
    rm -f "$script_log"
    echo "P57_FILE_ENTRYPOINT_NEGATIVE_PASS script_mode=rejected"
    env -u PYTHONPATH JAX_PLATFORMS=cpu python3 -u -m \
      examples.frozenlake.train_frozenlake_qwen3 --help > /dev/null
    echo "P57_MODULE_ENTRYPOINT_PASS workload_import=complete"
    PYTHONPATH="$overlay" python3 \
      canon-zero-tim/src/engine_shims/models/qwen8b_tp8/p22xf_contract.py
    CANON_SHIM_ROOT="$overlay" PYTHONPATH="$overlay" python3 \
      canon-zero-tim/tests/p45_frozenlake_dp8_tp8/probe_overlay_import.py
    PYTHONPATH="$overlay" python3 \
      canon-zero-tim/tests/p38_serving/probe_fixed_lm_head_overlay.py \
      --hidden-size 4096 --tp-size 8
    PYTHONPATH="$overlay" python3 \
      canon-zero-tim/tests/p45_frozenlake_dp8_tp8/probe_qwen8b_tp8.py
    observer_state="$(mktemp -d /tmp/p57-stock-observer-state.XXXXXX)"
    cat > "$observer_state/env.sh" <<EOF
export CANON_PROFILE_FILE=cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-tim.env
export CANON_P57_RUN_KIND=train
export CANON_P57_TIM_ARM=mismatch
export CANON_P57_INFERENCE_REGIME=stock-fast
export CANON_PROMPT_PROCESSED_LOGPROBS=1
EOF
    printf "%s\n" /usr/local/lib/python3.12/site-packages/tpu_inference \
      > "$observer_state/tpu_inference_path"
    CANON_STATE="$observer_state" CANON_PKG=/workspace/canon-zero-tim \
      bash canon-zero-tim/cluster/steps/39_install_p57_stock_observer.sh
    env PATHWAYS_HEAD="" JAX_BACKEND_TARGET="" JAX_PLATFORMS=cpu \
      python3 canon-zero-tim/tests/p57_frozenlake_tim/probe_stock_prompt_observer.py
    rm -r "$observer_state"
    echo "P57_STOCK_OBSERVER_EXACT_IMAGE_PASS targets=absolute values=processed"
    echo "P45_EXACT_IMAGE_CPU_PASS overlay=qwen8b_tp8"
  '
