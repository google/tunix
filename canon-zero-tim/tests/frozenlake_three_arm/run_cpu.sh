#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

python3 -m unittest \
  canon-zero-tim/tests/frozenlake_three_arm/test_standard.py \
  canon-zero-tim/tests/frozenlake_three_arm/test_training_geometry.py \
  canon-zero-tim/tests/frozenlake_three_arm/test_renderer.py \
  canon-zero-tim/tests/frozenlake_three_arm/test_checkpoint_eval.py \
  canon-zero-tim/tests/frozenlake_three_arm/test_eval_cycle_counter.py \
  canon-zero-tim/tests/frozenlake_three_arm/test_perf_v2_step_boundary.py \
  canon-zero-tim/tests/frozenlake_three_arm/test_perf_v2_onehost.py \
  canon-zero-tim/tests/frozenlake_resident/test_checkpoint_contract.py \
  canon-zero-tim/tests/frozenlake_three_arm/test_eval_classifier.py \
  canon-zero-tim/tests/frozenlake_three_arm/test_inprocess_eval_classifier.py \
  canon-zero-tim/tests/frozenlake_three_arm/test_stock_classifier.py \
  canon-zero-tim/tests/frozenlake_three_arm/test_jobset_log_collector.py \
  canon-zero-tim/tests/frozenlake_three_arm/test_m15_alignment_warning.py \
  canon-zero-tim/tests/frozenlake_three_arm/test_m15_token_continuity.py \
  canon-zero-tim/tests/frozenlake_three_arm/test_tito_collection_classifier.py \
  canon-zero-tim/tests/frozenlake_three_arm/test_tito_full_record_classifier.py \
  canon-zero-tim/tests/frozenlake_three_arm/test_tito_onehost_neutrality.py \
  canon-zero-tim/tests/frozenlake_three_arm/test_tito_gcs_sync.py \
  canon-zero-tim/tests/frozenlake_three_arm/test_tito_diagnostic_renderer.py \
  canon-zero-tim/tests/frozenlake_three_arm/test_runtime_contract.py \
  canon-zero-tim/tests/frozenlake_three_arm/test_workloads.py \
  canon-zero-tim/tests/workloads/test_sampler_is_contract.py \
  canon-zero-tim/tests/workloads/test_classify_run.py \
  canon-zero-tim/tests/alignment_carrier_serving/test_fixed_lm_head.py \
  canon-zero-tim/tests/alignment_carrier_serving/test_fixed_lm_head_receipts.py \
  canon-zero-tim/tests/frozenlake_resident/test_qwen8b_tp8.py
python3 -m py_compile \
  canon-zero-tim/cluster/render_frozenlake_calibration.py \
  canon-zero-tim/cluster/render_frozenlake_three_arm.py \
  canon-zero-tim/tests/workloads/classify_run.py \
  canon-zero-tim/tests/frozenlake_three_arm/test_m15_alignment_warning.py \
  canon-zero-tim/workloads/frozenlake-three-arm/scripts/classify_checkpoint_eval.py \
  canon-zero-tim/workloads/frozenlake-three-arm/scripts/classify_inprocess_eval.py \
  canon-zero-tim/workloads/frozenlake-three-arm/scripts/classify_perf_v2_onehost.py \
  canon-zero-tim/workloads/frozenlake-three-arm/scripts/classify_stock_discovery.py \
  canon-zero-tim/workloads/frozenlake-three-arm/scripts/census_perf_v2_onehost.py \
  canon-zero-tim/workloads/frozenlake-three-arm/scripts/collect_jobset_logs_to_gcs.py \
  canon-zero-tim/workloads/frozenlake-three-arm/scripts/derive_calibration_provenance.py \
  canon-zero-tim/workloads/frozenlake-three-arm/scripts/verify_calibration_manifest.py \
  canon-zero-tim/workloads/frozenlake-three-arm/scripts/verify_three_arm_manifests.py \
  canon-zero-tim/workloads/frozenlake-three-arm/scripts/verify_eval_schedule.py \
  canon-zero-tim/workloads/token-continuity/scripts/extract_first_diff_capsule.py \
  canon-zero-tim/workloads/token-continuity/scripts/classify_tito_collection.py \
  canon-zero-tim/workloads/token-continuity/scripts/classify_tito_full_record.py \
  canon-zero-tim/workloads/token-continuity/scripts/judge_tito_onehost_neutrality.py \
  canon-zero-tim/workloads/token-continuity/scripts/sync_tito_evidence_to_gcs.py \
  canon-zero-tim/workloads/token-continuity/scripts/render_tito_diagnostic_pair.py \
  canon-zero-tim/cluster/steps/p57_probe_stock_engine.py \
  canon-zero-tim/src/p57_stock_prompt_observer.py \
  examples/frozenlake/p57_workloads.py \
  examples/frozenlake/train_frozenlake_qwen3.py \
  tunix/rl/agentic/agentic_rl_learner.py \
  tunix/rl/agentic/trajectory/trajectory_collect_engine.py \
  tunix/perf/experimental/timeline.py \
  tunix/perf/experimental/tracer.py \
  tunix/rl/frozenlake_checkpoint.py
bash -n \
  canon-zero-tim/cluster/steps/00_env.sh \
  canon-zero-tim/cluster/entrypoint.sh \
  canon-zero-tim/cluster/steps/37_install_stock_runtime.sh \
  canon-zero-tim/cluster/steps/38_verify_stock_engine.sh \
  canon-zero-tim/cluster/steps/39_install_p57_stock_observer.sh \
  canon-zero-tim/cluster/steps/p57_runtime_contract.sh \
  canon-zero-tim/cluster/steps/90_run.sh \
  canon-zero-tim/cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-tim.env \
  canon-zero-tim/cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-tito-diagnostic.env \
  canon-zero-tim/workloads/frozenlake-three-arm/scripts/render_eval_schedule.sh \
  canon-zero-tim/workloads/frozenlake-three-arm/scripts/run_perf_v2_onehost.sh \
  canon-zero-tim/workloads/token-continuity/scripts/p57_tito_gcs_worker.sh \
  canon-zero-tim/workloads/token-continuity/scripts/run_tito_onehost_neutrality_pair.sh \
  canon-zero-tim/tests/frozenlake_three_arm/run_cpu.sh
echo "P57_FROZENLAKE_TIM_CPU_PASS"
