#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

python3 -m unittest \
  canon-zero-tim/tests/p57_frozenlake_tim/test_renderer.py \
  canon-zero-tim/tests/p57_frozenlake_tim/test_checkpoint_eval.py \
  canon-zero-tim/tests/p57_frozenlake_tim/test_eval_cycle_counter.py \
  canon-zero-tim/tests/p57_frozenlake_tim/test_perf_v2_step_boundary.py \
  canon-zero-tim/tests/p57_frozenlake_tim/test_perf_v2_onehost.py \
  canon-zero-tim/tests/p45_frozenlake_dp8_tp8/test_checkpoint_contract.py \
  canon-zero-tim/tests/p57_frozenlake_tim/test_eval_classifier.py \
  canon-zero-tim/tests/p57_frozenlake_tim/test_inprocess_eval_classifier.py \
  canon-zero-tim/tests/p57_frozenlake_tim/test_stock_classifier.py \
  canon-zero-tim/tests/p57_frozenlake_tim/test_provenance_derivation.py \
  canon-zero-tim/tests/p57_frozenlake_tim/test_jobset_log_collector.py \
  canon-zero-tim/tests/p57_frozenlake_tim/test_m15_alignment_warning.py \
  canon-zero-tim/tests/p57_frozenlake_tim/test_m15_token_continuity.py \
  canon-zero-tim/tests/p57_frozenlake_tim/test_tito_collection_classifier.py \
  canon-zero-tim/tests/p57_frozenlake_tim/test_tito_full_record_classifier.py \
  canon-zero-tim/tests/p57_frozenlake_tim/test_tito_onehost_neutrality.py \
  canon-zero-tim/tests/p57_frozenlake_tim/test_tito_gcs_sync.py \
  canon-zero-tim/tests/p57_frozenlake_tim/test_tito_diagnostic_renderer.py \
  canon-zero-tim/tests/p57_frozenlake_tim/test_runtime_contract.py \
  canon-zero-tim/tests/p57_frozenlake_tim/test_workloads.py \
  canon-zero-tim/tests/p33_workloads/test_sampler_is_contract.py \
  canon-zero-tim/tests/p33_workloads/test_classify_run.py \
  canon-zero-tim/tests/p38_serving/test_fixed_lm_head.py \
  canon-zero-tim/tests/p38_serving/test_fixed_lm_head_receipts.py \
  canon-zero-tim/tests/p45_frozenlake_dp8_tp8/test_qwen8b_tp8.py
python3 -m py_compile \
  canon-zero-tim/cluster/render_p57_calibration.py \
  canon-zero-tim/cluster/render_p57_frozenlake_tim.py \
  canon-zero-tim/tests/p33_workloads/classify_run.py \
  canon-zero-tim/tests/p57_frozenlake_tim/test_m15_alignment_warning.py \
  canon-zero-tim/tasks/p57-frozenlake-tim-causal-study/scripts/classify_checkpoint_eval.py \
  canon-zero-tim/tasks/p57-frozenlake-tim-causal-study/scripts/classify_inprocess_eval.py \
  canon-zero-tim/tasks/p57-frozenlake-tim-causal-study/scripts/classify_perf_v2_onehost.py \
  canon-zero-tim/tasks/p57-frozenlake-tim-causal-study/scripts/classify_stock_discovery.py \
  canon-zero-tim/tasks/p57-frozenlake-tim-causal-study/scripts/census_perf_v2_onehost.py \
  canon-zero-tim/tasks/p57-frozenlake-tim-causal-study/scripts/collect_jobset_logs_to_gcs.py \
  canon-zero-tim/tasks/p57-frozenlake-tim-causal-study/scripts/derive_calibration_provenance.py \
  canon-zero-tim/tasks/p57-frozenlake-tim-causal-study/scripts/verify_calibration_manifest.py \
  canon-zero-tim/tasks/p57-frozenlake-tim-causal-study/scripts/verify_three_arm_manifests.py \
  canon-zero-tim/tasks/p57-frozenlake-tim-causal-study/scripts/verify_eval_schedule.py \
  canon-zero-tim/tasks/multiturn-tito-cross-workload/scripts/extract_first_diff_capsule.py \
  canon-zero-tim/tasks/multiturn-tito-cross-workload/scripts/classify_tito_collection.py \
  canon-zero-tim/tasks/multiturn-tito-cross-workload/scripts/classify_tito_full_record.py \
  canon-zero-tim/tasks/multiturn-tito-cross-workload/scripts/judge_tito_onehost_neutrality.py \
  canon-zero-tim/tasks/multiturn-tito-cross-workload/scripts/sync_tito_evidence_to_gcs.py \
  canon-zero-tim/tasks/multiturn-tito-cross-workload/scripts/render_tito_diagnostic_pair.py \
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
  canon-zero-tim/tasks/p57-frozenlake-tim-causal-study/scripts/render_eval_schedule.sh \
  canon-zero-tim/tasks/p57-frozenlake-tim-causal-study/scripts/run_perf_v2_onehost.sh \
  canon-zero-tim/tasks/multiturn-tito-cross-workload/scripts/p57_tito_gcs_worker.sh \
  canon-zero-tim/tasks/multiturn-tito-cross-workload/scripts/run_tito_onehost_neutrality_pair.sh \
  canon-zero-tim/tests/p57_frozenlake_tim/run_cpu.sh
echo "P57_FROZENLAKE_TIM_CPU_PASS"
