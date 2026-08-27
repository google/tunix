#!/usr/bin/env python3
"""Contracts for the P58.7 Zero-HP target postflight."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    ROOT / "canon-zero-tim/tasks/p58-deepswe-native-zero-comparison/scripts/"
    "classify_zero_hp_full.py"
)
SPEC = importlib.util.spec_from_file_location("p58_zero_hp_classifier", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
classifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(classifier)


def _env() -> dict[str, str]:
  return {
      "CANON_PROFILE_FILE": (
          "cluster/profiles/qwen3-4b-dp8-tp8-deepswe-v1-hp.env"
      ),
      "CANON_V1_HP_FULL": "1",
      "CANON_P58_DEEPSWE_TIM": "1",
      "CANON_P58_TIM_ADMITTED": "1",
      "CANON_P58_TIM_ARM": "zero",
      "CANON_P58_EXPECTED_UPDATES": "1000",
      "CANON_P34_RUN_STAGE": "full",
      "CANON_P34_NO_COMMIT": "0",
      "CANON_DEEPSWE_ALIGNMENT_WARN_ONLY": "0",
      "CANON_P38_FIXED_LM_HEAD": "1",
      "CANON_CONTINUE_DECODE": "8",
      "CANON_FIXED_AR_GATHER": "1",
      "CANON_PALLAS_GATHERED_LOGPROBS": "1",
      "CANON_LOGPROB_STEP_FUSION": "1",
      "CANON_VLLM_ENABLE_PREFIX_CACHING": "0",
      "CANON_P59_RANK_PARALLEL_BACKWARD": "1",
      "CANON_P59_CHECKED_VMA": "1",
      "CANON_P66_P59_CHECK_VMA": "1",
      "CANON_V1_HP_FIRST_UPDATE_GATE": "1",
      "CANON_P63_OVERFLOW_SAFE_CLIP": "1",
      "CANON_P28_BATCHED_REPORT": "1",
      "CANON_P28_BATCHED_REVERSE": "0",
      "CANON_BATCHED_EVIDENCE": "0",
      "CANON_FUSED_TREE_OPS": "0",
      "CANON_PALLAS_NORM_MATMUL": "0",
      "CANON_PALLAS_INPUT_FUSION": "0",
      "CANON_SAMPLE_SPLIT_FUSION": "0",
      "CANON_ENGINE_LOGPROB_READBACK": "0",
      "CANON_ANCHOR_OVERLAP": "0",
      "CANON_OPT_STATE_RESIDENT": "1",
      "CANON_P30_OPT_STATE_OFFLOAD": "0",
      "CANON_XPROF_PHASE": "update",
      "CANON_XPROF_SKIP_STEPS": "2",
      "CANON_XPROF_STEPS": "1",
      "CANON_XPROF_PYTHON_TRACER": "0",
      "CANON_XPROF_HOST_TRACER": "1",
      "CANON_XPROF_TPU_TRACE_MODE": "TRACE_COMPUTE",
      "CANON_XPROF_LABELS": "1",
      "CANON_PERF_TRACE_EXPORT_STEP": "2",
  }


def _update(index: int) -> dict:
  return {
      "contract_name": "p58-qwen4b-tim-128",
      "dp_size": 8,
      "tp_size": 8,
      "global_m": 2048,
      "microsteps": 16,
      "dp_rank_pullbacks_per_transaction": 8,
      "dp_pullback_invocations_per_transaction": 1,
      "dp_replicas_exact": True,
      "gradient_finite": True,
      "optimizer_placement": "device-resident",
      "verdict": "PASS",
      "commits": 1,
      "elapsed_seconds": 4.0,
      "train_steps_before": index,
      "train_steps_after": index + 1,
      "commit_evidence": {
          "overflow_safe_clip": {
              "enabled": True,
              "all_finite": True,
              "naive_norm": 0.5,
              "naive_norm_finite": True,
              "stable_norm": 0.5,
              "selected_norm": 0.5,
              "fallback_used": False,
              "clip_factor": 1.0,
              "max_norm": 1.0,
          },
      },
  }


def _skip(index: int) -> dict:
  row = _update(index)
  row.update({
      "mode": "compact-filtered-no-commit",
      "commits": 0,
      "elapsed_seconds": 3.0,
      "train_steps_after": index,
  })
  row.pop("commit_evidence")
  return row


def _fixture(
    root: Path, *, skip_before_first_commit: bool = False
) -> tuple[Path, Path, Path]:
  (root / "env.sh").write_text(
      "".join(f"export {key}={value}\n" for key, value in _env().items())
  )
  updates = root / "updates.jsonl"
  update_rows = [_update(index) for index in range(1000)]
  if skip_before_first_commit:
    update_rows.insert(0, _skip(0))
  updates.write_text("".join(json.dumps(row) + "\n" for row in update_rows))
  base = root / "base.json"
  base.write_text(json.dumps({
      "verdict": "PASS",
      "arm": "zero",
      "stage": "full",
      "expected_commits": 1000,
      "observed_commits": 1000,
      "checks": {"zero_all_boundaries_exact": True},
  }))
  lines = [
      "[CANON_ADAPTER.PLACEMENT] PASS relation=disjoint "
      "rollout_devices=64 trainer_devices=64 execution_role=trainer",
      "[CANON_ADAPTER.PLACEMENT] trainer logprob scorer rebound "
      "relation=disjoint implementation=factory-identical "
      "mesh_bound_instances=2",
      "[CANON_ADAPTER.PLACEMENT] trainer model callables rebuilt "
      "relation=disjoint graph=abstract-clone mesh_bound_jits=2",
      "[CANON_ADAPTER.PLACEMENT] trainer state contract PASS "
      "relation=disjoint leaves=398 "
      "normalized_loader_metadata=_is_loaded live_markers=398 "
      "reconstruction_markers=0",
      "[P57.CONTINUE_DECODE] on-device decode loop enabled max_decode_steps=8 workload=deepswe",
      "[P56.GATHERED_LOGPROBS] installed",
      "[P56.LOGPROB_STEP_FUSION] active",
      "CANON_FIXED_AR=1 gather-ordered-sum",
      "[CANON_XPROF_LABELS] continue-decode stage callables cached",
      "[P63.STABLE_CLIP] configured enabled=1 mode=hybrid "
      "stock_finite=stock_exact overflow_fallback=max_scaled_l2 "
      "nonfinite=fatal max_norm=1.0 workload=p58-qwen4b-tim-128",
      "[V1.FIRST_UPDATE] " + json.dumps({
          "schema": "canon-v1-first-update-precommit-v1",
          "update": 0,
          "workload": "p58-qwen4b-tim-128",
          "dp": 8,
          "tp": 8,
          "microsteps": 16,
          "accumulator_denominator": 16.0,
          "stable_norm_max": 1.0e6,
          "all_finite": True,
          "any_nonzero": True,
          "stable_norm": 0.5,
      }, sort_keys=True, separators=(",", ":")),
      "[V1.FIRST_UPDATE] " + json.dumps({
          "schema": "canon-v1-first-update-commit-v1",
          "update": 0,
          "workload": "p58-qwen4b-tim-128",
          "dp": 8,
          "tp": 8,
          "train_steps_before": 0,
          "train_steps_after": 1,
          "optimizer_transaction_valid": True,
          "gradient_finite": True,
          "parameter_delta_finite": True,
          "parameter_changed_elements": 10,
          "effective_learning_rate": 1.0e-6,
          "outer_weight_sync_pending": True,
      }, sort_keys=True, separators=(",", ":")),
      "[P51.XPROF] phase=update armed step=2",
      "[P51.XPROF] phase=update started step=2",
      "[P51.XPROF] phase=update stopped step=3",
      "[V1.PERFETTO] captured training_step=2 timelines=3",
  ]
  if skip_before_first_commit:
    lines.extend((
        "[P59.DP8] gradient_reducer_ready dp_axis=dp dp_size=8 staging=parallel_table",
        "[P59.CHECKED_VMA] enabled=1 "
        "workload=p58-qwen4b-tim-128 dp=8 tp=8 global_M=2048 "
        "manual_axes=data,model compatibility_alias=1",
        "[PERF] stage=p32_vag_forward seconds=0.8",
        "[PERF] stage=p32_vag_reverse seconds=1.6",
        "[PERF] stage=segmented_value_and_grad seconds=2.4",
        "[DEEPSWE.COMPACT_FILTER] canonical all_filtered=1 "
        "train_steps=0 optimizer_commits=0",
        "[P58.COMPACT_FILTER] all_filtered=1 optimizer_commits=0 "
        "train_steps=0 global_steps=0 policy_version=0 weight_sync=0",
    ))
  for index in range(1000):
    lines.extend((
        "[P59.DP8] gradient_reducer_ready dp_axis=dp dp_size=8 staging=parallel_table",
        "[P59.CHECKED_VMA] enabled=1 "
        "workload=p58-qwen4b-tim-128 dp=8 tp=8 global_M=2048 "
        "manual_axes=data,model compatibility_alias=1",
        f"[P63.STABLE_CLIP] update={index + 1} all_finite=1 "
        "naive_norm=0.5 naive_norm_finite=1 stable_norm=0.5 "
        "selected_norm=0.5 fallback_used=0 clip_factor=1.0 max_norm=1.0",
        "[PERF] stage=p32_vag_forward seconds=1.0",
        "[PERF] stage=p32_vag_reverse seconds=2.0",
        "[PERF] stage=segmented_value_and_grad seconds=3.0",
        "[PERF] stage=optimizer_transaction seconds=0.5",
        "[PERF] stage=weight_sync seconds=0.2",
        f"Global step {index + 1} completed in 5.0 seconds.",
    ))
  log = root / "run.log"
  log.write_text("\n".join(lines) + "\n")
  capture = root / "xprof-update/plugins/profile/run"
  capture.mkdir(parents=True)
  (capture / "device.xplane.pb").write_bytes(b"xplane")
  (capture / "device.trace.json.gz").write_bytes(b"trace")
  perfetto = root / "perfetto"
  perfetto.mkdir()
  (perfetto / "perfetto_trace_v2_1.pb").write_bytes(b"perfetto")
  (root / "p38_fixed_lm_head_receipts.json").write_text("{}\n")
  return log, updates, base


class ZeroHpFullClassifierTest(unittest.TestCase):

  def test_complete_target_fixture_passes_and_reports_timing(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      log, updates, base = _fixture(root)
      result = classifier.classify(
          state=root,
          run_log=log,
          update_report=updates,
          base_classification=base,
      )
      self.assertEqual(result["verdict"], "PASS", result["reasons"])
      self.assertEqual(
          result["timing"]["steady_steps2_plus_excluding_profile_count"],
          998,
      )
      self.assertFalse(result["serial_adamw_trajectory_identity_claimed"])

  def test_compact_filtered_attempt_is_reconciled_without_a_fake_commit(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      log, updates, base = _fixture(root, skip_before_first_commit=True)
      result = classifier.classify(
          state=root,
          run_log=log,
          update_report=updates,
          base_classification=base,
      )
      self.assertEqual(result["verdict"], "PASS", result["reasons"])
      self.assertEqual(result["updates"]["observed"], 1000)
      self.assertEqual(result["attempts"], {"observed": 1001, "skipped": 1})
      self.assertEqual(
          result["timing"]["steady_steps2_plus_excluding_profile_count"],
          998,
      )

  def test_partial_bundle_and_p59_serialization_are_rejected(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      log, updates, base = _fixture(root)
      env = root / "env.sh"
      env.write_text(
          env.read_text().replace(
              "export CANON_CONTINUE_DECODE=8",
              "export CANON_CONTINUE_DECODE=0",
          )
      )
      rows = [json.loads(line) for line in updates.read_text().splitlines()]
      rows[0]["dp_pullback_invocations_per_transaction"] = 8
      updates.write_text("".join(json.dumps(row) + "\n" for row in rows))
      result = classifier.classify(
          state=root,
          run_log=log,
          update_report=updates,
          base_classification=base,
      )
      self.assertEqual(result["verdict"], "FAIL")
      self.assertTrue(any("resolved_env" in value for value in result["reasons"]))
      self.assertTrue(any("update[0].p59" in value for value in result["reasons"]))

  def test_missing_device_trace_is_capture_failure(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      log, updates, base = _fixture(root)
      next((root / "xprof-update").rglob("*.xplane.pb")).unlink()
      result = classifier.classify(
          state=root,
          run_log=log,
          update_report=updates,
          base_classification=base,
      )
      self.assertEqual(result["verdict"], "FAIL")
      self.assertIn("artifact.xplane", result["reasons"])

  def test_missing_checked_vma_first_update_or_clip_receipt_is_rejected(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      log, updates, base = _fixture(root)
      rows = [json.loads(line) for line in updates.read_text().splitlines()]
      rows[0]["commit_evidence"].pop("overflow_safe_clip")
      updates.write_text("".join(json.dumps(row) + "\n" for row in rows))
      log.write_text(
          "\n".join(
              line
              for line in log.read_text().splitlines()
              if not line.startswith("[V1.FIRST_UPDATE]")
              and not line.startswith("[P59.CHECKED_VMA]")
          )
          + "\n"
      )
      result = classifier.classify(
          state=root,
          run_log=log,
          update_report=updates,
          base_classification=base,
      )
      self.assertEqual(result["verdict"], "FAIL")
      self.assertTrue(any("p63_missing" in value for value in result["reasons"]))
      self.assertTrue(any("checked_vma" in value for value in result["reasons"]))
      self.assertTrue(any("first_update" in value for value in result["reasons"]))

  def test_missing_trainer_model_jit_rebind_receipt_is_rejected(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      log, updates, base = _fixture(root)
      log.write_text(
          "\n".join(
              line
              for line in log.read_text().splitlines()
              if "trainer model callables rebuilt" not in line
          )
          + "\n"
      )
      result = classifier.classify(
          state=root,
          run_log=log,
          update_report=updates,
          base_classification=base,
      )
      self.assertEqual(result["verdict"], "FAIL")
      self.assertIn("marker.trainer_model_jits=0", result["reasons"])

  def test_missing_trainer_state_contract_receipt_is_rejected(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      log, updates, base = _fixture(root)
      log.write_text(
          "\n".join(
              line
              for line in log.read_text().splitlines()
              if "trainer state contract PASS" not in line
          )
          + "\n"
      )
      result = classifier.classify(
          state=root,
          run_log=log,
          update_report=updates,
          base_classification=base,
      )
      self.assertEqual(result["verdict"], "FAIL")
      self.assertIn("marker.trainer_state_contract=0", result["reasons"])


if __name__ == "__main__":
  unittest.main()
