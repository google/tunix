"""Tests for the fail-closed P33 run classifier."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


_CLASSIFIER_PATH = Path(__file__).with_name("classify_run.py")
_MODULE_SPEC = importlib.util.spec_from_file_location(
    "classify_p33_run", _CLASSIFIER_PATH
)
assert _MODULE_SPEC is not None and _MODULE_SPEC.loader is not None
classifier = importlib.util.module_from_spec(_MODULE_SPEC)
sys.modules[_MODULE_SPEC.name] = classifier
_MODULE_SPEC.loader.exec_module(classifier)


def _policy(enabled: bool, workload: str = "gsm8k") -> dict:
  return {
      "id": (
          classifier._FROZENLAKE_WARNING_POLICY_ID
          if workload == "frozenlake"
          else classifier._WARNING_POLICY_ID
      ),
      "enabled": enabled,
      "warning_only": enabled,
      "bounded_ab_only": False,
      "workload": workload if enabled else "",
      "stage": "full" if enabled else "",
      "max_abs_limit": None if enabled else 1.0e-4,
      "byte_fraction_limit": None if enabled else 4.0e-3,
      "claim_level": "convergence-only" if enabled else "strict-zero-tim",
  }


def _boundary(*, drift: bool = False) -> dict:
  return {
      "valid": True,
      "finite": True,
      "differing_bytes": 4 if drift else 0,
      "differing_elements": 1 if drift else 0,
      "byte_fraction": 0.25 if drift else 0.0,
      "element_fraction": 0.5 if drift else 0.0,
      "max_abs": 3.0 if drift else 0.0,
  }


def _alignment(
    step: int,
    *,
    optimizer_skipped: bool,
    warning_policy: bool = False,
    warned: bool = False,
    policy_workload: str = "gsm8k",
) -> dict:
  record = {
      "verdict": "PASS",
      "reds": [],
      "blocking_reds": [],
      "reported_reds": [],
      "warning_reds": [],
      "admission_policy": _policy(warning_policy, policy_workload),
      "execution_mode": "train",
      "step": step,
      "N_action": 4,
      "boundaries": {
          name: _boundary()
          for name in classifier._BOUNDARIES
      },
      "exact": {name: True for name in classifier._EXACT_KEYS},
      "ratio_finite": True,
      "ratio_stats": {
          name: {"min": 1.0, "max": 1.0} for name in ("w", "r", "wr")
      },
      "clip_hits": 0,
      "tis_hits": 0,
      "optimizer_skipped": optimizer_skipped,
      "gradient": {"finite": True, "nonzero": True, "norm": 1.0},
  }
  if warned:
    record["verdict"] = "PASS_WITH_ALIGNMENT_WARNINGS"
    record["reds"] = [
        "S_decode_vs_S_prefill",
        "S_prefill_vs_T_old",
        "T_old_vs_T_current",
        "w_all_exactly_1",
        "r_all_exactly_1",
        "wr_all_exactly_1",
        "clip_hits=3",
        "tis_hits=2",
    ]
    record["warning_reds"] = list(record["reds"])
    record["boundaries"] = {
        name: _boundary(drift=True) for name in classifier._BOUNDARIES
    }
    record["exact"] = {name: False for name in classifier._EXACT_KEYS}
    record["clip_hits"] = 3
    record["tis_hits"] = 2
    record["ratio_stats"] = {
        "w": {"min": 0.1, "max": 20.0},
        "r": {"min": 0.2, "max": 5.0},
        "wr": {"min": 0.02, "max": 100.0},
    }
  return record


def _pre_alignment(
    step: int,
    *,
    warning_policy: bool = False,
    warned: bool = False,
    policy_workload: str = "gsm8k",
) -> dict:
  record = {
      "verdict": "PASS",
      "reds": [],
      "blocking_reds": [],
      "reported_reds": [],
      "warning_reds": [],
      "admission_policy": _policy(warning_policy, policy_workload),
      "step": step,
      "N_action": 4,
      "boundaries": {
          name: _boundary()
          for name in classifier._PRE_BOUNDARIES
      },
  }
  if warned:
    record["verdict"] = "PASS_WITH_ALIGNMENT_WARNINGS"
    record["reds"] = list(classifier._PRE_BOUNDARIES)
    record["warning_reds"] = list(record["reds"])
    record["boundaries"] = {
        name: _boundary(drift=True) for name in classifier._PRE_BOUNDARIES
    }
  return record


def _update(
    index: int,
    *,
    placement: str = "pinned-host-offload",
    dp_size: int = 16,
    tp_size: int = 4,
) -> dict:
  effective_lr = 0.0 if index == 0 else 4.0e-9
  parameter_changed = 0 if index == 0 else 1
  memory_kind = (
      "device" if placement == "device-resident" else "pinned_host"
  )
  return {
      "verdict": "PASS",
      "dp_axis": "data",
      "dp_size": dp_size,
      "tp_size": tp_size,
      "global_m": dp_size * 256,
      "microsteps": 256 // dp_size,
      "commits": 1,
      "train_steps_before": index,
      "train_steps_after": index + 1,
      "gradient_activity": [True] * (256 // dp_size),
      "alignment_hashes": [{"T_current": "a"}] * (256 // dp_size),
      "micro_gradient_norms": [1.0] * (256 // dp_size),
      "optimizer_placement": placement,
      "optimizer_memory_kinds_before": [memory_kind],
      "optimizer_memory_kinds_after": [memory_kind],
      "accumulator_changed_paths": [],
      "reference_changed_paths": [],
      "commit_gradient_norm": 1.0,
      "optimizer_transaction_valid": True,
      "parameter_mutation": (
          "zero_lr_unchanged" if index == 0 else "observed_nonzero"
      ),
      "commit_evidence": {
          "effective_learning_rate": effective_lr,
          "gradient_nonzero_elements": 1,
          "gradient_max_abs": 1.0,
          "gradient_finite": True,
          "parameter_changed_elements": parameter_changed,
          "parameter_total_elements": 1,
          "parameter_delta_max_abs": 0.0 if index == 0 else 1.0e-8,
          "parameter_delta_finite": True,
          "optimizer_timing": {
              "optimizer_logical_bytes": 1024,
              "optimizer_h2d_seconds": 0.0,
              "adam_commit_seconds": 1.0,
              "optimizer_d2h_seconds": 0.0,
              "optimizer_transaction_seconds": 1.0,
          },
      },
  }


class ClassifyP33RunTest(unittest.TestCase):

  def _write_jsonl(self, path: Path, records) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

  def test_full_gsm8k_positive(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      run_log = root / "run.log"
      updates = root / "updates.jsonl"
      pre_alignments = root / "pre_alignment.jsonl"
      alignments = root / "alignment.jsonl"
      run_log.write_text(
          "[CANON_P33_WANDB] ONLINE_RUN_PASS\n"
          "[CANON_P31_METRICS] monotonic_direct last_step=199 events=200 regressions=0\n"
          + "[CANON_P33_DP16] update_step_committed\n" * 200,
          encoding="utf-8",
      )
      self._write_jsonl(updates, (_update(index) for index in range(200)))
      self._write_jsonl(
          pre_alignments,
          (_pre_alignment(index, warning_policy=True) for index in range(200)),
      )
      self._write_jsonl(
          alignments,
          (
              _alignment(index, optimizer_skipped=False, warning_policy=True)
              for index in range(3200)
          ),
      )
      record = classifier.classify(
          workload="gsm8k",
          stage="full",
          run_log=run_log,
          pre_alignment_report=pre_alignments,
          update_report=updates,
          alignment_report=alignments,
      )
      self.assertEqual(record["verdict"], "PASS_WITH_ALIGNMENT_WARNINGS")
      self.assertEqual(record["observed_updates"], 200)
      self.assertEqual(record["observed_alignments"], 3200)

  def test_full_gsm8k_accepts_finite_alignment_warnings(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      run_log = root / "run.log"
      updates = root / "updates.jsonl"
      pre_alignments = root / "pre_alignment.jsonl"
      alignments = root / "alignment.jsonl"
      run_log.write_text(
          "[CANON_P33_WANDB] ONLINE_RUN_PASS\n"
          "[CANON_P31_METRICS] monotonic_direct last_step=199 events=200 regressions=0\n"
          + "[CANON_P33_DP16] update_step_committed\n" * 200,
          encoding="utf-8",
      )
      self._write_jsonl(updates, (_update(index) for index in range(200)))
      pre_rows = [
          _pre_alignment(index, warning_policy=True) for index in range(200)
      ]
      pre_rows[0] = _pre_alignment(0, warning_policy=True, warned=True)
      self._write_jsonl(pre_alignments, pre_rows)
      alignment_rows = [
          _alignment(index, optimizer_skipped=False, warning_policy=True)
          for index in range(3200)
      ]
      alignment_rows[0] = _alignment(
          0, optimizer_skipped=False, warning_policy=True, warned=True
      )
      self._write_jsonl(alignments, alignment_rows)
      record = classifier.classify(
          workload="gsm8k",
          stage="full",
          run_log=run_log,
          pre_alignment_report=pre_alignments,
          update_report=updates,
          alignment_report=alignments,
      )
      self.assertEqual(record["verdict"], "PASS_WITH_ALIGNMENT_WARNINGS")
      self.assertEqual(record["pre_alignment_warning_records"], 1)
      self.assertEqual(record["alignment_warning_records"], 1)
      self.assertEqual(record["claim_level"], "convergence-only")

  def test_warning_policy_rejects_nonfinite_and_frozenlake_scope(self):
    nonfinite = _pre_alignment(0, warning_policy=True, warned=True)
    nonfinite["boundaries"]["S_decode_vs_S_prefill"]["finite"] = False
    reasons = []
    classifier._validate_pre_alignment_records(
        [nonfinite],
        expected_count=1,
        workload="gsm8k",
        stage="full",
        reasons=reasons,
    )
    self.assertIn(
        "pre_alignment[0].S_decode_vs_S_prefill.finite", reasons
    )

    frozenlake = _pre_alignment(0, warning_policy=True, warned=True)
    reasons = []
    classifier._validate_pre_alignment_records(
        [frozenlake],
        expected_count=1,
        workload="frozenlake",
        stage="backward-no-commit",
        reasons=reasons,
    )
    self.assertIn("pre_alignment[0].policy", reasons)
    self.assertIn("pre_alignment[0].verdict", reasons)

  def test_full_frozenlake_positive(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      run_log = root / "run.log"
      updates = root / "updates.jsonl"
      pre_alignments = root / "pre_alignment.jsonl"
      alignments = root / "alignment.jsonl"
      run_log.write_text(
          "[CANON_P33_WANDB] ONLINE_RUN_PASS\n"
          "[CANON_P33_EVAL] DISABLED workload=frozenlake\n"
          "[CANON_P31_METRICS] monotonic_direct last_step=449 events=450 regressions=0\n"
          + "[CANON_P33_DP16] update_step_committed\n" * 450,
          encoding="utf-8",
      )
      self._write_jsonl(updates, (_update(index) for index in range(450)))
      self._write_jsonl(
          pre_alignments,
          (
              _pre_alignment(
                  index,
                  warning_policy=True,
                  policy_workload="frozenlake",
              )
              for index in range(450)
          ),
      )
      self._write_jsonl(
          alignments,
          (
              _alignment(
                  index,
                  optimizer_skipped=False,
                  warning_policy=True,
                  policy_workload="frozenlake",
              )
              for index in range(7200)
          ),
      )
      record = classifier.classify(
          workload="frozenlake",
          stage="full",
          run_log=run_log,
          pre_alignment_report=pre_alignments,
          update_report=updates,
          alignment_report=alignments,
      )
      self.assertEqual(record["verdict"], "PASS_WITH_ALIGNMENT_WARNINGS")
      self.assertEqual(record["observed_updates"], 450)
      self.assertEqual(record["observed_alignments"], 7200)

  def test_full_frozenlake_evaluation_inventory_positive(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      run_log = root / "run.log"
      updates = root / "updates.jsonl"
      pre_alignments = root / "pre_alignment.jsonl"
      alignments = root / "alignment.jsonl"
      eval_rows = "".join(
          "[CANON_FROZENLAKE_P31] eval_reward_inventory "
          f"step={step} prompts=100 generations=8 rewards=800 "
          "expected=800 verdict=PASS\n"
          for step in range(0, 450, 10)
      )
      eval_summaries = "".join(
          "[CANON_FROZENLAKE_P42_JSON] "
          + json.dumps({
              "n": 800,
              "policy_step": step,
              "reward": 0.5,
              "solve": 0.25,
              "wall_seconds": 2.0,
          }, sort_keys=True, separators=(",", ":"))
          + "\n"
          for step in range(0, 450, 10)
      )
      run_log.write_text(
          "[CANON_P33_WANDB] ONLINE_RUN_PASS\n"
          "[CANON_P33_EVAL] ENABLED workload=frozenlake cadence=10 "
          "held_out_rows=100 generations=8\n"
          + eval_rows
          + eval_summaries
          + "[CANON_P31_METRICS] monotonic_direct last_step=449 "
          "events=450 regressions=0\n"
          + "[CANON_P33_DP16] update_step_committed\n" * 450,
          encoding="utf-8",
      )
      self._write_jsonl(updates, (_update(index) for index in range(450)))
      self._write_jsonl(
          pre_alignments,
          (
              _pre_alignment(
                  index,
                  warning_policy=True,
                  policy_workload="frozenlake",
              )
              for index in range(450)
          ),
      )
      self._write_jsonl(
          alignments,
          (
              _alignment(
                  index,
                  optimizer_skipped=False,
                  warning_policy=True,
                  policy_workload="frozenlake",
              )
              for index in range(7200)
          ),
      )
      record = classifier.classify(
          workload="frozenlake",
          stage="full",
          run_log=run_log,
          pre_alignment_report=pre_alignments,
          update_report=updates,
          alignment_report=alignments,
      )
      self.assertEqual(record["verdict"], "PASS_WITH_ALIGNMENT_WARNINGS")
      self.assertTrue(record["evaluation_enabled"])

  def test_backward_no_commit_positive(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      run_log = root / "run.log"
      updates = root / "updates.jsonl"
      pre_alignments = root / "pre_alignment.jsonl"
      alignments = root / "alignment.jsonl"
      run_log.write_text(
          "[CANON_P33_WANDB] ONLINE_RUN_PASS\n"
          "[CANON_P33_EVAL] DISABLED workload=frozenlake\n"
          "[CANON_P31_METRICS] monotonic_direct last_step=0 events=1 regressions=0\n"
          "[CANON_P33_DP16] backward_no_commit verdict=PASS\n",
          encoding="utf-8",
      )
      record = {
          "verdict": "PASS",
          "dp_axis": "data",
          "dp_size": 16,
          "tp_size": 4,
          "global_m": 4096,
          "mode": "backward-no-commit",
          "microsteps": 16,
          "commits": 0,
          "train_steps_before": 0,
          "train_steps_after": 0,
          "gradient_activity": [True] * 16,
          "alignment_hashes": [{"T_current": "a"}] * 16,
          "micro_gradient_norms": [1.0] * 16,
          "optimizer_memory_kinds_before": ["pinned_host"],
          "optimizer_placement": "pinned-host-offload",
          "model_changed_paths": [],
          "optimizer_changed_paths": [],
          "accumulator_changed_paths": [],
          "reference_changed_paths": [],
      }
      updates.write_text(json.dumps(record), encoding="utf-8")
      self._write_jsonl(pre_alignments, [_pre_alignment(0)])
      self._write_jsonl(
          alignments,
          (_alignment(index, optimizer_skipped=True) for index in range(16)),
      )
      result = classifier.classify(
          workload="frozenlake",
          stage="backward-no-commit",
          run_log=run_log,
          pre_alignment_report=pre_alignments,
          update_report=updates,
          alignment_report=alignments,
      )
      self.assertEqual(result["verdict"], "PASS")

  def test_short_alignment_positive_is_diagnostic_only(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      run_log = root / "run.log"
      updates = root / "updates.jsonl"
      pre_alignments = root / "pre_alignment.jsonl"
      alignments = root / "alignment.jsonl"
      run_log.write_text(
          "[CANON_P33_WANDB] ONLINE_RUN_PASS\n"
          "[CANON_P33_EVAL] DISABLED workload=frozenlake\n"
          "[CANON_P31_METRICS] monotonic_direct last_step=0 events=1 regressions=0\n"
          "[CANON_P33_DP16] backward_no_commit verdict=PASS\n",
          encoding="utf-8",
      )
      update = {
          "verdict": "PASS",
          "dp_axis": "data",
          "dp_size": 16,
          "tp_size": 4,
          "global_m": 4096,
          "mode": "alignment-short",
          "microsteps": 16,
          "commits": 0,
          "train_steps_before": 0,
          "train_steps_after": 0,
          "gradient_activity": [True] * 16,
          "alignment_hashes": [{"T_current": "a"}] * 16,
          "micro_gradient_norms": [1.0] * 16,
          "optimizer_memory_kinds_before": ["pinned_host"],
          "optimizer_placement": "pinned-host-offload",
          "model_changed_paths": [],
          "optimizer_changed_paths": [],
          "accumulator_changed_paths": [],
          "reference_changed_paths": [],
      }
      updates.write_text(json.dumps(update), encoding="utf-8")
      self._write_jsonl(pre_alignments, [_pre_alignment(0)])
      self._write_jsonl(
          alignments,
          (_alignment(index, optimizer_skipped=True) for index in range(16)),
      )
      result = classifier.classify(
          workload="frozenlake",
          stage="alignment-short",
          run_log=run_log,
          pre_alignment_report=pre_alignments,
          update_report=updates,
          alignment_report=alignments,
      )
      self.assertEqual(result["verdict"], "PASS")
      self.assertTrue(result["diagnostic_only"])

  def test_negative_control_rejects_pre_backward_boundary(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      run_log = root / "run.log"
      updates = root / "updates.jsonl"
      pre_alignments = root / "pre_alignment.jsonl"
      alignments = root / "alignment.jsonl"
      run_log.write_text(
          "[CANON_P33_WANDB] ONLINE_RUN_PASS\n"
          "[CANON_P31_METRICS] monotonic_direct last_step=0 events=1 regressions=0\n"
          "[CANON_P33_DP16] update_step_committed\n",
          encoding="utf-8",
      )
      self._write_jsonl(updates, [_update(0)])
      pre = _pre_alignment(0)
      pre["boundaries"]["S_decode_vs_S_prefill"]["differing_bytes"] = 1
      self._write_jsonl(pre_alignments, [pre])
      self._write_jsonl(
          alignments,
          (_alignment(index, optimizer_skipped=False) for index in range(16)),
      )
      result = classifier.classify(
          workload="gsm8k",
          stage="one-update",
          run_log=run_log,
          pre_alignment_report=pre_alignments,
          update_report=updates,
          alignment_report=alignments,
      )
      self.assertEqual(result["verdict"], "FAIL")
      self.assertIn(
          "pre_alignment[0].S_decode_vs_S_prefill.strict_drift",
          result["reasons"],
      )

  def test_one_update_accepts_device_resident_optimizer(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      run_log = root / "run.log"
      updates = root / "updates.jsonl"
      pre_alignments = root / "pre_alignment.jsonl"
      alignments = root / "alignment.jsonl"
      run_log.write_text(
          "[CANON_P33_WANDB] ONLINE_RUN_PASS\n"
          "[CANON_P31_METRICS] monotonic_direct last_step=0 events=1 regressions=0\n"
          "[CANON_P33_DP16] update_step_committed\n",
          encoding="utf-8",
      )
      self._write_jsonl(
          updates, [_update(0, placement="device-resident")]
      )
      self._write_jsonl(pre_alignments, [_pre_alignment(0)])
      self._write_jsonl(
          alignments,
          (_alignment(index, optimizer_skipped=False) for index in range(16)),
      )
      result = classifier.classify(
          workload="gsm8k",
          stage="one-update",
          run_log=run_log,
          pre_alignment_report=pre_alignments,
          update_report=updates,
          alignment_report=alignments,
      )
      self.assertEqual(result["verdict"], "PASS")

  def test_p45_dp8_tp8_resident_topology_and_cadence(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      run_log = root / "run.log"
      updates = root / "updates.jsonl"
      pre_alignments = root / "pre_alignment.jsonl"
      alignments = root / "alignment.jsonl"
      run_log.write_text(
          "[CANON_P33_WANDB] ONLINE_RUN_PASS\n"
          "[CANON_P33_EVAL] DISABLED workload=frozenlake\n"
          "[CANON_P31_METRICS] monotonic_direct last_step=0 events=1 regressions=0\n"
          "[CANON_P33_DP8] update_step_committed\n",
          encoding="utf-8",
      )
      self._write_jsonl(
          updates,
          [_update(0, placement="device-resident", dp_size=8, tp_size=8)],
      )
      self._write_jsonl(pre_alignments, [_pre_alignment(0)])
      self._write_jsonl(
          alignments,
          (_alignment(index, optimizer_skipped=False) for index in range(32)),
      )
      result = classifier.classify(
          workload="frozenlake-dp8-tp8",
          stage="one-update",
          run_log=run_log,
          pre_alignment_report=pre_alignments,
          update_report=updates,
          alignment_report=alignments,
          dp_size=8,
          tp_size=8,
      )
      self.assertEqual(result["verdict"], "PASS")
      self.assertEqual(result["local_gradient_groups"], 32)
      self.assertEqual(result["topology"], {"dp": 8, "tp": 8})

  def test_p45_rejects_offloaded_optimizer(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      run_log = root / "run.log"
      updates = root / "updates.jsonl"
      pre_alignments = root / "pre_alignment.jsonl"
      alignments = root / "alignment.jsonl"
      run_log.write_text(
          "[CANON_P33_WANDB] ONLINE_RUN_PASS\n"
          "[CANON_P33_EVAL] DISABLED workload=frozenlake\n"
          "[CANON_P31_METRICS] monotonic_direct last_step=0 events=1 regressions=0\n"
          "[CANON_P33_DP8] update_step_committed\n",
          encoding="utf-8",
      )
      self._write_jsonl(
          updates,
          [_update(0, placement="pinned-host-offload", dp_size=8, tp_size=8)],
      )
      self._write_jsonl(pre_alignments, [_pre_alignment(0)])
      self._write_jsonl(
          alignments,
          (_alignment(index, optimizer_skipped=False) for index in range(32)),
      )
      result = classifier.classify(
          workload="frozenlake-dp8-tp8",
          stage="one-update",
          run_log=run_log,
          pre_alignment_report=pre_alignments,
          update_report=updates,
          alignment_report=alignments,
          dp_size=8,
          tp_size=8,
      )
      self.assertEqual(result["verdict"], "FAIL")
      self.assertIn("update[0].p45_optimizer_placement", result["reasons"])

  def test_negative_control_rejects_unattested_optimizer_placement(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      run_log = root / "run.log"
      updates = root / "updates.jsonl"
      pre_alignments = root / "pre_alignment.jsonl"
      alignments = root / "alignment.jsonl"
      run_log.write_text(
          "[CANON_P33_WANDB] ONLINE_RUN_PASS\n"
          "[CANON_P31_METRICS] monotonic_direct last_step=0 events=1 regressions=0\n"
          "[CANON_P33_DP16] update_step_committed\n",
          encoding="utf-8",
      )
      update = _update(0)
      del update["optimizer_placement"]
      self._write_jsonl(updates, [update])
      self._write_jsonl(pre_alignments, [_pre_alignment(0)])
      self._write_jsonl(
          alignments,
          (_alignment(index, optimizer_skipped=False) for index in range(16)),
      )
      result = classifier.classify(
          workload="gsm8k",
          stage="one-update",
          run_log=run_log,
          pre_alignment_report=pre_alignments,
          update_report=updates,
          alignment_report=alignments,
      )
      self.assertEqual(result["verdict"], "FAIL")
      self.assertIn("update[0].optimizer_placement", result["reasons"])

  def test_negative_control_rejects_parameter_change_at_zero_lr(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      run_log = root / "run.log"
      updates = root / "updates.jsonl"
      pre_alignments = root / "pre_alignment.jsonl"
      alignments = root / "alignment.jsonl"
      run_log.write_text(
          "[CANON_P33_WANDB] ONLINE_RUN_PASS\n"
          "[CANON_P31_METRICS] monotonic_direct last_step=0 events=1 regressions=0\n"
          "[CANON_P33_DP16] update_step_committed\n",
          encoding="utf-8",
      )
      update = _update(0)
      update["commit_evidence"]["parameter_changed_elements"] = 1
      update["commit_evidence"]["parameter_delta_max_abs"] = 1.0e-8
      update["parameter_mutation"] = "observed_nonzero"
      self._write_jsonl(updates, [update])
      self._write_jsonl(pre_alignments, [_pre_alignment(0)])
      self._write_jsonl(
          alignments,
          (_alignment(index, optimizer_skipped=False) for index in range(16)),
      )
      result = classifier.classify(
          workload="gsm8k",
          stage="one-update",
          run_log=run_log,
          pre_alignment_report=pre_alignments,
          update_report=updates,
          alignment_report=alignments,
      )
      self.assertEqual(result["verdict"], "FAIL")
      self.assertIn(
          "update[0].zero_lr_model_unchanged", result["reasons"]
      )

  def test_warning_policy_negative_control_rejects_nonfinite_boundary(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      run_log = root / "run.log"
      updates = root / "updates.jsonl"
      pre_alignments = root / "pre_alignment.jsonl"
      alignments = root / "alignment.jsonl"
      run_log.write_text(
          "[CANON_P33_WANDB] ONLINE_RUN_PASS\n"
          "[CANON_P31_METRICS] monotonic_direct last_step=199 events=200 regressions=0\n"
          + "[CANON_P33_DP16] update_step_committed\n" * 200,
          encoding="utf-8",
      )
      self._write_jsonl(updates, (_update(index) for index in range(200)))
      self._write_jsonl(
          pre_alignments,
          (_pre_alignment(index, warning_policy=True) for index in range(200)),
      )
      rows = [
          _alignment(index, optimizer_skipped=False, warning_policy=True)
          for index in range(3200)
      ]
      rows[17]["boundaries"]["T_old_vs_T_current"]["finite"] = False
      self._write_jsonl(alignments, rows)
      record = classifier.classify(
          workload="gsm8k",
          stage="full",
          run_log=run_log,
          pre_alignment_report=pre_alignments,
          update_report=updates,
          alignment_report=alignments,
      )
      self.assertEqual(record["verdict"], "FAIL")
      self.assertIn(
          "alignment[17].T_old_vs_T_current.finite",
          record["reasons"],
      )

  def test_negative_control_rejects_wrong_dp_axis(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      run_log = root / "run.log"
      updates = root / "updates.jsonl"
      pre_alignments = root / "pre_alignment.jsonl"
      alignments = root / "alignment.jsonl"
      run_log.write_text(
          "[CANON_P33_WANDB] ONLINE_RUN_PASS\n"
          "[CANON_P31_METRICS] monotonic_direct last_step=0 events=1 regressions=0\n"
          "[CANON_P33_DP16] update_step_committed\n",
          encoding="utf-8",
      )
      update = _update(0)
      update["dp_axis"] = "dp"
      self._write_jsonl(updates, [update])
      self._write_jsonl(pre_alignments, [_pre_alignment(0)])
      self._write_jsonl(
          alignments,
          (_alignment(index, optimizer_skipped=False) for index in range(16)),
      )
      record = classifier.classify(
          workload="gsm8k",
          stage="one-update",
          run_log=run_log,
          pre_alignment_report=pre_alignments,
          update_report=updates,
          alignment_report=alignments,
      )
      self.assertEqual(record["verdict"], "FAIL")
      self.assertIn("update[0].dp_axis", record["reasons"])

  def test_bounded_stage_budgets_remain_supported(self):
    self.assertEqual(
        classifier._expected_updates("frozenlake", "alignment-short"), 1
    )
    self.assertEqual(classifier._expected_updates("gsm8k", "one-update"), 1)
    self.assertEqual(
        classifier._expected_updates("frozenlake", "three-update"), 3
    )


if __name__ == "__main__":
  unittest.main()
