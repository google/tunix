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


def _alignment(step: int, *, optimizer_skipped: bool) -> dict:
  return {
      "verdict": "PASS",
      "reds": [],
      "execution_mode": "train",
      "step": step,
      "N_action": 4,
      "boundaries": {
          name: {"differing_bytes": 0}
          for name in classifier._BOUNDARIES
      },
      "exact": {name: True for name in classifier._EXACT_KEYS},
      "clip_hits": 0,
      "tis_hits": 0,
      "optimizer_skipped": optimizer_skipped,
      "gradient": {"finite": True, "nonzero": True, "norm": 1.0},
  }


def _pre_alignment(step: int) -> dict:
  return {
      "verdict": "PASS",
      "reds": [],
      "step": step,
      "N_action": 4,
      "boundaries": {
          name: {"differing_bytes": 0}
          for name in classifier._PRE_BOUNDARIES
      },
  }


def _update(index: int) -> dict:
  return {
      "verdict": "PASS",
      "dp_axis": "data",
      "microsteps": 16,
      "commits": 1,
      "train_steps_before": index,
      "train_steps_after": index + 1,
      "gradient_activity": [True] * 16,
      "alignment_hashes": [{"T_current": "a"}] * 16,
      "micro_gradient_norms": [1.0] * 16,
      "optimizer_memory_kinds_before": ["pinned_host"],
      "optimizer_memory_kinds_after": ["pinned_host"],
      "accumulator_changed_paths": [],
      "reference_changed_paths": [],
      "commit_gradient_norm": 1.0,
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
          pre_alignments, (_pre_alignment(index) for index in range(200))
      )
      self._write_jsonl(
          alignments,
          (_alignment(index, optimizer_skipped=False) for index in range(3200)),
      )
      record = classifier.classify(
          workload="gsm8k",
          stage="full",
          run_log=run_log,
          pre_alignment_report=pre_alignments,
          update_report=updates,
          alignment_report=alignments,
      )
      self.assertEqual(record["verdict"], "PASS")
      self.assertEqual(record["observed_updates"], 200)
      self.assertEqual(record["observed_alignments"], 3200)

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
          pre_alignments, (_pre_alignment(index) for index in range(450))
      )
      self._write_jsonl(
          alignments,
          (_alignment(index, optimizer_skipped=False) for index in range(7200)),
      )
      record = classifier.classify(
          workload="frozenlake",
          stage="full",
          run_log=run_log,
          pre_alignment_report=pre_alignments,
          update_report=updates,
          alignment_report=alignments,
      )
      self.assertEqual(record["verdict"], "PASS")
      self.assertEqual(record["observed_updates"], 450)
      self.assertEqual(record["observed_alignments"], 7200)

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
          "mode": "backward-no-commit",
          "microsteps": 16,
          "commits": 0,
          "train_steps_before": 0,
          "train_steps_after": 0,
          "gradient_activity": [True] * 16,
          "alignment_hashes": [{"T_current": "a"}] * 16,
          "micro_gradient_norms": [1.0] * 16,
          "optimizer_memory_kinds_before": ["pinned_host"],
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
          "mode": "alignment-short",
          "microsteps": 16,
          "commits": 0,
          "train_steps_before": 0,
          "train_steps_after": 0,
          "gradient_activity": [True] * 16,
          "alignment_hashes": [{"T_current": "a"}] * 16,
          "micro_gradient_norms": [1.0] * 16,
          "optimizer_memory_kinds_before": ["pinned_host"],
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
          "pre_alignment[0].S_decode_vs_S_prefill.differing_bytes",
          result["reasons"],
      )

  def test_negative_control_rejects_one_changed_boundary(self):
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
          pre_alignments, (_pre_alignment(index) for index in range(200))
      )
      rows = [_alignment(index, optimizer_skipped=False) for index in range(3200)]
      rows[17]["boundaries"]["T_old_vs_T_current"]["differing_bytes"] = 1
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
          "alignment[17].T_old_vs_T_current.differing_bytes",
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
