"""Tests for the isolated P57 evaluation classifier."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    ROOT
    / "canon-zero-tim/tasks/p57-frozenlake-tim-causal-study/scripts/classify_checkpoint_eval.py"
)
SPEC = importlib.util.spec_from_file_location("p57_eval_classifier", MODULE_PATH)
assert SPEC and SPEC.loader
classifier = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = classifier
SPEC.loader.exec_module(classifier)


def _record():
  rewards = [float(group % 2) for group in range(100) for _ in range(8)]
  return {
      "schema": "p57-frozenlake-isolated-evaluation-v1",
      "arm": "zero",
      "fixed_lm_head": "1",
      "source_commit": "a" * 40,
      "expected_updates": 200,
      "checkpoint_step": 20,
      "checkpoint_tag": "p57-campaign-zero",
      "temperature": 0.0,
      "seed": 42,
      "held_out_rows": 100,
      "workload_candidate": "",
      "data_split": "",
      "dataset_eval_sha256": "",
      "reward": 0.5,
      "solve": 0.5,
      "n": 800,
      "wall_seconds": 12.5,
      "policy_step": 20,
      "prompts": 100,
      "generations": 8,
      "batches": 100,
      "rewards": rewards,
  }


class P57EvalClassifierTest(unittest.TestCase):

  def _classify(self, record, *, suffix=""):
    temporary = tempfile.TemporaryDirectory()
    self.addCleanup(temporary.cleanup)
    root = Path(temporary.name)
    evaluation = root / "evaluation.json"
    run_log = root / "run.log"
    evaluation.write_text(json.dumps(record), encoding="utf-8")
    run_log.write_text(
        "[CANON_P57_EVAL_JSON] "
        + json.dumps(record, sort_keys=True, separators=(",", ":"))
        + "\n[CANON_P57_EVAL] COMPLETE arm=zero step=20 prompts=100 "
        "generations=8 rewards=800 solve=0.500000 backward=0 "
        "optimizer_commits=0 checkpoint_writes=0\n"
        + suffix,
        encoding="utf-8",
    )
    return classifier.classify(
        evaluation_path=evaluation,
        run_log_path=run_log,
        arm="zero",
        source_commit="a" * 40,
        checkpoint_tag="p57-campaign-zero",
        checkpoint_step=20,
        expected_updates=200,
    )

  def test_accepts_complete_no_update_evaluation(self):
    result = self._classify(_record())
    self.assertEqual(result["verdict"], "PASS")
    self.assertEqual(result["reasons"], [])

  def test_rejects_reward_drift_or_training_entry(self):
    record = _record()
    record["solve"] = 0.25
    result = self._classify(
        record, suffix="Global step 21 completed in 1.0 seconds.\n"
    )
    self.assertEqual(result["verdict"], "FAIL")
    self.assertTrue(any("solve rate" in reason for reason in result["reasons"]))
    self.assertTrue(any("train loop" in reason for reason in result["reasons"]))

  def test_rejects_wrong_treatment_or_incomplete_coverage(self):
    record = _record()
    record["fixed_lm_head"] = "0"
    record["rewards"] = record["rewards"][:-1]
    result = self._classify(record)
    self.assertEqual(result["verdict"], "FAIL")
    self.assertTrue(any("contract fields" in reason for reason in result["reasons"]))
    self.assertTrue(any("exactly 800" in reason for reason in result["reasons"]))

  def test_rejects_divergent_deterministic_replicas(self):
    record = _record()
    record["rewards"][1] = 1.0
    result = self._classify(record)
    self.assertEqual(result["verdict"], "FAIL")
    self.assertTrue(any("replicas diverged" in reason for reason in result["reasons"]))

  def test_materialized_eval_requires_matching_dataset_attestation(self):
    record = _record()
    record["workload_candidate"] = "m10"
    record["data_split"] = "main"
    record["dataset_eval_sha256"] = "a" * 64
    temporary = tempfile.TemporaryDirectory()
    self.addCleanup(temporary.cleanup)
    root = Path(temporary.name)
    evaluation = root / "evaluation.json"
    run_log = root / "run.log"
    evaluation.write_text(json.dumps(record), encoding="utf-8")
    run_log.write_text(
        "[CANON_P57_EVAL_JSON] "
        + json.dumps(record, sort_keys=True, separators=(",", ":"))
        + "\n[CANON_P57_EVAL] COMPLETE arm=zero step=20 prompts=100 "
        "generations=8 rewards=800 solve=0.500000 backward=0 "
        "optimizer_commits=0 checkpoint_writes=0\n",
        encoding="utf-8",
    )
    result = classifier.classify(
        evaluation_path=evaluation,
        run_log_path=run_log,
        arm="zero",
        source_commit="a" * 40,
        checkpoint_tag="p57-campaign-zero",
        checkpoint_step=20,
        expected_updates=200,
        workload_candidate="m10",
        data_split="main",
    )
    self.assertEqual(result["verdict"], "PASS")


if __name__ == "__main__":
  unittest.main()
