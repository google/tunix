"""Contract tests for the P57 300-update in-process eval curve."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

from examples.frozenlake import p57_workloads


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    ROOT
    / "canon-zero-tim/tasks/p57-frozenlake-tim-causal-study/scripts/classify_inprocess_eval.py"
)
SPEC = importlib.util.spec_from_file_location("p57_inprocess_eval", SCRIPT)
assert SPEC and SPEC.loader
classifier = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = classifier
SPEC.loader.exec_module(classifier)


def _log(*, omit_step: int | None = None, bad_n: bool = False) -> str:
  lines = [
      "[CANON_" "P33_EVAL] ENABLED workload=frozenlake cadence=50 "
      "held_out_rows=100 generations=8",
      "[P57.DATASET] MATERIALIZED_PASS candidate=p45 split=legacy "
      "train_rows=10000 eval_rows=100 "
      "train_sha256="
      + p57_workloads.PRIMARY_DATASET_SHA256[
          ("p45", "legacy", "train", 10_000)
      ]
      + " eval_sha256="
      + p57_workloads.PRIMARY_DATASET_SHA256[
          ("p45", "legacy", "eval", 100)
      ],
      "[P57.SEED] CONTRACT_PASS data_shuffle_seed=42 "
      "vllm_global_seed=0 per_request_seed=unsupported",
  ]
  last = None
  for step in range(0, 301, 50):
    if step == omit_step:
      continue
    last = {
        "n": 799 if bad_n and step == 100 else 800,
        "policy_step": step,
        "reward": step / 600.0,
        "solve": step / 600.0,
        "wall_seconds": 1.25,
    }
    lines.append(
        "[CANON_" "FROZENLAKE_P42_JSON] "
        + json.dumps(last, sort_keys=True, separators=(",", ":"))
    )
  assert last is not None
  lines.append(
      "[P57.EVAL] FINAL policy_step=300 prompts=100 generations=8 n=800 "
      f"reward={last['reward']:.6f} solve={last['solve']:.6f} "
      "backward=0 optimizer_commits=0 evaluation_checkpoint_writes=0"
  )
  return "\n".join(lines) + "\n"


class InprocessEvalClassifierTest(unittest.TestCase):

  def _classify(self, text: str):
    with tempfile.TemporaryDirectory() as tmp:
      path = Path(tmp) / "run.log"
      path.write_text(text, encoding="utf-8")
      return classifier.classify(
          path,
          expected_updates=300,
          interval=50,
          held_out_rows=100,
          generations=8,
          workload_candidate="",
          data_split="",
      )

  def test_complete_curve_passes(self):
    result = self._classify(_log())
    self.assertEqual(result["verdict"], "PASS")
    self.assertEqual(result["steps"], [0, 50, 100, 150, 200, 250, 300])
    print("P57_INPROCESS_EVAL_CLASSIFIER_PASS steps=7", flush=True)

  def test_missing_point_and_bad_coverage_fail(self):
    with self.assertRaisesRegex(ValueError, "schedule incomplete"):
      self._classify(_log(omit_step=150))
    with self.assertRaisesRegex(ValueError, "coverage drifted"):
      self._classify(_log(bad_n=True))

  def test_dataset_and_seed_drift_fail(self):
    with self.assertRaisesRegex(ValueError, "dataset identity drifted"):
      self._classify(_log().replace("ddc96fd9", "0dc96fd9", 1))
    with self.assertRaisesRegex(ValueError, "seed receipt drifted"):
      self._classify(_log().replace("data_shuffle_seed=42", "data_shuffle_seed=43"))


if __name__ == "__main__":
  unittest.main()
