#!/usr/bin/env python3
"""Tests for the matched TiTO record-full one-host neutrality judge."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / (
    "canon-zero-tim/tasks/multiturn-tito-cross-workload/scripts/"
    "judge_tito_onehost_neutrality.py"
)
SPEC = importlib.util.spec_from_file_location("p57_tito_onehost_judge", SCRIPT)
if SPEC is None or SPEC.loader is None:
  raise RuntimeError("cannot import P57 TiTO one-host judge")
judge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(judge)


def _write_json(path: Path, value: dict) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, values: list[dict]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(
      "".join(json.dumps(value) + "\n" for value in values),
      encoding="utf-8",
  )


def _arm(root: Path, arm: str) -> None:
  _write_json(
      root / "classification.json",
      {
          "verdict": "PASS",
          "neutrality_arm": arm,
          "semantic_event_counts": {
              "advantage_computation": 1,
              "data_loading": 1,
              "peft_train": 2,
              "rollout": 16,
              "weight_sync": 2,
          },
      },
  )
  updates = [
      {
          "commit_gradient_norm": value,
          "alignment_hashes": [
              {key: f"{step}-{group}-{key}" for key in judge._ALIGNMENT_HASH_KEYS}
              for group in range(4)
          ],
          "state_fingerprints_before": {"model": f"before-{step}"},
      }
      for step, value in enumerate(judge._R7_GRADIENT_NORM_ANCHOR)
  ]
  _write_jsonl(root / "updates.jsonl", updates)
  alignment = [
      {
          "context": {
              "canonical_c": {
                  "implementation_id": judge._R7_IMPLEMENTATION_ID
              }
          }
      }
      for _ in range(12)
  ]
  _write_jsonl(root / "alignment.jsonl", alignment)


class TitoOnehostNeutralityTest(unittest.TestCase):

  def test_exact_anchor_and_cross_arm_numerics_are_required(self):
    with tempfile.TemporaryDirectory() as tmp:
      off = Path(tmp) / "off"
      on = Path(tmp) / "on"
      _arm(off, "tito-off")
      _arm(on, "tito-on")
      result = judge.judge(off_root=off, on_root=on)
      self.assertEqual(result["verdict"], "PASS")
      self.assertTrue(result["claims"]["record_full_observer_neutral"])
      self.assertFalse(result["claims"]["target_dp8_tp8_certified"])

      updates = [json.loads(line) for line in (on / "updates.jsonl").read_text().splitlines()]
      updates[1]["commit_gradient_norm"] += 1.0
      _write_jsonl(on / "updates.jsonl", updates)
      red = judge.judge(off_root=off, on_root=on)
      self.assertEqual(red["verdict"], "FAIL")
      self.assertIn("on_r7_gradient_anchor", red["reasons"])
      self.assertIn("update_numerics:1", red["reasons"])

  def test_input_drift_is_inconclusive_and_missing_hashes_fail(self):
    with tempfile.TemporaryDirectory() as tmp:
      off = Path(tmp) / "off"
      on = Path(tmp) / "on"
      _arm(off, "tito-off")
      _arm(on, "tito-on")
      updates = [
          json.loads(line) for line in (on / "updates.jsonl").read_text().splitlines()
      ]
      updates[0]["alignment_hashes"][0]["tokens"] = "different"
      _write_jsonl(on / "updates.jsonl", updates)
      inconclusive = judge.judge(off_root=off, on_root=on)
      self.assertEqual(
          inconclusive["verdict"], "INCONCLUSIVE_INPUT_MISMATCH"
      )
      self.assertEqual(inconclusive["input_verdict"], "MISMATCH")
      self.assertIn("input_hashes:0", inconclusive["input_mismatch_reasons"])

      del updates[0]["alignment_hashes"]
      _write_jsonl(on / "updates.jsonl", updates)
      malformed = judge.judge(off_root=off, on_root=on)
      self.assertEqual(malformed["verdict"], "FAIL")
      self.assertIn("on_seven_hash_contract:0", malformed["reasons"])


if __name__ == "__main__":
  unittest.main()
