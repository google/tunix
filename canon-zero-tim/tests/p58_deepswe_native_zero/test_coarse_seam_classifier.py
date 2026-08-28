#!/usr/bin/env python3
"""Unit contracts for the P58.19 per-round and aggregate classifiers."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / (
    "canon-zero-tim/tasks/p58-deepswe-native-zero-comparison/scripts"
)


def _load(name: str):
  spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
  if spec is None or spec.loader is None:
    raise RuntimeError(name)
  module = importlib.util.module_from_spec(spec)
  sys.modules[name] = module
  spec.loader.exec_module(module)
  return module


ROUND = _load("classify_p58_coarse_seam_round")
AGGREGATE = _load("classify_p58_coarse_seam_three_round")


def _underlying(round_index: int = 0):
  return {
      "status": "PASS",
      "diagnostic_round": round_index,
      "alignment": {
          "a_b_differing_bytes": 17,
          "b_c_differing_bytes": 0,
          "n_action": 128,
      },
      "coverage": {
          "total_red_points": 3,
          "standard_joinable_red_points": 1,
          "unobserved_red_points": 2,
      },
      "seam_inventory": {"records": 2},
      "tail_inventory": {"records": 2},
      "first_difference_signatures": [
          {"layer": 9, "checkpoint": "layer_output"},
      ],
      "mixed_first_difference_signatures": False,
      "selected_layer": 9,
      "last_exact_boundary": {"layer": 9, "checkpoint": "layer_input"},
      "first_red_boundary": {"layer": 9, "checkpoint": "layer_output"},
      "source_interval": {"last_exact": {}, "first_red": {}},
      "anchors": [{"source_row": 0}],
      "next_action": "fine layer 9",
  }


def _round_report(round_index: int, layer: int = 9):
  return {
      "schema": "canon.p58.coarse-seam-round-classification.v1",
      "verdict": "PASS",
      "outcome": "COARSE_FIRST_RED_INTERVAL",
      "diagnostic_round": round_index,
      "alignment": {
          "a_b_differing_bytes": 17 + round_index,
          "b_c_differing_bytes": 0,
      },
      "first_difference_signatures": [
          {"layer": layer, "checkpoint": "layer_output"},
      ],
  }


class P58CoarseSeamClassifierTest(unittest.TestCase):

  def test_round_wrapper_preserves_bounded_claim(self):
    with mock.patch.object(ROUND._M15, "classify", return_value=_underlying()):
      report = ROUND.classify(
          directory=Path("observer"),
          alignment_report=Path("alignment.jsonl"),
          capsules=[Path("capsule.npz")],
      )
    self.assertEqual(report["verdict"], "PASS")
    self.assertEqual(report["selected_layer"], 9)
    self.assertEqual(report["backward"], 0)
    self.assertIn("standard-path", report["claim_ceiling"])

  def test_round_wrapper_rejects_control_drift(self):
    underlying = _underlying()
    underlying["alignment"]["b_c_differing_bytes"] = 1
    with mock.patch.object(ROUND._M15, "classify", return_value=underlying):
      with self.assertRaisesRegex(ValueError, "B-C"):
        ROUND.classify(
            directory=Path("observer"),
            alignment_report=Path("alignment.jsonl"),
            capsules=[Path("capsule.npz")],
        )

  def test_three_round_common_signature_passes_without_training(self):
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      paths = []
      for round_index in range(3):
        path = root / f"round-{round_index}.json"
        path.write_text(json.dumps(_round_report(round_index)))
        paths.append(path)
      log = root / "run.log"
      log.write_text(
          "[CANON_P38] PRECHECK_ROUND_COMPLETE \n" * 3
          + "[CANON_P38] CONTROLLED_EXIT code=42 backward=0 "
          "optimizer_commits=0\n"
      )
      report = AGGREGATE.classify(rounds=paths, run_log=log)
    self.assertEqual(report["verdict"], "PASS")
    self.assertEqual(
        report["selected_signature"],
        {"layer": 9, "checkpoint": "layer_output"},
    )
    self.assertEqual(report["optimizer_commits"], 0)

  def test_three_round_nonrepeating_signature_is_inconclusive(self):
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      paths = []
      for round_index in range(3):
        path = root / f"round-{round_index}.json"
        path.write_text(json.dumps(_round_report(round_index, 9 + round_index)))
        paths.append(path)
      log = root / "run.log"
      log.write_text(
          "[CANON_P38] PRECHECK_ROUND_COMPLETE \n" * 3
          + "[CANON_P38] CONTROLLED_EXIT code=42 backward=0 "
          "optimizer_commits=0\n"
      )
      report = AGGREGATE.classify(rounds=paths, run_log=log)
    self.assertEqual(report["verdict"], "INCONCLUSIVE")
    self.assertIn("no_common_first_red_signature", report["reasons"])


if __name__ == "__main__":
  unittest.main()
