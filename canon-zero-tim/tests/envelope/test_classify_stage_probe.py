"""Tests for the fail-closed P35.3c stage-probe classifier."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


PATH = Path(__file__).with_name("classify_stage_probe.py")
SPEC = importlib.util.spec_from_file_location("p35_stage_probe_classifier", PATH)
assert SPEC is not None and SPEC.loader is not None
classifier = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = classifier
SPEC.loader.exec_module(classifier)


def _events():
  return [
      {
          "schema_version": 1,
          "event": "ready",
          "replay": "R0_live_first",
          "record_index": 1,
          "record_count": 2,
          "stage": stage,
          "ordinal": ordinal,
          "stage_count": len(classifier.REQUIRED_STAGES),
          "leaf_shapes": [[256, 8]],
      }
      for ordinal, stage in enumerate(classifier.REQUIRED_STAGES, start=1)
  ]


class StageProbeClassifierTest(unittest.TestCase):

  def test_accepts_exactly_one_ordered_first_record(self):
    result = classifier.classify(_events())
    self.assertEqual(result["measurement_verdict"], "COMPLETE")
    self.assertEqual(
        result["classification"], "first_record_stage_probe_complete"
    )
    self.assertFalse(result["numerical_verdict"])

  def test_rejects_missing_stage(self):
    result = classifier.classify(_events()[:-1])
    self.assertEqual(result["measurement_verdict"], "INCONCLUSIVE")
    self.assertEqual(result["last_ready_stage"], "target_gathers")
    self.assertEqual(result["first_missing_stage"], "record_outputs")

  def test_empty_report_localizes_model_as_first_missing(self):
    result = classifier.classify([])
    self.assertEqual(result["measurement_verdict"], "INCONCLUSIVE")
    self.assertIsNone(result["last_ready_stage"])
    self.assertEqual(result["first_missing_stage"], "model")

  def test_rejects_out_of_order_stage(self):
    events = _events()
    events[2], events[3] = events[3], events[2]
    result = classifier.classify(events)
    self.assertEqual(result["measurement_verdict"], "INCONCLUSIVE")

  def test_rejects_second_record(self):
    events = _events()
    events[0]["record_index"] = 2
    result = classifier.classify(events)
    self.assertEqual(result["measurement_verdict"], "INCONCLUSIVE")

  def test_rejects_duplicate_stage(self):
    events = _events()
    events.append(dict(events[-1]))
    result = classifier.classify(events)
    self.assertEqual(result["measurement_verdict"], "INCONCLUSIVE")

  def test_rejects_record_count_drift(self):
    events = _events()
    events[-1]["record_count"] = 3
    result = classifier.classify(events)
    self.assertEqual(result["measurement_verdict"], "INCONCLUSIVE")


if __name__ == "__main__":
  unittest.main()
