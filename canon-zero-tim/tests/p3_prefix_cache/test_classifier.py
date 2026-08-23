#!/usr/bin/env python3

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
MODULE = ROOT / (
    "canon-zero-tim/tasks/v1-phase3-prefix-cache/scripts/"
    "classify_p3_alignment.py"
)
SPEC = importlib.util.spec_from_file_location("p3_apc_classifier", MODULE)
assert SPEC is not None and SPEC.loader is not None
classifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(classifier)


def _record(round_index: int, ab: int, bc: int = 0) -> dict:
  hashes = {
      key: key * 8
      for key in (
          "S_decode", "S_prefill", "T_old", "tokens", "action_mask",
          "policy_version",
      )
  }
  return {
      "diagnostic_round": round_index,
      "N_action": 4,
      "boundaries": {
          "S_decode_vs_S_prefill": {
              "valid": True, "finite": True, "differing_bytes": ab,
          },
          "S_prefill_vs_T_old": {
              "valid": True, "finite": True, "differing_bytes": bc,
          },
      },
      "hashes": hashes,
      "masked_hashes": {
          key: hashes[key] for key in ("S_decode", "S_prefill", "T_old")
      },
  }


class AlignmentClassifierTest(unittest.TestCase):

  def _fixture(self, expect_apc: bool, records: list[dict], hit: float = 0.0):
    temporary = tempfile.TemporaryDirectory()
    root = Path(temporary.name)
    raw = root / "raw.log"
    report = root / "pre_alignment.jsonl"
    raw.write_text(
        f"[P3_APC_CONFIG] enabled={int(expect_apc)} "
        "workload=frozenlake reader=train_frozenlake_qwen3\n"
        f"Prefix cache hit rate: {hit}%\n",
        encoding="utf-8",
    )
    report.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    return temporary, raw, report

  def test_apc_off_three_round_control_is_green(self):
    tmp, raw, report = self._fixture(
        False, [_record(index, 0) for index in range(3)]
    )
    with tmp:
      result = classifier.classify(raw, report, False)
      self.assertEqual(result["status"], "CONTROL_GREEN")

  def test_apc_on_requires_a_positive_hit_and_red_ab(self):
    tmp, raw, report = self._fixture(True, [_record(0, 7)], hit=25.0)
    with tmp:
      result = classifier.classify(raw, report, True)
      self.assertEqual(result["status"], "REPRODUCED_RED")

  def test_apc_on_three_round_certification_is_green(self):
    tmp, raw, report = self._fixture(
        True, [_record(index, 0) for index in range(3)], hit=83.4
    )
    with tmp:
      result = classifier.classify(
          raw, report, True, purpose="certification"
      )
      self.assertEqual(result["status"], "GB_GC_CERTIFICATION_GREEN")

  def test_apc_on_certification_rejects_red_ab(self):
    tmp, raw, report = self._fixture(
        True, [_record(0, 0), _record(1, 1), _record(2, 0)], hit=83.4
    )
    with tmp, self.assertRaisesRegex(
        classifier.ClassificationError, "observed an A-B byte difference"
    ):
      classifier.classify(raw, report, True, purpose="certification")

  def test_apc_on_zero_hit_is_rejected(self):
    tmp, raw, report = self._fixture(True, [_record(0, 7)], hit=0.0)
    with tmp, self.assertRaisesRegex(
        classifier.ClassificationError, "no positive cache hit"
    ):
      classifier.classify(raw, report, True)

  def test_b_c_red_is_always_rejected(self):
    tmp, raw, report = self._fixture(True, [_record(0, 7, bc=1)], hit=25.0)
    with tmp, self.assertRaisesRegex(
        classifier.ClassificationError, "B-C changed"
    ):
      classifier.classify(raw, report, True)


if __name__ == "__main__":
  unittest.main()
