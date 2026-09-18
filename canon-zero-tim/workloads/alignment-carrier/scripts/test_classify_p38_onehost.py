#!/usr/bin/env python3
"""Negative controls for the P38 one-host classifier."""

from __future__ import annotations

import unittest

from classify_p38_onehost import ClassificationError
from classify_p38_onehost import classify


def _record(a_b: int = 0, b_c: int = 0):
  return {
      "verdict": "PASS" if not (a_b or b_c) else "FAIL",
      "N_action": 8,
      "boundaries": {
          "S_decode_vs_S_prefill": {
              "valid": True,
              "differing_bytes": a_b,
          },
          "S_prefill_vs_T_old": {
              "valid": True,
              "differing_bytes": b_c,
          },
      },
  }


def _log(*, stop: bool = True, backward: bool = False):
  lines = [
      "[P38.ONEHOST] OVERLAY_BYTE_IDENTITY PASS files=6",
      "[P38.ONEHOST] devices=4 ids=[0, 1, 2, 3] platform=tpu",
      "CANON_FIXED_AR=1 fixed-order tree",
      "CANON_FIXED_AR_EMBED=1 fixed-order embed gather",
      "CANON_LOGPROB_M on",
      "runner_sampling_adapter_same_object=True",
  ]
  if stop:
    lines.append("[CANON_P38] PRECHECK_COMPLETE STOP_BEFORE_BACKWARD")
  if backward:
    lines.append("CANON_PROCESSED_LOGPROB_VJP backward")
  return "\n".join(lines)


class P38OneHostClassifierTest(unittest.TestCase):

  def test_exact_is_local_not_reproduced(self):
    self.assertEqual(
        classify([_record()], _log())["verdict"], "LOCAL_NOT_REPRODUCED"
    )

  def test_one_bit_ab_drift_is_local_reproduced(self):
    self.assertEqual(
        classify([_record(a_b=1)], _log(stop=False))["verdict"],
        "LOCAL_REPRODUCED",
    )

  def test_bc_drift_is_regression(self):
    self.assertEqual(
        classify([_record(b_c=1)], _log(stop=False))["verdict"],
        "VOID_REGRESSION",
    )

  def test_missing_row_and_backward_are_rejected(self):
    with self.assertRaises(ClassificationError):
      classify([], _log())
    with self.assertRaises(ClassificationError):
      classify([_record()], _log(backward=True))

  def test_bad_environment_and_missing_topology_are_rejected(self):
    with self.assertRaises(ClassificationError):
      classify([_record()], _log() + "\nC7/C8 violation [post-import]")
    with self.assertRaises(ClassificationError):
      classify([_record()], _log().replace(" platform=tpu", " platform=cpu"))


if __name__ == "__main__":
  unittest.main()
