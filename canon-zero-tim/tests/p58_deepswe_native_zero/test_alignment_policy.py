#!/usr/bin/env python3
"""P58 native A-B observer and zero strict-alignment policies."""

from __future__ import annotations

import os
from unittest import mock
import unittest

from tunix.rl import alignment


def _env(arm: str) -> dict[str, str]:
  return {
      "CANON_P34_DEEPSWE": "1",
      "CANON_P34_RUN_STAGE": "three-update",
      "CANON_P34_NO_COMMIT": "0",
      "CANON_P58_DEEPSWE_TIM": "1",
      "CANON_P58_TIM_ARM": arm,
      "CANON_DEEPSWE_ALIGNMENT_WARN_ONLY": "1" if arm == "native" else "0",
      "CANON_ALIGNMENT_GATE": "1",
      "CANON_ALIGNMENT_GATE_ONLY": "0",
      "CANON_ALIGNMENT_UPDATE_CANARY": "0",
      "CANON_ALIGNMENT_TRAIN": "1",
  }


class P58AlignmentPolicyTest(unittest.TestCase):

  def test_native_warns_only_for_registered_a_b_treatment(self):
    with mock.patch.dict(os.environ, _env("native"), clear=True):
      policy = alignment.gsm8k_ab_report_policy()
    self.assertTrue(policy["warning_only"])
    self.assertEqual(policy["warning_boundaries"], ("S_decode_vs_S_prefill",))
    self.assertTrue(alignment._policy_warns(policy, "S_decode_vs_S_prefill"))
    self.assertFalse(alignment._policy_warns(policy, "S_prefill_vs_T_old"))
    self.assertTrue(alignment._policy_warns(policy, "w_all_exactly_1"))
    self.assertFalse(alignment._policy_warns(policy, "r_all_exactly_1"))

  def test_zero_is_strict(self):
    with mock.patch.dict(os.environ, _env("zero"), clear=True):
      policy = alignment.gsm8k_ab_report_policy()
    self.assertFalse(policy["warning_only"])
    self.assertIsNone(policy["warning_boundaries"])
    self.assertFalse(alignment._policy_warns(policy, "S_decode_vs_S_prefill"))


if __name__ == "__main__":
  unittest.main()
