#!/usr/bin/env python3
"""P58 native A-B observer and zero strict-alignment policies."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
from unittest import mock
import unittest


ROOT = Path(__file__).resolve().parents[3]
ALIGNMENT_SPEC = importlib.util.spec_from_file_location(
    "p58_alignment", ROOT / "tunix/rl/alignment.py"
)
if ALIGNMENT_SPEC is None or ALIGNMENT_SPEC.loader is None:
  raise RuntimeError("cannot import alignment policy")
alignment = importlib.util.module_from_spec(ALIGNMENT_SPEC)
sys.modules[ALIGNMENT_SPEC.name] = alignment
ALIGNMENT_SPEC.loader.exec_module(alignment)


def _env(arm: str, stage: str = "three-update") -> dict[str, str]:
  expected_updates = "3" if stage == "three-update" else "1000"
  return {
      "CANON_P34_DEEPSWE": "1",
      "CANON_P34_RUN_STAGE": stage,
      "CANON_P34_NO_COMMIT": "0",
      "CANON_P58_DEEPSWE_TIM": "1",
      "CANON_P58_TIM_ADMITTED": "1",
      "CANON_P58_TIM_ARM": arm,
      "CANON_P58_EXPECTED_UPDATES": expected_updates,
      "CANON_DEEPSWE_ALIGNMENT_WARN_ONLY": "1" if arm == "native" else "0",
      "CANON_ALIGNMENT_GATE": "1",
      "CANON_ALIGNMENT_GATE_ONLY": "0",
      "CANON_ALIGNMENT_UPDATE_CANARY": "0",
      "CANON_ALIGNMENT_TRAIN": "1",
  }


class P58AlignmentPolicyTest(unittest.TestCase):

  def test_native_warns_only_for_registered_a_b_treatment(self):
    for stage in ("three-update", "full"):
      with self.subTest(stage=stage), mock.patch.dict(
          os.environ, _env("native", stage), clear=True
      ):
        policy = alignment.gsm8k_ab_report_policy()
      self.assertTrue(policy["warning_only"])
      self.assertEqual(
          policy["warning_boundaries"], ("S_decode_vs_S_prefill",)
      )
      self.assertTrue(alignment._policy_warns(policy, "S_decode_vs_S_prefill"))
      self.assertFalse(alignment._policy_warns(policy, "S_prefill_vs_T_old"))
      self.assertTrue(alignment._policy_warns(policy, "w_all_exactly_1"))
      self.assertFalse(alignment._policy_warns(policy, "r_all_exactly_1"))

  def test_zero_is_strict(self):
    for stage in ("three-update", "full"):
      with self.subTest(stage=stage), mock.patch.dict(
          os.environ, _env("zero", stage), clear=True
      ):
        policy = alignment.gsm8k_ab_report_policy()
      self.assertFalse(policy["warning_only"])
      self.assertIsNone(policy["warning_boundaries"])
      self.assertFalse(alignment._policy_warns(policy, "S_decode_vs_S_prefill"))

  def test_native_full_requires_signed_admission_and_horizon(self):
    valid = _env("native", "full")
    for changed in (
        {"CANON_P58_TIM_ADMITTED": "0"},
        {"CANON_P58_EXPECTED_UPDATES": "3"},
        {"CANON_P58_EXPECTED_UPDATES": "999"},
        {"CANON_P44_DEEPSWE_PARITY": "1"},
    ):
      with self.subTest(changed=changed), mock.patch.dict(
          os.environ, {**valid, **changed}, clear=True
      ), self.assertRaisesRegex(
          alignment.AlignmentGateError, "signed P58 native"
      ):
        alignment.gsm8k_ab_report_policy()

  def test_native_three_update_requires_exact_three_update_horizon(self):
    values = _env("native", "three-update")
    values["CANON_P58_EXPECTED_UPDATES"] = "1000"
    with (
        mock.patch.dict(os.environ, values, clear=True),
        self.assertRaisesRegex(
            alignment.AlignmentGateError, "signed P58 native"
        ),
    ):
      alignment.gsm8k_ab_report_policy()

  def test_zero_cannot_enable_native_warning_policy(self):
    values = _env("zero", "full")
    values["CANON_DEEPSWE_ALIGNMENT_WARN_ONLY"] = "1"
    with (
        mock.patch.dict(os.environ, values, clear=True),
        self.assertRaisesRegex(
            alignment.AlignmentGateError, "zero requires strict alignment"
        ),
    ):
      alignment.gsm8k_ab_report_policy()


if __name__ == "__main__":
  unittest.main()
