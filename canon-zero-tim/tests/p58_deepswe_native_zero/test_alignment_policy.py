#!/usr/bin/env python3
"""P58 native mismatch observers and zero strict-alignment policies."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import types
from unittest import mock
import unittest

import numpy as np


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
      alignment.PRE_GATE_ENV: "1",
  }


class P58AlignmentPolicyTest(unittest.TestCase):

  def test_native_warns_for_registered_serving_trainer_treatment(self):
    for stage in ("three-update", "full"):
      with self.subTest(stage=stage), mock.patch.dict(
          os.environ, _env("native", stage), clear=True
      ):
        policy = alignment.gsm8k_ab_report_policy()
      self.assertTrue(policy["warning_only"])
      self.assertEqual(
          policy["warning_boundaries"],
          ("S_decode_vs_S_prefill", "S_prefill_vs_T_old"),
      )
      self.assertTrue(alignment._policy_warns(policy, "S_decode_vs_S_prefill"))
      self.assertTrue(alignment._policy_warns(policy, "S_prefill_vs_T_old"))
      self.assertTrue(alignment._policy_warns(policy, "w_all_exactly_1"))
      self.assertFalse(alignment._policy_warns(policy, "r_all_exactly_1"))

  def test_native_full_finite_a_b_and_b_c_are_nonblocking(self):
    s_decode = np.asarray([[0.0, 0.25, 0.5]], dtype=np.float32)
    s_prefill = s_decode + np.float32(0.1)
    t_old = s_prefill + np.float32(0.2)
    sidecar = alignment.ObservedTrainExample(
        train_example=types.SimpleNamespace(),
        s_decode=s_decode,
        s_prefill=s_prefill,
        t_old=t_old,
        action_mask=np.ones_like(s_decode, dtype=np.bool_),
        completion_valid_mask=np.ones_like(s_decode, dtype=np.bool_),
        prompt_mask=np.zeros_like(s_decode, dtype=np.bool_),
        tokens=np.asarray([[10, 11, 12]], dtype=np.int32),
        policy_version=np.asarray([0], dtype=np.int32),
        sampling_values=np.asarray([[1.0, 0.0, 1.0]], dtype=np.float32),
    )
    with tempfile.TemporaryDirectory() as root:
      values = {
          **_env("native", "full"),
          alignment.PRE_REPORT_ENV: str(Path(root) / "pre.jsonl"),
          alignment.REPORT_ENV: str(Path(root) / "post.jsonl"),
      }
      with mock.patch.dict(os.environ, values, clear=True):
        record = alignment.check_pre_backward(sidecar, step=0)
        post = alignment.check_batch(
            sidecar,
            t_current=t_old,
            gradient_norm=np.asarray(1.0, dtype=np.float32),
            optimizer_skipped=np.asarray(0, dtype=np.int32),
            step=0,
        )
    self.assertEqual(record["verdict"], "PASS_WITH_ALIGNMENT_WARNINGS")
    self.assertEqual(record["blocking_reds"], [])
    self.assertEqual(
        set(record["warning_reds"]),
        {"S_decode_vs_S_prefill", "S_prefill_vs_T_old"},
    )
    self.assertEqual(post["blocking_reds"], [])
    self.assertTrue(post["exact"]["r_all_exactly_1"])

  def test_native_full_nonfinite_b_c_remains_blocking(self):
    s_decode = np.asarray([[0.0, 0.25, 0.5]], dtype=np.float32)
    s_prefill = s_decode + np.float32(0.1)
    t_old = s_prefill.copy()
    t_old[0, 1] = np.nan
    sidecar = alignment.ObservedTrainExample(
        train_example=types.SimpleNamespace(),
        s_decode=s_decode,
        s_prefill=s_prefill,
        t_old=t_old,
        action_mask=np.ones_like(s_decode, dtype=np.bool_),
        completion_valid_mask=np.ones_like(s_decode, dtype=np.bool_),
        prompt_mask=np.zeros_like(s_decode, dtype=np.bool_),
        tokens=np.asarray([[10, 11, 12]], dtype=np.int32),
        policy_version=np.asarray([0], dtype=np.int32),
        sampling_values=np.asarray([[1.0, 0.0, 1.0]], dtype=np.float32),
    )
    with tempfile.TemporaryDirectory() as root:
      values = {
          **_env("native", "full"),
          alignment.PRE_REPORT_ENV: str(Path(root) / "pre.jsonl"),
      }
      with mock.patch.dict(os.environ, values, clear=True), self.assertRaisesRegex(
          alignment.AlignmentGateError, "S_prefill_vs_T_old"
      ):
        alignment.check_pre_backward(sidecar, step=0)

  def test_native_full_trainer_repeat_remains_blocking(self):
    s_decode = np.asarray([[0.0, 0.25, 0.5]], dtype=np.float32)
    s_prefill = s_decode + np.float32(0.1)
    t_old = s_prefill + np.float32(0.2)
    sidecar = alignment.ObservedTrainExample(
        train_example=types.SimpleNamespace(),
        s_decode=s_decode,
        s_prefill=s_prefill,
        t_old=t_old,
        action_mask=np.ones_like(s_decode, dtype=np.bool_),
        completion_valid_mask=np.ones_like(s_decode, dtype=np.bool_),
        prompt_mask=np.zeros_like(s_decode, dtype=np.bool_),
        tokens=np.asarray([[10, 11, 12]], dtype=np.int32),
        policy_version=np.asarray([0], dtype=np.int32),
        sampling_values=np.asarray([[1.0, 0.0, 1.0]], dtype=np.float32),
    )
    t_current = t_old.copy()
    t_current[0, 1] = np.nextafter(t_current[0, 1], np.float32(np.inf))
    with tempfile.TemporaryDirectory() as root:
      values = {
          **_env("native", "full"),
          alignment.REPORT_ENV: str(Path(root) / "post.jsonl"),
      }
      with mock.patch.dict(os.environ, values, clear=True), self.assertRaisesRegex(
          alignment.AlignmentGateError, "T_old_vs_T_current"
      ):
        alignment.check_batch(
            sidecar,
            t_current=t_current,
            gradient_norm=np.asarray(1.0, dtype=np.float32),
            optimizer_skipped=np.asarray(0, dtype=np.int32),
            step=0,
        )

  def test_zero_is_strict(self):
    for stage in ("three-update", "full"):
      with self.subTest(stage=stage), mock.patch.dict(
          os.environ, _env("zero", stage), clear=True
      ):
        policy = alignment.gsm8k_ab_report_policy()
      self.assertFalse(policy["warning_only"])
      self.assertIsNone(policy["warning_boundaries"])
      self.assertFalse(alignment._policy_warns(policy, "S_decode_vs_S_prefill"))
      self.assertFalse(alignment._policy_warns(policy, "S_prefill_vs_T_old"))

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
