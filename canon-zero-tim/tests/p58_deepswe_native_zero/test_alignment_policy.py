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


def _env(
    arm: str,
    stage: str = "three-update",
    *,
    zero_hp_warning: bool = False,
) -> dict[str, str]:
  if zero_hp_warning and (arm != "zero" or stage != "full"):
    raise ValueError("Zero-HP warning admission requires Zero full")
  expected_updates = "3" if stage == "three-update" else "1000"
  values = {
      "CANON_P34_DEEPSWE": "1",
      "CANON_P34_RUN_STAGE": stage,
      "CANON_P34_NO_COMMIT": "0",
      "CANON_P58_DEEPSWE_TIM": "1",
      "CANON_P58_TIM_ADMITTED": "1",
      "CANON_P58_TIM_ARM": arm,
      "CANON_P58_EXPECTED_UPDATES": expected_updates,
      "CANON_DEEPSWE_ALIGNMENT_WARN_ONLY": (
          "1" if arm == "native" or zero_hp_warning else "0"
      ),
      "CANON_ALIGNMENT_GATE": "1",
      "CANON_ALIGNMENT_GATE_ONLY": "0",
      "CANON_ALIGNMENT_UPDATE_CANARY": "0",
      "CANON_ALIGNMENT_TRAIN": "1",
      alignment.PRE_GATE_ENV: "1",
  }
  if zero_hp_warning:
    values.update({
        "CANON_PROFILE_FILE": (
            "cluster/profiles/qwen3-4b-dp8-tp8-deepswe-v1-hp.env"
        ),
        "CANON_PROFILE": "qwen3-4b-dp8-tp8-deepswe-v1-hp",
        "CANON_V1_HP_FULL": "1",
        "CANON_DP_SIZE": "8",
        "CANON_TP_SIZE": "8",
        "CANON_GLOBAL_TRAJECTORIES": "128",
    })
  return values


def _sidecar(
    *,
    ab_drift: bool = False,
    bc_drift: bool = False,
    nonfinite_ab: bool = False,
) -> alignment.ObservedTrainExample:
  s_prefill = np.asarray([[0.0, 0.25, 0.5]], dtype=np.float32)
  s_decode = s_prefill.copy()
  t_old = s_prefill.copy()
  if ab_drift:
    s_decode[0, 1] += np.float32(0.1)
  if bc_drift:
    t_old[0, 1] += np.float32(0.2)
  if nonfinite_ab:
    s_decode[0, 1] = np.nan
  return alignment.ObservedTrainExample(
      train_example=types.SimpleNamespace(),
      s_decode=s_decode,
      s_prefill=s_prefill,
      t_old=t_old,
      action_mask=np.ones_like(s_decode, dtype=np.bool_),
      completion_valid_mask=np.ones_like(s_decode, dtype=np.bool_),
      prompt_mask=np.zeros_like(s_decode, dtype=np.bool_),
      tokens=np.asarray([[10, 11, 12]], dtype=np.int32),
      policy_version=np.asarray([0], dtype=np.int32),
      sampling_values=np.asarray([[0.7, 0.0, 1.0]], dtype=np.float32),
  )


class P58AlignmentPolicyTest(unittest.TestCase):

  def _empty_sidecar(self, *, all_compact_filtered: bool):
    values = np.zeros((2, 3), dtype=np.float32)
    return alignment.ObservedTrainExample(
        train_example=types.SimpleNamespace(),
        s_decode=values,
        s_prefill=values.copy(),
        t_old=values.copy(),
        action_mask=np.zeros_like(values, dtype=np.bool_),
        completion_valid_mask=np.zeros_like(values, dtype=np.bool_),
        prompt_mask=np.ones((2, 2), dtype=np.bool_),
        tokens=np.zeros((2, 3), dtype=np.int32),
        policy_version=np.zeros((2,), dtype=np.int32),
        sampling_values=np.asarray(
            [[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]], dtype=np.float32
        ),
        all_compact_filtered=all_compact_filtered,
    )

  def test_signed_all_compact_filtered_batch_is_no_signal_not_alignment_red(self):
    sidecar = self._empty_sidecar(all_compact_filtered=True)
    with tempfile.TemporaryDirectory() as root:
      values = {
          **_env("native", "full"),
          alignment.PRE_REPORT_ENV: str(Path(root) / "pre.jsonl"),
          alignment.REPORT_ENV: str(Path(root) / "post.jsonl"),
      }
      with mock.patch.dict(os.environ, values, clear=True):
        pre = alignment.check_pre_backward(sidecar, step=0)
        post = alignment.check_batch(
            sidecar,
            t_current=np.zeros((2, 3), dtype=np.float32),
            gradient_norm=np.asarray(0.0, dtype=np.float32),
            optimizer_skipped=np.asarray(0, dtype=np.int32),
            step=0,
        )
    self.assertEqual(pre["verdict"], "PASS")
    self.assertEqual(post["verdict"], "PASS")
    self.assertTrue(pre["no_signal_admitted"])
    self.assertTrue(post["no_signal_admitted"])
    self.assertEqual(pre["N_action"], 0)
    self.assertEqual(post["N_action"], 0)

  def test_zero_action_without_compact_filter_provenance_remains_blocking(self):
    sidecar = self._empty_sidecar(all_compact_filtered=False)
    with tempfile.TemporaryDirectory() as root:
      values = {
          **_env("native", "full"),
          alignment.PRE_REPORT_ENV: str(Path(root) / "pre.jsonl"),
      }
      with mock.patch.dict(os.environ, values, clear=True), self.assertRaisesRegex(
          alignment.AlignmentGateError, "N_action=0"
      ):
        alignment.check_pre_backward(sidecar, step=0)

  def test_native_warns_for_registered_serving_trainer_treatment(self):
    for stage in ("three-update", "full"):
      with self.subTest(stage=stage), mock.patch.dict(
          os.environ, _env("native", stage), clear=True
      ):
        policy = alignment.gsm8k_ab_report_policy()
      self.assertTrue(policy["warning_only"])
      self.assertEqual(
          policy["warning_boundaries"],
          (
              "S_decode_vs_S_prefill",
              "S_prefill_vs_T_old",
              "T_old_vs_T_current",
          ),
      )
      self.assertTrue(alignment._policy_warns(policy, "S_decode_vs_S_prefill"))
      self.assertTrue(alignment._policy_warns(policy, "S_prefill_vs_T_old"))
      self.assertTrue(alignment._policy_warns(policy, "T_old_vs_T_current"))
      self.assertTrue(alignment._policy_warns(policy, "w_all_exactly_1"))
      self.assertTrue(alignment._policy_warns(policy, "r_all_exactly_1"))

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
            t_current=t_old + np.float32(0.05),
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
    self.assertFalse(post["exact"]["r_all_exactly_1"])
    self.assertIn("T_old_vs_T_current", post["warning_reds"])
    self.assertIn("r_all_exactly_1", post["warning_reds"])

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

  def test_native_full_finite_trainer_program_drift_is_nonblocking(self):
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
      with mock.patch.dict(os.environ, values, clear=True):
        record = alignment.check_batch(
            sidecar,
            t_current=t_current,
            gradient_norm=np.asarray(1.0, dtype=np.float32),
            optimizer_skipped=np.asarray(0, dtype=np.int32),
            step=0,
        )
    self.assertEqual(record["blocking_reds"], [])
    self.assertIn("T_old_vs_T_current", record["warning_reds"])
    self.assertIn("r_all_exactly_1", record["warning_reds"])

  def test_native_full_nonfinite_trainer_program_drift_remains_blocking(self):
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
    t_current[0, 1] = np.nan
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

  def test_regular_zero_cannot_enable_warning_policy(self):
    values = _env("zero", "full")
    values["CANON_DEEPSWE_ALIGNMENT_WARN_ONLY"] = "1"
    with (
        mock.patch.dict(os.environ, values, clear=True),
        self.assertRaisesRegex(
            alignment.AlignmentGateError, "DeepSWE warning policy"
        ),
    ):
      alignment.gsm8k_ab_report_policy()

  def test_zero_hp_full_warns_only_for_finite_a_b_and_derivatives(self):
    values = _env("zero", "full", zero_hp_warning=True)
    with tempfile.TemporaryDirectory() as root:
      values[alignment.PRE_REPORT_ENV] = str(Path(root) / "pre.jsonl")
      with mock.patch.dict(os.environ, values, clear=True):
        policy = alignment.gsm8k_ab_report_policy()
        record = alignment.check_pre_backward(
            _sidecar(ab_drift=True), step=0
        )
    self.assertEqual(policy["id"], "deepswe-zero-hp-ab-warning-v1")
    self.assertEqual(policy["claim_level"], "convergence-only")
    self.assertEqual(
        policy["warning_boundaries"], ("S_decode_vs_S_prefill",)
    )
    for item in (
        "S_decode_vs_S_prefill",
        "w_all_exactly_1",
        "wr_all_exactly_1",
        "clip_hits",
        "tis_hits",
    ):
      self.assertTrue(alignment._policy_warns(policy, item), item)
    for item in (
        "S_prefill_vs_T_old",
        "T_old_vs_T_current",
        "r_all_exactly_1",
        "gradient_nonfinite",
    ):
      self.assertFalse(alignment._policy_warns(policy, item), item)
    self.assertEqual(record["verdict"], "PASS_WITH_ALIGNMENT_WARNINGS")
    self.assertEqual(record["blocking_reds"], [])
    self.assertEqual(record["warning_reds"], ["S_decode_vs_S_prefill"])

  def test_zero_hp_full_b_c_and_nonfinite_a_b_remain_blocking(self):
    for sidecar, expected in (
        (_sidecar(bc_drift=True), "S_prefill_vs_T_old"),
        (_sidecar(nonfinite_ab=True), "S_decode_vs_S_prefill"),
    ):
      with self.subTest(expected=expected), tempfile.TemporaryDirectory() as root:
        values = _env("zero", "full", zero_hp_warning=True)
        values[alignment.PRE_REPORT_ENV] = str(Path(root) / "pre.jsonl")
        with mock.patch.dict(
            os.environ, values, clear=True
        ), self.assertRaisesRegex(alignment.AlignmentGateError, expected):
          alignment.check_pre_backward(sidecar, step=0)

  def test_zero_hp_full_trainer_repeat_remains_blocking(self):
    sidecar = _sidecar(ab_drift=True)
    t_current = sidecar.t_old.copy()
    t_current[0, 1] = np.nextafter(
        t_current[0, 1], np.float32(np.inf)
    )
    with tempfile.TemporaryDirectory() as root:
      values = _env("zero", "full", zero_hp_warning=True)
      values[alignment.REPORT_ENV] = str(Path(root) / "post.jsonl")
      with mock.patch.dict(
          os.environ, values, clear=True
      ), self.assertRaisesRegex(
          alignment.AlignmentGateError, "T_old_vs_T_current"
      ):
        alignment.check_batch(
            sidecar,
            t_current=t_current,
            gradient_norm=np.asarray(1.0, dtype=np.float32),
            optimizer_skipped=np.asarray(0, dtype=np.int32),
            step=0,
        )

  def test_zero_hp_full_admits_unset_and_zero_precheck_only(self):
    for precheck_val in (None, "0"):
      with self.subTest(precheck_val=precheck_val):
        values = _env("zero", "full", zero_hp_warning=True)
        if precheck_val is not None:
          values["CANON_P38_PRECHECK_ONLY"] = precheck_val
        with mock.patch.dict(os.environ, values, clear=True):
          policy = alignment.gsm8k_ab_report_policy()
        self.assertEqual(policy["id"], "deepswe-zero-hp-ab-warning-v1")

    with mock.patch.dict(
        os.environ,
        {**_env("zero", "full", zero_hp_warning=True), "CANON_P38_PRECHECK_ONLY": "1"},
        clear=True,
    ), self.assertRaisesRegex(alignment.AlignmentGateError, "DeepSWE warning policy"):
      alignment.gsm8k_ab_report_policy()


if __name__ == "__main__":
  unittest.main()
