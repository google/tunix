#!/usr/bin/env python3
"""Exact-arm admission tests for the M15 Zero A-B warning lane."""

from __future__ import annotations

import os
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
ALIGNMENT_SPEC = importlib.util.spec_from_file_location(
    "p57_m15_alignment", ROOT / "tunix/rl/alignment.py"
)
if ALIGNMENT_SPEC is None or ALIGNMENT_SPEC.loader is None:
  raise RuntimeError("cannot import alignment policy")
alignment = importlib.util.module_from_spec(ALIGNMENT_SPEC)
sys.modules[ALIGNMENT_SPEC.name] = alignment
ALIGNMENT_SPEC.loader.exec_module(alignment)


def _environment(
    report: str, *, candidate: str = "m15", data_split: str = "main"
) -> dict[str, str]:
  return {
      alignment.GATE_ONLY_ENV: "0",
      alignment.UPDATE_CANARY_ENV: "0",
      alignment.TRAIN_ENV: "1",
      alignment.PRE_GATE_ENV: "1",
      alignment.GSM8K_AB_REPORT_ONLY_ENV: "0",
      alignment.GSM8K_ALIGNMENT_WARN_ONLY_ENV: "0",
      alignment.FROZENLAKE_ALIGNMENT_WARN_ONLY_ENV: "1",
      alignment.DEEPSWE_ALIGNMENT_WARN_ONLY_ENV: "0",
      alignment.PRE_REPORT_ENV: report,
      "CANON_P32_WORKLOAD": "frozenlake-dp8-tp8",
      "CANON_P33_RUN_STAGE": "full",
      "CANON_P33_NO_COMMIT": "0",
      "CANON_PROFILE_FILE": (
          "cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-v1-hp.env"
      ),
      "CANON_PROFILE": "qwen3-8b-dp8-tp8-frozenlake-v1-hp",
      "CANON_V1_HP_FULL": "1",
      "CANON_P57_RUN_KIND": "train",
      "CANON_P57_TIM_ARM": "zero",
      "CANON_P57_EXPECTED_UPDATES": "300",
      "CANON_P57_STOP_AFTER_STEP": "300",
      "CANON_P57_WORKLOAD_CANDIDATE": candidate,
      "CANON_P57_DATA_SPLIT": data_split,
      "CANON_P33_ENABLE_EVAL": "0",
      "CANON_P33_DISABLE_EVAL": "1",
      "CANON_P31_ENABLE_EVAL": "0",
      "CANON_FROZENLAKE_CKPT_MODE": "disabled",
  }


def _sidecar(*, ab_drift: bool = False, bc_drift: bool = False,
             nonfinite_ab: bool = False):
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


class M15AlignmentWarningTest(unittest.TestCase):

  def test_exact_m15_policy_warns_only_for_ab_and_derived_ratios(self):
    with tempfile.TemporaryDirectory() as root, mock.patch.dict(
        os.environ,
        _environment(f"{root}/pre.jsonl"),
        clear=True,
    ):
      policy = alignment.gsm8k_ab_report_policy()
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

  def test_finite_ab_drift_is_durable_warning(self):
    with tempfile.TemporaryDirectory() as root, mock.patch.dict(
        os.environ,
        _environment(f"{root}/pre.jsonl"),
        clear=True,
    ):
      record = alignment.check_pre_backward(_sidecar(ab_drift=True), step=0)
    self.assertEqual(record["verdict"], "PASS_WITH_ALIGNMENT_WARNINGS")
    self.assertEqual(record["blocking_reds"], [])
    self.assertEqual(record["warning_reds"], ["S_decode_vs_S_prefill"])

  def test_bc_drift_and_nonfinite_ab_remain_blocking(self):
    for sidecar, expected in (
        (_sidecar(bc_drift=True), "S_prefill_vs_T_old"),
        (_sidecar(nonfinite_ab=True), "S_decode_vs_S_prefill"),
    ):
      with self.subTest(expected=expected), tempfile.TemporaryDirectory() as root:
        with mock.patch.dict(
            os.environ,
            _environment(f"{root}/pre.jsonl"),
            clear=True,
        ), self.assertRaisesRegex(alignment.AlignmentGateError, expected):
          alignment.check_pre_backward(sidecar, step=0)

  def test_exact_p45_identity_uses_the_same_narrow_ab_warning_lane(self):
    with tempfile.TemporaryDirectory() as root:
      values = _environment(
          f"{root}/pre.jsonl", candidate="", data_split=""
      )
      with mock.patch.dict(os.environ, values, clear=True):
        policy = alignment.gsm8k_ab_report_policy()
    self.assertEqual(policy["claim_level"], "convergence-only")
    self.assertEqual(
        policy["warning_boundaries"], ("S_decode_vs_S_prefill",)
    )
    self.assertFalse(alignment._policy_warns(policy, "S_prefill_vs_T_old"))

  def test_partial_or_foreign_p45_identity_is_rejected(self):
    with tempfile.TemporaryDirectory() as root:
      values = _environment(
          f"{root}/pre.jsonl", candidate="", data_split="main"
      )
      with mock.patch.dict(os.environ, values, clear=True), self.assertRaisesRegex(
          alignment.AlignmentGateError, "exact P45 or M15/main"
      ):
        alignment.gsm8k_ab_report_policy()


if __name__ == "__main__":
  unittest.main()
