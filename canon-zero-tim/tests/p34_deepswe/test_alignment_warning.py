"""Finite-only warning admission for the P34 DeepSWE full run."""

from __future__ import annotations

import os
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

from tunix.rl import alignment


def _sidecar(*, nonfinite_bc: bool = False):
  base = np.asarray([[0.0, 0.25, 0.5]], dtype=np.float32)
  t_old = base + np.float32(0.2)
  if nonfinite_bc:
    t_old[0, 1] = np.nan
  return alignment.ObservedTrainExample(
      train_example=types.SimpleNamespace(),
      s_decode=base - np.float32(0.1),
      s_prefill=base,
      t_old=t_old,
      action_mask=np.ones_like(base, dtype=np.bool_),
      completion_valid_mask=np.ones_like(base, dtype=np.bool_),
      prompt_mask=np.zeros_like(base, dtype=np.bool_),
      tokens=np.asarray([[10, 11, 12]], dtype=np.int32),
      policy_version=np.asarray([0], dtype=np.int32),
      sampling_values=np.asarray([[0.7, 0.0, 1.0]], dtype=np.float32),
  )


def _environment(report: str) -> dict[str, str]:
  return {
      alignment.GATE_ONLY_ENV: "0",
      alignment.UPDATE_CANARY_ENV: "0",
      alignment.TRAIN_ENV: "1",
      alignment.PRE_GATE_ENV: "1",
      alignment.GSM8K_AB_REPORT_ONLY_ENV: "0",
      alignment.GSM8K_ALIGNMENT_WARN_ONLY_ENV: "0",
      alignment.FROZENLAKE_ALIGNMENT_WARN_ONLY_ENV: "0",
      alignment.DEEPSWE_ALIGNMENT_WARN_ONLY_ENV: "1",
      alignment.PRE_REPORT_ENV: report,
      "CANON_P34_DEEPSWE": "1",
      "CANON_P34_RUN_STAGE": "full",
      "CANON_P34_NO_COMMIT": "0",
      "CANON_P39_64CHIP_PILOT": "0",
      "CANON_P43_DEEPSWE_DEBUG": "0",
      "CANON_P44_DEEPSWE_PARITY": "0",
  }


class P34AlignmentWarningTest(unittest.TestCase):

  def test_finite_ab_and_bc_are_durable_nonblocking_warnings(self):
    with tempfile.TemporaryDirectory() as root, mock.patch.dict(
        os.environ, _environment(os.path.join(root, "pre.jsonl")), clear=False
    ):
      record = alignment.check_pre_backward(_sidecar(), step=0)
    self.assertEqual(record["verdict"], "PASS_WITH_ALIGNMENT_WARNINGS")
    self.assertEqual(record["blocking_reds"], [])
    self.assertEqual(
        set(record["warning_reds"]),
        {"S_decode_vs_S_prefill", "S_prefill_vs_T_old"},
    )
    self.assertEqual(
        record["admission_policy"]["claim_level"], "convergence-only"
    )

  def test_nonfinite_bc_remains_fail_closed(self):
    with tempfile.TemporaryDirectory() as root, mock.patch.dict(
        os.environ, _environment(os.path.join(root, "pre.jsonl")), clear=False
    ), self.assertRaisesRegex(
        alignment.AlignmentGateError, "S_prefill_vs_T_old"
    ):
      alignment.check_pre_backward(_sidecar(nonfinite_bc=True), step=0)

  def test_warning_policy_is_not_admitted_for_short_p34_stage(self):
    values = _environment("/tmp/not-used-p34-pre.jsonl")
    values["CANON_P34_RUN_STAGE"] = "one-update"
    with mock.patch.dict(os.environ, values, clear=False), self.assertRaisesRegex(
        alignment.AlignmentGateError, "P34 full training"
    ):
      alignment.gsm8k_ab_report_policy()


if __name__ == "__main__":
  unittest.main()
