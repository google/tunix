#!/usr/bin/env python3

from __future__ import annotations

import unittest
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "v1_first_update_gate", ROOT / "tunix/rl/v1_first_update_gate.py"
)
assert SPEC is not None and SPEC.loader is not None
gate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = gate
SPEC.loader.exec_module(gate)


class FirstUpdateGateTest(unittest.TestCase):

  def _precommit(self):
    return {
        "schema": "canon-v1-first-update-precommit-v1",
        "update": 0,
        "workload": "gsm8k",
        "dp": 16,
        "tp": 4,
        "microsteps": 16,
        "accumulator_denominator": 16.0,
        "stable_norm_max": gate.STABLE_NORM_MAX,
        "all_finite": True,
        "any_nonzero": True,
        "stable_norm": 2.0,
    }

  def _commit(self):
    return {
        "schema": "canon-v1-first-update-commit-v1",
        "update": 0,
        "workload": "gsm8k",
        "dp": 16,
        "tp": 4,
        "train_steps_before": 0,
        "train_steps_after": 1,
        "optimizer_transaction_valid": True,
        "gradient_finite": True,
        "parameter_delta_finite": True,
        "parameter_changed_elements": 10,
        "effective_learning_rate": 1.0e-6,
        "outer_weight_sync_pending": True,
    }

  def test_green_precommit_and_commit(self):
    self.assertEqual(
        gate.validate_precommit(
            self._precommit(), workload="gsm8k", dp=16, tp=4,
            microsteps=16,
        ),
        (),
    )
    self.assertEqual(
        gate.validate_commit(
            self._commit(), workload="gsm8k", dp=16, tp=4,
        ),
        (),
    )

  def test_precommit_rejects_nonfinite_zero_huge_and_wrong_denominator(self):
    mutations = (
        {"all_finite": False, "stable_norm": float("inf")},
        {"any_nonzero": False, "stable_norm": 0.0},
        {"stable_norm": 1.0e21},
        {"accumulator_denominator": 8.0},
    )
    for mutation in mutations:
      with self.subTest(mutation=mutation):
        record = {**self._precommit(), **mutation}
        self.assertTrue(gate.validate_precommit(
            record, workload="gsm8k", dp=16, tp=4, microsteps=16,
        ))

  def test_commit_rejects_nonfinite_delta_or_positive_lr_without_change(self):
    for mutation in (
        {"parameter_delta_finite": False},
        {"parameter_changed_elements": 0},
        {"train_steps_after": 2},
    ):
      with self.subTest(mutation=mutation):
        record = {**self._commit(), **mutation}
        self.assertTrue(gate.validate_commit(
            record, workload="gsm8k", dp=16, tp=4,
        ))

  def test_zero_learning_rate_allows_unchanged_parameters(self):
    record = {
        **self._commit(),
        "parameter_changed_elements": 0,
        "effective_learning_rate": 0.0,
    }
    self.assertEqual(
        gate.validate_commit(record, workload="gsm8k", dp=16, tp=4),
        (),
    )


if __name__ == "__main__":
  unittest.main()
