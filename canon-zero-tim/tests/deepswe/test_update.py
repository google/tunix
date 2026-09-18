"""Deterministic DP16 reducer and token-denominator CPU gates."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "p34_dp_training", ROOT / "tunix/rl/dp_training.py"
)
if SPEC is None or SPEC.loader is None:
  raise RuntimeError("cannot import DP training contract")
dp_training = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = dp_training
SPEC.loader.exec_module(dp_training)


def _numpy_tree(values):
  current = [np.asarray(value, dtype=np.float64) for value in values]
  while len(current) > 1:
    current = [
        current[index] + current[index + 1]
        for index in range(0, len(current), 2)
    ]
  return current[0]


class DeepSWEUpdateTest(unittest.TestCase):

  def test_dp16_schedule_has_four_reduce_and_four_broadcast_rounds(self):
    reduce_rounds, broadcast_rounds = dp_training.fixed_dp_tree_permutations(16)
    self.assertEqual((len(reduce_rounds), len(broadcast_rounds)), (4, 4))
    self.assertEqual(dp_training.fixed_dp_collective_count(16), 8)

  def test_fixed_sum_matches_independent_fp64_tree(self):
    values = [np.array([rank, rank + 0.5], np.float32) for rank in range(16)]
    actual = np.asarray(dp_training.fixed_dp_sum(tuple(jnp.asarray(x) for x in values)))
    expected = _numpy_tree(values).astype(np.float32)
    np.testing.assert_array_equal(actual, expected)

  def test_global_token_denominator_differs_from_mean_of_rank_means(self):
    token_counts = np.arange(1, 17, dtype=np.float64)
    gradient_sums = token_counts * np.arange(1, 17, dtype=np.float64)
    correct = gradient_sums.sum() / token_counts.sum()
    wrong = np.mean(gradient_sums / token_counts)
    self.assertNotEqual(correct, wrong)
    self.assertAlmostEqual(correct, 11.0)
    self.assertAlmostEqual(wrong, 8.5)

  def test_rank_order_fault_changes_non_associative_tree(self):
    values = [np.float32(1.0e8), np.float32(1.0), np.float32(-1.0e8), np.float32(1.0)]
    values.extend(np.float32(0.0) for _ in range(12))
    base = np.asarray(dp_training.fixed_dp_sum(tuple(jnp.asarray(x) for x in values)))
    values[1], values[2] = values[2], values[1]
    changed = np.asarray(dp_training.fixed_dp_sum(tuple(jnp.asarray(x) for x in values)))
    self.assertFalse(np.array_equal(base, changed))

  def test_non_power_of_two_width_is_rejected(self):
    with self.assertRaisesRegex(ValueError, "power-of-two"):
      dp_training.fixed_dp_tree_permutations(15)


if __name__ == "__main__":
  unittest.main()
