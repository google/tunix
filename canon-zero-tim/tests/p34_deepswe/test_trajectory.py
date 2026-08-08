"""Multi-turn mask, boundary, and stale-weight negative controls."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "p34_deepswe_contract_trajectory", ROOT / "tunix/rl/deepswe_contract.py"
)
if SPEC is None or SPEC.loader is None:
  raise RuntimeError("cannot import DeepSWE contract")
deepswe_contract = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = deepswe_contract
SPEC.loader.exec_module(deepswe_contract)


class DeepSWETrajectoryTest(unittest.TestCase):

  def test_environment_tokens_remain_context_but_not_actions(self):
    valid = np.array([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]], dtype=np.bool_)
    action = np.array([[0, 1, 0, 1, 0], [0, 1, 0, 0, 0]], dtype=np.bool_)
    report = deepswe_contract.validate_multiturn_masks(valid, action)
    self.assertEqual(report["action_tokens"], 3)
    self.assertEqual(report["context_only_tokens"], 4)

  def test_action_outside_context_is_rejected(self):
    valid = np.array([[1, 0]], dtype=np.bool_)
    action = np.array([[0, 1]], dtype=np.bool_)
    with self.assertRaisesRegex(ValueError, "not a subset"):
      deepswe_contract.validate_multiturn_masks(valid, action)

  def test_four_boundaries_and_ratios_are_array_exact(self):
    values = np.arange(64 * 7, dtype=np.float32).reshape(64, 7) / 32
    mask = np.ones(values.shape, dtype=np.bool_)
    report = deepswe_contract.validate_four_boundaries(
        values, values.copy(), values.copy(), values.copy(), mask
    )
    self.assertTrue(report["all_boundaries_exact"])
    self.assertTrue(report["ratio_exact"])
    self.assertEqual(report["clip_hits"], 0)
    self.assertEqual(report["tis_hits"], 0)

  def test_one_bit_boundary_corruption_is_rejected(self):
    values = np.zeros((64, 7), dtype=np.float32)
    changed = values.copy()
    changed.view(np.uint32)[9] ^= np.uint32(1)
    with self.assertRaisesRegex(ValueError, "not exact"):
      deepswe_contract.validate_four_boundaries(
          values,
          changed,
          values,
          values,
          np.ones(values.shape, dtype=np.bool_),
      )

  def test_stale_rollout_weight_is_rejected(self):
    deepswe_contract.require_weight_sync("abc", "abc")
    with self.assertRaisesRegex(ValueError, "fingerprints differ"):
      deepswe_contract.require_weight_sync("abc", "def")


if __name__ == "__main__":
  unittest.main()
