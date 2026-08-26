#!/usr/bin/env python3
"""CPU positive and negative controls for the P66 same-point VJP oracle."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import jax.numpy as jnp


ROOT = Path(__file__).resolve().parents[3]
PATH = ROOT / "tunix/rl/p66_vjp_oracle.py"
SPEC = importlib.util.spec_from_file_location("p66_vjp_oracle_tested", PATH)
assert SPEC is not None and SPEC.loader is not None
ORACLE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ORACLE
SPEC.loader.exec_module(ORACLE)


class P66VjpOracleTest(unittest.TestCase):

  def test_exact_pair_passes(self):
    reference = (jnp.asarray([1.0, -2.0, 3.0], jnp.float32),)
    result = ORACLE.compare(
        reference, reference, endpoint="exact", emit=False
    )
    self.assertEqual(result["verdict"], "PASS")
    self.assertTrue(result["array_exact"])
    self.assertEqual(result["metrics"]["rel_l2"], 0.0)

  def test_small_in_envelope_difference_passes(self):
    reference = (jnp.asarray([1.0, -2.0, 3.0], jnp.float32),)
    candidate = (jnp.asarray([1.001, -2.0, 3.0], jnp.float32),)
    result = ORACLE.compare(
        reference, candidate, endpoint="close", emit=False
    )
    self.assertEqual(result["verdict"], "PASS", result)
    self.assertFalse(result["array_exact"])

  def test_normal_value_fault_is_rejected(self):
    reference = (jnp.asarray([1.0, -2.0, 3.0], jnp.float32),)
    candidate = (jnp.asarray([1.0, -1.0, 3.0], jnp.float32),)
    result = ORACLE.compare(
        reference, candidate, endpoint="fault", emit=False
    )
    self.assertEqual(result["verdict"], "FAIL")
    self.assertGreater(result["metrics"]["rel_l2"], ORACLE.CAPS["rel_l2"])
    self.assertTrue(ORACLE.negative_control())

  def test_dead_candidate_leaf_is_rejected(self):
    reference = (
        jnp.asarray([1.0, -2.0], jnp.float32),
        jnp.asarray([0.5], jnp.float32),
    )
    candidate = (
        jnp.asarray([1.0, -2.0], jnp.float32),
        jnp.asarray([0.0], jnp.float32),
    )
    result = ORACLE.compare(
        reference, candidate, endpoint="dead", emit=False
    )
    self.assertEqual(result["verdict"], "FAIL")
    self.assertEqual(result["dead_candidate_leaves"], 1)

  def test_shape_mismatch_is_inconclusive_contract_error(self):
    with self.assertRaisesRegex(ORACLE.OracleContractError, "contract changed"):
      ORACLE.compare(
          (jnp.ones((2,), jnp.float32),),
          (jnp.ones((3,), jnp.float32),),
          endpoint="shape",
          emit=False,
      )

  def test_unit_rank_unstage_rejects_nonunit_axis(self):
    value = ORACLE.unstage_unit_rank(
        (jnp.ones((1, 3), jnp.float32),), endpoint="unit"
    )
    self.assertEqual(value[0].shape, (3,))
    with self.assertRaisesRegex(ORACLE.OracleContractError, "rank staging"):
      ORACLE.unstage_unit_rank(
          (jnp.ones((2, 3), jnp.float32),), endpoint="nonunit"
      )


if __name__ == "__main__":
  unittest.main()
