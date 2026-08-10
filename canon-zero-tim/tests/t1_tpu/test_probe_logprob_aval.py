"""CPU contract tests for the P38 model-free aval discriminator."""

from __future__ import annotations

import unittest

import numpy as np

import probe_logprob_aval as target


class ProbeLogprobAvalTest(unittest.TestCase):

  def test_registered_topologies(self):
    local = target.shape_contract(4)
    self.assertEqual((local.data_size, local.global_m), (1, 256))
    production = target.shape_contract(64)
    self.assertEqual((production.data_size, production.global_m), (16, 4096))
    with self.assertRaises(target.ProbeContractError):
      target.shape_contract(8)

  def test_exact_and_one_bit_comparisons(self):
    value = np.asarray([0.25, -0.5], dtype=np.float32)
    self.assertTrue(target.bitwise_comparison(value, value)["exact"])
    changed = value.copy()
    changed.view(np.uint32)[0] ^= np.uint32(1)
    report = target.bitwise_comparison(value, changed)
    self.assertFalse(report["exact"])
    self.assertEqual(report["differing_elements"], 1)

  def test_classification_table(self):
    exact = {"exact": True}
    red = {"exact": False}
    base = {
        "raw_target": exact,
        "sampled_token": exact,
        "processed_target": exact,
        "target_logprob": exact,
        "implied_normalizer": exact,
    }
    negative = {"differing_elements": 1}
    self.assertEqual(
        target.classify(base, negative), "MODEL_FREE_NOT_REPRODUCED"
    )
    transform = dict(base, processed_target=red, implied_normalizer=red)
    self.assertEqual(
        target.classify(transform, negative), "TRANSFORM_AVAL_CARRIER"
    )
    score = dict(base, target_logprob=red, implied_normalizer=red)
    self.assertEqual(target.classify(score, negative), "SCORE_AVAL_CARRIER")
    both = dict(
        base,
        processed_target=red,
        target_logprob=red,
        implied_normalizer=red,
    )
    self.assertEqual(
        target.classify(both, negative), "TRANSFORM_AND_SCORE_AVAL_CARRIER"
    )

  def test_fail_closed_contracts(self):
    exact = {"exact": True}
    base = {
        "raw_target": exact,
        "sampled_token": exact,
        "processed_target": exact,
        "target_logprob": exact,
        "implied_normalizer": exact,
    }
    with self.assertRaises(target.ProbeContractError):
      target.classify(dict(base, raw_target={"exact": False}), {"differing_elements": 1})
    with self.assertRaises(target.ProbeContractError):
      target.classify(base, {"differing_elements": 0})
    with self.assertRaises(target.ProbeContractError):
      target.classify(dict(base, implied_normalizer={"exact": False}), {"differing_elements": 1})


if __name__ == "__main__":
  unittest.main()
