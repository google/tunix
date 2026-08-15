# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
from tunix.sft import utils


class UtilsTest(absltest.TestCase):

  def test_make_causal_attn_mask(self):
    input_mask = jnp.array([
        [True, True, True, True],
        [True, True, True, False],
        [False, True, True, False],
    ])
    attn_mask = utils.make_causal_attn_mask(input_mask)
    expected_value = jnp.array([
        [
            [True, False, False, False],
            [True, True, False, False],
            [True, True, True, False],
            [True, True, True, True],
        ],
        [
            [True, False, False, False],
            [True, True, False, False],
            [True, True, True, False],
            [True, True, True, False],
        ],
        [
            [False, False, False, False],
            [False, True, False, False],
            [False, True, True, False],
            [False, True, True, False],
        ],
    ])
    np.testing.assert_allclose(attn_mask, expected_value)

  def test_build_positions_from_mask(self):
    input_mask = jnp.array(
        [[1, 1, 1, 1], [0, 1, 1, 1], [1, 1, 1, 0], [0, 1, 1, 0]]
    )
    positions = utils.build_positions_from_mask(input_mask)
    expected_value = jnp.array([
        [0, 1, 2, 3],
        [0, 0, 1, 2],
        [0, 1, 2, 2],
        [0, 0, 1, 1],
    ])
    np.testing.assert_array_equal(positions, expected_value)


class WeightedMetricTest(absltest.TestCase):
  """Isolated tests for WeightedMetric's deferred division and safeguards.

  WeightedMetric stores an unreduced sum and a denominator and only divides in
  compute(). compute_scale() builds 1 / denominator with three safeguards, in
  order: add eps (if set), clamp to min_denom (if set), then a zero-guard that
  maps denominator == 0 to a 0.0 scale (so an empty batch contributes nothing
  and does not poison gradients with NaN/Inf).
  """

  def _compute(self, sum_val, denom, **kwargs):
    metric = utils.WeightedMetric(
        jnp.array(sum_val, dtype=jnp.float32),
        jnp.array(denom, dtype=jnp.float32),
        **kwargs,
    )
    return float(metric.compute())

  def test_basic_compute(self):
    metric = utils.WeightedMetric(
        jnp.array(6.0, dtype=jnp.float32), jnp.array(3.0, dtype=jnp.float32)
    )
    self.assertAlmostEqual(float(metric.compute()), 2.0, places=5)
    self.assertAlmostEqual(float(metric.compute_scale()), 1.0 / 3.0, places=5)

  def test_zero_denominator_is_safe(self):
    metric = utils.WeightedMetric(
        jnp.array(5.0, dtype=jnp.float32), jnp.array(0.0, dtype=jnp.float32)
    )
    scale = float(metric.compute_scale())
    value = float(metric.compute())
    self.assertEqual(scale, 0.0)
    self.assertEqual(value, 0.0)
    self.assertFalse(np.isnan(value) or np.isinf(value))

  def test_eps_is_negligible_for_nonzero_denominator(self):
    self.assertAlmostEqual(self._compute(6.0, 3.0, eps=1e-6), 2.0, places=4)

  def test_eps_bypasses_zero_guard(self):
    # With eps set, denominator == 0 becomes eps (not 0), so the zero-guard is
    # NOT triggered: the result is sum / eps (a large, finite number), not 0.
    # eps prevents NaN, it does not zero out empty batches.
    value = self._compute(6.0, 0.0, eps=1e-6)
    self.assertAlmostEqual(value, 6.0 / 1e-6, delta=1.0)
    self.assertFalse(np.isinf(value))
    self.assertNotEqual(value, 0.0)

  def test_min_denom_clamps_small_denominator(self):
    self.assertAlmostEqual(self._compute(6.0, 1.0, min_denom=3.0), 2.0, places=5)

  def test_min_denom_noop_when_denominator_is_larger(self):
    self.assertAlmostEqual(self._compute(6.0, 5.0, min_denom=3.0), 1.2, places=5)

  def test_min_denom_acts_as_zero_safeguard(self):
    self.assertAlmostEqual(self._compute(6.0, 0.0, min_denom=2.0), 3.0, places=5)

  def test_eps_and_min_denom_applied_in_order(self):
    # denominator 0 -> +eps (1e-6) -> max(1e-6, 2.0) = 2.0 -> 6 / 2 = 3.0.
    value = self._compute(6.0, 0.0, eps=1e-6, min_denom=2.0)
    self.assertAlmostEqual(value, 3.0, places=5)

  def test_gradient_is_finite_at_zero_denominator(self):
    # The zero-guard uses a double jnp.where so the backward pass stays finite:
    # gradient of the numerator at denominator == 0 is exactly 0, not NaN.
    def loss(sum_val):
      metric = utils.WeightedMetric(sum_val, jnp.array(0.0, dtype=jnp.float32))
      return metric.compute()

    grad = float(jax.grad(loss)(jnp.array(5.0, dtype=jnp.float32)))
    self.assertEqual(grad, 0.0)
    self.assertFalse(np.isnan(grad))

  def test_is_a_pytree_with_two_dynamic_leaves(self):
    # sum and denominator are dynamic leaves; eps and min_denom are static.
    metric = utils.WeightedMetric(
        jnp.array(6.0, dtype=jnp.float32),
        jnp.array(3.0, dtype=jnp.float32),
        eps=1e-8,
        min_denom=1.0,
    )
    leaves = jax.tree_util.tree_leaves(metric)
    self.assertLen(leaves, 2)


if __name__ == '__main__':
  absltest.main()
