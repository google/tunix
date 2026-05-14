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
from absl.testing import parameterized
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


class WeightedMetricTest(parameterized.TestCase):
  """Unit tests for `WeightedMetric` — the building block of `LossOutput`.

  The class implements deferred reduction: it holds an unreduced sum plus a
  weight (denominator) and only divides at `compute()` / `compute_scale()`
  time. This shape is essential to the gradient-accumulation contract — the
  caller can sum unreduced losses + denominators across micro-batches and
  divide once at the end, which is the only way to obtain
  ``grad(Σ_i sum_loss_i / Σ_i denom_i)`` (the correct full-batch gradient)
  when ``denom_i`` differs across micro-batches.
  """

  def test_compute_scale_basic(self):
    """No bounds: scale is the plain 1/denom."""
    m = utils.WeightedMetric(jnp.asarray(2.0), jnp.asarray(4.0))
    np.testing.assert_allclose(m.compute_scale(), 0.25, rtol=1e-7)
    np.testing.assert_allclose(m.compute(), 0.5, rtol=1e-7)

  @parameterized.named_parameters(
      dict(testcase_name='positive_eps', eps=1e-8, denom=4.0,
           expected_scale=1.0 / (4.0 + 1e-8)),
      dict(testcase_name='larger_eps', eps=0.5, denom=2.0,
           expected_scale=1.0 / 2.5),
      dict(testcase_name='eps_dominates_small_denom', eps=1.0, denom=0.01,
           expected_scale=1.0 / 1.01),
  )
  def test_compute_scale_with_eps(self, eps, denom, expected_scale):
    """`eps` is added to the denominator before inversion."""
    m = utils.WeightedMetric(jnp.asarray(1.0), jnp.asarray(denom), eps=eps)
    np.testing.assert_allclose(m.compute_scale(), expected_scale, rtol=1e-6)

  @parameterized.named_parameters(
      dict(testcase_name='min_dominates', min_denom=2.0, denom=1.0,
           expected_scale=0.5),
      dict(testcase_name='denom_dominates', min_denom=2.0, denom=8.0,
           expected_scale=1.0 / 8.0),
      dict(testcase_name='equal', min_denom=4.0, denom=4.0,
           expected_scale=0.25),
  )
  def test_compute_scale_with_min_denom(self, min_denom, denom,
                                        expected_scale):
    """`min_denom` clips the denominator from below before inversion."""
    m = utils.WeightedMetric(
        jnp.asarray(1.0), jnp.asarray(denom), min_denom=min_denom
    )
    np.testing.assert_allclose(m.compute_scale(), expected_scale, rtol=1e-7)

  def test_compute_scale_eps_then_min_denom_ordering(self):
    """When both bounds are set, `eps` is added BEFORE the `max(.., min_denom)`.

    Locking in the documented ordering — flipping it would change the value
    when `denom + eps < min_denom < denom`, an edge case the production
    grad-accumulation path may exercise with tiny float32 denominators.
    """
    eps = 0.5
    min_denom = 2.0
    denom = 1.0
    # eps-first: (1.0 + 0.5) = 1.5, max(1.5, 2.0) = 2.0 -> scale 0.5.
    # min-first: max(1.0, 2.0) = 2.0, + 0.5 = 2.5 -> scale 0.4.
    m = utils.WeightedMetric(
        jnp.asarray(1.0), jnp.asarray(denom), eps=eps, min_denom=min_denom
    )
    np.testing.assert_allclose(m.compute_scale(), 0.5, rtol=1e-7)

  def test_compute_scale_zero_denom_no_bounds(self):
    """Pure-zero denominator with no bounds yields scale 0 (safe division).

    This is the unreduced-loss contract for sequence-packing micro-batches:
    if a row is fully masked, both `unreduced_sum` and `denominator` are
    zero, and `compute()` must return 0 (not NaN) so the row neither
    pollutes the loss nor poisons gradients through downstream sums.
    """
    m = utils.WeightedMetric(jnp.asarray(0.0), jnp.asarray(0.0))
    np.testing.assert_allclose(m.compute_scale(), 0.0)
    np.testing.assert_allclose(m.compute(), 0.0)
    self.assertTrue(jnp.isfinite(m.compute_scale()))

  def test_compute_scale_zero_denom_with_min_denom(self):
    """Zero denominator + `min_denom` clamps to `1/min_denom` (not zero).

    Distinct from the no-bounds case: callers using `min_denom=1.0` rely
    on getting a well-defined divisor of 1, not the "safe division"
    sentinel. This is the path used by `aggregate_loss("token-mean")` and
    the loss-function aux metrics.
    """
    m = utils.WeightedMetric(
        jnp.asarray(0.0), jnp.asarray(0.0), min_denom=1.0
    )
    np.testing.assert_allclose(m.compute_scale(), 1.0)

  def test_compute_value_equals_sum_times_scale(self):
    """Concrete check on `compute()` = `unreduced_sum * compute_scale()`."""
    m = utils.WeightedMetric(
        jnp.asarray(7.5), jnp.asarray(0.0), eps=1e-3, min_denom=2.0
    )
    np.testing.assert_allclose(m.compute(), 7.5 * 0.5, rtol=1e-6)

  def test_pytree_round_trip(self):
    """`WeightedMetric` is a flax struct-dataclass and must survive `tree_map`.

    The trainer threads `WeightedMetric` through `nnx.value_and_grad`'s
    `has_aux=True` channel, which requires the auxiliary value to be a
    valid pytree. Confirm `tree_map` round-trips both fields and that
    `eps`/`min_denom` (static, `pytree_node=False`) survive the trip
    unchanged.
    """
    m = utils.WeightedMetric(
        jnp.asarray(3.0), jnp.asarray(6.0), eps=1e-4, min_denom=1.0
    )
    doubled = jax.tree.map(lambda x: x * 2.0, m)
    np.testing.assert_allclose(doubled.unreduced_sum, 6.0)
    np.testing.assert_allclose(doubled.denominator, 12.0)
    self.assertEqual(doubled.eps, 1e-4)
    self.assertEqual(doubled.min_denom, 1.0)
    np.testing.assert_allclose(doubled.compute(), 6.0 / (12.0 + 1e-4),
                               rtol=1e-6)


class LossOutputTest(absltest.TestCase):
  """Unit tests for `LossOutput`."""

  def test_lossoutput_round_trips_through_tree_map(self):
    """`LossOutput` must be a valid pytree (used as the aux channel of grad)."""
    primary = utils.WeightedMetric(jnp.asarray(2.0), jnp.asarray(4.0))
    aux = {
        'foo': utils.WeightedMetric(
            jnp.asarray(10.0), jnp.asarray(5.0), min_denom=1.0
        ),
        'bar': utils.WeightedMetric(jnp.asarray(6.0), jnp.asarray(2.0)),
    }
    out = utils.LossOutput(primary_loss=primary, aux_metrics=aux)
    doubled = jax.tree.map(lambda x: x * 2.0, out)
    np.testing.assert_allclose(doubled.primary_loss.compute(),
                               4.0 / 8.0, rtol=1e-7)
    np.testing.assert_allclose(doubled.aux_metrics['foo'].compute(),
                               20.0 / 10.0, rtol=1e-7)
    np.testing.assert_allclose(doubled.aux_metrics['bar'].compute(),
                               12.0 / 4.0, rtol=1e-7)


if __name__ == '__main__':
  absltest.main()
