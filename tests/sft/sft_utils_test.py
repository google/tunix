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
  """Tests the deferred-reduction contract of `WeightedMetric`.

  `compute_scale` must reproduce every normalization pattern the loss
  functions rely on — plain ``1/denom``, ``1/clip(denom, min_denom)`` (e.g.
  ``min=1.0`` for token-mean, ``min=1e-6`` for the seq-sum modes) and
  ``1/(denom + eps)`` (the default cross-entropy loss) — while never emitting
  a NaN when the denominator is zero (a NaN scale would poison every gradient).
  """

  @parameterized.named_parameters(
      dict(
          testcase_name='plain_division',
          unreduced_sum=10.0, denominator=5.0, eps=None, min_denom=None,
          expected_scale=0.2, expected_value=2.0,
      ),
      dict(
          testcase_name='zero_denominator_is_safe',
          unreduced_sum=10.0, denominator=0.0, eps=None, min_denom=None,
          expected_scale=0.0, expected_value=0.0,
      ),
      dict(
          testcase_name='min_denom_clamps_small',
          unreduced_sum=1.0, denominator=0.5, eps=None, min_denom=1.0,
          expected_scale=1.0, expected_value=1.0,
      ),
      dict(
          testcase_name='min_denom_clamps_zero',
          unreduced_sum=3.0, denominator=0.0, eps=None, min_denom=1.0,
          expected_scale=1.0, expected_value=3.0,
      ),
      dict(
          testcase_name='min_denom_no_effect_when_larger',
          unreduced_sum=10.0, denominator=4.0, eps=None, min_denom=1.0,
          expected_scale=0.25, expected_value=2.5,
      ),
      dict(
          testcase_name='eps_prevents_zero_division',
          unreduced_sum=5.0, denominator=0.0, eps=1.0, min_denom=None,
          expected_scale=1.0, expected_value=5.0,
      ),
      dict(
          testcase_name='eps_applied_before_min_denom',
          unreduced_sum=5.0, denominator=0.0, eps=1e-8, min_denom=1.0,
          expected_scale=1.0, expected_value=5.0,
      ),
  )
  def test_compute_scale_and_value(
      self, unreduced_sum, denominator, eps, min_denom,
      expected_scale, expected_value,
  ):
    metric = utils.WeightedMetric(
        jnp.asarray(unreduced_sum, dtype=jnp.float32),
        jnp.asarray(denominator, dtype=jnp.float32),
        eps=eps,
        min_denom=min_denom,
    )
    np.testing.assert_allclose(
        metric.compute_scale(), expected_scale, rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        metric.compute(), expected_value, rtol=1e-6, atol=1e-6
    )

  def test_is_jittable_pytree_with_static_bounds(self):
    # WeightedMetric is returned as aux through `nnx.value_and_grad`, so it must
    # be a pytree whose only leaves are the two arrays; `eps`/`min_denom` are
    # static (pytree_node=False) so they neither become tracers nor trigger
    # recompiles.
    metric = utils.WeightedMetric(
        jnp.asarray(6.0, dtype=jnp.float32),
        jnp.asarray(2.0, dtype=jnp.float32),
        min_denom=1.0,
    )
    self.assertLen(jax.tree_util.tree_leaves(metric), 2)
    np.testing.assert_allclose(
        jax.jit(lambda m: m.compute())(metric), 3.0, rtol=1e-6, atol=1e-6
    )


if __name__ == '__main__':
  absltest.main()
