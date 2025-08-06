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
"""Unit tests for PPO helper functions."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import numpy as np
from tunix.rl.ppo import ppo_helpers


def _ref_compute_gae_advantages(
    rewards: jax.Array,
    values: jax.Array,
    gamma: float,
    gae_lambda: float,
    seq_len: int,
) -> jax.Array:
  lastgaelam = 0
  advantages_reversed = []
  for t in reversed(range(seq_len)):
    nextvalues = values[:, t + 1] if t < seq_len - 1 else 0.0
    delta = rewards[:, t] + gamma * nextvalues - values[:, t]
    lastgaelam = delta + gamma * gae_lambda * lastgaelam
    advantages_reversed.append(lastgaelam)

  advantages = np.stack(advantages_reversed[::-1], axis=1)
  return advantages


class PpoHelpersTest(parameterized.TestCase):

  def test_compute_gae_advantages(self):
    bsz, seq_len = 2, 4
    rewards = jax.random.uniform(jax.random.PRNGKey(0), (bsz, seq_len))
    values = jax.random.uniform(jax.random.PRNGKey(1), (bsz, seq_len))

    advantages, _ = ppo_helpers.compute_gae_advantages(
        rewards, values, gamma=0.9, gae_lambda=0.7
    )
    expected_advantages = _ref_compute_gae_advantages(
        rewards, values, gamma=0.9, gae_lambda=0.7, seq_len=seq_len
    )

    np.testing.assert_allclose(
        advantages, expected_advantages, rtol=1e-5, atol=1e-5
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='1d',
          x=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
          mask=np.array([True, True, False, False, True]),
          axis=None,
          expected_mean=np.array(2.666667),
      ),
      dict(
          testcase_name='2d_no_axis',
          x=np.array([[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]]),
          mask=np.array(
              [[True, True, False, False, True], [True, True, True, True, True]]
          ),
          axis=None,
          expected_mean=np.array(6.0),
      ),
      dict(
          testcase_name='2d_with_axis',
          x=np.array([[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]]),
          mask=np.array(
              [[True, True, False, False, True], [True, True, True, True, True]]
          ),
          axis=1,
          expected_mean=np.array([2.666667, 8.0]),
      ),
  )
  def test_masked_mean(self, x, mask, axis, expected_mean):
    computed = ppo_helpers.masked_mean(x, mask, axis)
    np.testing.assert_allclose(computed, expected_mean, rtol=1e-5, atol=1e-5)

  @parameterized.named_parameters(
      dict(
          testcase_name='1d',
          x=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
          mask=np.array([True, True, False, False, True]),
          expected_var=np.array(4.333333),
      ),
      dict(
          testcase_name='2d',
          x=np.array([[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]]),
          mask=np.array(
              [[True, True, False, False, True], [True, True, True, True, True]]
          ),
          expected_var=np.array(10.285715),
      ),
  )
  def test_masked_var(self, x, mask, expected_var):
    computed = ppo_helpers.masked_var(x, mask=mask)
    np.testing.assert_allclose(computed, expected_var, rtol=1e-5, atol=1e-5)


if __name__ == '__main__':
  absltest.main()
