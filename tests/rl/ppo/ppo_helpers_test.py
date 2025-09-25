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
import jax.numpy as jnp
import numpy as np
from tunix.rl.ppo import ppo_helpers


def _ref_compute_gae_advantages(
    token_level_rewards,
    values,
    response_mask,
    gamma,
    lam,
):
  """Verl implementation of GAE advantages computation."""

  def masked_sum(values, mask, axis=None):
    return (values * mask).sum(axis=axis)

  def masked_mean(values, mask, axis=None):
    s = masked_sum(values, mask, axis)
    return s / (mask.sum(axis=axis) + 1e-8)

  def masked_var(values, mask):
    mean = masked_mean(values, mask)
    centered_values = values - mean
    variance = masked_mean(centered_values**2, mask)
    mask_sum = mask.sum()

    bessel_correction = mask_sum / (mask_sum - 1)
    variance = variance * bessel_correction
    return variance

  def masked_whiten(values, mask):
    mean, var = masked_mean(values, mask), masked_var(values, mask)
    whitened = (values - mean) / np.sqrt(var + 1e-8)
    return whitened

  nextvalues = 0
  lastgaelam = 0
  advantages_reversed = []
  gen_len = token_level_rewards.shape[-1]

  for t in reversed(range(gen_len)):
    delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
    lastgaelam_ = delta + gamma * lam * lastgaelam

    # skip values and TD-error on observation tokens
    nextvalues = (
        values[:, t] * response_mask[:, t]
        + (1 - response_mask[:, t]) * nextvalues
    )
    lastgaelam = (
        lastgaelam_ * response_mask[:, t]
        + (1 - response_mask[:, t]) * lastgaelam
    )

    advantages_reversed.append(lastgaelam)
  advantages = np.stack(advantages_reversed[::-1], axis=1)

  returns = advantages + values
  advantages = masked_whiten(advantages, response_mask)
  return advantages, returns


class PPOHelpersTest(parameterized.TestCase):

  def test_compute_gae_advantages(self):
    bsz, seq_len = 3, 10
    rewards = jax.random.uniform(jax.random.PRNGKey(0), (bsz, seq_len))
    values = jax.random.uniform(jax.random.PRNGKey(1), (bsz, seq_len))
    response_mask = jnp.array([
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],  # masking at the end
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # no masking
        [1, 1, 0, 0, 1, 1, 0, 1, 0, 0],  # arbitrary mask
    ])

    advantages, returns = ppo_helpers.compute_gae_advantages(
        np.array(rewards),
        np.array(values),
        np.array(response_mask),
        gamma=0.9,
        gae_lambda=0.7,
    )
    expected_advantages, expected_returns = _ref_compute_gae_advantages(
        rewards, values, response_mask, gamma=0.9, lam=0.7
    )

    np.testing.assert_allclose(
        advantages, expected_advantages, rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(returns, expected_returns, rtol=1e-5, atol=1e-5)

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
