# Copyright 2026 Google LLC
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

"""Pure-NumPy unit tests for `tunix.rl.advantages`."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from tunix.rl import advantages
from tunix.rl import function_registry


class AdvantagesTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("grpo", "grpo", advantages.compute_advantages),
      ("grpo_loo", "grpo-loo", advantages.compute_grpo_loo_advantages),
      ("rloo", "rloo", advantages.compute_rloo_advantages),
      ("drgrpo", "drgrpo", advantages.compute_drgrpo_advantages),
  )
  def test_registered_in_function_registry_and_returns_float32_ndarray(
      self, name, fn
  ):
    self.assertIs(function_registry.get_advantage_estimator(name), fn)
    rewards = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    out = fn(rewards, num_generations=4)
    self.assertIsInstance(out, np.ndarray)
    self.assertEqual(out.dtype, np.float32)
    self.assertEqual(out.shape, (4,))

  def test_rloo_and_drgrpo_values_and_valid_mask(self):
    rewards = np.array([1.0, 2.0, 3.0, 0.0], dtype=np.float32)
    valid_mask = np.array([True, True, True, False])

    rloo = advantages.compute_rloo_advantages(
        rewards, num_generations=4, valid_mask=valid_mask
    )
    np.testing.assert_allclose(rloo, [-1.5, 0.0, 1.5, 0.0], rtol=1e-5)

    drgrpo = advantages.compute_drgrpo_advantages(
        rewards, num_generations=4, valid_mask=valid_mask
    )
    np.testing.assert_allclose(drgrpo, [-1.0, 0.0, 1.0, 0.0], rtol=1e-5)

  def test_grpo_and_grpo_loo_relation(self):
    rewards = np.array([1.0, 2.0, 4.0], dtype=np.float32)
    grpo = advantages.compute_advantages(rewards, num_generations=3)
    grpo_loo = advantages.compute_grpo_loo_advantages(
        rewards, num_generations=3
    )
    # For k = 3 valid samples, A_i^{grpo-loo} = (k / (k - 1)) * A_i^{grpo}.
    np.testing.assert_allclose(grpo_loo, 1.5 * grpo, rtol=1e-5)

  @parameterized.parameters(
      (advantages.compute_advantages,),
      (advantages.compute_grpo_loo_advantages,),
      (advantages.compute_rloo_advantages,),
      (advantages.compute_drgrpo_advantages,),
  )
  def test_degenerate_group_and_non_finite_rewards_safe(self, estimator):
    rewards = np.array([1.0, np.nan, 3.0, np.inf], dtype=np.float32)
    valid_mask = np.array([True, True, True, False])
    out = estimator(rewards, num_generations=4, valid_mask=valid_mask)
    self.assertIsInstance(out, np.ndarray)
    self.assertEqual(out.dtype, np.float32)
    self.assertTrue(bool(np.all(np.isfinite(out))))
    self.assertEqual(float(out[1]), 0.0)
    self.assertEqual(float(out[3]), 0.0)

    single_valid = estimator(
        np.array([5.0, 1.0, 2.0, 3.0], dtype=np.float32),
        num_generations=4,
        valid_mask=np.array([True, False, False, False]),
    )
    np.testing.assert_array_equal(single_valid, np.zeros(4, dtype=np.float32))

  def test_low_num_generations_returns_zero_float32_ndarray(self):
    for fn in (
        advantages.compute_grpo_loo_advantages,
        advantages.compute_rloo_advantages,
    ):
      out = fn(np.array([[1.0, 2.0]], dtype=np.float64), num_generations=1)
      self.assertIsInstance(out, np.ndarray)
      self.assertEqual(out.dtype, np.float32)
      np.testing.assert_array_equal(out, np.zeros((1, 2), dtype=np.float32))


if __name__ == "__main__":
  absltest.main()
