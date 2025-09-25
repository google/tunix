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
from tunix.rl.grpo import grpo_helpers

jax.config.update("jax_threefry_partitionable", False)


class GRPOHelpersTest(absltest.TestCase):

  def test_compute_advantages(self):
    rng = jax.random.PRNGKey(0)
    rewards = jax.random.uniform(rng, shape=(1, 6))
    advantages = grpo_helpers.compute_advantages(rewards, num_generations=3)
    expected_value = jnp.array(
        [[0.307407, -1.117304, 0.809897, 1.094044, -0.22857, -0.865474]]
    )
    np.testing.assert_allclose(advantages, expected_value, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
  absltest.main()
