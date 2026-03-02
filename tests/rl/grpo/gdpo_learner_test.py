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
from flax import nnx
import jax.numpy as jnp
from tunix.rl import function_registry as fr
from tunix.rl.grpo import gdpo_learner as gdpo_lib
from tunix.tests import test_common as tc
import numpy as np


class GDPOlearnerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

  def test_compute_advantages(self):
    rewards = jnp.array([[0, 2], [0, 1], [0, 0], [1, 1], [0, 1], [2, 0]])
    advantages = gdpo_lib.compute_advantages(rewards, num_generations=3)
    expected_array = jnp.array([
        1.5380104e00,
        2.4953168e-08,
        -1.5380104e00,
        8.8789117e-01,
        -6.5011925e-01,
        -2.3777203e-01,
    ])
    np.testing.assert_allclose(advantages, expected_array, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
  absltest.main()
