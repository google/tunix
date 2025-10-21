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


if __name__ == '__main__':
  absltest.main()
