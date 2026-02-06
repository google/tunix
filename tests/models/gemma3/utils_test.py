# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np
from tunix.models.gemma3 import utils
from tunix.models.gemma3 import vision


class UtilsTest(parameterized.TestCase):

  def test_get_positions_and_attention_mask_not_multimodal(self):
    tokens = jnp.array([[1, 2, 3, utils._PADDING_ID, utils._PADDING_ID]])
    result = utils.get_positions_and_attention_mask(tokens)
    positions = result['positions']
    attention_mask = result['attention_mask']

    expected_positions = jnp.array([[0, 1, 2, 2, 2]])
    expected_attention_mask = jnp.array(
        [[
            [1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0],
        ]],
        dtype=jnp.bool_,
    )
    np.testing.assert_array_equal(positions, expected_positions)
    np.testing.assert_array_equal(attention_mask, expected_attention_mask)

  def test_get_positions_and_attention_mask_multimodal(self):
    tokens = jnp.array([[
        1,
        2,
        vision.TOKEN_PLACEHOLDER,
        vision.TOKEN_PLACEHOLDER,
        3,
        utils._PADDING_ID,
    ]])
    result = utils.get_positions_and_attention_mask(tokens)
    positions = result['positions']
    attention_mask = result['attention_mask']

    expected_positions = jnp.array([[0, 1, 2, 3, 4, 4]])
    expected_attention_mask = jnp.array(
        [[
            [1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 0],
        ]],
        dtype=jnp.bool_,
    )
    np.testing.assert_array_equal(positions, expected_positions)
    np.testing.assert_array_equal(attention_mask, expected_attention_mask)

  def test_get_positions_and_attention_mask_precomputed_mask(self):
    tokens = jnp.array([[1, 2, 3, utils._PADDING_ID, utils._PADDING_ID]])
    inputs_mask = jnp.array([[1, 0, 1, 0, 0]])
    result = utils.get_positions_and_attention_mask(
        tokens, inputs_mask=inputs_mask
    )
    positions = result['positions']
    attention_mask = result['attention_mask']

    expected_positions = jnp.array([[0, 0, 1, 1, 1]])
    expected_attention_mask = jnp.array(
        [[
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 0, 1, 0, 0],
            [1, 0, 1, 0, 0],
            [1, 0, 1, 0, 0],
        ]],
        dtype=jnp.bool_,
    )
    np.testing.assert_array_equal(positions, expected_positions)
    np.testing.assert_array_equal(attention_mask, expected_attention_mask)


if __name__ == '__main__':
  absltest.main()
