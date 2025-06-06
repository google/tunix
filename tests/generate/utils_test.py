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
from tunix.generate import utils


class UtilsTest(absltest.TestCase):

  def test_compute_attention_mask(self):
    # Check that the input mask is correctly applied when total sampling steps
    # is lower than the max cache length.
    input_mask = jnp.array([[1, 1, 0, 0, 0], [1, 1, 0, 1, 0]], dtype=jnp.bool_)
    seq_len = 8
    time_step = 4
    attn_mask = utils.compute_attention_masks(time_step, seq_len, input_mask)
    expected_attn_mask = jnp.array(
        [[0, 0, 1, 1, 1, 0, 0, 0], [0, 0, 1, 0, 1, 0, 0, 0]], dtype=jnp.bool_
    )

    self.assertTrue((attn_mask.squeeze(1) == expected_attn_mask).all())

    # Check that the input mask is correctly applied when total sampling steps
    # is *longer* than the max cache length.
    seq_len = 4
    time_step = 4
    attn_mask = utils.compute_attention_masks(time_step, seq_len, input_mask)
    expected_attn_mask = jnp.array(
        [[0, 1, 1, 1], [0, 1, 0, 1]], dtype=jnp.bool_
    )

    self.assertTrue((attn_mask.squeeze(1) == expected_attn_mask).all())

  def test_make_causal_attn_mask(self):
    input_mask = jnp.array([[0, 1, 1, 0], [1, 1, 1, 0]])
    attn_mask = utils.make_causal_attn_mask(input_mask, 5)
    expected = jnp.array([
        [
            [0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 1, 1, 0, 0],
        ],
        [
            [1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0],
        ],
    ])
    np.testing.assert_array_equal(attn_mask, expected)

  def test_next_power_of_2(self):
    self.assertEqual(utils.next_power_of_2(0), 1)
    self.assertEqual(utils.next_power_of_2(2), 2)
    self.assertEqual(utils.next_power_of_2(3), 4)
    self.assertEqual(utils.next_power_of_2(4), 4)
    self.assertEqual(utils.next_power_of_2(5), 8)


if __name__ == '__main__':
  absltest.main()
