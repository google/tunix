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


class Logprob:

  def __init__(self, logprob, rank=None, decoded_token=None):
    self.logprob = logprob
    self.rank = rank
    self.decoded_token = decoded_token


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

  def test_find_first_non_pad_idx(self):
    data = [
        ([1, 2, 3, 4, 5, 6], 0),
        ([0, 0, 1, 2, 3, 4], 2),
        ([0, 1, 2, 3, 0, 0], 1),
    ]
    for ids, expected in data:
      self.assertEqual(
          utils.find_first_non_pad_idx(jnp.array(ids), 0), expected
      )

  def test_find_first_eos_idx(self):
    data = [
        ([1, 2, 3, 4, 5, -1], 5),
        ([1, 2, -1, 4, -1, 0], 2),
        ([1, 2, 3, 4, 5, 6], 6),
    ]
    for ids, expected in data:
      self.assertEqual(utils.find_first_eos_idx(jnp.array(ids), -1), expected)

  def test_find_last_non_pad_idx(self):
    data = [
        ([1, 2, 3, 4, 5, 6], 5),
        ([1, 2, 3, 0, 0, 0], 2),
        ([0, 1, 2, 3, 0, 0], 3),
    ]
    for ids, expected in data:
      self.assertEqual(utils.find_last_non_pad_idx(jnp.array(ids), 0), expected)

  def test_basic_extraction(self):
    token_ids = [271, 567, 15166]
    logprobs = [
        {271: Logprob(-1.71), 198: Logprob(-0.52)},
        {567: Logprob(-0.37)},
        {15166: Logprob(0.0)},
    ]
    expected = [-1.71, -0.37, 0.0]
    self.assertEqual(
        utils.get_logprobs_from_vllm_output(token_ids, logprobs),
        expected,
    )

  def test_missing_token(self):
    token_ids = [100, 200]
    logprobs = [{101: Logprob(-0.5)}, {200: Logprob(-1.2)}]
    with self.assertRaises(ValueError):
      utils.get_logprobs_from_vllm_output(token_ids, logprobs)


if __name__ == '__main__':
  absltest.main()
