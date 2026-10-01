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
"""Tests for padded_swiglu (CPU, Pallas interpret mode)."""

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np
from tunix.experimental.zero_tim_kernel import model_contracts
from tunix.experimental.zero_tim_kernel import padded_swiglu
from tunix.experimental.zero_tim_kernel import pallas_swiglu
from tunix.experimental.zero_tim_kernel import test_utils


def setUpModule():
  test_utils.configure_cpu()


class PaddedSwigluTest(parameterized.TestCase):

  @parameterized.parameters(
      ("qwen8b", 3072, 3072),
      ("qwen4b", 1216, 1280),
      ("qwen4b_tp4", 2432, 2560),
      ("qwen32b", 3200, 3328),
  )
  def test_feature_extent(self, name, feature, expected):
    contract = model_contracts.get_contract(name)
    self.assertEqual(
        padded_swiglu.padded_feature_extent(feature, contract=contract),
        expected,
    )

  @parameterized.parameters(("qwen8b", 1000), ("qwen4b", 0))
  def test_rejects_unadmitted_feature(self, name, feature):
    contract = model_contracts.get_contract(name)
    with self.assertRaises(ValueError):
      padded_swiglu.padded_feature_extent(feature, contract=contract)

  @parameterized.parameters(1, 8, 64, 127)
  def test_decode_rows_equal_the_128_row_program(self, m):
    contract = model_contracts.get_contract("qwen8b")
    rng = np.random.default_rng(m)
    gate = test_utils.random_bf16(rng, (128, 512), scale=3.0)
    up = test_utils.random_bf16(rng, (128, 512))
    full = padded_swiglu.swiglu(gate, up, contract=contract, interpret=True)
    part = padded_swiglu.swiglu(
        gate[:m], up[:m], contract=contract, interpret=True
    )
    test_utils.assert_bitwise_equal(part, full[:m])

  def test_feature_padding_is_sliced_off(self):
    # qwen4b TP8 MLP width 1216 runs at 1280; real columns are unchanged.
    contract = model_contracts.get_contract("qwen4b")
    rng = np.random.default_rng(1)
    gate = test_utils.random_bf16(rng, (16, 1216), scale=3.0)
    up = test_utils.random_bf16(rng, (16, 1216))
    out = padded_swiglu.swiglu(gate, up, contract=contract, interpret=True)
    self.assertEqual(out.shape, (16, 1216))
    pad = ((0, 112), (0, 64))
    expected = pallas_swiglu.swiglu(
        jnp.pad(gate, pad), jnp.pad(up, pad), interpret=True
    )[:16, :1216]
    test_utils.assert_bitwise_equal(out, expected)

  def test_rejects_mismatched_or_empty_inputs(self):
    contract = model_contracts.get_contract("qwen8b")
    with self.assertRaises(ValueError):
      padded_swiglu.swiglu(
          jnp.zeros((8, 256), jnp.bfloat16),
          jnp.zeros((8, 512), jnp.bfloat16),
          contract=contract,
          interpret=True,
      )
    with self.assertRaises(ValueError):
      padded_swiglu.swiglu(
          jnp.zeros((0, 256), jnp.bfloat16),
          jnp.zeros((0, 256), jnp.bfloat16),
          contract=contract,
          interpret=True,
      )


if __name__ == "__main__":
  absltest.main()
