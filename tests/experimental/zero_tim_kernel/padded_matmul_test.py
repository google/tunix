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
"""Tests for padded_matmul (CPU, Pallas interpret mode)."""

import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np
from tunix.experimental.zero_tim_kernel import model_contracts
from tunix.experimental.zero_tim_kernel import padded_matmul
from tunix.experimental.zero_tim_kernel import test_utils


def setUpModule():
  test_utils.configure_cpu()


def _contract(name):
  return model_contracts.get_contract(name)


class ExtentsTest(parameterized.TestCase):

  @parameterized.parameters(
      ("qwen8b", 4096, 37984, (4096, 38144)),
      ("qwen8b", 4096, 3072, (4096, 3072)),
      ("qwen4b", 1216, 2560, (1280, 2560)),
      ("qwen4b", 2560, 1216, (2560, 1280)),
      ("qwen4b_tp4", 2432, 2560, (2432, 2560)),
      ("qwen4b_tp4", 2432, 2560, (2560, 2560), 256),
      ("qwen8b_tp8", 4096, 18992, (4096, 19200)),
  )
  def test_padded_extents(self, name, k, n, expected, block_k=None):
    self.assertEqual(
        padded_matmul.padded_matmul_extents(
            k, n, contract=_contract(name), block_k=block_k
        ),
        expected,
    )

  @parameterized.parameters(
      ("qwen8b", 4096, 1000),  # N not admitted.
      ("qwen8b", 1000, 4096),  # K not admitted.
      ("qwen4b", 0, 128),  # Non-positive width.
  )
  def test_rejects_unadmitted_widths(self, name, k, n):
    with self.assertRaises(ValueError):
      padded_matmul.padded_matmul_extents(k, n, contract=_contract(name))

  def test_wide_n_tiles(self):
    qwen4b = _contract("qwen4b")
    qwen4b_tp4 = _contract("qwen4b_tp4")
    tiles = qwen4b.tiles
    # 4B MLP widths get a wide tile on their contract-padded width.
    self.assertEqual(
        padded_matmul.wide_n_tiles(
            128, 2560, 1216, tiles, qwen4b.matmul_n_padding
        ),
        (1280, 1280),
    )
    self.assertEqual(
        padded_matmul.wide_n_tiles(
            128, 2560, 2432, tiles, qwen4b_tp4.matmul_n_padding
        ),
        (2560, 1280),
    )
    # A padded lm_head width no wide candidate divides keeps the contract.
    self.assertIsNone(
        padded_matmul.wide_n_tiles(
            128, 2560, 37984, tiles, qwen4b_tp4.matmul_n_padding
        )
    )
    # Widths the policy already tiles, explicit tiles, and unmapped widths.
    self.assertIsNone(
        padded_matmul.wide_n_tiles(
            128, 2560, 2560, tiles, qwen4b.matmul_n_padding
        )
    )
    self.assertIsNone(
        padded_matmul.wide_n_tiles(
            128, 2560, 1216, (128, 128, 256), qwen4b.matmul_n_padding
        )
    )
    self.assertIsNone(padded_matmul.wide_n_tiles(128, 2560, 1216, tiles, {}))


class PaddedMatmulTest(parameterized.TestCase):

  @parameterized.parameters(1, 7, 8, 100, 127)
  def test_decode_rows_equal_the_128_row_program(self, m):
    # Every M <= 128 runs the identical 128-row program, so a decode row is
    # bitwise the same row inside a 128-row prefill/scoring batch.
    contract = _contract("qwen8b")
    rng = np.random.default_rng(m)
    x = test_utils.random_bf16(rng, (128, 512))
    y = test_utils.random_bf16(rng, (512, 1024), scale=0.05)
    full = padded_matmul.matmul(x, y, contract=contract, interpret=True)
    part = padded_matmul.matmul(x[:m], y, contract=contract, interpret=True)
    test_utils.assert_bitwise_equal(part, full[:m])

  def test_m_padding_to_256_rows(self):
    contract = _contract("qwen8b")
    rng = np.random.default_rng(1)
    x = test_utils.exact_bf16(rng, (200, 256))
    y = test_utils.exact_bf16(rng, (256, 512))
    out = padded_matmul.matmul(x, y, contract=contract, interpret=True)
    self.assertEqual(out.shape, (200, 512))
    test_utils.assert_bitwise_equal(
        out, test_utils.exact_matmul_reference(x, y)
    )

  def test_contract_k_and_n_padding(self):
    contract = dataclasses.replace(
        _contract("qwen4b"),
        matmul_k_padding={200: 256},
        matmul_n_padding={200: 256},
    )
    rng = np.random.default_rng(2)
    x = test_utils.exact_bf16(rng, (16, 200))
    y = test_utils.exact_bf16(rng, (200, 200))
    out = padded_matmul.matmul(x, y, contract=contract, interpret=True)
    self.assertEqual(out.shape, (16, 200))
    test_utils.assert_bitwise_equal(
        out, test_utils.exact_matmul_reference(x, y)
    )

  def test_wide_padded_width_is_bitwise_neutral(self):
    # qwen4b gate/up: N=1216 -> 1280 with a 1280-wide tile.
    contract = _contract("qwen4b")
    rng = np.random.default_rng(3)
    x = test_utils.exact_bf16(rng, (8, 256))
    y = test_utils.exact_bf16(rng, (256, 1216))
    out = padded_matmul.matmul(x, y, contract=contract, interpret=True)
    self.assertEqual(out.shape, (8, 1216))
    test_utils.assert_bitwise_equal(
        out, test_utils.exact_matmul_reference(x, y)
    )

  def test_rejects(self):
    contract = _contract("qwen8b")
    with self.assertRaises(ValueError):
      padded_matmul.matmul(
          jnp.zeros((8, 1000), jnp.bfloat16),
          jnp.zeros((1000, 256), jnp.bfloat16),
          contract=contract,
          interpret=True,
      )
    with self.assertRaises(ValueError):
      padded_matmul.matmul(
          jnp.zeros((8, 256), jnp.bfloat16),
          jnp.zeros((512, 256), jnp.bfloat16),
          contract=contract,
          interpret=True,
      )
    with self.assertRaises(ValueError):
      padded_matmul.matmul(
          jnp.zeros((0, 256), jnp.bfloat16),
          jnp.zeros((256, 256), jnp.bfloat16),
          contract=contract,
          interpret=True,
      )


if __name__ == "__main__":
  absltest.main()
