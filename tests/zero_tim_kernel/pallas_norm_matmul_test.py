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
"""Tests for pallas_norm_matmul (CPU, Pallas interpret mode)."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from jax.experimental import pallas as pl
import jax.numpy as jnp
import numpy as np
from tunix.zero_tim_kernel import pallas_matmul
from tunix.zero_tim_kernel import pallas_norm_matmul
from tunix.zero_tim_kernel import pallas_rmsnorm
from tunix.zero_tim_kernel import test_utils

_EPS = 1e-6


def setUpModule():
  test_utils.configure_cpu()


def _two_kernel_chain(x, gamma, y, **tiles):
  normed = pallas_rmsnorm.rmsnorm(x, gamma, epsilon=_EPS, interpret=True)
  return pallas_matmul.matmul(normed, y, interpret=True, **tiles)


class NormMatmulTest(parameterized.TestCase):

  def test_fused_equals_two_kernel_chain_at_equal_tiles(self):
    # M=128/N=256 keep the policy at the fused kernel's (128, 256, 256), so
    # both programs run identical per-tile dots on random gamma/y.  Exact
    # rows keep the sum of squares order-free (the fused kernel normalizes
    # 128 rows per block, the standalone kernel 8).
    rng = np.random.default_rng(0)
    x = test_utils.exact_bf16(rng, (128, 512))
    gamma = test_utils.random_bf16(rng, (512,))
    y = test_utils.random_bf16(rng, (512, 256), scale=0.05)
    fused = pallas_norm_matmul.norm_matmul(
        x, gamma, y, epsilon=_EPS, interpret=True
    )
    test_utils.assert_bitwise_equal(fused, _two_kernel_chain(x, gamma, y))

  @parameterized.parameters((256, 512, 1024), (384, 256, 512))
  def test_fused_equals_two_kernel_chain_on_exact_operands(self, m, k, n):
    # With x in {0, +-1/8, +-1/4} and gamma = 1 every normalized value is
    # +-bf16(inv) * 2^-j, so the projection sums are exact and the tile
    # shapes (fused 128/256 vs policy 256/1024) cannot change a bit.
    rng = np.random.default_rng(m + n)
    x = test_utils.exact_bf16(rng, (m, k))
    gamma = jnp.ones((k,), jnp.bfloat16)
    y = test_utils.exact_bf16(rng, (k, n))
    fused = pallas_norm_matmul.norm_matmul(
        x, gamma, y, epsilon=_EPS, interpret=True
    )
    test_utils.assert_bitwise_equal(fused, _two_kernel_chain(x, gamma, y))

  def test_wide_block_n_promotion_and_column_grouping_are_bitwise_identical(
      self,
  ):
    rng = np.random.default_rng(42)
    x = test_utils.random_bf16(rng, (16, 512))
    gamma = test_utils.random_bf16(rng, (512,))
    # N = 6144 exceeds MAX_GROUP_N = 3072: block_n promotes 256 -> 1024 and
    # group_n caps at 3072 -> 2 parallel column groups of 3072 each.
    y = test_utils.random_bf16(rng, (512, 6144), scale=0.05)
    with mock.patch.object(pl, "pallas_call", wraps=pl.pallas_call) as spy:
      fused = pallas_norm_matmul.norm_matmul(
          x, gamma, y, epsilon=_EPS, interpret=True, block_m=16
      )
    self.assertLen(spy.call_args_list, 1)
    call = spy.call_args_list[0]
    self.assertEqual(
        call.kwargs["name"],
        "canon_norm_matmul_bm16_bn1024_gn3072_bk256_f512",
    )
    self.assertEqual(tuple(call.kwargs["grid_spec"].grid), (1, 2, 2))
    test_utils.assert_bitwise_equal(
        fused, _two_kernel_chain(x, gamma, y, block_m=16)
    )

  def test_column_group_width(self):
    self.assertEqual(pallas_norm_matmul.column_group_width(1536, 512), 1536)
    self.assertEqual(pallas_norm_matmul.column_group_width(2560, 1280), 2560)
    self.assertEqual(pallas_norm_matmul.column_group_width(6144, 1024), 3072)
    self.assertEqual(pallas_norm_matmul.column_group_width(5120, 1024), 1024)
    self.assertEqual(pallas_norm_matmul.column_group_width(24576, 1024), 3072)
    with self.assertRaises(ValueError):
      pallas_norm_matmul.column_group_width(1024, 300)

  @parameterized.named_parameters(
      ("rank", (128, 256, 1), (256,), (256, 256), {}),
      ("gamma_mismatch", (128, 256), (128,), (256, 256), {}),
      ("m_not_tiled", (64, 256), (256,), (256, 256), {}),
      ("n_not_tiled", (128, 256), (256,), (256, 128), {}),
  )
  def test_rejects(self, x_shape, gamma_shape, y_shape, kwargs):
    with self.assertRaises(ValueError):
      pallas_norm_matmul.norm_matmul(
          jnp.zeros(x_shape, jnp.bfloat16),
          jnp.ones(gamma_shape, jnp.bfloat16),
          jnp.zeros(y_shape, jnp.bfloat16),
          epsilon=_EPS,
          interpret=True,
          **kwargs,
      )

  def test_rejects_bad_dtype_and_epsilon(self):
    x = jnp.zeros((128, 256), jnp.bfloat16)
    gamma = jnp.ones((256,), jnp.bfloat16)
    y = jnp.zeros((256, 256), jnp.bfloat16)
    with self.assertRaisesRegex(ValueError, "bf16"):
      pallas_norm_matmul.norm_matmul(
          x, gamma.astype(jnp.float32), y, epsilon=_EPS, interpret=True
      )
    with self.assertRaisesRegex(ValueError, "epsilon"):
      pallas_norm_matmul.norm_matmul(x, gamma, y, epsilon=0.0, interpret=True)


if __name__ == "__main__":
  absltest.main()
