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

import os
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tunix.experimental.zero_tim_kernel import model_contracts
from tunix.experimental.zero_tim_kernel import padded_matmul
from tunix.experimental.zero_tim_kernel import pallas_matmul
from tunix.experimental.zero_tim_kernel import pallas_norm_matmul
from tunix.experimental.zero_tim_kernel import pallas_rmsnorm
from tunix.experimental.zero_tim_kernel import test_utils

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


class ContinueDecodeTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    patcher = mock.patch.dict(os.environ)
    patcher.start()
    self.addCleanup(patcher.stop)
    os.environ.pop(pallas_norm_matmul.CONTINUE_DECODE_ENV, None)
    self.contract = model_contracts.get_contract("qwen8b")

  def _inputs(self, m):
    rng = np.random.default_rng(m)
    return (
        test_utils.random_bf16(rng, (m, 256), scale=2.0),
        test_utils.random_bf16(rng, (256,)),
        test_utils.random_bf16(rng, (256, 512), scale=0.05),
    )

  @parameterized.parameters(8, 16, 32)
  def test_runs_the_separate_certified_coats(self, m):
    os.environ[pallas_norm_matmul.CONTINUE_DECODE_ENV] = "8"
    x, gamma, y = self._inputs(m)
    out = pallas_norm_matmul.continue_decode_norm_matmul(
        x, gamma, y, epsilon=_EPS, contract=self.contract, interpret=True
    )
    normed = pallas_rmsnorm.rmsnorm(x, gamma, epsilon=_EPS, interpret=True)
    expected = padded_matmul.matmul(
        normed, y, contract=self.contract, interpret=True
    )
    test_utils.assert_bitwise_equal(out, expected)

  def test_is_differentiable_through_the_canonical_coats(self):
    os.environ[pallas_norm_matmul.CONTINUE_DECODE_ENV] = "8"
    x, gamma, y = self._inputs(16)

    def loss(a, g, b):
      out = pallas_norm_matmul.continue_decode_norm_matmul(
          a, g, b, epsilon=_EPS, contract=self.contract, interpret=True
      )
      return jnp.sum(out.astype(jnp.float32))

    def reference_loss(a, g, b):
      normed = pallas_rmsnorm.canonical_rmsnorm(a, g, epsilon=_EPS)
      return jnp.sum(jnp.dot(normed, b, preferred_element_type=jnp.float32))

    grads = jax.grad(loss, argnums=(0, 1, 2))(x, gamma, y)
    expected = jax.grad(reference_loss, argnums=(0, 1, 2))(x, gamma, y)
    for grad, want in zip(grads, expected):
      np.testing.assert_allclose(
          test_utils.as_f32(grad), test_utils.as_f32(want), rtol=5e-2, atol=5e-2
      )

  @parameterized.named_parameters(
      ("env_unset", None, 16),
      ("env_out_of_range", "65", 16),
      ("env_not_a_number", "x", 16),
      ("m_not_multiple_of_8", "8", 12),
      ("m_not_below_block_m", "8", 128),
  )
  def test_rejects(self, env_value, m):
    if env_value is not None:
      os.environ[pallas_norm_matmul.CONTINUE_DECODE_ENV] = env_value
    x, gamma, y = self._inputs(m)
    with self.assertRaises(ValueError):
      pallas_norm_matmul.continue_decode_norm_matmul(
          x, gamma, y, epsilon=_EPS, contract=self.contract, interpret=True
      )


if __name__ == "__main__":
  absltest.main()
