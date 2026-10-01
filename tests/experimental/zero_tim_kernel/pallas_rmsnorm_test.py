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
"""Tests for pallas_rmsnorm (CPU, Pallas interpret mode)."""

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np
from tunix.experimental.zero_tim_kernel import pallas_rmsnorm
from tunix.experimental.zero_tim_kernel import test_utils

_EPS = 1e-6


def setUpModule():
  test_utils.configure_cpu()


def _float64_rmsnorm(x, weight, epsilon):
  xf = np.asarray(x, np.float64)
  inv = 1.0 / np.sqrt(np.mean(xf * xf, axis=-1, keepdims=True) + epsilon)
  return xf * inv * np.asarray(weight, np.float64)


class RowTileTest(parameterized.TestCase):

  @parameterized.parameters(
      (256, 4096, 256),
      (512, 2048, 256),
      (256, 5120, pallas_rmsnorm.BM),
      (264, 128, pallas_rmsnorm.BM),
      (8, 2560, pallas_rmsnorm.BM),
  )
  def test_row_tile(self, m, f, expected):
    self.assertEqual(pallas_rmsnorm.row_tile(m, f), expected)

  @parameterized.named_parameters(
      ("x_rank", (8, 128, 1), (128,)),
      ("weight_rank", (8, 128), (1, 128)),
      ("feature_mismatch", (8, 128), (256,)),
      ("m_not_tiled", (12, 128), (128,)),
      ("f_not_tiled", (8, 200), (200,)),
  )
  def test_validate_shape_rejects(self, x_shape, weight_shape):
    with self.assertRaises(ValueError):
      pallas_rmsnorm.validate_shape(x_shape, weight_shape)


class RmsnormTest(parameterized.TestCase):

  @parameterized.parameters((8, 128), (16, 384), (256, 256), (512, 1024))
  def test_kernel_matches_declared_semantic_on_exact_rows(self, m, f):
    # Exact rows make the sum of squares order-free, so the Pallas program
    # and the declared canonical semantic must agree bit for bit on CPU.
    rng = np.random.default_rng(m * f)
    x = test_utils.exact_bf16(rng, (m, f))
    weight = test_utils.random_bf16(rng, (f,))
    out = pallas_rmsnorm.rmsnorm(x, weight, epsilon=_EPS, interpret=True)
    expected = pallas_rmsnorm.canonical_rmsnorm(x, weight, epsilon=_EPS)
    self.assertEqual(out.dtype, jnp.bfloat16)
    test_utils.assert_bitwise_equal(out, expected)

  def test_random_rows_close_to_float64(self):
    rng = np.random.default_rng(1)
    x = test_utils.random_bf16(rng, (64, 2048), scale=3.0)
    weight = test_utils.random_bf16(rng, (2048,))
    out = pallas_rmsnorm.rmsnorm(x, weight, epsilon=_EPS, interpret=True)
    np.testing.assert_allclose(
        test_utils.as_f32(out),
        _float64_rmsnorm(x, weight, _EPS),
        rtol=1e-2,
        atol=1e-2,
    )

  @parameterized.parameters((256, 512), (8, 16))
  def test_rows_do_not_depend_on_batch_size(self, small, large):
    # Both batch sizes use the same row tile, so each row is bitwise equal.
    rng = np.random.default_rng(2)
    x = test_utils.random_bf16(rng, (large, 1024), scale=2.0)
    weight = test_utils.random_bf16(rng, (1024,))
    self.assertEqual(
        pallas_rmsnorm.row_tile(small, 1024),
        pallas_rmsnorm.row_tile(large, 1024),
    )
    full = pallas_rmsnorm.rmsnorm(x, weight, epsilon=_EPS, interpret=True)
    for start in range(0, large, small):
      part = pallas_rmsnorm.rmsnorm(
          x[start : start + small], weight, epsilon=_EPS, interpret=True
      )
      test_utils.assert_bitwise_equal(full[start : start + small], part)

  def test_rejects_non_bf16_and_bad_epsilon(self):
    x = jnp.zeros((8, 128), jnp.bfloat16)
    weight = jnp.ones((128,), jnp.bfloat16)
    with self.assertRaisesRegex(ValueError, "bf16"):
      pallas_rmsnorm.rmsnorm(
          x.astype(jnp.float32), weight, epsilon=_EPS, interpret=True
      )
    with self.assertRaisesRegex(ValueError, "epsilon"):
      pallas_rmsnorm.rmsnorm(x, weight, epsilon=0.0, interpret=True)
    with self.assertRaisesRegex(ValueError, "epsilon"):
      pallas_rmsnorm.canonical_rmsnorm(x, weight, epsilon=-1.0)


if __name__ == "__main__":
  absltest.main()
