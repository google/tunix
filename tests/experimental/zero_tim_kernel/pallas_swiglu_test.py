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
"""Tests for pallas_swiglu (CPU, Pallas interpret mode)."""

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np
from tunix.experimental.zero_tim_kernel import pallas_swiglu
from tunix.experimental.zero_tim_kernel import test_utils


def setUpModule():
  test_utils.configure_cpu()


class SwigluTest(parameterized.TestCase):

  def test_matches_float64_reference(self):
    rng = np.random.default_rng(0)
    gate = test_utils.random_bf16(rng, (256, 512), scale=3.0)
    up = test_utils.random_bf16(rng, (256, 512))
    out = pallas_swiglu.swiglu(gate, up, interpret=True)
    g = np.asarray(gate, np.float64)
    expected = g / (1.0 + np.exp(-g)) * np.asarray(up, np.float64)
    self.assertEqual(out.dtype, jnp.bfloat16)
    np.testing.assert_allclose(
        test_utils.as_f32(out), expected, rtol=2e-2, atol=2e-2
    )

  def test_rows_do_not_depend_on_batch_size(self):
    rng = np.random.default_rng(1)
    gate = test_utils.random_bf16(rng, (384, 768), scale=3.0)
    up = test_utils.random_bf16(rng, (384, 768))
    full = pallas_swiglu.swiglu(gate, up, interpret=True)
    for start in (0, 128, 256):
      part = pallas_swiglu.swiglu(
          gate[start : start + 128], up[start : start + 128], interpret=True
      )
      test_utils.assert_bitwise_equal(full[start : start + 128], part)

  @parameterized.named_parameters(
      ("shape_mismatch", (128, 256), (128, 512)),
      ("rank", (128, 256, 1), (128, 256, 1)),
      ("m_not_tiled", (64, 256), (64, 256)),
      ("f_not_tiled", (128, 128), (128, 128)),
  )
  def test_rejects(self, gate_shape, up_shape):
    with self.assertRaises(ValueError):
      pallas_swiglu.swiglu(
          jnp.zeros(gate_shape, jnp.bfloat16),
          jnp.zeros(up_shape, jnp.bfloat16),
          interpret=True,
      )

  def test_rejects_non_bf16(self):
    with self.assertRaisesRegex(ValueError, "bf16"):
      pallas_swiglu.swiglu(
          jnp.zeros((128, 256), jnp.float32),
          jnp.zeros((128, 256), jnp.float32),
          interpret=True,
      )


if __name__ == "__main__":
  absltest.main()
