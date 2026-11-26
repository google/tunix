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
"""Tests for `pallas_swiglu` (CPU, Pallas interpret mode)."""

import os
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from jax.experimental import pallas as pl
import jax.numpy as jnp
import numpy as np
from tunix.zero_tim_kernel import pallas_matmul
from tunix.zero_tim_kernel import pallas_swiglu
from tunix.zero_tim_kernel import test_utils


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

  @parameterized.parameters(1, 8, 64, 127)
  def test_decode_rows_equal_the_128_row_program(self, m):
    rng = np.random.default_rng(m)
    gate = test_utils.random_bf16(rng, (128, 512), scale=3.0)
    up = test_utils.random_bf16(rng, (128, 512))
    full = pallas_swiglu.swiglu(gate, up, interpret=True)
    part = pallas_swiglu.swiglu(gate[:m], up[:m], interpret=True)
    test_utils.assert_bitwise_equal(part, full[:m])

  @parameterized.parameters((1, 8), (8, 8), (32, 32), (64, 64), (100, 128))
  def test_small_m_launches_adaptive_bm_grid(self, m, expected_bm):
    rng = np.random.default_rng(20 + m)
    gate = test_utils.random_bf16(rng, (m, 512), scale=3.0)
    up = test_utils.random_bf16(rng, (m, 512))
    with mock.patch.object(pl, "pallas_call", wraps=pl.pallas_call) as spy:
      out = pallas_swiglu.swiglu(gate, up, interpret=True)
    self.assertEqual(out.shape, (m, 512))
    self.assertLen(spy.call_args_list, 1)
    call = spy.call_args_list[0]
    self.assertEqual(call.kwargs["name"], f"canon_swiglu_bm{expected_bm}_bf256")
    self.assertEqual(tuple(call.kwargs["grid_spec"].grid), (1, 2))

  def test_feature_padding_is_sliced_off(self):
    rng = np.random.default_rng(1)
    gate = test_utils.random_bf16(rng, (16, 1216), scale=3.0)
    up = test_utils.random_bf16(rng, (16, 1216))
    out = pallas_swiglu.swiglu(gate, up, interpret=True)
    self.assertEqual(out.shape, (16, 1216))
    expected = pallas_swiglu.canonical_swiglu(gate, up)
    test_utils.assert_bitwise_equal(out, expected)
    with mock.patch.dict(os.environ, {pallas_matmul.INPUT_FUSION_ENV: "1"}):
      fused = pallas_swiglu.swiglu(gate, up, interpret=True)
    test_utils.assert_bitwise_equal(fused, out)

  @parameterized.named_parameters(
      ("shape_mismatch", (128, 256), (128, 512)),
      ("rank", (128, 256, 1), (128, 256, 1)),
      ("empty_m", (0, 256), (0, 256)),
      ("empty_f", (128, 0), (128, 0)),
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
