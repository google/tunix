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
"""Tests for pallas_matmul (CPU, Pallas interpret mode)."""

import os
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.experimental import pallas as pl
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import numpy as np
from tunix.experimental.zero_tim_kernel import pallas_matmul
from tunix.experimental.zero_tim_kernel import test_utils

_VMA_ENVS = (
    pallas_matmul.P66_CHECK_VMA_ENV,
    pallas_matmul.P67_SCOPE_ENV,
    pallas_matmul.P59_RANK_PARALLEL_BACKWARD_ENV,
    pallas_matmul.INPUT_FUSION_ENV,
)


def setUpModule():
  test_utils.configure_cpu(num_devices=4)


def _launches(fn):
  """Runs ``fn`` and returns its result and the (name, grid) of each launch."""
  with mock.patch.object(pl, "pallas_call", wraps=pl.pallas_call) as spy:
    result = fn()
  return result, [
      (call.kwargs["name"], tuple(call.kwargs["grid_spec"].grid))
      for call in spy.call_args_list
  ]


class TilePolicyTest(parameterized.TestCase):

  @parameterized.parameters(
      # (m, k, n, caller tiles, expected tiles)
      (256, 4096, 6144, (128, 256, 256), (256, 1024, 256)),
      (128, 4096, 2560, (128, 256, 256), (128, 1280, 256)),
      (512, 1216, 1280, (128, 128, 128), (256, 1280, 128)),
      (384, 256, 768, (128, 256, 256), (128, 256, 256)),
      (256, 4096, 1024, (128, 128, 128), (256, 1024, 128)),
      (256, 2560, 640, (128, 256, 256), (256, 640, 256)),
      (256, 2048, 1536, (128, 256, 256), (256, 512, 256)),
  )
  def test_tile_policy(self, m, k, n, tiles, expected):
    self.assertEqual(pallas_matmul.tile_policy(m, k, n, tiles), expected)

  def test_block_k_is_never_changed(self):
    for tiles in pallas_matmul.CONTRACT_TILES:
      for m in (128, 256, 384, 512):
        for n in (256, 512, 640, 1024, 1280, 2560, 38144):
          self.assertEqual(
              pallas_matmul.tile_policy(m, 4096, n, tiles)[2], tiles[2]
          )

  def test_contract_tiles_get_the_policy_and_others_are_honored(self):
    rng = np.random.default_rng(0)
    x = test_utils.exact_bf16(rng, (256, 512))
    y = test_utils.exact_bf16(rng, (512, 1024))
    _, launches = _launches(lambda: pallas_matmul.matmul(x, y, interpret=True))
    self.assertEqual(launches, [("canon_matmul_bm256_bn1024_bk256", (1, 1, 2))])
    _, launches = _launches(
        lambda: pallas_matmul.matmul(
            x, y, interpret=True, block_m=128, block_n=128, block_k=256
        )
    )
    self.assertEqual(launches, [("canon_matmul_bm128_bn128_bk256", (2, 8, 2))])


class MatmulNumericsTest(parameterized.TestCase):

  @parameterized.parameters(
      (256, 256, 256),
      (512, 256, 1024),
      (128, 512, 640),
      (384, 256, 512),
      (128, 1024, 256),
  )
  def test_exact_known_answer(self, m, k, n):
    rng = np.random.default_rng(m + k + n)
    x = test_utils.exact_bf16(rng, (m, k))
    y = test_utils.exact_bf16(rng, (k, n))
    out = pallas_matmul.matmul(x, y, interpret=True)
    self.assertEqual(out.dtype, jnp.bfloat16)
    test_utils.assert_bitwise_equal(
        out, test_utils.exact_matmul_reference(x, y)
    )

  def test_random_operands_close_to_f32_reference(self):
    rng = np.random.default_rng(1)
    x = test_utils.random_bf16(rng, (256, 768))
    y = test_utils.random_bf16(rng, (768, 512), scale=0.05)
    out = pallas_matmul.matmul(x, y, interpret=True)
    reference = np.asarray(x, np.float64) @ np.asarray(y, np.float64)
    np.testing.assert_allclose(
        test_utils.as_f32(out), reference, rtol=1e-2, atol=1e-2
    )

  def test_rows_do_not_depend_on_batch_size_or_other_rows(self):
    # Same tiles (bm=256) for M=256 and M=512: every row runs the identical
    # per-element chain, whatever the other rows hold.
    rng = np.random.default_rng(2)
    x = test_utils.random_bf16(rng, (256, 512))
    y = test_utils.random_bf16(rng, (512, 1024), scale=0.05)
    alone = pallas_matmul.matmul(x, y, interpret=True)
    for filler in (
        jnp.zeros_like(x),
        jnp.full_like(x, 1e4),
        jnp.full_like(x, jnp.nan),
        -x,
    ):
      below = pallas_matmul.matmul(
          jnp.concatenate([x, filler]), y, interpret=True
      )
      above = pallas_matmul.matmul(
          jnp.concatenate([filler, x]), y, interpret=True
      )
      test_utils.assert_bitwise_equal(below[:256], alone)
      test_utils.assert_bitwise_equal(above[256:], alone)

  def test_rows_invariant_at_block_m_128(self):
    # M=384 does not divide 256, so it keeps bm=128 like M=128.
    rng = np.random.default_rng(3)
    x = test_utils.random_bf16(rng, (384, 256))
    y = test_utils.random_bf16(rng, (256, 512), scale=0.05)
    full = pallas_matmul.matmul(x, y, interpret=True)
    for start in (0, 128, 256):
      part = pallas_matmul.matmul(x[start : start + 128], y, interpret=True)
      test_utils.assert_bitwise_equal(full[start : start + 128], part)

  @parameterized.parameters(
      (128, 128, 256), (256, 512, 256), (128, 1024, 256), (256, 256, 256)
  )
  def test_output_tiles_are_neutral_on_exact_operands(self, bm, bn, bk):
    rng = np.random.default_rng(4)
    x = test_utils.exact_bf16(rng, (512, 512))
    y = test_utils.exact_bf16(rng, (512, 1024))
    default = pallas_matmul.matmul(x, y, interpret=True)
    tiled = pallas_matmul.matmul(
        x, y, interpret=True, block_m=bm, block_n=bn, block_k=bk
    )
    test_utils.assert_bitwise_equal(tiled, default)

  def test_jit_matches_eager(self):
    rng = np.random.default_rng(5)
    x = test_utils.random_bf16(rng, (256, 256))
    y = test_utils.random_bf16(rng, (256, 256))
    eager = pallas_matmul.matmul(x, y, interpret=True)
    jitted = jax.jit(lambda a, b: pallas_matmul.matmul(a, b, interpret=True))(
        x, y
    )
    test_utils.assert_bitwise_equal(jitted, eager)

  def test_block_k_controls_numerics_while_block_m_and_n_are_neutral(self):
    rng = np.random.default_rng(7)
    x = test_utils.random_bf16(rng, (256, 256))
    y = test_utils.random_bf16(rng, (256, 256))
    # Put canceling large terms at k=0 (first 128-slice) and k=128 (second
    # 128-slice): with block_k=256 the single dot cancels k=0 and k=128 before
    # k=129..255 are accumulated, whereas with block_k=128 the second slice
    # accumulates k=128..255 starting from -256 (swamping low mantissa bits)
    # before adding to the first slice.
    x = x.at[:, 0].set(jnp.bfloat16(16.0)).at[:, 128].set(jnp.bfloat16(16.0))
    y = y.at[0, :].set(jnp.bfloat16(16.0)).at[128, :].set(jnp.bfloat16(-16.0))

    ref_bm128_bn256_bk256 = pallas_matmul.matmul(
        x, y, interpret=True, block_m=128, block_n=256, block_k=256
    )
    bm256_bn128_bk256 = pallas_matmul.matmul(
        x, y, interpret=True, block_m=256, block_n=128, block_k=256
    )
    bk128 = pallas_matmul.matmul(
        x, y, interpret=True, block_m=128, block_n=256, block_k=128
    )

    # Output tiles (BM, BN) only partition which tile writes which output
    # submatrix: they are 100% bitwise neutral even on ill-conditioned inputs.
    test_utils.assert_bitwise_equal(bm256_bn128_bk256, ref_bm128_bn256_bk256)
    # Contraction tile BK changes the K-reduction grouping and therefore drifts.
    self.assertFalse(
        np.array_equal(
            np.asarray(bk128.view(jnp.uint16)),
            np.asarray(ref_bm128_bn256_bk256.view(jnp.uint16)),
        )
    )


class MatmulValidationTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("rank", (2, 128, 256), (256, 256), {}),
      ("contracted_mismatch", (128, 256), (512, 256), {}),
      ("m_not_tiled", (100, 256), (256, 256), {}),
      ("k_not_tiled", (128, 200), (200, 256), {}),
      ("n_not_tiled", (128, 256), (256, 200), {}),
      ("non_positive_tile", (128, 256), (256, 256), {"block_m": 0}),
  )
  def test_rejects(self, x_shape, y_shape, kwargs):
    x = jnp.zeros(x_shape, jnp.bfloat16)
    y = jnp.zeros(y_shape, jnp.bfloat16)
    with self.assertRaises(ValueError):
      pallas_matmul.matmul(x, y, interpret=True, **kwargs)

  def test_rejects_non_bf16(self):
    with self.assertRaisesRegex(ValueError, "bf16"):
      pallas_matmul.matmul(
          jnp.zeros((128, 256), jnp.float32),
          jnp.zeros((256, 256), jnp.float32),
          interpret=True,
      )


class VmaHelpersTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    patcher = mock.patch.dict(os.environ)
    patcher.start()
    self.addCleanup(patcher.stop)
    for name in _VMA_ENVS:
      os.environ.pop(name, None)

  def test_unscoped_mode_is_process_wide(self):
    self.assertTrue(pallas_matmul.p67_p59_vma_context())

  def test_scope_requires_p59_manual_data_model_map(self):
    os.environ[pallas_matmul.P67_SCOPE_ENV] = "1"
    self.assertFalse(pallas_matmul.p67_p59_vma_context())
    os.environ[pallas_matmul.P59_RANK_PARALLEL_BACKWARD_ENV] = "1"
    self.assertFalse(pallas_matmul.p67_p59_vma_context())

    seen = []

    def body(x):
      seen.append(pallas_matmul.p67_p59_vma_context())
      return x

    mesh = test_utils.cpu_mesh((2, 2), ("data", "model"))
    jax.shard_map(
        body, mesh=mesh, in_specs=P(), out_specs=P(), check_vma=False
    )(jnp.zeros((), jnp.float32))
    self.assertEqual(seen, [True])

    seen.clear()
    model_only = test_utils.cpu_mesh((4,), ("model",))
    jax.shard_map(
        body, mesh=model_only, in_specs=P(), out_specs=P(), check_vma=False
    )(jnp.zeros((), jnp.float32))
    self.assertEqual(seen, [False])

  def test_invalid_scope_is_rejected(self):
    os.environ[pallas_matmul.P67_SCOPE_ENV] = "2"
    with self.assertRaises(ValueError):
      pallas_matmul.p67_p59_vma_context()

  def test_helpers_are_identity_when_vma_mode_is_off(self):
    x = jnp.ones((8, 128), jnp.bfloat16)
    y = jnp.ones((128, 128), jnp.bfloat16)
    aligned = pallas_matmul.p66_vma_align_operands(x, y)
    self.assertIs(aligned[0], x)
    self.assertIs(aligned[1], y)
    self.assertIsNone(pallas_matmul.p66_vma_output_manual_axis_type(x, y))

  def test_align_casts_replicated_operands_to_varying(self):
    os.environ[pallas_matmul.P66_CHECK_VMA_ENV] = "1"
    seen = {}

    def body(x_local, y):
      aligned = pallas_matmul.p66_vma_align_operands(x_local, y)
      seen["before"] = tuple(jax.typeof(v).mat.varying for v in (x_local, y))
      seen["after"] = tuple(jax.typeof(v).mat.varying for v in aligned)
      seen["out"] = pallas_matmul.p66_vma_output_manual_axis_type(*aligned)
      return aligned[0]

    mesh = test_utils.cpu_mesh((4,), ("model",))
    jax.shard_map(
        body,
        mesh=mesh,
        in_specs=(P("model", None), P()),
        out_specs=P("model", None),
        check_vma=True,
    )(jnp.ones((32, 128), jnp.bfloat16), jnp.ones((128, 128), jnp.bfloat16))
    self.assertEqual(seen["before"], (frozenset({"model"}), frozenset()))
    self.assertEqual(
        seen["after"], (frozenset({"model"}), frozenset({"model"}))
    )
    self.assertEqual(seen["out"].varying, frozenset({"model"}))

  def test_input_fusion_knob(self):
    self.assertFalse(pallas_matmul.input_fusion_enabled())
    os.environ[pallas_matmul.INPUT_FUSION_ENV] = "1"
    self.assertTrue(pallas_matmul.input_fusion_enabled())


if __name__ == "__main__":
  absltest.main()
