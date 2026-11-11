# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU-runnable unit tests for reference_attention and pure-JAX attention helpers."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import numpy as np
from tunix.models import gpu_kernels
from tunix.models import reference_attention as reference_attention_lib


class ReferenceAttentionTest(parameterized.TestCase):
  """Tests pure-JAX reference_attention, mask helpers, and sharding resolvers."""

  def test_extract_token_mask_right_and_left_padding(self):
    seq_len = 4
    causal = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
    input_mask = jnp.array(
        [[True, True, True, False], [False, True, True, True]], dtype=jnp.bool_
    )
    self.assertIsNone(reference_attention_lib.extract_token_mask(None, seq_len))
    np.testing.assert_array_equal(
        np.asarray(
            reference_attention_lib.extract_token_mask(input_mask, seq_len)
        ),
        np.asarray(input_mask),
    )
    attn_mask_3d = causal[None, :, :] & input_mask[:, None, :]
    np.testing.assert_array_equal(
        np.asarray(
            reference_attention_lib.extract_token_mask(attn_mask_3d, seq_len)
        ),
        np.asarray(input_mask),
    )
    attn_mask_4d = attn_mask_3d[:, None, :, :]
    np.testing.assert_array_equal(
        np.asarray(
            reference_attention_lib.extract_token_mask(attn_mask_4d, seq_len)
        ),
        np.asarray(input_mask),
    )
    decode_mask = jnp.array(
        [[[True, True, False, False]], [[False, False, False, False]]],
        dtype=jnp.bool_,
    )
    np.testing.assert_array_equal(
        np.asarray(reference_attention_lib.extract_token_mask(decode_mask, 1)),
        np.array([[True], [False]], dtype=np.bool_),
    )
    explicit_seg = jnp.array([[1, 1, 2, 2], [0, 1, 1, 2]], dtype=jnp.int32)
    np.testing.assert_array_equal(
        np.asarray(
            reference_attention_lib.extract_token_mask(
                attn_mask_3d, seq_len, segment_ids=explicit_seg, dtype=jnp.int32
            )
        ),
        np.asarray(explicit_seg),
    )

  def test_resolve_q_and_splash_sharding_compound_axes_and_mesh_contexts(self):
    devices = np.array(jax.devices()[:1]).reshape(1, 1)
    single_mesh = jax.sharding.Mesh(devices, ("fsdp", "tp"))
    with single_mesh:
      self.assertIsNone(
          gpu_kernels.resolve_q_sharding(("fsdp", None, "tp", None))
      )
      _, shd_b, shd_t, shd_n, shd_h, shd_n_kv, h_shards, q_shards = (
          reference_attention_lib.resolve_splash_sharding(
              ("fsdp", "seq", "tp", None), batch_size=2, kv_heads=2
          )
      )
      self.assertEqual((shd_b, shd_t, shd_n, shd_h), ("fsdp", None, "tp", None))
      self.assertEqual((shd_n_kv, h_shards, q_shards), ("tp", 1, 1))

    abstract_mesh = jax.sharding.AbstractMesh((2, 2, 4), ("dp", "fsdp", "tp"))
    with jax.sharding.use_abstract_mesh(abstract_mesh):
      # Compound tuple axis (('dp', 'fsdp'), None, 'tp', 'tp')
      q_shd = gpu_kernels.resolve_q_sharding(
          (("dp", "fsdp"), None, "tp", "tp"),
          query_shape=(4, 128, 8, 64),
          key_shape=(4, 128, 4, 64),
      )
      self.assertIsNotNone(q_shd)
      self.assertEqual(
          q_shd.spec,
          jax.sharding.PartitionSpec(("dp", "fsdp"), None, "tp", None),
      )
      _, shd_b, shd_t, shd_n, shd_h, shd_n_kv, h_shards, q_shards = (
          reference_attention_lib.resolve_splash_sharding(
              (("dp", "fsdp"), None, "tp", None), batch_size=4, kv_heads=4
          )
      )
      self.assertEqual(shd_b, ("dp", "fsdp"))
      self.assertIsNone(shd_t)
      self.assertEqual(shd_n, "tp")
      self.assertIsNone(shd_h)
      self.assertEqual(shd_n_kv, "tp")
      self.assertEqual((h_shards, q_shards), (4, 1))

      # Indivisible batch (b=2 vs 4) and indivisible kv_heads (kh=2 vs 4)
      _, shd_b2, _, shd_n2, _, shd_n_kv2, h_shards2, _ = (
          reference_attention_lib.resolve_splash_sharding(
              (("dp", "fsdp"), None, "tp", None), batch_size=2, kv_heads=2
          )
      )
      self.assertIsNone(shd_b2)
      self.assertEqual(shd_n2, "tp")
      self.assertIsNone(shd_n_kv2)
      self.assertEqual(h_shards2, 4)

    # Verify jax.set_mesh context fallback and jax.jit trace safety
    self.assertIsNone(
        jax.jit(
            lambda: gpu_kernels.resolve_q_sharding(("fsdp", None, "tp", None))
        )()
    )
    self.assertEqual(
        jax.jit(
            lambda: reference_attention_lib.resolve_splash_sharding(
                ("fsdp", None, "tp", None), batch_size=2, kv_heads=2
            )[1:]
        )(),
        (None, None, None, None, None, 1, 1),
    )
    with jax.set_mesh(single_mesh):
      mesh_out, shd_b_set, _, shd_n_set, _, _, _, _ = (
          reference_attention_lib.resolve_splash_sharding(
              ("fsdp", None, "tp", None), batch_size=2, kv_heads=2
          )
      )
      self.assertFalse(mesh_out.empty)
      self.assertEqual((shd_b_set, shd_n_set), ("fsdp", "tp"))

      @jax.jit
      def _check_in_jit() -> jax.Array:
        _, b_ax, t_ax, n_ax, _, _, h_shd, _ = (
            reference_attention_lib.resolve_splash_sharding(
                ("fsdp", None, "tp", None), batch_size=2, kv_heads=2
            )
        )
        self.assertEqual((b_ax, t_ax, n_ax), ("fsdp", None, "tp"))
        return jnp.int32(h_shd)

      self.assertEqual(int(_check_in_jit()), 1)

  def test_prepare_attention_mask_rectangular_and_validation(self):
    # Rectangular q_len < kv_len with all-False padding row 0 gets diag guard
    rect_mask = jnp.array(
        [[
            [False, False, False, False],
            [True, True, False, False],
        ]],
        dtype=jnp.bool_,
    )
    prepared = gpu_kernels._prepare_attention_mask(
        rect_mask,
        segment_ids=None,
        q_len=2,
        kv_len=4,
        is_causal=True,
        local_window_size=(1, 0),
    )
    self.assertEqual(prepared.shape, (1, 1, 2, 4))
    # Row 0 had no valid keys -> guarded at diagonal (0, 0); Row 1 untouched
    np.testing.assert_array_equal(
        np.asarray(prepared[0, 0]),
        np.array(
            [[True, False, False, False], [True, True, False, False]],
            dtype=np.bool_,
        ),
    )
    with self.assertRaisesRegex(ValueError, "mask must be 3D or 4D"):
      gpu_kernels._prepare_attention_mask(
          jnp.ones((1, 4), dtype=jnp.bool_),
          segment_ids=None,
          q_len=4,
          kv_len=4,
          is_causal=True,
      )
    with self.assertRaisesRegex(ValueError, "segment_ids sequence length"):
      gpu_kernels._prepare_attention_mask(
          None,
          segment_ids=jnp.ones((1, 2), dtype=jnp.int32),
          q_len=4,
          kv_len=4,
          is_causal=True,
      )

  def test_reference_attention_mha_gqa_and_validation(self):
    q = jnp.ones((1, 4, 4, 8), dtype=jnp.float32)
    k = jnp.ones((1, 4, 2, 8), dtype=jnp.float32)
    v = jnp.arange(1 * 4 * 2 * 8, dtype=jnp.float32).reshape((1, 4, 2, 8))
    out = reference_attention_lib.reference_attention(
        q,
        k,
        v,
        is_causal=True,
        local_window_size=(2, 0),
        logits_soft_cap=30.0,
        segment_ids=jnp.array([[1, 1, 2, 2]], dtype=jnp.int32),
    )
    self.assertEqual(out.shape, (1, 4, 4, 8))
    self.assertTrue(bool(jnp.all(jnp.isfinite(out))))
    with self.assertRaisesRegex(ValueError, "must be divisible by KV heads"):
      reference_attention_lib.reference_attention(
          q, jnp.ones((1, 4, 3, 8), dtype=jnp.float32), v
      )

  def test_gpu_attention_backend_resolution_sm80_vs_sm90(self):
    if jax.devices()[0].platform != "gpu":
      self.assertFalse(gpu_kernels.supports_mosaic_gpu_attention())
    default_impls = gpu_kernels.DEFAULT_GPU_ATTN_IMPLEMENTATIONS
    self.assertEqual(
        gpu_kernels.resolve_attn_implementations(
            default_impls, mosaic_supported=True
        ),
        ("mosaic", "triton", "cudnn", "xla_chunked", "xla"),
    )
    self.assertEqual(
        gpu_kernels.resolve_attn_implementations(
            default_impls, logits_soft_cap=50.0, mosaic_supported=True
        ),
        ("mosaic", "cudnn", "xla_chunked", "xla"),
    )
    self.assertEqual(
        gpu_kernels.resolve_attn_implementations(
            default_impls, mosaic_supported=False
        ),
        ("triton", "cudnn", "xla_chunked", "xla"),
    )
    self.assertEqual(
        gpu_kernels.resolve_attn_implementations(
            default_impls, logits_soft_cap=50.0, mosaic_supported=False
        ),
        ("cudnn", "xla_chunked", "xla"),
    )
    self.assertEqual(
        gpu_kernels.resolve_attn_implementations(
            ("triton",), logits_soft_cap=50.0, mosaic_supported=False
        ),
        ("triton",),
    )
    with self.assertRaisesRegex(NotImplementedError, "SM90"):
      gpu_kernels.resolve_attn_implementations(
          ("mosaic",), mosaic_supported=False
      )


if __name__ == "__main__":
  absltest.main()
