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

"""GPU unit tests for Tokamax Mosaic GPU flash attention and ragged dot in Tunix."""

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
from jax import numpy as jnp
import numpy as np
from tunix.models import gpu_kernels
from tunix.models import reference_attention as reference_attention_lib
from tunix.models.gemma import model as gemma_model
from tunix.models.gemma3 import model as gemma3_model
from tunix.models.gemma4 import attention as gemma4_attention
from tunix.models.gemma4 import config as gemma4_config
from tunix.models.gemma4 import moe as gemma4_moe
from tunix.models.llama3 import model as llama3_model
from tunix.models.qwen2 import model as qwen2_model
from tunix.models.qwen3 import model as qwen3_model
from tunix.models.qwen3p5 import model as qwen3p5_model


class GpuKernelsTest(parameterized.TestCase):
  """Tests GPU flash attention, ragged dot, and model forward/backward parity."""

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "gpu":
      self.skipTest("Requires GPU backend")

  @parameterized.named_parameters(
      ("bf16_causal", jnp.bfloat16, False, None, None, 128),
      ("fp32_causal", jnp.float32, False, None, None, 128),
      ("bf16_segment_ids", jnp.bfloat16, True, None, None, 128),
      ("bf16_sliding_window", jnp.bfloat16, False, (31, 0), None, 128),
      ("bf16_soft_cap", jnp.bfloat16, False, None, 50.0, 128),
      ("bf16_non_aligned_100", jnp.bfloat16, False, None, None, 100),
      ("bf16_non_aligned_130_seg", jnp.bfloat16, True, None, None, 130),
  )
  def test_gpu_flash_attention_fwd_and_bwd(
      self,
      dtype,
      use_segment_ids: bool,
      local_window_size: tuple[int, int] | None,
      logits_soft_cap: float | None,
      seq_len: int,
  ):
    self.assertEqual(jax.default_backend(), "gpu")
    b, t, qh, kh, d = 2, seq_len, 4, 2, 64
    scale = float(d**-0.5)
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(0), 3)
    q = jax.random.normal(k1, (b, t, qh, d), dtype=dtype) * 0.2
    k = jax.random.normal(k2, (b, t, kh, d), dtype=dtype) * 0.2
    v = jax.random.normal(k3, (b, t, kh, d), dtype=dtype) * 0.2
    segment_ids = None
    if use_segment_ids:
      s1 = t // 3
      s2 = t // 3
      s3 = t - s1 - s2
      seg = jnp.concatenate(
          [
              jnp.ones((b, s1), dtype=jnp.int32),
              jnp.full((b, s2), 2, jnp.int32),
              jnp.zeros((b, s3), dtype=jnp.int32),
          ],
          axis=-1,
      )
      segment_ids = seg

    def gpu_loss_fn(q_in, k_in, v_in):
      out = gpu_kernels.gpu_flash_attention(
          q_in,
          k_in,
          v_in,
          scale=scale,
          is_causal=True,
          segment_ids=segment_ids,
          local_window_size=local_window_size,
          logits_soft_cap=logits_soft_cap,
      )
      return jnp.mean(out.astype(jnp.float32) ** 2), out

    def ref_loss_fn(q_in, k_in, v_in):
      out = reference_attention_lib.reference_attention(
          q_in.astype(jnp.float32),
          k_in.astype(jnp.float32),
          v_in.astype(jnp.float32),
          scale=scale,
          is_causal=True,
          segment_ids=segment_ids,
          local_window_size=local_window_size,
          logits_soft_cap=logits_soft_cap,
      ).astype(dtype)
      return jnp.mean(out.astype(jnp.float32) ** 2), out

    (_, out_gpu), (dq_gpu, dk_gpu, dv_gpu) = jax.value_and_grad(
        gpu_loss_fn, argnums=(0, 1, 2), has_aux=True
    )(q, k, v)
    (_, out_ref), (dq_ref, dk_ref, dv_ref) = jax.value_and_grad(
        ref_loss_fn, argnums=(0, 1, 2), has_aux=True
    )(q, k, v)

    self.assertEqual(out_gpu.dtype, dtype)
    valid = (segment_ids != 0)[:, :, None, None] if use_segment_ids else True
    np.testing.assert_allclose(
        np.asarray(jnp.where(valid, out_gpu, 0.0), dtype=np.float32),
        np.asarray(jnp.where(valid, out_ref, 0.0), dtype=np.float32),
        rtol=2e-2,
        atol=2e-2,
    )
    for g_gpu, g_ref in ((dq_gpu, dq_ref), (dk_gpu, dk_ref), (dv_gpu, dv_ref)):
      self.assertTrue(bool(jnp.all(jnp.isfinite(g_gpu))))
      np.testing.assert_allclose(
          np.asarray(jnp.where(valid, g_gpu, 0.0), dtype=np.float32),
          np.asarray(jnp.where(valid, g_ref, 0.0), dtype=np.float32),
          rtol=3e-2,
          atol=3e-2,
      )

  def test_left_padding_and_bidirectional_vision_mask_parity(self):
    self.assertEqual(jax.default_backend(), "gpu")
    b, t, qh, kh, d = 2, 128, 4, 2, 64
    scale = float(d**-0.5)
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(7), 3)
    q = jax.random.normal(k1, (b, t, qh, d), dtype=jnp.bfloat16) * 0.2
    k = jax.random.normal(k2, (b, t, kh, d), dtype=jnp.bfloat16) * 0.2
    v = jax.random.normal(k3, (b, t, kh, d), dtype=jnp.bfloat16) * 0.2

    # 1. Left-padded sequence (first 32 tokens are <pad> == 0)
    left_seg = jnp.concatenate(
        [
            jnp.zeros((b, 32), dtype=jnp.int32),
            jnp.ones((b, 96), dtype=jnp.int32),
        ],
        axis=-1,
    )
    attn_impls: tuple[gpu_kernels.GpuAttentionBackend, ...] = (
        ("mosaic", "triton")
        if gpu_kernels.supports_mosaic_gpu_attention()
        else ("triton", "cudnn")
    )
    if not gpu_kernels.supports_mosaic_gpu_attention():
      with self.assertRaisesRegex(NotImplementedError, "SM90"):
        gpu_kernels.gpu_flash_attention(
            q,
            k,
            v,
            scale=scale,
            is_causal=True,
            segment_ids=left_seg,
            implementation=("mosaic",),
        )
    for impl in attn_impls:
      (loss_gpu, out_gpu), (dq_gpu, dk_gpu, dv_gpu) = jax.value_and_grad(
          lambda q_, k_, v_, im=impl: (
              jnp.mean(
                  (
                      out := gpu_kernels.gpu_flash_attention(
                          q_,
                          k_,
                          v_,
                          scale=scale,
                          is_causal=True,
                          segment_ids=left_seg,
                          implementation=(im,),
                      )
                  )[:, 32:, :, :].astype(jnp.float32)
                  ** 2
              ),
              out,
          ),
          argnums=(0, 1, 2),
          has_aux=True,
      )(q, k, v)
      (_, out_ref), (dq_ref, _, _) = jax.value_and_grad(
          lambda q_, k_, v_: (
              jnp.mean(
                  (
                      out := reference_attention_lib.reference_attention(
                          q_.astype(jnp.float32),
                          k_.astype(jnp.float32),
                          v_.astype(jnp.float32),
                          scale=scale,
                          is_causal=True,
                          segment_ids=left_seg,
                      ).astype(jnp.bfloat16)
                  )[:, 32:, :, :].astype(jnp.float32)
                  ** 2
              ),
              out,
          ),
          argnums=(0, 1, 2),
          has_aux=True,
      )(q, k, v)
      self.assertTrue(bool(jnp.isfinite(loss_gpu)))
      self.assertTrue(bool(jnp.all(jnp.isfinite(out_gpu))))
      self.assertTrue(bool(jnp.all(jnp.isfinite(dq_gpu))))
      self.assertTrue(bool(jnp.all(jnp.isfinite(dk_gpu))))
      self.assertTrue(bool(jnp.all(jnp.isfinite(dv_gpu))))
      np.testing.assert_allclose(
          np.asarray(out_gpu[:, 32:, :, :], dtype=np.float32),
          np.asarray(out_ref[:, 32:, :, :], dtype=np.float32),
          rtol=2e-2,
          atol=2e-2,
      )
      np.testing.assert_allclose(
          np.asarray(dq_gpu[:, 32:, :, :], dtype=np.float32),
          np.asarray(dq_ref[:, 32:, :, :], dtype=np.float32),
          rtol=3e-2,
          atol=3e-2,
      )

    # 2. Multimodal bidirectional vision mask (image block at 32..64)
    causal = jnp.tril(jnp.ones((t, t), dtype=jnp.bool_))
    idx = jnp.arange(t)
    img_block = (
        (idx[:, None] >= 32)
        & (idx[:, None] < 64)
        & (idx[None, :] >= 32)
        & (idx[None, :] < 64)
    )
    valid_tok = (
        idx >= 16
    )  # Combine left-padding (0..15) with bidirectional vision (32..63)
    bidir_mask = jnp.broadcast_to(
        (causal | img_block) & valid_tok[None, :] & valid_tok[:, None],
        (b, t, t),
    )
    x_in = jax.random.normal(
        jax.random.PRNGKey(8), (b, t, 128), dtype=jnp.bfloat16
    )
    pos = jnp.broadcast_to(jnp.arange(t, dtype=jnp.int32), (b, t))

    g3_flash_cfg = gemma3_model.ModelConfig(
        num_layers=1,
        num_embed=512,
        embed_dim=128,
        hidden_dim=256,
        num_heads=4,
        head_dim=64,
        num_kv_heads=2,
        sliding_window_size=64,
        use_flash_attention=True,
    )
    g3_eager_cfg = gemma3_model.ModelConfig(
        num_layers=1,
        num_embed=512,
        embed_dim=128,
        hidden_dim=256,
        num_heads=4,
        head_dim=64,
        num_kv_heads=2,
        sliding_window_size=64,
        use_flash_attention=False,
    )
    g3_flash = gemma3_model.Gemma3(g3_flash_cfg, rngs=nnx.Rngs(params=9))
    g3_eager = gemma3_model.Gemma3(g3_eager_cfg, rngs=nnx.Rngs(params=9))
    _, out_g3_flash = g3_flash.layers[0].attn(
        x_in, pos, cache=None, attn_mask=bidir_mask
    )
    _, out_g3_eager = g3_eager.layers[0].attn(
        x_in, pos, cache=None, attn_mask=bidir_mask
    )
    self.assertTrue(bool(jnp.all(jnp.isfinite(out_g3_flash))))
    np.testing.assert_allclose(
        np.asarray(out_g3_flash[:, 16:, :], dtype=np.float32),
        np.asarray(out_g3_eager[:, 16:, :], dtype=np.float32),
        rtol=2e-2,
        atol=2e-2,
    )

    g4_cfg_flash = gemma4_config.ModelConfig(
        num_layers=1,
        num_embed=512,
        embed_dim=128,
        hidden_dim=256,
        num_heads=4,
        head_dim=64,
        num_kv_heads=2,
        global_key_size=64,
        sliding_window_size=64,
        use_flash_attention=True,
        flash_attention_block_size=64,
        dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16,
    )
    g4_cfg_eager = gemma4_config.ModelConfig(
        num_layers=1,
        num_embed=512,
        embed_dim=128,
        hidden_dim=256,
        num_heads=4,
        head_dim=64,
        num_kv_heads=2,
        global_key_size=64,
        sliding_window_size=64,
        use_flash_attention=False,
        flash_attention_block_size=64,
        dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16,
    )
    g4_flash = gemma4_attention.Attention(
        config=g4_cfg_flash,
        attn_type=gemma4_config.AttentionType.LOCAL_SLIDING,
        rngs=nnx.Rngs(params=10),
    )
    g4_eager = gemma4_attention.Attention(
        config=g4_cfg_eager,
        attn_type=gemma4_config.AttentionType.LOCAL_SLIDING,
        rngs=nnx.Rngs(params=10),
    )
    _, out_g4_flash, _ = g4_flash(
        x_in,
        pos,
        cache=None,
        attn_mask=bidir_mask,
        segment_ids=valid_tok[None, :].astype(jnp.int32)
        * jnp.ones((b, 1), dtype=jnp.int32),
    )
    _, out_g4_eager, _ = g4_eager(
        x_in,
        pos,
        cache=None,
        attn_mask=bidir_mask,
        segment_ids=valid_tok[None, :].astype(jnp.int32)
        * jnp.ones((b, 1), dtype=jnp.int32),
    )
    self.assertTrue(bool(jnp.all(jnp.isfinite(out_g4_flash))))
    np.testing.assert_allclose(
        np.asarray(out_g4_flash[:, 16:, :], dtype=np.float32),
        np.asarray(out_g4_eager[:, 16:, :], dtype=np.float32),
        rtol=2e-2,
        atol=2e-2,
    )

    # 3. Gemma4 with cache is not None and q_len == kv_len routes via eager path
    g4_cache = {
        "k": jnp.zeros((b, t, 2, 64), dtype=jnp.bfloat16),
        "v": jnp.zeros((b, t, 2, 64), dtype=jnp.bfloat16),
        "end_index": jnp.zeros((b,), dtype=jnp.int32),
    }
    _, out_g4_cached_flash, _ = g4_flash(
        x_in, pos, cache=g4_cache, attn_mask=bidir_mask
    )
    _, out_g4_cached_eager, _ = g4_eager(
        x_in, pos, cache=g4_cache, attn_mask=bidir_mask
    )
    np.testing.assert_allclose(
        np.asarray(out_g4_cached_flash[:, 16:, :], dtype=np.float32),
        np.asarray(out_g4_cached_eager[:, 16:, :], dtype=np.float32),
        rtol=1e-3,
        atol=1e-3,
    )

    # 4. Right-padded sliding-window sequence (L_valid=8, W=64, T=128):
    # rows 71..127 have attn_mask[:, q, :8] == True outside [q-63, q].
    right_valid = idx < 8
    right_pad_mask = jnp.broadcast_to(causal & right_valid[None, :], (b, t, t))
    loss_g3_rp, grads_g3_rp = nnx.value_and_grad(
        lambda m: jnp.mean(
            m.layers[0]
            .attn(x_in, pos, cache=None, attn_mask=right_pad_mask)[1]
            .astype(jnp.float32)
        )
    )(g3_flash)
    self.assertTrue(bool(jnp.isfinite(loss_g3_rp)))
    for g in jax.tree_util.tree_leaves(grads_g3_rp):
      self.assertTrue(bool(jnp.all(jnp.isfinite(g))))

    loss_g4_rp, grads_g4_rp = nnx.value_and_grad(
        lambda m: jnp.mean(
            m(x_in, pos, cache=None, attn_mask=right_pad_mask)[1].astype(
                jnp.float32
            )
        )
    )(g4_flash)
    self.assertTrue(bool(jnp.isfinite(loss_g4_rp)))
    for g in jax.tree_util.tree_leaves(grads_g4_rp):
      self.assertTrue(bool(jnp.all(jnp.isfinite(g))))

    # 5. Off-diagonal 3D packed-sequence attn_mask (segment_ids=None) parity
    seg_id = jnp.where(idx < 16, 0, jnp.where(idx < 64, 1, 2))
    packed_3d_mask = jnp.broadcast_to(
        causal & (seg_id[:, None] == seg_id[None, :]) & (seg_id[:, None] > 0),
        (b, t, t),
    )
    q3_flash_cfg = qwen3_model.ModelConfig(
        num_layers=1,
        vocab_size=512,
        embed_dim=128,
        hidden_dim=256,
        num_heads=4,
        head_dim=64,
        num_kv_heads=2,
        rope_theta=1_000_000,
        norm_eps=1e-6,
        use_flash_attention=True,
    )
    q3_eager_cfg = qwen3_model.ModelConfig(
        num_layers=1,
        vocab_size=512,
        embed_dim=128,
        hidden_dim=256,
        num_heads=4,
        head_dim=64,
        num_kv_heads=2,
        rope_theta=1_000_000,
        norm_eps=1e-6,
        use_flash_attention=False,
    )
    q3_flash = qwen3_model.Qwen3(q3_flash_cfg, rngs=nnx.Rngs(params=11))
    q3_eager = qwen3_model.Qwen3(q3_eager_cfg, rngs=nnx.Rngs(params=11))
    _, out_q3_flash = q3_flash.layers[0].attn(
        x_in, pos, cache=None, attn_mask=packed_3d_mask
    )
    _, out_q3_eager = q3_eager.layers[0].attn(
        x_in, pos, cache=None, attn_mask=packed_3d_mask
    )
    np.testing.assert_allclose(
        np.asarray(out_q3_flash[:, 16:, :], dtype=np.float32),
        np.asarray(out_q3_eager[:, 16:, :], dtype=np.float32),
        rtol=2e-2,
        atol=2e-2,
    )

  def test_resolve_q_sharding_mesh_fallback_and_divisibility(self):
    devices = np.array(jax.devices()[:1]).reshape(1, 1)
    single_mesh = jax.sharding.Mesh(devices, ("fsdp", "tp"))
    with single_mesh:
      self.assertIsNone(
          gpu_kernels.resolve_q_sharding(("fsdp", None, "tp", None))
      )
    abstract_mesh = jax.sharding.AbstractMesh((2, 4), ("fsdp", "tp"))
    with jax.sharding.use_abstract_mesh(abstract_mesh):
      # 1. Undeclared axes and head_dim ('tp' at idx 3) are dropped
      shd_1 = gpu_kernels.resolve_q_sharding(
          ("batch", "seq", "tp", "tp"),
          query_shape=(2, 128, 8, 64),
          key_shape=(2, 128, 4, 64),
      )
      self.assertIsNotNone(shd_1)
      self.assertEqual(
          shd_1.spec, jax.sharding.PartitionSpec(None, None, "tp", None)
      )
      # 2. Indivisible batch (b=1 on fsdp=2) and KV heads (kh=2 on tp=4) dropped
      shd_2 = gpu_kernels.resolve_q_sharding(
          ("fsdp", None, "tp", None),
          query_shape=(1, 128, 8, 64),
          key_shape=(1, 128, 2, 64),
      )
      self.assertIsNone(shd_2)
      # 3. MQA (kh=1, qh=8 on tp=4) keeps 'tp' since Tokamax broadcasts k/v
      shd_3 = gpu_kernels.resolve_q_sharding(
          ("fsdp", None, "tp", None),
          query_shape=(2, 128, 8, 64),
          key_shape=(2, 128, 1, 64),
      )
      self.assertIsNotNone(shd_3)
      self.assertEqual(
          shd_3.spec, jax.sharding.PartitionSpec("fsdp", None, "tp", None)
      )

  def test_gpu_flash_attention_rejects_mismatched_segment_ids_length(self):
    q = jnp.zeros((1, 64, 4, 64), dtype=jnp.bfloat16)
    k = jnp.zeros((1, 128, 2, 64), dtype=jnp.bfloat16)
    v = jnp.zeros((1, 128, 2, 64), dtype=jnp.bfloat16)
    bad_seg = jnp.ones((1, 64), dtype=jnp.int32)
    with self.assertRaisesRegex(ValueError, "segment_ids sequence length"):
      gpu_kernels.gpu_flash_attention(q, k, v, segment_ids=bad_seg)
    with self.assertRaisesRegex(
        ValueError, "local_window_size must be positive"
    ):
      gpu_kernels.gpu_flash_attention(q, q, q, local_window_size=0)

  def test_extract_token_mask_right_and_left_padding(self):
    seq_len = 4
    causal = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
    # Row 0: right-padded [1, 1, 1, 0]; Row 1: left-padded [0, 1, 1, 1]
    input_mask = jnp.array(
        [[True, True, True, False], [False, True, True, True]], dtype=jnp.bool_
    )
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
    explicit_seg = jnp.array([[1, 1, 2, 2], [0, 1, 1, 2]], dtype=jnp.int32)
    np.testing.assert_array_equal(
        np.asarray(
            reference_attention_lib.extract_token_mask(
                attn_mask_3d, seq_len, segment_ids=explicit_seg, dtype=jnp.int32
            )
        ),
        np.asarray(explicit_seg),
    )

  @parameterized.named_parameters(
      (
          "bf16_uniform",
          jnp.bfloat16,
          jnp.bfloat16,
          jnp.bfloat16,
          (32, 32, 32, 32),
          gpu_kernels.DEFAULT_GPU_RAGGED_DOT_IMPLEMENTATIONS,
      ),
      (
          "fp32_accum",
          jnp.float32,
          jnp.float32,
          jnp.float32,
          (32, 32, 32, 32),
          gpu_kernels.DEFAULT_GPU_RAGGED_DOT_IMPLEMENTATIONS,
      ),
      (
          "mixed_bf16_fp32_infer_out",
          jnp.bfloat16,
          jnp.float32,
          None,
          (32, 32, 32, 32),
          gpu_kernels.DEFAULT_GPU_RAGGED_DOT_IMPLEMENTATIONS,
      ),
      (
          "skewed_zero_experts",
          jnp.bfloat16,
          jnp.bfloat16,
          jnp.bfloat16,
          (0, 7, 121, 0),
          gpu_kernels.DEFAULT_GPU_RAGGED_DOT_IMPLEMENTATIONS,
      ),
      (
          "non_aligned_m130",
          jnp.bfloat16,
          jnp.bfloat16,
          jnp.bfloat16,
          (0, 13, 117, 0),
          gpu_kernels.DEFAULT_GPU_RAGGED_DOT_IMPLEMENTATIONS,
      ),
      (
          "small_decode_m2",
          jnp.bfloat16,
          jnp.bfloat16,
          jnp.bfloat16,
          (0, 1, 1, 0),
          gpu_kernels.DEFAULT_GPU_RAGGED_DOT_IMPLEMENTATIONS,
      ),
      (
          "explicit_mosaic_only",
          jnp.bfloat16,
          jnp.bfloat16,
          jnp.bfloat16,
          (32, 32, 32, 32),
          ("mosaic",),
      ),
  )
  def test_gpu_ragged_dot_fwd_and_bwd(
      self, lhs_dtype, rhs_dtype, pref_dtype, groups, impl
  ):
    self.assertEqual(jax.default_backend(), "gpu")
    k_x, k_w = jax.random.split(jax.random.PRNGKey(42))
    m, k_dim, n_dim, num_groups = sum(groups), 256, 256, len(groups)
    lhs = jax.random.normal(k_x, (m, k_dim), dtype=lhs_dtype) * 0.1
    rhs = (
        jax.random.normal(k_w, (num_groups, k_dim, n_dim), dtype=rhs_dtype)
        * 0.1
    )
    group_sizes = jnp.array(groups, dtype=jnp.int32)
    expected_out_dtype = (
        jnp.result_type(lhs, rhs) if pref_dtype is None else pref_dtype
    )

    def gpu_fn(x, w):
      out = gpu_kernels.gpu_ragged_dot(
          x,
          w,
          group_sizes=group_sizes,
          preferred_element_type=pref_dtype,
          implementation=impl,
      )
      return jnp.sum(out.astype(jnp.float32) ** 2), out

    def ref_fn(x, w):
      in_dt = jnp.result_type(x, w)
      out = jax.lax.ragged_dot(
          x.astype(in_dt),
          w.astype(in_dt),
          group_sizes=group_sizes,
          preferred_element_type=expected_out_dtype,
      )
      return jnp.sum(out.astype(jnp.float32) ** 2), out

    (_, out_gpu), (dlhs_gpu, drhs_gpu) = jax.value_and_grad(
        gpu_fn, argnums=(0, 1), has_aux=True
    )(lhs, rhs)
    (_, out_ref), (dlhs_ref, drhs_ref) = jax.value_and_grad(
        ref_fn, argnums=(0, 1), has_aux=True
    )(lhs, rhs)

    self.assertEqual(out_gpu.dtype, expected_out_dtype)
    self.assertEqual(dlhs_gpu.dtype, lhs_dtype)
    self.assertEqual(drhs_gpu.dtype, rhs_dtype)
    np.testing.assert_allclose(
        np.asarray(out_gpu, dtype=np.float32),
        np.asarray(out_ref, dtype=np.float32),
        rtol=2e-2,
        atol=2e-2,
    )
    np.testing.assert_allclose(
        np.asarray(dlhs_gpu, dtype=np.float32),
        np.asarray(dlhs_ref, dtype=np.float32),
        rtol=3e-2,
        atol=3e-2,
    )
    np.testing.assert_allclose(
        np.asarray(drhs_gpu, dtype=np.float32),
        np.asarray(drhs_ref, dtype=np.float32),
        rtol=3e-2,
        atol=3e-2,
    )

  @parameterized.named_parameters(
      dict(testcase_name="qwen3", family="qwen3"),
      dict(testcase_name="qwen2", family="qwen2"),
      dict(testcase_name="qwen3p5", family="qwen3p5"),
      dict(testcase_name="gemma4", family="gemma4"),
      dict(testcase_name="llama3", family="llama3"),
      dict(testcase_name="gemma3", family="gemma3"),
      dict(testcase_name="gemma", family="gemma"),
  )
  def test_all_model_families_gpu_fwd_and_bwd(self, family: str):
    self.assertEqual(jax.default_backend(), "gpu")
    mesh = jax.sharding.Mesh(
        np.array(jax.devices()[:1]).reshape(1, 1), ("fsdp", "tp")
    )
    tokens = jnp.ones((2, 128), dtype=jnp.int32)
    positions = jnp.broadcast_to(jnp.arange(128, dtype=jnp.int32), (2, 128))
    valid_tok = jnp.arange(128) >= 1
    causal_mask = jnp.broadcast_to(
        jnp.tril(jnp.ones((128, 128), dtype=jnp.bool_))
        & valid_tok[None, :]
        & valid_tok[:, None],
        (2, 128, 128),
    )
    x_g4 = jax.random.normal(
        jax.random.PRNGKey(3), (2, 128, 128), dtype=jnp.bfloat16
    )

    def _assert_finite_loss_and_grads(loss, grads):
      self.assertTrue(bool(jnp.isfinite(loss)))
      leaves = jax.tree_util.tree_leaves(grads)
      self.assertNotEmpty(leaves)
      for g in leaves:
        self.assertTrue(bool(jnp.all(jnp.isfinite(g))))

    with mesh:
      if family == "qwen3":
        q3_cfg = qwen3_model.ModelConfig(
            num_layers=2,
            vocab_size=512,
            embed_dim=128,
            hidden_dim=256,
            num_heads=4,
            head_dim=64,
            num_kv_heads=2,
            rope_theta=1_000_000,
            norm_eps=1e-6,
            num_experts=4,
            num_experts_per_tok=2,
            use_flash_attention=True,
        )
        q3 = qwen3_model.Qwen3(q3_cfg, rngs=nnx.Rngs(params=0))
        loss_q3, grads_q3 = nnx.value_and_grad(
            lambda m: jnp.mean(
                m(tokens, positions, cache=None, attention_mask=causal_mask)[
                    0
                ].astype(jnp.float32)
            )
        )(q3)
        _assert_finite_loss_and_grads(loss_q3, grads_q3)
        cache_q3 = q3.init_cache(
            batch_size=2, cache_size=256, dtype=jnp.bfloat16
        )
        logits_cached, _ = q3(
            tokens[:, :64],
            positions[:, :64],
            cache=cache_q3,
            attention_mask=jnp.ones((2, 64, 256), dtype=jnp.bool_),
        )
        self.assertTrue(bool(jnp.all(jnp.isfinite(logits_cached))))
      elif family == "qwen2":
        q2_cfg = qwen2_model.ModelConfig(
            num_layers=2,
            vocab_size=512,
            embed_dim=128,
            hidden_dim=256,
            num_heads=4,
            head_dim=64,
            num_kv_heads=2,
            rope_theta=1_000_000,
            norm_eps=1e-6,
            use_flash_attention=True,
        )
        q2 = qwen2_model.Qwen2(q2_cfg, rngs=nnx.Rngs(params=1))
        loss_q2, grads_q2 = nnx.value_and_grad(
            lambda m: jnp.mean(
                m(tokens, positions, cache=None, attention_mask=causal_mask)[
                    0
                ].astype(jnp.float32)
            )
        )(q2)
        _assert_finite_loss_and_grads(loss_q2, grads_q2)
      elif family == "qwen3p5":
        q35_cfg = qwen3p5_model.ModelConfig(
            num_layers=2,
            vocab_size=512,
            embed_dim=128,
            hidden_dim=256,
            num_heads=4,
            head_dim=64,
            num_kv_heads=2,
            rope_theta=1_000_000,
            norm_eps=1e-6,
            use_flash_attention=True,
        )
        q35 = qwen3p5_model.Qwen3p5(q35_cfg, rngs=nnx.Rngs(params=2))
        loss_q35, grads_q35 = nnx.value_and_grad(
            lambda m: jnp.mean(
                m(tokens, positions, cache=None, attention_mask=causal_mask)[
                    0
                ].astype(jnp.float32)
            )
        )(q35)
        _assert_finite_loss_and_grads(loss_q35, grads_q35)
      elif family == "gemma4":
        g4_cfg = gemma4_config.ModelConfig(
            num_layers=2,
            num_embed=512,
            embed_dim=128,
            hidden_dim=256,
            num_heads=4,
            head_dim=64,
            num_kv_heads=2,
            global_key_size=64,
            sliding_window_size=64,
            use_flash_attention=True,
            flash_attention_block_size=64,
            enable_moe=True,
            num_experts=4,
            num_experts_per_tok=2,
            expert_dim=128,
            moe_dense_hidden_dim=128,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
        )
        for attn_type in (
            gemma4_config.AttentionType.LOCAL_SLIDING,
            gemma4_config.AttentionType.GLOBAL,
        ):
          g4_attn = gemma4_attention.Attention(
              config=g4_cfg, attn_type=attn_type, rngs=nnx.Rngs(params=4)
          )
          loss_attn, grads_attn = nnx.value_and_grad(
              lambda m: jnp.mean(
                  m(x_g4, positions, cache=None, attn_mask=causal_mask)[
                      1
                  ].astype(jnp.float32)
              )
          )(g4_attn)
          _assert_finite_loss_and_grads(loss_attn, grads_attn)
        g4_moe = gemma4_moe.MoERagged(config=g4_cfg, rngs=nnx.Rngs(params=5))
        loss_moe, grads_moe = nnx.value_and_grad(
            lambda m: jnp.mean(m(x_g4).astype(jnp.float32) ** 2)
        )(g4_moe)
        _assert_finite_loss_and_grads(loss_moe, grads_moe)
      elif family == "llama3":
        l3_cfg = llama3_model.ModelConfig(
            num_layers=2,
            vocab_size=512,
            embed_dim=128,
            hidden_dim=256,
            num_heads=4,
            head_dim=64,
            num_kv_heads=2,
            rope_theta=500_000,
            norm_eps=1e-5,
            use_flash_attention=True,
        )
        l3 = llama3_model.Llama3(l3_cfg, rngs=nnx.Rngs(params=6))
        loss_l3, grads_l3 = nnx.value_and_grad(
            lambda m: jnp.mean(
                m(tokens, positions, cache=None, attention_mask=causal_mask)[
                    0
                ].astype(jnp.float32)
            )
        )(l3)
        _assert_finite_loss_and_grads(loss_l3, grads_l3)
      elif family == "gemma3":
        g3_cfg = gemma3_model.ModelConfig(
            num_layers=2,
            num_embed=512,
            embed_dim=128,
            hidden_dim=256,
            num_heads=4,
            head_dim=64,
            num_kv_heads=2,
            sliding_window_size=64,
            use_flash_attention=True,
        )
        g3 = gemma3_model.Gemma3(g3_cfg, rngs=nnx.Rngs(params=7))
        loss_g3, grads_g3 = nnx.value_and_grad(
            lambda m: jnp.mean(
                m(tokens, positions, cache=None, attention_mask=causal_mask)[
                    0
                ].astype(jnp.float32)
            )
        )(g3)
        _assert_finite_loss_and_grads(loss_g3, grads_g3)
      elif family == "gemma":
        g2_attn = gemma_model.Attention(
            num_heads=4,
            num_kv_heads=2,
            features=128,
            head_dim=64,
            attn_type=gemma_model.AttentionType.LOCAL_SLIDING,
            sliding_window_size=64,
            attn_logits_soft_cap=50.0,
            use_flash_attention=True,
            rngs=nnx.Rngs(params=8),
        )
        nnx.update(
            g2_attn,
            jax.tree.map(lambda p: p.astype(jnp.bfloat16), nnx.state(g2_attn)),
        )
        loss_g2, grads_g2 = nnx.value_and_grad(
            lambda m: jnp.mean(
                m(x_g4, positions, cache=None, attn_mask=causal_mask)[1].astype(
                    jnp.float32
                )
            )
        )(g2_attn)
        _assert_finite_loss_and_grads(loss_g2, grads_g2)

  def test_jit_backends_parity_vs_reference(self):
    self.assertEqual(jax.default_backend(), "gpu")
    b, s, qh, kh, d = 2, 512, 16, 8, 128
    scale = float(d**-0.5)
    q = (
        jax.random.normal(jax.random.PRNGKey(s), (b, s, qh, d), jnp.bfloat16)
        * 0.2
    )
    k = (
        jax.random.normal(
            jax.random.PRNGKey(s + 1), (b, s, kh, d), jnp.bfloat16
        )
        * 0.2
    )
    v = (
        jax.random.normal(
            jax.random.PRNGKey(s + 2), (b, s, kh, d), jnp.bfloat16
        )
        * 0.2
    )
    seg = jnp.concatenate(
        [
            jnp.ones((b, s // 2), dtype=jnp.int32),
            jnp.full((b, s - s // 2), 2, dtype=jnp.int32),
        ],
        axis=-1,
    )

    for seg_arg in (None, seg):
      xla_fwd = jax.jit(
          lambda q_, k_, v_, sg=seg_arg: reference_attention_lib.reference_attention(
              q_, k_, v_, scale=scale, is_causal=True, segment_ids=sg
          )
      )
      xla_bwd = jax.jit(
          jax.value_and_grad(
              lambda q_, k_, v_, sg=seg_arg: jnp.mean(
                  reference_attention_lib.reference_attention(
                      q_, k_, v_, scale=scale, is_causal=True, segment_ids=sg
                  ).astype(jnp.float32)
              ),
              argnums=(0, 1, 2),
          )
      )
      ref_out = xla_fwd(q, k, v)
      ref_loss, (ref_dq, _, _) = xla_bwd(q, k, v)

      for impl_name in ("mosaic", "triton", "cudnn"):
        try:
          tk_fwd = jax.jit(
              lambda q_, k_, v_, name=impl_name, sg=seg_arg: gpu_kernels.gpu_flash_attention(
                  q_,
                  k_,
                  v_,
                  scale=scale,
                  is_causal=True,
                  segment_ids=sg,
                  implementation=(name,),
              )
          )
          tk_bwd = jax.jit(
              jax.value_and_grad(
                  lambda q_, k_, v_, name=impl_name, sg=seg_arg: jnp.mean(
                      gpu_kernels.gpu_flash_attention(
                          q_,
                          k_,
                          v_,
                          scale=scale,
                          is_causal=True,
                          segment_ids=sg,
                          implementation=(name,),
                      ).astype(jnp.float32)
                  ),
                  argnums=(0, 1, 2),
              )
          )
          tk_out = tk_fwd(q, k, v)
          tk_loss, (tk_dq, _, _) = tk_bwd(q, k, v)
          np.testing.assert_allclose(
              np.asarray(tk_out, dtype=np.float32),
              np.asarray(ref_out, dtype=np.float32),
              rtol=2e-2,
              atol=2e-2,
          )
          np.testing.assert_allclose(
              float(tk_loss), float(ref_loss), rtol=2e-2, atol=2e-2
          )
          np.testing.assert_allclose(
              np.asarray(tk_dq, dtype=np.float32),
              np.asarray(ref_dq, dtype=np.float32),
              rtol=3e-2,
              atol=3e-2,
          )
        except NotImplementedError:
          pass

    g, m_tok, k_dim, n_dim = 8, 512, 256, 256
    lhs = (
        jax.random.normal(
            jax.random.PRNGKey(m_tok), (m_tok, k_dim), jnp.bfloat16
        )
        * 0.1
    )
    rhs = (
        jax.random.normal(
            jax.random.PRNGKey(m_tok + 1), (g, k_dim, n_dim), jnp.bfloat16
        )
        * 0.1
    )
    gs = jnp.full((g,), m_tok // g, dtype=jnp.int32)
    ref_rd_out = jax.jit(
        lambda x_, w_: jax.lax.ragged_dot(x_, w_, group_sizes=gs)
    )(lhs, rhs)
    _, (ref_dlhs, ref_drhs) = jax.jit(
        jax.value_and_grad(
            lambda x_, w_: jnp.mean(
                jax.lax.ragged_dot(x_, w_, group_sizes=gs).astype(jnp.float32)
            ),
            argnums=(0, 1),
        )
    )(lhs, rhs)

    for impl_name in ("mosaic", "triton"):
      tk_rd_out = jax.jit(
          lambda x_, w_, name=impl_name: gpu_kernels.gpu_ragged_dot(
              x_, w_, group_sizes=gs, implementation=(name,)
          )
      )(lhs, rhs)
      _, (tk_dlhs, tk_drhs) = jax.jit(
          jax.value_and_grad(
              lambda x_, w_, name=impl_name: jnp.mean(
                  gpu_kernels.gpu_ragged_dot(
                      x_, w_, group_sizes=gs, implementation=(name,)
                  ).astype(jnp.float32)
              ),
              argnums=(0, 1),
          )
      )(lhs, rhs)
      np.testing.assert_allclose(
          np.asarray(tk_rd_out, dtype=np.float32),
          np.asarray(ref_rd_out, dtype=np.float32),
          rtol=2e-2,
          atol=2e-2,
      )
      np.testing.assert_allclose(
          np.asarray(tk_dlhs, dtype=np.float32),
          np.asarray(ref_dlhs, dtype=np.float32),
          rtol=3e-2,
          atol=3e-2,
      )
      np.testing.assert_allclose(
          np.asarray(tk_drhs, dtype=np.float32),
          np.asarray(ref_drhs, dtype=np.float32),
          rtol=3e-2,
          atol=3e-2,
      )


if __name__ == "__main__":
  absltest.main()
