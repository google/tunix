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

"""Tests for Qwen3.5 hybrid model and SafeTensors parameter loading."""

import contextlib
import os
import types
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import safetensors.numpy as safe_np
from tunix.generate import utils as gen_utils
from tunix.models.qwen3p5 import model as qwen3p5_model
from tunix.models.qwen3p5 import params as qwen3p5_params


class _ReferenceCausalSplashKernel:
  """CPU stand-in for splash.make_splash_mha computing GQA + segment_ids."""

  def manual_sharding_spec(self, unused_named_sharding):
    return None

  def __call__(self, q, k, v, segment_ids=None):
    # q: (qh, T, D) — already scaled by 1/sqrt(D)
    # k, v: (kh, T, D)
    qh, t, d = q.shape
    kh = k.shape[0]
    if qh % kh != 0:
      raise ValueError(f'num_q_heads ({qh}) not divisible by num_kv ({kh})')
    g = qh // kh
    q_grouped = q.reshape(kh, g, t, d)
    scores = jnp.einsum(
        'kgtd,ksd->kgts',
        q_grouped.astype(jnp.float32),
        k.astype(jnp.float32),
    )
    scores = scores.reshape(qh, t, t)
    causal = jnp.tril(jnp.ones((t, t), dtype=jnp.bool_))
    if segment_ids is not None:
      seg_mask = segment_ids.q[:, None] == segment_ids.kv[None, :]
      mask = causal & seg_mask
    else:
      mask = causal
    scores = jnp.where(mask[None, :, :], scores, qwen3p5_model.K_MASK)
    probs = jax.nn.softmax(scores, axis=-1).astype(v.dtype)
    probs = probs.reshape(kh, g, t, t)
    out = jnp.einsum('kgts,ksd->kgtd', probs, v)
    return out.reshape(qh, t, d)


def _identity_shard_map(fn=None, **unused_kwargs):
  return (lambda inner_fn: inner_fn) if fn is None else fn


@contextlib.contextmanager
def _patch_cpu_splash():
  with (
      mock.patch.object(qwen3p5_model, 'shard_map', _identity_shard_map),
      mock.patch.object(
          qwen3p5_model.shd, 'NamedSharding', lambda *a, **kw: None
      ),
      mock.patch.object(
          qwen3p5_model.splash,
          'make_splash_mha',
          lambda *a, **kw: _ReferenceCausalSplashKernel(),
      ),
  ):
    yield


def _make_tiny_qwen3p5_config(
    *, use_tied_embedding: bool = True
) -> qwen3p5_model.ModelConfig:
  return qwen3p5_model.ModelConfig(
      num_layers=4,
      vocab_size=64,
      embed_dim=32,
      hidden_dim=64,
      num_heads=4,
      head_dim=16,
      num_kv_heads=2,
      rope_theta=10_000_000,
      norm_eps=1e-6,
      use_tied_embedding=use_tied_embedding,
      use_norm_offset=True,
      attn_output_gate=True,
      partial_rotary_factor=0.25,
      full_attention_interval=4,
      linear_num_key_heads=4,
      linear_num_value_heads=4,
      linear_key_head_dim=8,
      linear_value_head_dim=8,
      linear_conv_kernel_dim=4,
      dtype=jnp.float32,
      param_dtype=jnp.float32,
  )


def _numpy_silu(x: np.ndarray) -> np.ndarray:
  return x / (1.0 + np.exp(-x))


def _numpy_softplus(x: np.ndarray) -> np.ndarray:
  return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


class Qwen3p5ModelTest(parameterized.TestCase):

  def test_qwen3p5_0p8b_config(self):
    cfg = qwen3p5_model.ModelConfig.qwen3p5_0p8b()
    self.assertEqual(cfg.num_layers, 24)
    self.assertEqual(cfg.vocab_size, 248320)
    self.assertEqual(cfg.embed_dim, 1024)
    self.assertEqual(cfg.hidden_dim, 3584)
    self.assertEqual(cfg.num_heads, 8)
    self.assertEqual(cfg.head_dim, 256)
    self.assertEqual(cfg.num_kv_heads, 2)
    self.assertEqual(cfg.rotary_dim, 64)
    self.assertTrue(cfg.use_tied_embedding)
    self.assertTrue(cfg.use_norm_offset)
    self.assertTrue(cfg.attn_output_gate)
    assert cfg.layer_types is not None
    full_indices = [
        i
        for i, t in enumerate(cfg.layer_types)
        if t == qwen3p5_model.LayerType.FULL_ATTENTION
    ]
    self.assertEqual(full_indices, [3, 7, 11, 15, 19, 23])

  def test_qwen3p5_2b_config(self):
    cfg = qwen3p5_model.ModelConfig.qwen3p5_2b()
    self.assertEqual(cfg.num_layers, 24)
    self.assertEqual(cfg.vocab_size, 248320)
    self.assertEqual(cfg.embed_dim, 2048)
    self.assertEqual(cfg.hidden_dim, 6144)
    self.assertEqual(cfg.num_heads, 8)
    self.assertEqual(cfg.head_dim, 256)
    self.assertEqual(cfg.num_kv_heads, 2)
    self.assertEqual(cfg.rotary_dim, 64)
    self.assertTrue(cfg.use_tied_embedding)
    self.assertTrue(cfg.use_norm_offset)
    self.assertTrue(cfg.attn_output_gate)
    assert cfg.layer_types is not None
    full_indices = [
        i
        for i, t in enumerate(cfg.layer_types)
        if t == qwen3p5_model.LayerType.FULL_ATTENTION
    ]
    self.assertEqual(full_indices, [3, 7, 11, 15, 19, 23])

  def test_gated_delta_net_prefill_matches_step_by_step_decode(self):
    cfg = _make_tiny_qwen3p5_config()
    gdn = qwen3p5_model.GatedDeltaNet(cfg, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.key(1), (2, 6, cfg.embed_dim))
    causal_mask = jnp.tril(jnp.ones((2, 6, 6), dtype=jnp.bool_))

    _, full_out = gdn(x, None, causal_mask)

    model = qwen3p5_model.Qwen3p5(cfg, rngs=nnx.Rngs(0))
    cache = model.init_cache(batch_size=2, cache_size=8, dtype=jnp.float32)[
        'layer_0'
    ]
    cache, prefill_out = gdn(x[:, :4, :], cache, causal_mask[:, :4, :4])
    cache, step4_out = gdn(
        x[:, 4:5, :], cache, jnp.ones((2, 1, 8), dtype=jnp.bool_)
    )
    cache, step5_out = gdn(
        x[:, 5:6, :], cache, jnp.ones((2, 1, 8), dtype=jnp.bool_)
    )

    chained_out = jnp.concatenate([prefill_out, step4_out, step5_out], axis=1)
    np.testing.assert_allclose(
        np.asarray(full_out), np.asarray(chained_out), rtol=1e-5, atol=1e-5
    )

  def test_gated_delta_net_against_numpy_reference(self):
    cfg = _make_tiny_qwen3p5_config()
    gdn = qwen3p5_model.GatedDeltaNet(cfg, rngs=nnx.Rngs(5))
    gdn.A_log.value = jax.random.normal(jax.random.key(6), (4,))
    gdn.dt_bias.value = jax.random.normal(jax.random.key(7), (4,))
    gdn.norm.w.value = jax.random.normal(jax.random.key(8), (8,))

    x_jnp = jax.random.normal(jax.random.key(9), (1, 3, cfg.embed_dim))
    cache = {
        'conv_state': jnp.zeros((1, 3, 96), dtype=jnp.float32),
        'recurrent_state': jnp.zeros((1, 4, 8, 8), dtype=jnp.float32),
        'end_index': jnp.zeros((1,), dtype=jnp.int32),
    }
    new_cache, gdn_out = gdn(x_jnp, cache, None)

    x_np = np.asarray(x_jnp, dtype=np.float32)
    qkv = x_np @ np.asarray(gdn.in_proj_qkv.kernel.value)
    z = (x_np @ np.asarray(gdn.in_proj_z.kernel.value)).reshape(1, 3, 4, 8)
    b = x_np @ np.asarray(gdn.in_proj_b.kernel.value)
    a = x_np @ np.asarray(gdn.in_proj_a.kernel.value)
    conv_w = np.asarray(gdn.conv1d.kernel.value)[:, 0, :]  # (K=4, conv_dim)
    padded_qkv = np.pad(qkv, ((0, 0), (3, 0), (0, 0)))
    conv_np = np.zeros_like(qkv)
    for k_idx in range(4):
      conv_np += padded_qkv[:, k_idx : k_idx + 3, :] * conv_w[k_idx]
    conv_np = _numpy_silu(conv_np)
    q_np, k_np, v_np = np.split(conv_np, [32, 64], axis=-1)
    q_np = q_np.reshape(1, 3, 4, 8)
    k_np = k_np.reshape(1, 3, 4, 8)
    v_np = v_np.reshape(1, 3, 4, 8)
    q_np = q_np / np.sqrt(np.sum(q_np**2, axis=-1, keepdims=True) + 1e-6)
    q_np = q_np * (8**-0.5)
    k_np = k_np / np.sqrt(np.sum(k_np**2, axis=-1, keepdims=True) + 1e-6)
    beta_np = 1.0 / (1.0 + np.exp(-b))
    g_np = -np.exp(np.asarray(gdn.A_log.value)) * _numpy_softplus(
        a + np.asarray(gdn.dt_bias.value)
    )
    state_np = np.zeros((1, 4, 8, 8), dtype=np.float32)
    core_np = np.zeros((1, 3, 4, 8), dtype=np.float32)
    for t_idx in range(3):
      state_np = state_np * np.exp(g_np[:, t_idx, :, None, None])
      kv_mem = np.sum(state_np * k_np[:, t_idx, :, :, None], axis=-2)
      delta = (v_np[:, t_idx] - kv_mem) * beta_np[:, t_idx, :, None]
      state_np = state_np + k_np[:, t_idx, :, :, None] * delta[:, :, None, :]
      core_np[:, t_idx] = np.sum(state_np * q_np[:, t_idx, :, :, None], axis=-2)
    rms = np.sqrt(np.mean(core_np**2, axis=-1, keepdims=True) + cfg.norm_eps)
    normed = (core_np / rms) * np.asarray(gdn.norm.w.value) * _numpy_silu(z)
    ref_gdn_out = normed.reshape(1, 3, 32) @ np.asarray(
        gdn.out_proj.kernel.value
    )
    np.testing.assert_allclose(
        np.asarray(gdn_out), ref_gdn_out, rtol=1e-5, atol=1e-5
    )
    assert new_cache is not None
    np.testing.assert_allclose(
        np.asarray(new_cache['recurrent_state']),
        state_np,
        rtol=1e-5,
        atol=1e-5,
    )

  def test_hybrid_model_prefill_and_decode_consistency(self):
    cfg = _make_tiny_qwen3p5_config()
    model = qwen3p5_model.Qwen3p5(cfg, rngs=nnx.Rngs(42))
    for layer in model.layers:
      layer.mlp.gate_proj.kernel.value = jax.random.normal(
          jax.random.key(10), layer.mlp.gate_proj.kernel.value.shape
      )
      layer.mlp.up_proj.kernel.value = jax.random.normal(
          jax.random.key(11), layer.mlp.up_proj.kernel.value.shape
      )
      layer.mlp.down_proj.kernel.value = jax.random.normal(
          jax.random.key(12), layer.mlp.down_proj.kernel.value.shape
      )

    tokens = jnp.array([[5, 12, 19, 23, 31, 44]], dtype=jnp.int32)
    positions = jnp.arange(6, dtype=jnp.int32)[None, :]
    full_mask = jnp.tril(jnp.ones((1, 6, 6), dtype=jnp.bool_))
    full_logits, _ = model(tokens, positions, None, full_mask)

    cache_size = 8
    cache = model.init_cache(
        batch_size=1, cache_size=cache_size, dtype=jnp.float32
    )
    prefill_mask = gen_utils.make_causal_attn_mask(
        jnp.ones((1, 4), dtype=jnp.bool_), cache_size
    )
    prefill_logits, cache = model(
        tokens[:, :4], positions[:, :4], cache, prefill_mask
    )

    decode_mask_4 = gen_utils.compute_attention_masks(
        4, cache_size, jnp.zeros((1, 8), dtype=jnp.bool_)
    )
    logits_4, cache = model(
        tokens[:, 4:5], positions[:, 4:5], cache, decode_mask_4
    )

    decode_mask_5 = gen_utils.compute_attention_masks(
        5, cache_size, jnp.zeros((1, 8), dtype=jnp.bool_)
    )
    logits_5, cache = model(
        tokens[:, 5:6], positions[:, 5:6], cache, decode_mask_5
    )

    combined = jnp.concatenate([prefill_logits, logits_4, logits_5], axis=1)
    np.testing.assert_allclose(
        np.asarray(full_logits), np.asarray(combined), rtol=1e-4, atol=1e-4
    )

  @parameterized.named_parameters(
      dict(testcase_name='eager', use_flash_attention=False),
      dict(testcase_name='flash', use_flash_attention=True),
  )
  def test_left_padded_prefill_and_decode_matches_unpadded(
      self, use_flash_attention: bool
  ):
    cfg = _make_tiny_qwen3p5_config()
    cfg.use_flash_attention = use_flash_attention
    cfg.flash_attention_block_size = 2
    model = qwen3p5_model.Qwen3p5(cfg, rngs=nnx.Rngs(7))

    with _patch_cpu_splash():
      unpadded_tokens = jnp.array([[11, 22, 33]], dtype=jnp.int32)
      unpadded_pos = jnp.array([[0, 1, 2]], dtype=jnp.int32)
      unpadded_mask = gen_utils.make_causal_attn_mask(
          jnp.ones((1, 3), dtype=jnp.bool_), cache_size=8
      )
      cache_unpadded = model.init_cache(1, 8, jnp.float32)
      logits_unpadded, cache_unpadded = model(
          unpadded_tokens, unpadded_pos, cache_unpadded, unpadded_mask
      )
      decode_mask_unpadded = gen_utils.compute_attention_masks(
          3, 8, jnp.zeros((1, 8), dtype=jnp.bool_)
      )
      next_logits_unpadded, _ = model(
          jnp.array([[44]], dtype=jnp.int32),
          jnp.array([[3]], dtype=jnp.int32),
          cache_unpadded,
          decode_mask_unpadded,
      )

      padded_tokens = jnp.array([[0, 0, 11, 22, 33]], dtype=jnp.int32)
      input_mask = jnp.array([[False, False, True, True, True]])
      padded_pos = gen_utils.build_positions_from_mask(input_mask)
      padded_attn_mask = gen_utils.make_causal_attn_mask(
          input_mask, cache_size=8
      )
      cache_padded = model.init_cache(1, 8, jnp.float32)
      logits_padded, cache_padded = model(
          padded_tokens, padded_pos, cache_padded, padded_attn_mask
      )
      full_pad_mask = jnp.array(
          [[True, True, False, False, False, False, False, False]]
      )
      decode_mask_padded = gen_utils.compute_attention_masks(
          5, 8, full_pad_mask
      )
      next_logits_padded, _ = model(
          jnp.array([[44]], dtype=jnp.int32),
          jnp.array([[3]], dtype=jnp.int32),
          cache_padded,
          decode_mask_padded,
      )

    np.testing.assert_allclose(
        np.asarray(logits_unpadded[:, -1, :]),
        np.asarray(logits_padded[:, -1, :]),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(next_logits_unpadded),
        np.asarray(next_logits_padded),
        rtol=1e-5,
        atol=1e-5,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='multimodal_prefix_tied',
          use_language_model_prefix=True,
          use_tied_embedding=True,
      ),
      dict(
          testcase_name='standalone_prefix_untied',
          use_language_model_prefix=False,
          use_tied_embedding=False,
      ),
  )
  def test_safetensors_loading(
      self, use_language_model_prefix: bool, use_tied_embedding: bool
  ):
    cfg = _make_tiny_qwen3p5_config(use_tied_embedding=use_tied_embedding)
    ref_model = qwen3p5_model.Qwen3p5(cfg, rngs=nnx.Rngs(99))
    for layer in ref_model.layers:
      layer.input_layernorm.w.value = jax.random.normal(
          jax.random.key(20), layer.input_layernorm.w.value.shape
      )
      layer.post_attention_layernorm.w.value = jax.random.normal(
          jax.random.key(21), layer.post_attention_layernorm.w.value.shape
      )
      layer.mlp.gate_proj.kernel.value = jax.random.normal(
          jax.random.key(22), layer.mlp.gate_proj.kernel.value.shape
      )
      layer.mlp.up_proj.kernel.value = jax.random.normal(
          jax.random.key(23), layer.mlp.up_proj.kernel.value.shape
      )
      layer.mlp.down_proj.kernel.value = jax.random.normal(
          jax.random.key(24), layer.mlp.down_proj.kernel.value.shape
      )

    ckpt_dir = self.create_tempdir().full_path
    root = 'model.language_model' if use_language_model_prefix else 'model'
    state = {
        f'{root}.embed_tokens.weight': np.asarray(
            ref_model.embedder.input_embedding.value
        ),
        f'{root}.norm.weight': np.asarray(ref_model.final_norm.w.value),
        'model.visual.patch_embed.proj.weight': np.ones(
            (4, 3, 2, 2), dtype=np.float32
        ),
        'mtp.fc.weight': np.ones((8, 8), dtype=np.float32),
    }
    if not use_tied_embedding:
      lm_head_key = (
          'model.language_model.lm_head.weight'
          if use_language_model_prefix
          else 'lm_head.weight'
      )
      state[lm_head_key] = np.asarray(ref_model.lm_head.w.value).T

    for i, layer in enumerate(ref_model.layers):
      p = f'{root}.layers.{i}'
      state[f'{p}.input_layernorm.weight'] = np.asarray(
          layer.input_layernorm.w.value
      )
      state[f'{p}.post_attention_layernorm.weight'] = np.asarray(
          layer.post_attention_layernorm.w.value
      )
      state[f'{p}.mlp.gate_proj.weight'] = np.asarray(
          layer.mlp.gate_proj.kernel.value
      ).T
      state[f'{p}.mlp.up_proj.weight'] = np.asarray(
          layer.mlp.up_proj.kernel.value
      ).T
      state[f'{p}.mlp.down_proj.weight'] = np.asarray(
          layer.mlp.down_proj.kernel.value
      ).T
      if layer.layer_type == qwen3p5_model.LayerType.FULL_ATTENTION:
        state[f'{p}.self_attn.q_proj.weight'] = (
            np.asarray(layer.attn.q_proj.w.value).reshape(cfg.embed_dim, -1).T
        )
        state[f'{p}.self_attn.k_proj.weight'] = (
            np.asarray(layer.attn.k_proj.w.value).reshape(cfg.embed_dim, -1).T
        )
        state[f'{p}.self_attn.v_proj.weight'] = (
            np.asarray(layer.attn.v_proj.w.value).reshape(cfg.embed_dim, -1).T
        )
        state[f'{p}.self_attn.o_proj.weight'] = (
            np.asarray(layer.attn.o_proj.w.value).reshape(-1, cfg.embed_dim).T
        )
        state[f'{p}.self_attn.q_norm.weight'] = np.asarray(
            layer.attn.q_norm.w.value
        )
        state[f'{p}.self_attn.k_norm.weight'] = np.asarray(
            layer.attn.k_norm.w.value
        )
      else:
        state[f'{p}.linear_attn.in_proj_qkv.weight'] = np.asarray(
            layer.linear_attn.in_proj_qkv.kernel.value
        ).T
        state[f'{p}.linear_attn.in_proj_z.weight'] = np.asarray(
            layer.linear_attn.in_proj_z.kernel.value
        ).T
        state[f'{p}.linear_attn.in_proj_b.weight'] = np.asarray(
            layer.linear_attn.in_proj_b.kernel.value
        ).T
        state[f'{p}.linear_attn.in_proj_a.weight'] = np.asarray(
            layer.linear_attn.in_proj_a.kernel.value
        ).T
        state[f'{p}.linear_attn.conv1d.weight'] = np.transpose(
            np.asarray(layer.linear_attn.conv1d.kernel.value), (2, 1, 0)
        )
        state[f'{p}.linear_attn.dt_bias'] = np.asarray(
            layer.linear_attn.dt_bias.value
        )
        state[f'{p}.linear_attn.A_log'] = np.asarray(
            layer.linear_attn.A_log.value
        )
        state[f'{p}.linear_attn.norm.weight'] = np.asarray(
            layer.linear_attn.norm.w.value
        )
        state[f'{p}.linear_attn.out_proj.weight'] = np.asarray(
            layer.linear_attn.out_proj.kernel.value
        ).T

    for k, v in state.items():
      state[k] = np.ascontiguousarray(v)
    safe_np.save_file(state, os.path.join(ckpt_dir, 'model.safetensors'))

    loaded_model = qwen3p5_params.create_model_from_safe_tensors(
        file_dir=ckpt_dir,
        config=cfg,
        mesh=None,
        dtype=jnp.float32,
    )

    tokens = jnp.array([[1, 4, 9, 16]], dtype=jnp.int32)
    positions = jnp.arange(4, dtype=jnp.int32)[None, :]
    mask = jnp.tril(jnp.ones((1, 4, 4), dtype=jnp.bool_))
    ref_logits, _ = ref_model(tokens, positions, None, mask)
    loaded_logits, _ = loaded_model(tokens, positions, None, mask)
    np.testing.assert_allclose(
        np.asarray(ref_logits), np.asarray(loaded_logits), rtol=1e-5, atol=1e-5
    )

  def test_save_lora_merged_model_as_safetensors_end_to_end(self):
    cfg = _make_tiny_qwen3p5_config(use_tied_embedding=True)
    base_model = qwen3p5_model.Qwen3p5(cfg, rngs=nnx.Rngs(33))
    base_dir = self.create_tempdir().full_path
    merged_dir = self.create_tempdir().full_path

    state = {
        'model.language_model.embed_tokens.weight': np.asarray(
            base_model.embedder.input_embedding.value
        ),
        'model.language_model.norm.weight': np.asarray(
            base_model.final_norm.w.value
        ),
    }
    for i, layer in enumerate(base_model.layers):
      p = f'model.language_model.layers.{i}'
      state[f'{p}.input_layernorm.weight'] = np.asarray(
          layer.input_layernorm.w.value
      )
      state[f'{p}.post_attention_layernorm.weight'] = np.asarray(
          layer.post_attention_layernorm.w.value
      )
      state[f'{p}.mlp.gate_proj.weight'] = np.asarray(
          layer.mlp.gate_proj.kernel.value
      ).T
      state[f'{p}.mlp.up_proj.weight'] = np.asarray(
          layer.mlp.up_proj.kernel.value
      ).T
      state[f'{p}.mlp.down_proj.weight'] = np.asarray(
          layer.mlp.down_proj.kernel.value
      ).T
      if layer.layer_type == qwen3p5_model.LayerType.FULL_ATTENTION:
        state[f'{p}.self_attn.q_proj.weight'] = (
            np.asarray(layer.attn.q_proj.w.value).reshape(cfg.embed_dim, -1).T
        )
        state[f'{p}.self_attn.k_proj.weight'] = (
            np.asarray(layer.attn.k_proj.w.value).reshape(cfg.embed_dim, -1).T
        )
        state[f'{p}.self_attn.v_proj.weight'] = (
            np.asarray(layer.attn.v_proj.w.value).reshape(cfg.embed_dim, -1).T
        )
        state[f'{p}.self_attn.o_proj.weight'] = (
            np.asarray(layer.attn.o_proj.w.value).reshape(-1, cfg.embed_dim).T
        )
        state[f'{p}.self_attn.q_norm.weight'] = np.asarray(
            layer.attn.q_norm.w.value
        )
        state[f'{p}.self_attn.k_norm.weight'] = np.asarray(
            layer.attn.k_norm.w.value
        )
      else:
        state[f'{p}.linear_attn.in_proj_qkv.weight'] = np.asarray(
            layer.linear_attn.in_proj_qkv.kernel.value
        ).T
        state[f'{p}.linear_attn.in_proj_z.weight'] = np.asarray(
            layer.linear_attn.in_proj_z.kernel.value
        ).T
        state[f'{p}.linear_attn.in_proj_b.weight'] = np.asarray(
            layer.linear_attn.in_proj_b.kernel.value
        ).T
        state[f'{p}.linear_attn.in_proj_a.weight'] = np.asarray(
            layer.linear_attn.in_proj_a.kernel.value
        ).T
        state[f'{p}.linear_attn.conv1d.weight'] = np.transpose(
            np.asarray(layer.linear_attn.conv1d.kernel.value), (2, 1, 0)
        )
        state[f'{p}.linear_attn.dt_bias'] = np.asarray(
            layer.linear_attn.dt_bias.value
        )
        state[f'{p}.linear_attn.A_log'] = np.asarray(
            layer.linear_attn.A_log.value
        )
        state[f'{p}.linear_attn.norm.weight'] = np.asarray(
            layer.linear_attn.norm.w.value
        )
        state[f'{p}.linear_attn.out_proj.weight'] = np.asarray(
            layer.linear_attn.out_proj.kernel.value
        ).T
    for k, v in state.items():
      state[k] = np.ascontiguousarray(v)
    safe_np.save_file(state, os.path.join(base_dir, 'model.safetensors'))

    rank = 2
    alpha = 4.0
    q_lora_a = jax.random.normal(jax.random.key(50), (cfg.embed_dim, rank))
    q_lora_b = jax.random.normal(
        jax.random.key(51), (rank, cfg.num_heads, cfg.head_dim * 2)
    )
    base_model.layers[3].attn.q_proj.lora_a = nnx.LoRAParam(q_lora_a)
    base_model.layers[3].attn.q_proj.lora_b = nnx.LoRAParam(q_lora_b)

    qwen3p5_params.save_lora_merged_model_as_safetensors(
        local_model_path=base_dir,
        output_dir=merged_dir,
        lora_model=base_model,
        rank=rank,
        alpha=alpha,
    )

    # Apply the same LoRA delta directly to `q_proj.w` and compare with reloaded
    base_model.layers[3].attn.q_proj.w.value = base_model.layers[
        3
    ].attn.q_proj.w.value + jnp.einsum('dr,rnh->dnh', q_lora_a, q_lora_b) * (
        alpha / rank
    )
    reloaded = qwen3p5_params.create_model_from_safe_tensors(
        file_dir=merged_dir, config=cfg, mesh=None, dtype=jnp.float32
    )
    tokens = jnp.array([[2, 6, 10, 14]], dtype=jnp.int32)
    positions = jnp.arange(4, dtype=jnp.int32)[None, :]
    mask = jnp.tril(jnp.ones((1, 4, 4), dtype=jnp.bool_))
    expected_logits, _ = base_model(tokens, positions, None, mask)
    reloaded_logits, _ = reloaded(tokens, positions, None, mask)
    np.testing.assert_allclose(
        np.asarray(expected_logits),
        np.asarray(reloaded_logits),
        rtol=1e-5,
        atol=1e-5,
    )

  def test_remat_and_skip_lm_head(self):
    tokens = jnp.array([[1, 4, 9, 16]], dtype=jnp.int32)
    positions = jnp.arange(4, dtype=jnp.int32)[None, :]
    mask = jnp.tril(jnp.ones((1, 4, 4), dtype=jnp.bool_))

    cfg_none = _make_tiny_qwen3p5_config()
    model_none = qwen3p5_model.Qwen3p5(cfg_none, rngs=nnx.Rngs(123))
    logits_none, _ = model_none(tokens, positions, None, mask)

    hidden, _ = model_none(tokens, positions, None, mask, skip_lm_head=True)
    self.assertEqual(hidden.shape, (1, 4, cfg_none.embed_dim))
    logits_from_hidden = model_none.compute_final_logits(hidden)
    np.testing.assert_allclose(
        np.asarray(logits_none),
        np.asarray(logits_from_hidden),
        rtol=1e-5,
        atol=1e-5,
    )

    for remat_cfg in (
        qwen3p5_model.RematConfig.BLOCK,
        qwen3p5_model.RematConfig.DECODER,
    ):
      cfg = _make_tiny_qwen3p5_config()
      cfg.remat_config = remat_cfg
      model = qwen3p5_model.Qwen3p5(cfg, rngs=nnx.Rngs(123))
      logits_remat, _ = model(tokens, positions, None, mask)
      np.testing.assert_allclose(
          np.asarray(logits_none),
          np.asarray(logits_remat),
          rtol=1e-5,
          atol=1e-5,
      )

  def test_bf16_model_upcasts_final_logits_to_float32(self):
    cfg = _make_tiny_qwen3p5_config()
    cfg.dtype = jnp.bfloat16
    cfg.param_dtype = jnp.bfloat16
    model = qwen3p5_model.Qwen3p5(cfg, rngs=nnx.Rngs(0))
    tokens = jnp.array([[1, 2, 3]], dtype=jnp.int32)
    positions = jnp.arange(3, dtype=jnp.int32)[None, :]
    mask = jnp.tril(jnp.ones((1, 3, 3), dtype=jnp.bool_))
    logits, _ = model(tokens, positions, None, mask)
    self.assertEqual(logits.dtype, jnp.float32)

  def test_moe_config_raises_not_implemented_error(self):
    cfg = _make_tiny_qwen3p5_config()
    cfg.num_experts = 8
    cfg.num_experts_per_tok = 2
    with self.assertRaises(NotImplementedError):
      qwen3p5_model.Qwen3p5(cfg, rngs=nnx.Rngs(0))

  def test_backend_mappings_unavailable_raises_runtime_error(self):
    with self.assertRaises(RuntimeError):
      qwen3p5_model.Qwen3p5.mapping_for('vllm_jax')
    with self.assertRaises(RuntimeError):
      qwen3p5_model.Qwen3p5.mapping_for('sglang_jax')

  def test_segment_ids_with_cache_raises_value_error(self):
    cfg = _make_tiny_qwen3p5_config()
    model = qwen3p5_model.Qwen3p5(cfg, rngs=nnx.Rngs(0))
    cache = model.init_cache(batch_size=1, cache_size=8, dtype=jnp.float32)
    tokens = jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)
    positions = jnp.arange(4, dtype=jnp.int32)[None, :]
    segment_ids = jnp.array([[1, 1, 1, 1]], dtype=jnp.int32)
    with self.assertRaises(ValueError):
      model(tokens, positions, cache, None, segment_ids=segment_ids)

  def test_state_key_to_safetensors_key_prefix_handling(self):
    self.assertEqual(
        qwen3p5_params._qwen3p5_state_key_to_safetensors_key(
            'layers.3.attn.q_proj'
        ),
        'model.language_model.layers.3.self_attn.q_proj.weight',
    )
    self.assertEqual(
        qwen3p5_params._qwen3p5_state_key_to_safetensors_key(
            'layers.3.attn.q_proj', use_language_model_prefix=False
        ),
        'model.layers.3.self_attn.q_proj.weight',
    )

  def test_flash_attention_matches_eager_and_supports_segment_ids(self):
    cfg_eager = _make_tiny_qwen3p5_config()
    cfg_flash = _make_tiny_qwen3p5_config()
    cfg_flash.use_flash_attention = True
    cfg_flash.flash_attention_block_size = 4

    model_eager = qwen3p5_model.Qwen3p5(cfg_eager, rngs=nnx.Rngs(77))
    model_flash = qwen3p5_model.Qwen3p5(cfg_flash, rngs=nnx.Rngs(77))

    tokens = jnp.array([[3, 7, 11, 15], [0, 5, 9, 13]], dtype=jnp.int32)
    positions = jnp.array([[0, 1, 2, 3], [0, 0, 1, 2]], dtype=jnp.int32)
    causal_mask = jnp.tril(jnp.ones((2, 4, 4), dtype=jnp.bool_))
    segment_ids = jnp.array([[1, 1, 2, 2], [0, 1, 1, 1]], dtype=jnp.int32)

    with _patch_cpu_splash():
      logits_eager_no_seg, _ = model_eager(tokens, positions, None, causal_mask)
      logits_flash_no_seg, _ = model_flash(tokens, positions, None, causal_mask)
      np.testing.assert_allclose(
          np.asarray(logits_eager_no_seg),
          np.asarray(logits_flash_no_seg),
          rtol=1e-5,
          atol=1e-5,
      )

      logits_eager_seg, _ = model_eager(
          tokens, positions, None, causal_mask, segment_ids=segment_ids
      )
      logits_flash_seg, _ = model_flash(
          tokens, positions, None, causal_mask, segment_ids=segment_ids
      )
      np.testing.assert_allclose(
          np.asarray(logits_eager_seg),
          np.asarray(logits_flash_seg),
          rtol=1e-5,
          atol=1e-5,
      )

  def test_flash_attention_tokamax_backend(self):
    cfg_eager = _make_tiny_qwen3p5_config()
    cfg_tokamax = _make_tiny_qwen3p5_config()
    cfg_tokamax.use_flash_attention = True
    cfg_tokamax.flash_attention_block_size = 4
    cfg_tokamax.splash_attention_impl = (
        qwen3p5_model.SplashAttentionImpl.TOKAMAX
    )

    model_eager = qwen3p5_model.Qwen3p5(cfg_eager, rngs=nnx.Rngs(77))
    model_tokamax = qwen3p5_model.Qwen3p5(cfg_tokamax, rngs=nnx.Rngs(77))

    tokens = jnp.array([[3, 7, 11, 15]], dtype=jnp.int32)
    positions = jnp.array([[0, 1, 2, 3]], dtype=jnp.int32)
    segment_ids = jnp.array([[1, 1, 2, 2]], dtype=jnp.int32)

    fake_tokamax_splash = types.SimpleNamespace(
        SplashConfig=lambda **kwargs: kwargs,
        make_splash_mha=lambda *a, **kw: _ReferenceCausalSplashKernel(),
        SegmentIds=qwen3p5_model.splash.SegmentIds,
    )
    fake_tokamax_mask = types.SimpleNamespace(
        make_causal_mask=lambda shape: shape,
    )

    with (
        mock.patch.object(qwen3p5_model, 'shard_map', _identity_shard_map),
        mock.patch.object(
            qwen3p5_model.shd, 'NamedSharding', lambda *a, **kw: None
        ),
        mock.patch.object(
            qwen3p5_model,
            '_tokamax_splash_libs',
            return_value=(fake_tokamax_splash, fake_tokamax_mask),
        ),
    ):
      logits_eager, _ = model_eager(
          tokens, positions, None, None, segment_ids=segment_ids
      )
      logits_tokamax, _ = model_tokamax(
          tokens, positions, None, None, segment_ids=segment_ids
      )
      np.testing.assert_allclose(
          np.asarray(logits_eager),
          np.asarray(logits_tokamax),
          rtol=1e-5,
          atol=1e-5,
      )

  def test_align_kv_heads_for_splash_gqa_head_sharding(self):
    # Qwen3.5-0.8B ratio: 8 query heads, 2 KV heads, sharded across tp=4 or tp=8
    q = jax.random.normal(jax.random.key(1), (1, 8, 4, 16))
    k = jax.random.normal(jax.random.key(2), (1, 2, 4, 16))
    v = jax.random.normal(jax.random.key(3), (1, 2, 4, 16))
    kernel = _ReferenceCausalSplashKernel()

    ref_out = jax.vmap(kernel)(q, k, v)

    for head_shards in (2, 4, 8):
      k_aligned, v_aligned, kh_eff = qwen3p5_model._align_kv_heads_for_splash(
          k, v, num_q_heads=8, head_shards=head_shards
      )
      self.assertEqual(kh_eff % head_shards, 0)
      q_per_shard = 8 // head_shards
      kv_per_shard = kh_eff // head_shards
      shard_outs = []
      for s in range(head_shards):
        q_s = q[:, s * q_per_shard : (s + 1) * q_per_shard]
        k_s = k_aligned[:, s * kv_per_shard : (s + 1) * kv_per_shard]
        v_s = v_aligned[:, s * kv_per_shard : (s + 1) * kv_per_shard]
        shard_outs.append(jax.vmap(kernel)(q_s, k_s, v_s))
      sharded_out = jnp.concatenate(shard_outs, axis=1)
      np.testing.assert_allclose(
          np.asarray(ref_out), np.asarray(sharded_out), rtol=1e-6, atol=1e-6
      )

    with self.assertRaises(ValueError):
      qwen3p5_model._align_kv_heads_for_splash(
          k, v, num_q_heads=8, head_shards=3
      )

  @parameterized.named_parameters(
      dict(testcase_name='eager', use_flash_attention=False),
      dict(testcase_name='flash', use_flash_attention=True),
  )
  def test_sequence_packing_segment_ids_isolation_without_attn_mask(
      self, use_flash_attention: bool
  ):
    cfg = _make_tiny_qwen3p5_config()
    cfg.use_flash_attention = use_flash_attention
    cfg.flash_attention_block_size = 2
    model = qwen3p5_model.Qwen3p5(cfg, rngs=nnx.Rngs(101))

    with _patch_cpu_splash():
      seq_a = jnp.array([[4, 8]], dtype=jnp.int32)
      pos_a = jnp.array([[0, 1]], dtype=jnp.int32)
      causal_a = jnp.tril(jnp.ones((1, 2, 2), dtype=jnp.bool_))
      logits_a, _ = model(seq_a, pos_a, None, causal_a)

      seq_b = jnp.array([[15, 29, 42]], dtype=jnp.int32)
      pos_b = jnp.array([[0, 1, 2]], dtype=jnp.int32)
      causal_b = jnp.tril(jnp.ones((1, 3, 3), dtype=jnp.bool_))
      logits_b, _ = model(seq_b, pos_b, None, causal_b)

      # Packed sequence with attention_mask=None (matching rl/common.py)
      packed_tokens = jnp.concatenate([seq_a, seq_b], axis=1)
      packed_positions = jnp.concatenate([pos_a, pos_b], axis=1)
      packed_segment_ids = jnp.array([[1, 1, 2, 2, 2]], dtype=jnp.int32)

      packed_logits, _ = model(
          packed_tokens,
          packed_positions,
          None,
          None,
          segment_ids=packed_segment_ids,
      )
    np.testing.assert_allclose(
        np.asarray(logits_a),
        np.asarray(packed_logits[:, :2, :]),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(logits_b),
        np.asarray(packed_logits[:, 2:, :]),
        rtol=1e-5,
        atol=1e-5,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='chunk2_t7_unpacked',
          chunk_size=2,
          seq_len=7,
          packed=False,
      ),
      dict(
          testcase_name='chunk4_t11_unpacked',
          chunk_size=4,
          seq_len=11,
          packed=False,
      ),
      dict(
          testcase_name='chunk64_t7_unpacked',
          chunk_size=64,
          seq_len=7,
          packed=False,
      ),
      dict(
          testcase_name='chunk2_t7_packed',
          chunk_size=2,
          seq_len=7,
          packed=True,
      ),
      dict(
          testcase_name='chunk4_t11_packed',
          chunk_size=4,
          seq_len=11,
          packed=True,
      ),
  )
  def test_chunk_gated_delta_rule_matches_token_scan_forward_and_backward(
      self, chunk_size: int, seq_len: int, packed: bool
  ):
    batch_size, num_heads, head_k_dim, head_v_dim = 2, 4, 8, 8
    q = qwen3p5_model._l2norm(
        jax.random.normal(
            jax.random.key(1), (batch_size, seq_len, num_heads, head_k_dim)
        )
    ) * (head_k_dim**-0.5)
    k = qwen3p5_model._l2norm(
        jax.random.normal(
            jax.random.key(2), (batch_size, seq_len, num_heads, head_k_dim)
        )
    )
    v = jax.random.normal(
        jax.random.key(3), (batch_size, seq_len, num_heads, head_v_dim)
    )
    g = -jax.nn.softplus(
        jax.random.normal(jax.random.key(4), (batch_size, seq_len, num_heads))
    )
    beta = jax.nn.sigmoid(
        jax.random.normal(jax.random.key(5), (batch_size, seq_len, num_heads))
    )
    init_state = jax.random.normal(
        jax.random.key(6), (batch_size, num_heads, head_k_dim, head_v_dim)
    )

    if packed:
      if seq_len == 7:
        valid_seg = jnp.array(
            [[1, 1, 1, 2, 2, 0, 0], [0, 1, 1, 2, 3, 3, 3]], dtype=jnp.int32
        )
      else:
        valid_seg = jnp.array(
            [
                [1, 1, 2, 2, 2, 3, 3, 0, 4, 4, 0],
                [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
            ],
            dtype=jnp.int32,
        )
      token_mask = valid_seg != 0
      k = jnp.where(token_mask[:, :, None, None], k, 0.0)
      v = jnp.where(token_mask[:, :, None, None], v, 0.0)
      g = jnp.where(token_mask[:, :, None], g, 0.0)
      beta = jnp.where(token_mask[:, :, None], beta, 0.0)
    else:
      valid_seg = None

    def _ref_scan(q_in, k_in, v_in, g_in, beta_in, state_in):
      q_t = jnp.transpose(q_in, (1, 0, 2, 3))
      k_t = jnp.transpose(k_in, (1, 0, 2, 3))
      v_t = jnp.transpose(v_in, (1, 0, 2, 3))
      g_t = jnp.transpose(g_in, (1, 0, 2))
      beta_t = jnp.transpose(beta_in, (1, 0, 2))
      if valid_seg is not None:
        prev_seg = jnp.pad(
            valid_seg[:, :-1], ((0, 0), (1, 0)), constant_values=0
        )
        keep_t = jnp.transpose(
            (valid_seg == prev_seg) & (valid_seg != 0), (1, 0)
        )
      else:
        keep_t = jnp.ones((seq_len, batch_size), dtype=jnp.bool_)

      def _step(state, xs):
        q_i, k_i, v_i, g_i, beta_i, keep_i = xs
        state = jnp.where(keep_i[:, None, None, None], state, 0.0)
        state = state * jnp.exp(g_i)[..., None, None]
        kv_mem = jnp.sum(state * k_i[..., None], axis=-2)
        delta = (v_i - kv_mem) * beta_i[..., None]
        state = state + k_i[..., :, None] * delta[..., None, :]
        out_i = jnp.sum(state * q_i[..., None], axis=-2)
        return state, out_i

      final_s, out_t = jax.lax.scan(
          _step, state_in, (q_t, k_t, v_t, g_t, beta_t, keep_t)
      )
      return final_s, jnp.transpose(out_t, (1, 0, 2, 3))

    ref_state, ref_out = _ref_scan(q, k, v, g, beta, init_state)
    chunk_state, chunk_out = qwen3p5_model._chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        init_state,
        valid_seg=valid_seg,
        chunk_size=chunk_size,
    )
    if packed and valid_seg is not None:
      ref_out = jnp.where((valid_seg != 0)[:, :, None, None], ref_out, 0.0)
      chunk_out = jnp.where((valid_seg != 0)[:, :, None, None], chunk_out, 0.0)

    np.testing.assert_allclose(
        np.asarray(chunk_out), np.asarray(ref_out), rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        np.asarray(chunk_state), np.asarray(ref_state), rtol=1e-5, atol=1e-5
    )

    cotangent_out = jax.random.normal(jax.random.key(7), ref_out.shape)
    cotangent_state = jax.random.normal(jax.random.key(8), ref_state.shape)
    if packed and valid_seg is not None:
      cotangent_out = jnp.where(
          (valid_seg != 0)[:, :, None, None], cotangent_out, 0.0
      )

    def _loss_ref(q_in, k_in, v_in, g_in, beta_in, state_in):
      s_out, o_out = _ref_scan(q_in, k_in, v_in, g_in, beta_in, state_in)
      return jnp.sum(o_out * cotangent_out) + jnp.sum(s_out * cotangent_state)

    def _loss_chunk(q_in, k_in, v_in, g_in, beta_in, state_in):
      s_out, o_out = qwen3p5_model._chunk_gated_delta_rule(
          q_in,
          k_in,
          v_in,
          g_in,
          beta_in,
          state_in,
          valid_seg=valid_seg,
          chunk_size=chunk_size,
      )
      return jnp.sum(o_out * cotangent_out) + jnp.sum(s_out * cotangent_state)

    grads_ref = jax.grad(_loss_ref, argnums=(0, 1, 2, 3, 4, 5))(
        q, k, v, g, beta, init_state
    )
    grads_chunk = jax.grad(_loss_chunk, argnums=(0, 1, 2, 3, 4, 5))(
        q, k, v, g, beta, init_state
    )
    for g_r, g_c in zip(grads_ref, grads_chunk):
      np.testing.assert_allclose(
          np.asarray(g_c), np.asarray(g_r), rtol=1e-4, atol=1e-4
      )

  def test_sharding_config_alignment(self):
    shd_train = qwen3p5_model.ShardingConfig.get_default_sharding(
        is_sampling=False
    )
    self.assertEqual(shd_train.act_btd, ('fsdp', None, None))
    self.assertEqual(shd_train.rms_norm_weight, (None,))
    self.assertEqual(shd_train.head_norm_weight, (None,))
    self.assertEqual(shd_train.linear_ba_dh, ('fsdp', 'tp'))
    self.assertEqual(shd_train.linear_head_h, ('tp',))

    cfg = _make_tiny_qwen3p5_config()
    model = qwen3p5_model.Qwen3p5(cfg, rngs=nnx.Rngs(0))
    pspecs = nnx.get_partition_spec(nnx.state(model))
    part_spec = jax.sharding.PartitionSpec
    self.assertEqual(
        pspecs.layers[0].linear_attn.norm.w.value,
        part_spec(*shd_train.head_norm_weight),
    )
    self.assertEqual(
        pspecs.layers[0].linear_attn.dt_bias.value,
        part_spec(*shd_train.linear_head_h),
    )
    self.assertEqual(
        pspecs.layers[0].linear_attn.A_log.value,
        part_spec(*shd_train.linear_head_h),
    )
    self.assertEqual(
        pspecs.layers[3].attn.q_norm.w.value,
        part_spec(*shd_train.head_norm_weight),
    )
    self.assertEqual(
        pspecs.layers[3].attn.k_norm.w.value,
        part_spec(*shd_train.head_norm_weight),
    )


if __name__ == '__main__':
  absltest.main()
