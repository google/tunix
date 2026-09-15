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

"""Tests for Gemma 4 Attention module and Pallas Splash kernels."""

from __future__ import annotations

import dataclasses
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask as mask_lib
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import numpy as np
from tunix.models.gemma4 import attention as attention_lib
from tunix.models.gemma4 import config as config_lib
from tunix.models.gemma4 import model as model_lib


class FlashAttentionMaskTest(parameterized.TestCase):
  """Mask correctness unit tests (pure numpy — no model needed)."""

  def test_local_mask_matches_manual(self):
    """Verify LocalMask with offset produces the correct sliding window mask."""
    chunk_len = 1024
    sw_size = 512
    cache_len = sw_size
    kv_len = cache_len + chunk_len
    prefix_len = cache_len

    # Splash mask with offset
    splash_mask = mask_lib.LocalMask(
        (chunk_len, kv_len),
        window_size=(sw_size - 1, 0),
        offset=prefix_len,
    )
    splash_array = splash_mask[np.s_[:, :]]

    # Reference mask for local sliding window with offset.
    position_offset = prefix_len
    valid_cache_len = prefix_len
    row_pos = np.arange(chunk_len) + position_offset
    col_pos_cache = np.arange(cache_len) + (position_offset - valid_cache_len)
    col_pos_suffix = np.arange(chunk_len) + position_offset
    col_pos = np.concatenate([col_pos_cache, col_pos_suffix])
    manual_mask = (col_pos[None, :] > (row_pos[:, None] - sw_size)) & (
        col_pos[None, :] <= row_pos[:, None]
    )

    np.testing.assert_array_equal(splash_array, manual_mask)

  def test_causal_mask_matches_manual(self):
    """Verify CausalMask with offset for GLOBAL chunked prefill."""
    chunk_len = 1024
    prefix_len = 2048
    kv_len = prefix_len + chunk_len

    splash_mask = mask_lib.CausalMask(
        (chunk_len, kv_len),
        offset=prefix_len,
    )
    splash_array = splash_mask[np.s_[:, :]]

    # Manual: q[i] can attend to kv[j] where i + offset >= j
    row = np.arange(chunk_len)[:, None] + prefix_len
    col = np.arange(kv_len)[None, :]
    manual_mask = row >= col

    np.testing.assert_array_equal(splash_array, manual_mask)

  @parameterized.parameters(
      # (chunk_len, sw_size) — various sizes to test edge cases
      (256, 128),
      (512, 256),
      (1024, 512),
      (2048, 1024),
  )
  def test_local_mask_offset_parameterized(self, chunk_len, sw_size):
    """LocalMask with offset is correct for various chunk/window sizes."""
    cache_len = sw_size
    kv_len = cache_len + chunk_len

    splash_mask = mask_lib.LocalMask(
        (chunk_len, kv_len),
        window_size=(sw_size - 1, 0),
        offset=cache_len,
    )
    splash_array = splash_mask[np.s_[:, :]]

    # Each Q position q[i] at logical position (i + cache_len) should attend
    # to KV positions in [i + cache_len - (sw_size - 1), i + cache_len].
    for i in range(0, chunk_len, max(1, chunk_len // 8)):
      logical_q = i + cache_len
      expected_start = max(0, logical_q - (sw_size - 1))
      expected_end = logical_q
      # Verify True positions in row i
      true_cols = np.where(splash_array[i])[0]
      if len(true_cols) > 0:
        self.assertEqual(true_cols[0], expected_start)
        self.assertEqual(true_cols[-1], expected_end)
        self.assertLen(true_cols, expected_end - expected_start + 1)

  def test_local_mask_square_no_offset(self):
    """Square LocalMask (chunk 1) should produce standard sliding window."""
    seq_len = 512
    sw_size = 128

    splash_mask = mask_lib.LocalMask(
        (seq_len, seq_len),
        window_size=(sw_size - 1, 0),
        offset=0,
    )
    splash_array = splash_mask[np.s_[:, :]]

    # Manual: standard causal sliding window
    row = np.arange(seq_len)[:, None]
    col = np.arange(seq_len)[None, :]
    manual_mask = (col <= row) & (col > row - sw_size)

    np.testing.assert_array_equal(splash_array, manual_mask)

  def test_build_flash_mask_local_sliding_rectangular(self):
    config = model_lib.ModelConfig.gemma4_e2b()
    config.sliding_window_size = 512
    attn = attention_lib.Attention(
        config=config,
        attn_type=model_lib.AttentionType.LOCAL_SLIDING,
        rngs=nnx.Rngs(0),
    )
    q_len, kv_len, sw = 128, 512, 512
    offset = kv_len - q_len
    mask = attn._build_flash_mask(q_len=q_len, kv_len=kv_len, offset=offset)
    mask_array = mask[np.s_[:, :]]

    q_ids = np.arange(q_len) + offset
    kv_ids = np.arange(kv_len)
    expected = (kv_ids[None, :] > (q_ids[:, None] - sw)) & (
        kv_ids[None, :] <= q_ids[:, None]
    )
    np.testing.assert_array_equal(mask_array, expected)

  def test_eager_attention_local_sliding_rectangular_mask(self):
    """Verify eager attention local sliding window mask in rectangular prefill."""
    q_len, kv_len, sw = 128, 512, 256
    offset = kv_len - q_len
    all_ones = jnp.ones((1, q_len, kv_len), dtype=jnp.bool_)
    sliding_mask = jnp.triu(all_ones, offset - sw + 1) * jnp.tril(
        all_ones, offset + sw - 1
    )

    q_ids = np.arange(q_len) + offset
    kv_ids = np.arange(kv_len)
    expected_sliding = (kv_ids[None, :] > (q_ids[:, None] - sw)) & (
        kv_ids[None, :] < (q_ids[:, None] + sw)
    )
    np.testing.assert_array_equal(sliding_mask[0], expected_sliding)

    # Combined with causal mask:
    causal_mask = jnp.tril(all_ones, offset)
    expected_causal_sliding = expected_sliding & (
        kv_ids[None, :] <= q_ids[:, None]
    )
    np.testing.assert_array_equal(
        (sliding_mask * causal_mask)[0], expected_causal_sliding
    )

  @parameterized.named_parameters(
      dict(testcase_name='2d_mask', mask_3d=False),
      dict(testcase_name='3d_mask', mask_3d=True),
  )
  def test_eager_attention_local_sliding_rectangular_execution(self, mask_3d):
    """Verify _eager_attention with local sliding window on rectangular shapes."""
    config = model_lib.ModelConfig.gemma4_e2b()
    config.sliding_window_size = 4
    attn = attention_lib.Attention(
        config=config,
        attn_type=model_lib.AttentionType.LOCAL_SLIDING,
        rngs=nnx.Rngs(0),
    )
    b, q_len, kv_len, d = 2, 4, 16, config.head_dim
    h, kh = config.num_heads, config.num_kv_heads
    offset = kv_len - q_len

    q = jnp.ones((b, q_len, h, d))
    k = jnp.ones((b, kv_len, kh, d))
    v = jnp.arange(kv_len, dtype=jnp.float32)[None, :, None, None]
    v = jnp.broadcast_to(v, (b, kv_len, kh, d))
    mask_shape = (b, q_len, kv_len) if mask_3d else (q_len, kv_len)
    attn_mask = jnp.ones(mask_shape, dtype=jnp.bool_)
    segment_pos = jnp.broadcast_to(
        jnp.arange(offset, kv_len, dtype=jnp.int32)[None, :], (b, q_len)
    )

    out = attn._eager_attention(
        query_proj=q,
        key_proj=k,
        value_proj=v,
        attn_mask=attn_mask,
        segment_pos=segment_pos,
        cache=None,
        kv_shared_cache=None,
        seq_len=q_len,
    )
    self.assertEqual(out.shape, (b, q_len, h, d))
    self.assertFalse(jnp.isnan(out).any())
    # offset = 12. For query 0, valid kv positions are [9, 10, 11, 12, 13, 14,
    # 15]. Mean is 12.0.
    np.testing.assert_allclose(out[0, 0, 0, 0], 12.0, atol=1e-3)


class FlashAttentionBlockSizeTest(parameterized.TestCase):
  """Block-size divisibility parameterized test."""

  @parameterized.parameters(
      model_lib.ModelConfig.gemma4_e2b,
      model_lib.ModelConfig.gemma4_e4b,
      model_lib.ModelConfig.gemma4_31b,
      model_lib.ModelConfig.gemma4_26b_a4b,
  )
  def test_block_kv_divisibility_and_chunk_multipliers(self, config_factory):
    """block_kv must be 128-aligned and divide kv_len across chunk sizes."""
    config = config_factory()
    sw = config.sliding_window_size
    block_q = config.flash_attention_block_size
    block_kv = min(block_q, sw)

    self.assertEqual(
        block_kv % 128,
        0,
        f'block_kv={block_kv} not a multiple of 128 (NUM_LANES)',
    )

    for multiplier in (1, 2, 4):
      chunk_len = block_q * multiplier
      kv_len = sw + chunk_len
      self.assertEqual(
          chunk_len % block_q,
          0,
          f'chunk_len={chunk_len} not divisible by block_q={block_q}',
      )
      self.assertEqual(
          kv_len % block_kv,
          0,
          f'kv_len={kv_len} not divisible by block_kv={block_kv} '
          f'(multiplier={multiplier})',
      )


class AttentionTest(parameterized.TestCase):

  def _make_mock_splash(self, return_lse=False):
    """Returns a 1x1 mesh and a mock splash kernel that outputs zeros."""
    devices = np.array(jax.devices()[:1]).reshape(1, 1)
    mesh = jax.sharding.Mesh(devices, ('fsdp', 'tp'))

    def mock_kernel(q_in, k_in, v_in, segment_ids=None):
      del k_in, v_in, segment_ids
      out = jnp.zeros_like(q_in)
      if return_lse:
        lse = jnp.zeros(q_in.shape[:-1], dtype=jnp.float32)
        return out, (lse,)
      return out

    return mesh, mock_kernel

  def test_attention_with_segment_ids_rectangular_routing(self):
    config = model_lib.ModelConfig.gemma4_e2b()
    config.use_flash_attention = True
    config.flash_attention_block_size = 16

    attn = attention_lib.Attention(
        config=config,
        attn_type=model_lib.AttentionType.LOCAL_SLIDING,
        rngs=nnx.Rngs(0),
    )

    b, t, h, d = 2, 32, config.num_heads, config.head_dim
    x = jnp.zeros((b, t, config.embed_dim))
    segment_pos = jnp.zeros((b, t), dtype=jnp.int32)
    attn_mask = jnp.ones((b, t, t), dtype=jnp.bool_)

    # Case 1: Square sequence, segment_ids is not None -> should use FLASH
    with mock.patch.object(
        attn, '_flash_attention_single'
    ) as mock_flash, mock.patch.object(
        attn, '_eager_attention'
    ) as mock_eager, mock.patch.object(
        attn, '_make_sharding_specs'
    ) as mock_sharding, mock.patch.object(
        attn, '_make_splash_kernel'
    ) as mock_kernel:

      mock_flash.return_value = (
          jnp.zeros((b, t, h, d)),
          jnp.zeros((b, t, config.num_kv_heads, d)),
          jnp.zeros((b, t, config.num_kv_heads, d)),
      )
      mock_eager.return_value = jnp.zeros((b, t, h, d))
      mock_sharding.return_value = (None,) * 4 + (1, 1) + (None,) * 3
      mock_kernel.return_value = (None, None)

      segment_ids = jnp.zeros((b, t), dtype=jnp.int32)

      attn.block(
          x,
          segment_pos,
          cache=None,
          attn_mask=attn_mask,
          segment_ids=segment_ids,
      )

      mock_flash.assert_called_once()
      mock_eager.assert_not_called()

    # Case 2: Rectangular sequence, segment_ids is not None -> should use EAGER
    kv_len = 64
    kv_shared_cache = {
        'k': jnp.zeros((b, kv_len, config.num_kv_heads, d)),
        'v': jnp.zeros((b, kv_len, config.num_kv_heads, d)),
    }
    attn_mask_rect = jnp.ones((b, t, kv_len), dtype=jnp.bool_)

    with mock.patch.object(
        attn, '_flash_attention_single'
    ) as mock_flash, mock.patch.object(attn, '_eager_attention') as mock_eager:

      mock_flash.return_value = (
          jnp.zeros((b, t, h, d)),
          jnp.zeros((b, kv_len, config.num_kv_heads, d)),
          jnp.zeros((b, kv_len, config.num_kv_heads, d)),
      )
      mock_eager.return_value = jnp.zeros((b, t, h, d))

      segment_ids = jnp.zeros((b, t), dtype=jnp.int32)

      attn.block(
          x,
          segment_pos,
          cache=None,
          attn_mask=attn_mask_rect,
          kv_shared_cache=kv_shared_cache,
          segment_ids=segment_ids,
      )

      mock_flash.assert_not_called()
      mock_eager.assert_called_once()

  def test_make_block_sizes(self):
    config = model_lib.ModelConfig.gemma4_e2b()
    config.flash_attention_block_size = 128
    config.sliding_window_size = 64
    self.assertGreater(
        config.flash_attention_block_size, config.sliding_window_size
    )

    global_attn = attention_lib.Attention(
        config=config,
        attn_type=model_lib.AttentionType.GLOBAL,
        rngs=nnx.Rngs(0),
    )
    local_attn = attention_lib.Attention(
        config=config,
        attn_type=model_lib.AttentionType.LOCAL_SLIDING,
        rngs=nnx.Rngs(0),
    )

    # GLOBAL rectangular uses the full block size (kills `if is_rectangular:`).
    self.assertEqual(
        global_attn._make_block_sizes(is_rectangular=True).block_kv,
        config.flash_attention_block_size,
    )

    # LOCAL_SLIDING square uses the full block size (kills `if self.attn_type == LOCAL_SLIDING:`).
    self.assertEqual(
        local_attn._make_block_sizes(is_rectangular=False).block_kv,
        config.flash_attention_block_size,
    )

    # LOCAL_SLIDING rectangular uses min(block_size, window_size).
    self.assertEqual(
        local_attn._make_block_sizes(is_rectangular=True).block_kv,
        min(config.flash_attention_block_size, config.sliding_window_size),
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='none_mesh',
          act_btnh=None,
          mesh_shape=None,
          expected_head_shards=1,
          expected_q_seq_shards=1,
      ),
      dict(
          testcase_name='axis_not_in_mesh_defaults_to_one',
          # act_btnh unpacks to (shd_b, shd_t, shd_n, shd_h). Use axis names for
          # shd_t ('seq_axis') and shd_n ('model_axis') that are non-None but
          # NOT present in the mesh.
          act_btnh=P('fsdp', 'seq_axis', 'model_axis', None),
          # Mesh axes are ('fsdp', 'x'); 'seq_axis' and 'model_axis' are absent.
          mesh_shape={'fsdp': 1, 'x': 1},
          # shd_n ('model_axis') and shd_t ('seq_axis') are not in the mesh, so
          # both must fall back to 1. This kills the mutants that drop the
          # `and shd_n in mesh.shape` / `and shd_t in mesh.shape` guards.
          expected_head_shards=1,
          expected_q_seq_shards=1,
      ),
      dict(
          testcase_name='sharded_mesh',
          act_btnh=P('fsdp', 'seq_axis', 'model_axis', None),
          mesh_shape={'fsdp': 1, 'seq_axis': 4, 'model_axis': 2},
          expected_head_shards=2,
          expected_q_seq_shards=4,
      ),
  )
  def test_make_sharding_specs(
      self,
      act_btnh,
      mesh_shape,
      expected_head_shards,
      expected_q_seq_shards,
  ):
    config = model_lib.ModelConfig.gemma4_e2b()
    if act_btnh is not None:
      # ShardingConfig is frozen, so replace it.
      config.shd_config = dataclasses.replace(
          config.shd_config,
          act_btnh=act_btnh,
      )
    attn = attention_lib.Attention(
        config=config,
        attn_type=model_lib.AttentionType.GLOBAL,
        rngs=nnx.Rngs(0),
    )
    b = 1
    kh = config.num_kv_heads
    mesh = mock.MagicMock(shape=mesh_shape) if mesh_shape is not None else None
    if mesh is not None and 'fsdp' in mesh.shape:
      self.assertEqual(b % mesh.shape['fsdp'], 0)

    specs = attn._make_sharding_specs(b, kh, mesh)
    # Return tuple index 4 is head_shards, index 5 is q_seq_shards.
    head_shards = specs[4]
    q_seq_shards = specs[5]

    self.assertEqual(head_shards, expected_head_shards)
    self.assertEqual(q_seq_shards, expected_q_seq_shards)

  def test_attention_flash_rectangular_offset(self):
    config = model_lib.ModelConfig.gemma4_e2b()
    config.use_flash_attention = True
    config.flash_attention_block_size = 16

    attn = attention_lib.Attention(
        config=config,
        attn_type=model_lib.AttentionType.LOCAL_SLIDING,
        rngs=nnx.Rngs(0),
    )

    b, q_len, kv_len = 2, 32, 64
    h, d = config.num_heads, config.head_dim
    x = jnp.zeros((b, q_len, config.embed_dim))
    segment_pos = jnp.zeros((b, q_len), dtype=jnp.int32)
    attn_mask = jnp.ones((b, q_len, kv_len), dtype=jnp.bool_)
    kv_shared_cache = {
        'k': jnp.zeros((b, kv_len, config.num_kv_heads, d)),
        'v': jnp.zeros((b, kv_len, config.num_kv_heads, d)),
    }

    with mock.patch.object(
        attn, '_build_flash_mask', wraps=attn._build_flash_mask
    ) as mock_mask, mock.patch.object(
        attn, '_flash_attention_single'
    ) as mock_flash, mock.patch.object(
        attn, '_make_sharding_specs'
    ) as mock_sharding, mock.patch.object(
        attn, '_make_splash_kernel'
    ) as mock_kernel:

      mock_flash.return_value = (
          jnp.zeros((b, q_len, h, d)),
          jnp.zeros((b, kv_len, config.num_kv_heads, d)),
          jnp.zeros((b, kv_len, config.num_kv_heads, d)),
      )
      mock_sharding.return_value = (None,) * 4 + (1, 1) + (None,) * 3
      mock_kernel.return_value = (None, None)

      attn.block(
          x,
          segment_pos,
          cache=None,
          attn_mask=attn_mask,
          kv_shared_cache=kv_shared_cache,
          segment_ids=None,
      )

      # Verify flash attention is called and the exact positive offset is passed
      # (kv_len - q_len = 64 - 32 = 32), killing mutants that negate offset or
      # pass 0.
      mock_flash.assert_called_once()
      mock_mask.assert_called_once_with(q_len, kv_len, kv_len - q_len)

  @parameterized.named_parameters(
      dict(
          testcase_name='own_cache',
          # Case 1: Own cache is present (cache is not None).
          # end_index = 4 -> position 4 is valid and attended to.
          end_index=4,
          use_shared_cache=False,
          mask_indices=None,
      ),
      dict(
          testcase_name='shared_cache_adjusted_index',
          # Case 2: Shared cache is present (cache is None, kv_shared_cache is
          # not None). end_index = 5 -> adjusted to 4 by `end_idx - 1`.
          end_index=5,
          use_shared_cache=True,
          mask_indices=None,
      ),
      dict(
          testcase_name='gapped_mask_logical_end',
          # Case 3: Gapped mask (_has_physical_gap is True). Uses logical_end
          # (count - 1 = 4) instead of end_index, killing mutant at line 1374.
          end_index=0,
          use_shared_cache=False,
          mask_indices=[0, 1, 2, 4, 6],
      ),
  )
  def test_eager_attention_decoding_sliding_window_cache_indexing(
      self, end_index, use_shared_cache, mask_indices=None
  ):
    config = model_lib.ModelConfig.gemma4_e2b()
    config.sliding_window_size = 8
    config.use_sliding_window_kv_cache = True
    attn = attention_lib.Attention(
        config=config,
        attn_type=model_lib.AttentionType.LOCAL_SLIDING,
        rngs=nnx.Rngs(0),
    )
    b, q_len, cache_len, d = 1, 1, 8, config.head_dim
    h, kh = config.num_heads, config.num_kv_heads

    q = jnp.ones((b, q_len, h, d))
    k = jnp.zeros((b, cache_len, kh, d))
    k = k.at[:, 4, :, :].set(10.0)
    k = k.at[:, 6, :, :].set(20.0)
    v = jnp.ones((b, cache_len, kh, d))
    v = v.at[:, 4, :, :].set(5.0)
    v = v.at[:, 6, :, :].set(100.0)

    if mask_indices is not None:
      attn_mask = (
          jnp.zeros((b, q_len, cache_len), dtype=jnp.bool_)
          .at[:, :, mask_indices]
          .set(True)
      )
    else:
      attn_mask = jnp.ones((b, q_len, cache_len), dtype=jnp.bool_)
    segment_pos = jnp.array([[4]], dtype=jnp.int32)

    cache_dict = {'end_index': jnp.array([end_index])}
    cache = None if use_shared_cache else cache_dict
    kv_shared_cache = cache_dict if use_shared_cache else None

    out = attn._eager_attention(
        query_proj=q,
        key_proj=k,
        value_proj=v,
        attn_mask=attn_mask,
        segment_pos=segment_pos,
        cache=cache,
        kv_shared_cache=kv_shared_cache,
        seq_len=1,
    )
    np.testing.assert_allclose(out, jnp.full_like(out, 5.0), atol=1e-2)

  def test_sliding_window_kv_cache_prefill_over_cache_len(self):
    config = model_lib.ModelConfig.gemma4_e2b()
    config.sliding_window_size = 8
    b, seq_len, cache_len = 1, 10, 8
    d = config.head_dim
    kh = config.num_kv_heads

    x = jax.random.normal(jax.random.PRNGKey(0), (b, seq_len, config.embed_dim))
    segment_pos = jnp.arange(seq_len, dtype=jnp.int32)[None, :]
    attn_mask = jnp.ones((b, seq_len, seq_len), dtype=jnp.bool_)
    cache = {
        'k': jnp.zeros((b, cache_len, kh, d)),
        'v': jnp.zeros((b, cache_len, kh, d)),
        'end_index': jnp.zeros((b,), dtype=jnp.int32),
    }

    # 1. When use_sliding_window_kv_cache=True, circular update succeeds.
    config.use_sliding_window_kv_cache = True
    attn_sliding = attention_lib.Attention(
        config=config,
        attn_type=model_lib.AttentionType.LOCAL_SLIDING,
        rngs=nnx.Rngs(0),
    )
    new_cache, _, (k_proj, v_proj, *_) = attn_sliding.block(
        x,
        segment_pos,
        cache=cache,
        attn_mask=attn_mask,
        force_eager=True,
    )

    self.assertIsNotNone(new_cache)
    self.assertEqual(new_cache['end_index'][0], seq_len)
    valid_indices = (seq_len - cache_len + jnp.arange(cache_len)) % cache_len
    np.testing.assert_allclose(
        new_cache['k'][:, valid_indices, ...], k_proj[:, -cache_len:, ...]
    )
    np.testing.assert_allclose(
        new_cache['v'][:, valid_indices, ...], v_proj[:, -cache_len:, ...]
    )

    # 2. When use_sliding_window_kv_cache=False, non-sliding prefill cannot
    # exceed cache_len (dynamic_update_slice raises). This kills the mutant
    # at line 419 that drops `self.config.use_sliding_window_kv_cache and`.
    config.use_sliding_window_kv_cache = False
    attn_standard = attention_lib.Attention(
        config=config,
        attn_type=model_lib.AttentionType.LOCAL_SLIDING,
        rngs=nnx.Rngs(0),
    )
    with self.assertRaises(TypeError):
      attn_standard.block(
          x,
          segment_pos,
          cache=cache,
          attn_mask=attn_mask,
          force_eager=True,
      )

  def test_eager_attention_mha_non_gqa(self):
    config = model_lib.ModelConfig.gemma4_e2b()
    config.num_heads = 4
    config.num_kv_heads = 4
    attn = attention_lib.Attention(
        config=config,
        attn_type=model_lib.AttentionType.GLOBAL,
        rngs=nnx.Rngs(0),
    )
    self.assertFalse(attn.use_gqa)
    b, q_len, kv_len, d = 2, 4, 8, config.head_dim
    h = config.num_heads

    q = jnp.ones((b, q_len, h, d))
    k = jnp.zeros((b, kv_len, h, d))
    k = k.at[:, 2, :, :].set(10.0)
    v = jnp.zeros((b, kv_len, h, d))
    v = v.at[:, 2, :, :].set(3.0)

    attn_mask = jnp.ones((b, q_len, kv_len), dtype=jnp.bool_)
    segment_pos = jnp.broadcast_to(
        jnp.arange(q_len, dtype=jnp.int32)[None, :], (b, q_len)
    )

    out = attn._eager_attention(
        query_proj=q,
        key_proj=k,
        value_proj=v,
        attn_mask=attn_mask,
        segment_pos=segment_pos,
        cache=None,
        kv_shared_cache=None,
        seq_len=q_len,
    )
    self.assertEqual(out.shape, (b, q_len, h, d))
    np.testing.assert_allclose(out, jnp.full_like(out, 3.0), atol=1e-2)

  def test_kv_cache_prefill_within_cache_len_and_decode_step(self):
    config = model_lib.ModelConfig.gemma4_e2b()
    config.use_sliding_window_kv_cache = False
    attn = attention_lib.Attention(
        config=config,
        attn_type=model_lib.AttentionType.GLOBAL,
        rngs=nnx.Rngs(0),
    )
    b, prefill_len, cache_len = 1, 4, 16
    d = attn.head_dim
    kh = attn.num_kv_heads

    # 1. Prefill step with seq_len <= cache_len
    x_prefill = jax.random.normal(
        jax.random.PRNGKey(0), (b, prefill_len, config.embed_dim)
    )
    pos_prefill = jnp.arange(prefill_len, dtype=jnp.int32)[None, :]
    mask_prefill = jnp.ones((b, prefill_len, prefill_len), dtype=jnp.bool_)
    cache = {
        'k': jnp.zeros((b, cache_len, kh, d)),
        'v': jnp.zeros((b, cache_len, kh, d)),
        'end_index': jnp.zeros((b,), dtype=jnp.int32),
    }

    cache, _, (k_prefill, v_prefill, *_) = attn.block(
        x_prefill,
        pos_prefill,
        cache=cache,
        attn_mask=mask_prefill,
        force_eager=True,
    )
    self.assertEqual(cache['end_index'][0], prefill_len)
    np.testing.assert_allclose(cache['k'][:, :prefill_len, ...], k_prefill)
    np.testing.assert_allclose(cache['v'][:, :prefill_len, ...], v_prefill)

    # Overwrite cache with 1.0 to explicitly test decode preservation
    cache['k'] = cache['k'].at[:, :prefill_len].set(1.0)
    cache['v'] = cache['v'].at[:, :prefill_len].set(1.0)

    # 2. Decode step with seq_len == 1
    x_decode = jax.random.normal(
        jax.random.PRNGKey(1), (b, 1, config.embed_dim)
    )
    pos_decode = jnp.array([[prefill_len]], dtype=jnp.int32)
    mask_decode = jnp.ones((b, 1, cache_len), dtype=jnp.bool_)

    cache, _, (k_decode, v_decode, *_) = attn.block(
        x_decode,
        pos_decode,
        cache=cache,
        attn_mask=mask_decode,
        force_eager=True,
    )
    self.assertEqual(cache['end_index'][0], prefill_len + 1)
    np.testing.assert_allclose(cache['k'], k_decode)
    np.testing.assert_allclose(cache['v'], v_decode)
    # Verify prior cache slots are preserved (end_index non-zero update).
    np.testing.assert_allclose(cache['k'][:, :4], 1.0)
    np.testing.assert_allclose(cache['v'][:, :4], 1.0)
    # Verify future cache slots remain untouched (0.0).
    np.testing.assert_allclose(cache['k'][:, 5:], 0.0)
    np.testing.assert_allclose(cache['v'][:, 5:], 0.0)
    np.testing.assert_allclose(cache['k'], k_decode)
    np.testing.assert_allclose(cache['v'], v_decode)

  def test_eager_attention_decoding_without_sliding_window_cache(self):
    config = model_lib.ModelConfig.gemma4_e2b()
    config.sliding_window_size = 4
    config.use_sliding_window_kv_cache = False
    attn = attention_lib.Attention(
        config=config,
        attn_type=model_lib.AttentionType.LOCAL_SLIDING,
        rngs=nnx.Rngs(0),
    )
    b, q_len, kv_len, d = 1, 1, 8, config.head_dim
    h, kh = config.num_heads, config.num_kv_heads

    q = jnp.ones((b, q_len, h, d))
    k = jnp.zeros((b, kv_len, kh, d))
    # Query at position 6 with window 4 -> valid window [3, 6].
    k = k.at[:, 5, :, :].set(10.0)
    k = k.at[:, 1, :, :].set(20.0)
    v = jnp.zeros((b, kv_len, kh, d))
    v = v.at[:, 5, :, :].set(4.0)
    v = v.at[:, 1, :, :].set(99.0)

    # Causal mask up to position 6.
    attn_mask = (
        jnp.zeros((b, q_len, kv_len), dtype=jnp.bool_).at[:, :, :7].set(True)
    )
    segment_pos = jnp.array([[6]], dtype=jnp.int32)

    out = attn._eager_attention(
        query_proj=q,
        key_proj=k,
        value_proj=v,
        attn_mask=attn_mask,
        segment_pos=segment_pos,
        cache=None,
        kv_shared_cache=None,
        seq_len=1,
    )
    self.assertEqual(out.shape, (b, q_len, h, d))
    np.testing.assert_allclose(out, jnp.full_like(out, 4.0), atol=1e-2)

  def test_eager_attention_decoding_missing_cache_raises(self):
    config = model_lib.ModelConfig.gemma4_e2b()
    config.sliding_window_size = 4
    config.use_sliding_window_kv_cache = True
    attn = attention_lib.Attention(
        config=config,
        attn_type=model_lib.AttentionType.LOCAL_SLIDING,
        rngs=nnx.Rngs(0),
    )
    b, q_len, kv_len, d = 1, 1, 8, config.head_dim
    h, kh = config.num_heads, config.num_kv_heads

    q = jnp.ones((b, q_len, h, d))
    k = jnp.ones((b, kv_len, kh, d))
    v = jnp.ones((b, kv_len, kh, d))
    attn_mask = jnp.ones((b, q_len, kv_len), dtype=jnp.bool_)
    segment_pos = jnp.array([[4]], dtype=jnp.int32)

    with self.assertRaisesRegex(
        ValueError, 'Cache or shared cache is required'
    ):
      attn._eager_attention(
          query_proj=q,
          key_proj=k,
          value_proj=v,
          attn_mask=attn_mask,
          segment_pos=segment_pos,
          cache=None,
          kv_shared_cache=None,
          seq_len=1,
      )

  def test_eager_attention_chunked_prefill_mask_construction(self):
    """Verify _eager_attention constructs chunked prefill mask for rectangular KV."""
    config = model_lib.ModelConfig.gemma4_e2b()
    config.sliding_window_size = 8
    attn = attention_lib.Attention(
        config=config,
        attn_type=model_lib.AttentionType.GLOBAL,
        rngs=nnx.Rngs(0),
    )
    b, q_len, kv_len = 2, 4, 12
    h, kh, d = config.num_heads, config.num_kv_heads, config.head_dim
    q = jnp.ones((b, q_len, h, d))
    k = jnp.zeros((b, kv_len, kh, d))
    v = jnp.zeros((b, kv_len, kh, d))

    # Active Probe at valid prefix slot 2 (< prior_end_index=6)
    k = k.at[:, 2, :, :].set(10.0)
    v = v.at[:, 2, :, :].set(5.0)

    # Trap Probe 1 at uninitialized prefix position 7 (> prior_end_index=6)
    k = k.at[:, 7, :, :].set(100.0)
    v = v.at[:, 7, :, :].set(999.0)

    # Trap Probe 2 at future suffix position 11 (for query 0)
    k = k.at[:, 11, :, :].set(100.0)
    v = v.at[:, 11, :, :].set(999.0)

    prefix_mask = jnp.ones((b, q_len, 8), dtype=jnp.bool_)
    suffix_causal = jnp.broadcast_to(
        jnp.tril(jnp.ones((q_len, q_len), dtype=jnp.bool_))[None, :, :],
        (b, q_len, q_len),
    )
    attn_mask = jnp.concatenate([prefix_mask, suffix_causal], axis=-1)
    segment_pos = jnp.broadcast_to(
        jnp.arange(8, 12, dtype=jnp.int32)[None, :], (b, q_len)
    )
    cache = {
        'k': jnp.zeros((b, 8, kh, d)),
        'v': jnp.zeros((b, 8, kh, d)),
        'end_index': jnp.array([6]),
    }
    out = attn._eager_attention(
        query_proj=q,
        key_proj=k,
        value_proj=v,
        attn_mask=attn_mask,
        segment_pos=segment_pos,
        cache=cache,
        kv_shared_cache=None,
        prior_end_index=jnp.array([6]),
        prefix_length=8,
        seq_len=q_len,
        is_chunked_prefill=True,
    )
    self.assertEqual(out.shape, (b, q_len, h, d))
    self.assertFalse(jnp.isnan(out).any())
    np.testing.assert_allclose(out[:, 0, :, :], 5.0, atol=1e-2)

  @parameterized.named_parameters(
      ('with_segments', True),
      ('without_segments', False),
  )
  def test_flash_attention_single_segment_ids(self, with_segments):
    """Verify _flash_attention_single execution path with and without segment_ids."""
    config = model_lib.ModelConfig.gemma4_e2b()
    config.use_flash_attention = True
    config.flash_attention_block_size = 16
    attn = attention_lib.Attention(
        config=config,
        attn_type=model_lib.AttentionType.GLOBAL,
        rngs=nnx.Rngs(0),
    )
    b, t = 2, 16
    x = jnp.zeros((b, t, config.embed_dim))
    segment_pos = jnp.zeros((b, t), dtype=jnp.int32)
    attn_mask = jnp.ones((b, t, t), dtype=jnp.bool_)
    segment_ids = jnp.zeros((b, t), dtype=jnp.int32) if with_segments else None

    kernel_called = []
    mesh, mock_kernel = self._make_mock_splash()

    def wrapped_mock_kernel(q_in, k_in, v_in, segment_ids=None):
      kernel_called.append(segment_ids)
      if with_segments:
        self.assertIsNotNone(segment_ids)
      return mock_kernel(q_in, k_in, v_in, segment_ids)

    with mesh, mock.patch.object(
        attn, '_make_splash_kernel', return_value=(wrapped_mock_kernel, None)
    ):
      new_cache, out, (k_proj, v_proj, *_) = attn.block(
          x,
          segment_pos,
          cache=None,
          attn_mask=attn_mask,
          segment_ids=segment_ids,
      )
      self.assertTrue(kernel_called)
      if not with_segments:
        self.assertEqual(kernel_called, [None])
      self.assertEqual(out.shape, (b, t, config.embed_dim))
      self.assertEqual(k_proj.shape, (b, t, attn.num_kv_heads, attn.head_dim))
      self.assertEqual(v_proj.shape, (b, t, attn.num_kv_heads, attn.head_dim))
      self.assertFalse(jnp.isnan(out).any())

  def test_ragged_ring_buffer_decode_matches_unbatched_ground_truth(self):
    """Ragged LOCAL_SLIDING prefill+decode must match B=1 unpadded ground truth."""
    config = model_lib.ModelConfig.gemma4_e2b()
    config.use_sliding_window_kv_cache = True
    config.sliding_window_size = 4
    attn = attention_lib.Attention(
        config=config,
        attn_type=model_lib.AttentionType.LOCAL_SLIDING,
        rngs=nnx.Rngs(0),
    )
    cache_len, kh, d = 4, config.num_kv_heads, config.head_dim
    lens = jnp.array([12, 6], dtype=jnp.int32)  # row 1 gap = 6 > window (4)
    x_pre = jax.random.normal(jax.random.PRNGKey(0), (2, 12, config.embed_dim))
    x_dec = jax.random.normal(jax.random.PRNGKey(1), (2, 1, config.embed_dim))

    def _run(row: int | None) -> jnp.ndarray:
      idx = slice(None) if row is None else slice(row, row + 1)
      b, seq = (2, 12) if row is None else (1, int(lens[row]))
      in_mask = jnp.arange(seq)[None, :] < lens[idx, None]
      causal = jnp.tril(jnp.ones((b, seq, seq), dtype=jnp.bool_))
      cache = {
          'k': jnp.zeros((b, cache_len, kh, d)),
          'v': jnp.zeros((b, cache_len, kh, d)),
          'end_index': jnp.zeros((b,), dtype=jnp.int32),
      }
      cache, _, _ = attn.block(
          x_pre[idx, :seq],
          jnp.broadcast_to(jnp.arange(seq, dtype=jnp.int32), (b, seq)),
          cache=cache,
          attn_mask=causal & in_mask[:, None, :],
          input_mask=in_mask,
          is_chunked_prefill=True,
          force_eager=True,
      )
      dec_mask = jnp.zeros((b, 1, seq + 1), dtype=jnp.bool_)
      dec_mask = dec_mask.at[:, 0, :seq].set(in_mask)
      dec_mask = dec_mask.at[:, 0, seq].set(True)
      _, out, _ = attn.block(
          x_dec[idx],
          lens[idx, None],
          cache=cache,
          attn_mask=dec_mask,
          force_eager=True,
      )
      return out

    ragged_out = _run(None)
    for r in range(len(lens)):
      np.testing.assert_allclose(
          ragged_out[r : r + 1],
          _run(r),
          atol=1e-2,
          rtol=1e-2,
          err_msg=f'Row {r} diverged from batch=1 ground truth!',
      )

  @parameterized.named_parameters(
      dict(
          testcase_name='sliding_window_within_window',
          attn_type=model_lib.AttentionType.LOCAL_SLIDING,
          sliding_window_size=16,
          prefix_len=8,
          suffix_len=4,
      ),
      dict(
          testcase_name='sliding_window_at_window_boundary',
          attn_type=model_lib.AttentionType.LOCAL_SLIDING,
          sliding_window_size=16,
          prefix_len=16,
          suffix_len=4,
      ),
      dict(
          testcase_name='sliding_window_beyond_window_wrapped',
          attn_type=model_lib.AttentionType.LOCAL_SLIDING,
          sliding_window_size=16,
          prefix_len=24,
          suffix_len=4,
      ),
      dict(
          testcase_name='global_prefix_reuse',
          attn_type=model_lib.AttentionType.GLOBAL,
          sliding_window_size=None,
          prefix_len=24,
          suffix_len=4,
      ),
  )
  def test_chunked_prefill_prefix_reuse_matches_full_prefill(
      self, attn_type, sliding_window_size, prefix_len, suffix_len
  ):
    config = model_lib.ModelConfig.gemma4_e2b()
    config.sliding_window_size = sliding_window_size
    config.use_sliding_window_kv_cache = True
    config.use_flash_attention = False

    attn = attention_lib.Attention(
        config=config,
        attn_type=attn_type,
        rngs=nnx.Rngs(0),
    )

    b = 2
    total_len = prefix_len + suffix_len
    x_full = jax.random.normal(
        jax.random.PRNGKey(0), (b, total_len, config.embed_dim)
    )
    pos_full = jnp.broadcast_to(
        jnp.arange(total_len, dtype=jnp.int32)[None, :], (b, total_len)
    )
    mask_full = jnp.tril(jnp.ones((b, total_len, total_len), dtype=jnp.bool_))

    # 1. Full monolithic prefill
    max_seq_len = total_len + 16
    cache_full = attn.init_cache(b, max_seq_len, dtype=jnp.float32)
    cache_full, out_full, _ = attn.block(
        x_full,
        pos_full,
        cache=cache_full,
        attn_mask=mask_full,
        is_chunked_prefill=False,
    )

    # 2. Chunked prefill: Chunk 1 (prefix)
    cache_chunked = attn.init_cache(b, max_seq_len, dtype=jnp.float32)
    x_prefix = x_full[:, :prefix_len, :]
    pos_prefix = pos_full[:, :prefix_len]
    mask_prefix = mask_full[:, :prefix_len, :]
    cache_chunked, out_prefix_chunked, _ = attn.block(
        x_prefix,
        pos_prefix,
        cache=cache_chunked,
        attn_mask=mask_prefix,
        is_chunked_prefill=True,
        prefix_length=0,
        force_eager=True,
    )
    np.testing.assert_allclose(
        out_prefix_chunked, out_full[:, :prefix_len, :], atol=2e-5, rtol=1e-4
    )

    # 3. Chunked prefill: Chunk 2 (suffix reusing prefix cache)
    x_suffix = x_full[:, prefix_len:, :]
    pos_suffix = pos_full[:, prefix_len:]
    mask_suffix = mask_full[:, prefix_len:, :]
    cache_chunked, out_suffix_chunked, _ = attn.block(
        x_suffix,
        pos_suffix,
        cache=cache_chunked,
        attn_mask=mask_suffix,
        is_chunked_prefill=True,
        prefix_length=prefix_len,
        force_eager=True,
    )

    # Verify suffix representation matches the suffix region of monolithic prefill
    out_suffix_expected = out_full[:, prefix_len:, :]
    np.testing.assert_allclose(
        out_suffix_chunked, out_suffix_expected, atol=2e-5, rtol=1e-4
    )

    # 4. Decode step at position total_len
    x_decode = jax.random.normal(
        jax.random.PRNGKey(1), (b, 1, config.embed_dim)
    )
    pos_decode = jnp.array([[total_len], [total_len]], dtype=jnp.int32)
    mask_decode = jnp.broadcast_to(
        (jnp.arange(max_seq_len)[None, None, :] <= total_len),
        (b, 1, max_seq_len),
    )

    cache_full, out_decode_full, _ = attn.block(
        x_decode,
        pos_decode,
        cache=cache_full,
        attn_mask=mask_decode,
        is_chunked_prefill=False,
    )
    cache_chunked, out_decode_chunked, _ = attn.block(
        x_decode,
        pos_decode,
        cache=cache_chunked,
        attn_mask=mask_decode,
        is_chunked_prefill=False,
    )
    np.testing.assert_allclose(
        out_decode_chunked, out_decode_full, atol=2e-5, rtol=1e-4
    )

  def test_init_cache_on_device_sharded_allocation(self):
    """Verifies init_cache creates NamedSharding and calls _zeros on device."""
    act_btnh = ('fsdp', None, 'tp', None)
    config = model_lib.ModelConfig.gemma4_e2b()
    config.shd_config = mock.MagicMock(act_btnh=act_btnh)
    attn = attention_lib.Attention(
        config=config,
        attn_type=model_lib.AttentionType.GLOBAL,
        rngs=nnx.Rngs(0),
    )
    b, max_seq_len = 2, 16
    devices = np.array(jax.devices()[:1]).reshape(1, 1)
    mesh = jax.sharding.Mesh(devices, ('fsdp', 'tp'))
    expected_k_sharding = jax.sharding.NamedSharding(mesh, P(*act_btnh))
    expected_idx_sharding = jax.sharding.NamedSharding(mesh, P(*act_btnh[:1]))

    with mesh:
      cache = attn.init_cache(
          batch_size=b, max_seq_len=max_seq_len, dtype=jnp.float32
      )

    self.assertEqual(
        cache['k'].shape, (b, max_seq_len, attn.num_kv_heads, attn.head_dim)
    )
    self.assertEqual(
        cache['v'].shape, (b, max_seq_len, attn.num_kv_heads, attn.head_dim)
    )
    self.assertEqual(cache['end_index'].shape, (b,))
    self.assertEqual(cache['k'].sharding, expected_k_sharding)
    self.assertEqual(cache['v'].sharding, expected_k_sharding)
    self.assertEqual(cache['end_index'].sharding, expected_idx_sharding)
    self.assertTrue((cache['k'] == 0.0).all())
    self.assertTrue((cache['v'] == 0.0).all())
    self.assertTrue((cache['end_index'] == 0).all())

  def test_init_cache_unsharded_fallback(self):
    """Verifies init_cache gracefully falls back to jnp.zeros when un-sharded."""
    config = model_lib.ModelConfig.gemma4_e2b()
    attn = attention_lib.Attention(
        config=config,
        attn_type=model_lib.AttentionType.GLOBAL,
        rngs=nnx.Rngs(0),
    )
    b, max_seq_len = 2, 16
    cache = attn.init_cache(
        batch_size=b, max_seq_len=max_seq_len, dtype=jnp.float32
    )
    self.assertEqual(
        cache['k'].shape, (b, max_seq_len, attn.num_kv_heads, attn.head_dim)
    )
    self.assertEqual(
        cache['v'].shape, (b, max_seq_len, attn.num_kv_heads, attn.head_dim)
    )
    self.assertEqual(cache['end_index'].shape, (b,))
    self.assertTrue((cache['k'] == 0.0).all())
    self.assertTrue((cache['v'] == 0.0).all())
    self.assertTrue((cache['end_index'] == 0).all())

  @parameterized.named_parameters(
      dict(
          testcase_name='global_unpadded',
          attn_type=model_lib.AttentionType.GLOBAL,
          use_input_mask=False,
      ),
      dict(
          testcase_name='global_padded',
          attn_type=model_lib.AttentionType.GLOBAL,
          use_input_mask=True,
      ),
      dict(
          testcase_name='local_sliding_unpadded',
          attn_type=model_lib.AttentionType.LOCAL_SLIDING,
          use_input_mask=False,
      ),
      dict(
          testcase_name='local_sliding_padded',
          attn_type=model_lib.AttentionType.LOCAL_SLIDING,
          use_input_mask=True,
      ),
  )
  def test_split_attention_eager_fallback_concatenates_prefix(
      self, attn_type, use_input_mask
  ):
    config = model_lib.ModelConfig.gemma4_e2b()
    config.sliding_window_size = 16
    config.use_sliding_window_kv_cache = True

    # 1. Baseline: use_split_attention = False
    config_base = dataclasses.replace(
        config, use_flash_attention=False, use_split_attention=False
    )
    attn_base = attention_lib.Attention(
        config=config_base,
        attn_type=attn_type,
        rngs=nnx.Rngs(0),
    )

    b, prefix_len, suffix_len = 2, 8, 4
    cache_len = 16
    kh, d = attn_base.num_kv_heads, attn_base.head_dim

    cache_split = {
        'k': jax.random.normal(jax.random.PRNGKey(10), (b, cache_len, kh, d)),
        'v': jax.random.normal(jax.random.PRNGKey(11), (b, cache_len, kh, d)),
        'end_index': jnp.array([prefix_len, prefix_len], dtype=jnp.int32),
    }
    cache_base = jax.tree.map(jnp.copy, cache_split)

    x = jax.random.normal(
        jax.random.PRNGKey(0), (b, suffix_len, config.embed_dim)
    )
    segment_pos = jnp.broadcast_to(
        jnp.arange(prefix_len, prefix_len + suffix_len, dtype=jnp.int32)[
            None, :
        ],
        (b, suffix_len),
    )
    total_len = prefix_len + suffix_len
    attn_mask = jnp.ones((b, suffix_len, total_len), dtype=jnp.bool_)
    if use_input_mask:
      input_mask = jnp.ones((b, suffix_len), dtype=jnp.bool_)
      input_mask = input_mask.at[1, -1].set(False)
    else:
      input_mask = None
    _, out_base, layers_kvs_base = attn_base.block(
        x,
        segment_pos,
        cache=cache_base,
        attn_mask=attn_mask,
        is_chunked_prefill=True,
        prefix_length=prefix_len,
        input_mask=input_mask,
        force_eager=True,
    )

    # 2. Split attention with eager fallback: use_split_attention = True,
    # force_eager = True
    config_split = dataclasses.replace(
        config, use_flash_attention=True, use_split_attention=True
    )
    attn_split = attention_lib.Attention(
        config=config_split,
        attn_type=attn_type,
        rngs=nnx.Rngs(0),
    )
    _, out_split, layers_kvs_split = attn_split.block(
        x,
        segment_pos,
        cache=cache_split,
        attn_mask=attn_mask,
        is_chunked_prefill=True,
        prefix_length=prefix_len,
        input_mask=input_mask,
        force_eager=True,
    )

    np.testing.assert_allclose(out_split, out_base, atol=1e-5, rtol=1e-5)
    self.assertLen(layers_kvs_split, 6)
    k_proj, v_proj, _, _, split_k, split_v = layers_kvs_split
    self.assertIsNone(split_k)
    self.assertIsNone(split_v)
    np.testing.assert_allclose(k_proj, layers_kvs_base[0], atol=1e-5)
    np.testing.assert_allclose(v_proj, layers_kvs_base[1], atol=1e-5)

  @parameterized.named_parameters(
      ('has_split_prefix', True),
      ('no_split_prefix', False),
  )
  def test_follower_layer_routing_with_shared_cache(self, has_split_prefix):
    """Verifies follower layer routes to split or single flash attention."""
    config = model_lib.ModelConfig.gemma4_e2b()
    config.use_flash_attention = True
    config.use_split_attention = True
    config.flash_attention_block_size = 8
    attn = attention_lib.Attention(
        config=config,
        attn_type=model_lib.AttentionType.GLOBAL,
        rngs=nnx.Rngs(0),
    )

    b, q_len, kv_len, d = 1, 8, 16, config.head_dim
    x = jnp.zeros((b, q_len, config.embed_dim))
    segment_pos = jnp.zeros((b, q_len), dtype=jnp.int32)
    attn_mask = jnp.ones((b, q_len, kv_len), dtype=jnp.bool_)

    dummy_k = jnp.zeros((b, kv_len, config.num_kv_heads, d))
    dummy_v = jnp.zeros((b, kv_len, config.num_kv_heads, d))
    kv_shared_cache = {
        'k': dummy_k,
        'v': dummy_v,
        'valid_mask': jnp.ones((b, kv_len), dtype=jnp.bool_),
        'prior_end_index': jnp.array([kv_len - q_len], dtype=jnp.int32),
        'split_prefix_k': dummy_k if has_split_prefix else None,
        'split_prefix_v': dummy_v if has_split_prefix else None,
    }

    # For _flash_attention_split, we need mock_kernel to return (q, (lse,))
    # For _flash_attention_single, we need mock_kernel to return q
    def mock_kernel(q_in, k_in, v_in, segment_ids=None):
      del k_in, v_in, segment_ids
      if has_split_prefix:
        lse = jnp.zeros(q_in.shape[:-1], dtype=jnp.float32)
        return jnp.zeros_like(q_in), (lse,)
      return jnp.zeros_like(q_in)

    mesh = jax.sharding.Mesh(
        np.array(jax.devices()[:1]).reshape(1, 1), ('fsdp', 'tp')
    )

    with mesh, mock.patch.object(
        attn, '_flash_attention_split', wraps=attn._flash_attention_split
    ) as mock_split, mock.patch.object(
        attn, '_flash_attention_single', wraps=attn._flash_attention_single
    ) as mock_single, mock.patch.object(
        attn, '_make_splash_kernel', return_value=(mock_kernel, None)
    ):
      attn.block(
          x,
          segment_pos,
          cache=None,
          attn_mask=attn_mask,
          kv_shared_cache=kv_shared_cache,
          is_chunked_prefill=True,
          prefix_length=kv_len - q_len,
      )

      if has_split_prefix:
        mock_split.assert_called_once()
        mock_single.assert_not_called()
      else:
        mock_split.assert_not_called()
        mock_single.assert_called_once()

  @parameterized.named_parameters(
      dict(
          testcase_name='local_sliding_q_len_greater_than_window',
          attn_type=model_lib.AttentionType.LOCAL_SLIDING,
          q_len=32,
          window_size=16,
          expected_mask_type=mask_lib.LocalMask,
      ),
      dict(
          testcase_name='global_full_causal',
          attn_type=model_lib.AttentionType.GLOBAL,
          q_len=32,
          window_size=16,
          expected_mask_type=mask_lib.CausalMask,
      ),
      dict(
          testcase_name='local_sliding_q_len_equal_window',
          attn_type=model_lib.AttentionType.LOCAL_SLIDING,
          q_len=16,
          window_size=16,
          expected_mask_type=mask_lib.LocalMask,
      ),
  )
  def test_split_attention_suffix_mask_respects_window_size(
      self, attn_type, q_len, window_size, expected_mask_type
  ):
    """Verifies that _flash_attention_split constructs the correct suffix mask."""
    config = model_lib.ModelConfig.gemma4_e2b()
    config.sliding_window_size = window_size
    config.flash_attention_block_size = 16
    config.use_split_attention = True
    config.use_flash_attention = True
    attn = attention_lib.Attention(
        config=config, attn_type=attn_type, rngs=nnx.Rngs(0)
    )

    captured_masks = []
    mock_kernel = lambda q, k, v, segment_ids=None: (
        q,
        (jnp.zeros(q.shape[:-1], dtype=jnp.float32),),
    )
    dummy = jnp.zeros((1, 1, 1, 1))
    mesh = jax.sharding.Mesh(
        np.array(jax.devices()[:1]).reshape(1, 1), ('fsdp', 'tp')
    )

    with mock.patch.object(
        attn,
        '_make_splash_kernel',
        side_effect=lambda m, *a, **kw: (
            captured_masks.append(m.masks[0]),
            (mock_kernel, None),
        )[1],
    ):
      attn._flash_attention_split(
          query_proj=dummy,
          key_proj=dummy,
          value_proj=dummy,
          split_prefix_k=dummy,
          split_prefix_v=dummy,
          q_len=q_len,
          prefix_kv_len=16,
          qh=config.num_heads,
          block_sizes=attn._make_block_sizes(is_rectangular=True, q_len=q_len),
          head_shards=1,
          q_seq_shards=1,
          mesh=mesh,
          shd_b=None,
          shd_n=None,
          shd_t=None,
          shd_h=None,
          shd_n_kv=None,
          shd_spec=P(None, None, None, None),
      )

    self.assertLen(captured_masks, 2)
    prefix_mask = captured_masks[0]
    suffix_mask = captured_masks[1]
    self.assertIsInstance(suffix_mask, expected_mask_type)

    row = np.arange(q_len)[:, None]
    col = np.arange(q_len)[None, :]
    if attn_type == model_lib.AttentionType.LOCAL_SLIDING:
      expected_suffix = (col <= row) & (col > row - window_size)
    else:
      expected_suffix = col <= row
    np.testing.assert_array_equal(suffix_mask[np.s_[:, :]], expected_suffix)

    if (
        q_len > window_size
        and attn_type == model_lib.AttentionType.LOCAL_SLIDING
    ):
      self.assertFalse(prefix_mask[np.s_[:, :]][window_size:, :].any())

  def test_flash_attention_single_reconstructs_concatenated_prefix_kv(self):
    """Verifies single-kernel flash attention concatenates split_prefix_k/v.

    Triggered when split attention is bypassed.
    """
    config = model_lib.ModelConfig.gemma4_e2b()
    attn = attention_lib.Attention(
        config=config,
        attn_type=model_lib.AttentionType.GLOBAL,
        rngs=nnx.Rngs(0),
    )
    b, q_len, prefix_len = 1, 8, 16
    h, kh, d = config.num_heads, config.num_kv_heads, config.head_dim
    dummy_q = jnp.zeros((b, h, q_len, d))
    dummy_k = jnp.zeros((b, kh, q_len, d))
    dummy_v = jnp.zeros((b, kh, q_len, d))
    split_prefix_k = jnp.ones((b, prefix_len, kh, d))
    split_prefix_v = jnp.ones((b, prefix_len, kh, d))

    captured_kv_shapes = []

    def mock_kernel(q_in, k_in, v_in, segment_ids=None):
      del segment_ids
      captured_kv_shapes.append((k_in.shape, v_in.shape))
      return q_in

    mesh = jax.sharding.Mesh(
        np.array(jax.devices()[:1]).reshape(1, 1), ('fsdp', 'tp')
    )
    _, new_k, new_v = attn._flash_attention_single(
        query_proj=dummy_q,
        key_proj=dummy_k,
        value_proj=dummy_v,
        split_prefix_k=split_prefix_k,
        split_prefix_v=split_prefix_v,
        segment_ids=None,
        splash_attn_kernel=mock_kernel,
        kernel_spec=None,
        shd_spec=P(None, None, None, None),
        unsharded_seq_kv=P(None, None, None, None),
        mesh=mesh,
        shd_b=None,
        shd_t=None,
    )

    expected_seq_len = prefix_len + q_len
    self.assertEqual(captured_kv_shapes[0][0][1], expected_seq_len)
    self.assertEqual(captured_kv_shapes[0][1][1], expected_seq_len)
    self.assertEqual(new_k.shape[1], expected_seq_len)
    self.assertEqual(new_v.shape[1], expected_seq_len)


if __name__ == '__main__':
  absltest.main()
