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

"""Unit tests for model-agnostic cache and chunked prefill utilities."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tunix.models import cache_utils


class CacheUtilsTest(parameterized.TestCase):

  def test_pow2_buckets(self):
    pow2 = cache_utils.pow2_buckets(max_len=1024)
    self.assertEqual(pow2, (0, 128, 256, 512, 1024))
    pow2_default = cache_utils.pow2_buckets()
    self.assertEqual(pow2_default[0], 0)
    self.assertEqual(pow2_default[1], 128)
    self.assertEqual(pow2_default[-1], 131072)

  def test_linear_buckets(self):
    linear = cache_utils.linear_buckets(step=256, max_len=1024)
    self.assertEqual(linear, (0, 256, 512, 768, 1024))
    linear_default = cache_utils.linear_buckets()
    self.assertEqual(linear_default[0], 0)
    self.assertEqual(linear_default[1], 512)
    self.assertEqual(linear_default[-1], 131072)

    self.assertEqual(
        cache_utils.linear_buckets(step=1, max_len=5), (0, 1, 2, 3, 4, 5)
    )
    self.assertEqual(cache_utils.linear_buckets(step=3, max_len=8), (0, 3, 6))

  @parameterized.named_parameters(
      dict(
          testcase_name='exact_boundary',
          prefix_length=128,
          cache_len=1024,
          boundaries=(0, 128, 256),
          expected=128,
      ),
      dict(
          testcase_name='in_between_rounds_up_to_next_boundary',
          prefix_length=100,
          cache_len=1024,
          boundaries=(0, 128, 256),
          expected=128,
      ),
      dict(
          testcase_name='overflow_past_boundaries_falls_back_to_cache_len',
          prefix_length=500,
          cache_len=1024,
          boundaries=(0, 128, 256),
          expected=1024,
      ),
      dict(
          testcase_name='beyond_cache_len_clamped_to_cache_len',
          prefix_length=2000,
          cache_len=1024,
          boundaries=(0, 128, 256),
          expected=1024,
      ),
      dict(
          testcase_name='boundary_exceeds_cache_len_clamped_to_cache_len',
          prefix_length=300,
          cache_len=256,
          boundaries=(0, 128, 512),
          expected=256,
      ),
      dict(
          testcase_name='empty_boundaries_ladder_falls_back_to_cache_len',
          prefix_length=100,
          cache_len=256,
          boundaries=(),
          expected=256,
      ),
  )
  def test_bucket_prefix_length(
      self, prefix_length, cache_len, boundaries, expected
  ):
    self.assertEqual(
        cache_utils.bucket_prefix_length(prefix_length, cache_len, boundaries),
        expected,
    )

  def test_maybe_bucket_prefix_length(self):
    cache = {'v': jnp.zeros((1, 1024, 1, 64))}
    boundaries = (0, 128, 256)

    # Chunked prefill with LayerCache
    self.assertEqual(
        cache_utils.maybe_bucket_prefix_length(
            100, cache, is_chunked_prefill=True, boundaries=boundaries
        ),
        128,
    )
    # Chunked prefill with integer cache length
    self.assertEqual(
        cache_utils.maybe_bucket_prefix_length(
            100, 1024, is_chunked_prefill=True, boundaries=boundaries
        ),
        128,
    )
    # Chunked prefill with None cache_or_len
    self.assertEqual(
        cache_utils.maybe_bucket_prefix_length(
            100, None, is_chunked_prefill=True, boundaries=boundaries
        ),
        100,
    )
    # Non-chunked prefill bypasses bucketing
    self.assertEqual(
        cache_utils.maybe_bucket_prefix_length(
            100, cache, is_chunked_prefill=False, boundaries=boundaries
        ),
        100,
    )
    # prefix_length <= 0 bypasses bucketing
    self.assertEqual(
        cache_utils.maybe_bucket_prefix_length(
            0, cache, is_chunked_prefill=True, boundaries=boundaries
        ),
        0,
    )
    # Empty boundaries ladder bypasses bucketing
    self.assertEqual(
        cache_utils.maybe_bucket_prefix_length(
            100, cache, is_chunked_prefill=True, boundaries=()
        ),
        100,
    )

  def test_find_last_one_index(self):
    mask = jnp.array(
        [
            [[1, 1, 1, 0, 0]],
            [[1, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0]],
        ],
        dtype=jnp.int32,
    )
    last_indices = cache_utils.find_last_one_index(mask)
    np.testing.assert_array_equal(
        last_indices,
        np.array([2, 0, 0], dtype=np.int32),
    )

  def test_create_logical_sliding_window_mask_gapped_boundaries(self):
    cache_len = 16
    sw = 4
    mask_indices = [0, 1, 2, 3, 4, 14]
    attn_mask = jnp.zeros((1, 1, cache_len), dtype=jnp.int32)
    attn_mask = attn_mask.at[0, 0, mask_indices].set(1)

    result = cache_utils.create_logical_sliding_window_mask(
        attn_mask, sliding_window_size=sw
    )

    self.assertFalse(bool(result[0, 0, 1]))
    self.assertTrue(bool(result[0, 0, 2]))
    self.assertEqual(int(jnp.sum(result)), 4)
    expected = jnp.zeros((1, 1, cache_len), dtype=jnp.bool_)
    expected = expected.at[0, 0, [2, 3, 4, 14]].set(True)
    np.testing.assert_array_equal(result, expected)

  def test_create_logical_sliding_window_mask_contiguous_matches_physical(self):
    attn_mask = jnp.array([[[1, 1, 1, 1, 0, 0]]], dtype=jnp.int32)
    sw = 2
    logical_mask = cache_utils.create_logical_sliding_window_mask(
        attn_mask, sliding_window_size=sw
    )
    physical_mask = cache_utils.create_sliding_window_mask(
        attn_mask, sliding_window_size=sw
    )
    np.testing.assert_array_equal(logical_mask, physical_mask)

  def test_has_physical_gap_batched(self):
    cache_len = 16
    attn_mask = jnp.zeros((7, 1, cache_len), dtype=jnp.int32)
    # row 0: contiguous [1, 1, 1, 1, 0, ...] -> False
    attn_mask = attn_mask.at[0, 0, [0, 1, 2, 3]].set(1)
    # row 1: multi-gap [1, 1, 0, 0, 1, 0, ...] -> True
    attn_mask = attn_mask.at[1, 0, [0, 1, 4]].set(1)
    # row 2: all zeros [0, 0, 0, ...] -> False
    # row 3: single token [0, 0, 1, 0, ...] -> False
    attn_mask = attn_mask.at[3, 0, [2]].set(1)
    # row 4: single-token gap [1, 0, 1, 0, ...] (count=2, span=3) -> True
    attn_mask = attn_mask.at[4, 0, [0, 2]].set(1)
    # row 5: single-token gap with prefix [1, 1, 0, 1, 0, ...] (count=3, span=4) -> True
    attn_mask = attn_mask.at[5, 0, [0, 1, 3]].set(1)
    # row 6: offset contiguous [0, 0, 1, 1, 1, 0, ...] (count=3, span=3) -> False
    attn_mask = attn_mask.at[6, 0, [2, 3, 4]].set(1)

    result = cache_utils.has_physical_gap(attn_mask)
    self.assertEqual(result.shape, (7, 1, 1))
    expected = jnp.array([
        [[False]],
        [[True]],
        [[False]],
        [[False]],
        [[True]],
        [[True]],
        [[False]],
    ])
    np.testing.assert_array_equal(result, expected)

  def test_read_prefix_kv_passthrough(self):
    b, s, kh, d = 2, 4, 2, 16
    key_proj = jnp.ones((b, s, kh, d))
    value_proj = jnp.ones((b, s, kh, d)) * 2
    cache = {
        'k': jnp.zeros((b, 16, kh, d)),
        'v': jnp.zeros((b, 16, kh, d)),
        'end_index': jnp.array([4]),
    }

    # Not chunked prefill
    res = cache_utils.read_prefix_kv(
        cache, key_proj, value_proj, is_chunked_prefill=False, prefix_length=4
    )
    np.testing.assert_array_equal(res.key, key_proj)
    np.testing.assert_array_equal(res.value, value_proj)
    self.assertIsNone(res.valid_mask)

    # prefix_length = 0
    res = cache_utils.read_prefix_kv(
        cache, key_proj, value_proj, is_chunked_prefill=True, prefix_length=0
    )
    np.testing.assert_array_equal(res.key, key_proj)
    np.testing.assert_array_equal(res.value, value_proj)
    self.assertIsNone(res.valid_mask)

  def test_read_prefix_kv_linear(self):
    b, s, kh, d = 1, 4, 1, 8
    cache_len = 16
    prefix_length = 8
    k_cached = jnp.broadcast_to(
        (jnp.arange(cache_len, dtype=jnp.float32) + 1.0)[None, :, None, None],
        (b, cache_len, kh, d),
    )
    v_cached = k_cached * 2.0
    cache = {
        'k': k_cached,
        'v': v_cached,
        'end_index': jnp.array([6]),  # Only first 6 are valid
    }
    key_proj = jnp.full((b, s, kh, d), 100.0)
    value_proj = jnp.full((b, s, kh, d), 200.0)

    res = cache_utils.read_prefix_kv(
        cache,
        key_proj,
        value_proj,
        is_chunked_prefill=True,
        prefix_length=prefix_length,
        is_ring_buffer=False,
    )

    self.assertEqual(res.key.shape, (b, prefix_length + s, kh, d))
    self.assertEqual(res.value.shape, (b, prefix_length + s, kh, d))
    self.assertIsNone(res.valid_mask)
    # Positions 0..5 should have cached values; 6..7 zeroed out because prior_end_index=6
    expected_k_prefix = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 0.0])
    np.testing.assert_allclose(
        res.key[0, :prefix_length, 0, 0], expected_k_prefix
    )
    np.testing.assert_allclose(res.key[0, prefix_length:, 0, 0], 100.0)

  def test_read_prefix_kv_ring_buffer_unrolling(self):
    b, cache_len, seq_len = 1, 8, 4
    kh, d = 1, 4
    k_cached = jnp.broadcast_to(
        (jnp.arange(cache_len, dtype=jnp.float32) * 10.0)[None, :, None, None],
        (b, cache_len, kh, d),
    )
    v_cached = jnp.broadcast_to(
        (jnp.arange(cache_len, dtype=jnp.float32) * 10.0 + 1.0)[
            None, :, None, None
        ],
        (b, cache_len, kh, d),
    )
    cache = {
        'k': k_cached,
        'v': v_cached,
        'end_index': jnp.array([12]),
    }
    key_proj = jnp.full((b, seq_len, kh, d), 100.0)
    value_proj = jnp.full((b, seq_len, kh, d), 101.0)
    prior_end_index = jnp.array([12])

    res = cache_utils.read_prefix_kv(
        cache,
        key_proj,
        value_proj,
        is_chunked_prefill=True,
        prefix_length=8,
        is_ring_buffer=True,
        prior_end_index=prior_end_index,
    )

    expected_k_prefix = jnp.array(
        [40.0, 50.0, 60.0, 70.0, 0.0, 10.0, 20.0, 30.0]
    )
    expected_v_prefix = jnp.array(
        [41.0, 51.0, 61.0, 71.0, 1.0, 11.0, 21.0, 31.0]
    )

    self.assertEqual(res.key.shape, (b, cache_len + seq_len, kh, d))
    self.assertEqual(res.value.shape, (b, cache_len + seq_len, kh, d))
    np.testing.assert_allclose(res.key[0, :cache_len, 0, 0], expected_k_prefix)
    np.testing.assert_allclose(
        res.value[0, :cache_len, 0, 0], expected_v_prefix
    )
    np.testing.assert_allclose(
        res.key[0, cache_len:, 0, 0], jnp.full((seq_len,), 100.0)
    )
    np.testing.assert_allclose(
        res.value[0, cache_len:, 0, 0], jnp.full((seq_len,), 101.0)
    )
    self.assertIsNotNone(res.valid_mask)
    np.testing.assert_array_equal(
        res.valid_mask, jnp.ones((cache_len,), dtype=jnp.bool_)
    )

  def test_write_cache_prefill_linear(self):
    b, seq_len, cache_len, kh, d = 1, 4, 16, 1, 8
    key_proj = jnp.ones((b, seq_len, kh, d)) * 5.0
    value_proj = jnp.ones((b, seq_len, kh, d)) * 6.0
    cache = {
        'k': jnp.zeros((b, cache_len, kh, d)),
        'v': jnp.zeros((b, cache_len, kh, d)),
        'end_index': jnp.array([4]),
    }

    new_cache = cache_utils.write_cache_prefill(
        cache,
        key_proj,
        value_proj,
        seq_len=seq_len,
        is_chunked_prefill=True,
        input_mask=None,
        is_ring_buffer=False,
    )

    np.testing.assert_allclose(new_cache['k'][0, 4:8, 0, 0], 5.0)
    np.testing.assert_allclose(new_cache['v'][0, 4:8, 0, 0], 6.0)
    np.testing.assert_array_equal(new_cache['end_index'], jnp.array([8]))

  def test_write_cache_prefill_ring_buffer_ragged(self):
    b, seq_len, cache_len = 2, 10, 16
    kh, d = 1, 4
    key_proj = jnp.zeros((b, seq_len, kh, d))
    value_proj = jnp.zeros((b, seq_len, kh, d))

    # Row 0: 7 real tokens, Row 1: 4 real tokens. Max = 7.
    input_mask = jnp.array(
        [[1] * 7 + [0] * 3, [1] * 4 + [0] * 6], dtype=jnp.bool_
    )
    cache = {
        'k': jnp.zeros((b, cache_len, kh, d)),
        'v': jnp.zeros((b, cache_len, kh, d)),
        'end_index': jnp.array([5, 5], dtype=jnp.int32),
    }
    updated_cache = cache_utils.write_cache_prefill(
        cache,
        key_proj,
        value_proj,
        seq_len=seq_len,
        is_chunked_prefill=True,
        input_mask=input_mask,
        is_ring_buffer=True,
    )
    np.testing.assert_array_equal(
        updated_cache['end_index'], jnp.array([12, 12], dtype=jnp.int32)
    )

    # input_mask is None -> advance by seq_len
    updated_cache_none = cache_utils.write_cache_prefill(
        cache,
        key_proj,
        value_proj,
        seq_len=seq_len,
        is_chunked_prefill=True,
        input_mask=None,
        is_ring_buffer=True,
    )
    np.testing.assert_array_equal(
        updated_cache_none['end_index'], jnp.array([15, 15], dtype=jnp.int32)
    )

  def test_build_dense_chunked_prefill_mask_global(self):
    b, q_len, prefix_length = 1, 4, 8
    kv_len = prefix_length + q_len
    attn_mask = jnp.ones((b, q_len, kv_len), dtype=jnp.bool_)
    prior_end_index = jnp.array([6])

    mask = cache_utils.build_dense_chunked_prefill_mask(
        attn_mask=attn_mask,
        q_len=q_len,
        kv_len=kv_len,
        prior_end_index=prior_end_index,
        prefix_length=prefix_length,
        is_ring_buffer=False,
    )

    self.assertEqual(mask.shape, (b, q_len, kv_len))
    # Positions 0..5 are valid prefix; 6..7 are masked out; 8..11 are suffix (causal/ones)
    self.assertTrue(bool(mask[0, 0, 5]))
    self.assertFalse(bool(mask[0, 0, 6]))
    self.assertFalse(bool(mask[0, 0, 7]))
    self.assertTrue(bool(mask[0, 0, 8]))

  def test_build_dense_chunked_prefill_mask_ring_buffer(self):
    b, q_len, prefix_kv_len, sw = 1, 4, 8, 4
    kv_len = prefix_kv_len + q_len
    attn_mask = jnp.concatenate(
        [
            jnp.ones((b, q_len, prefix_kv_len), dtype=jnp.bool_),
            jnp.broadcast_to(
                jnp.tril(jnp.ones((q_len, q_len), dtype=jnp.bool_))[None, :, :],
                (b, q_len, q_len),
            ),
        ],
        axis=-1,
    )

    mask = cache_utils.build_dense_chunked_prefill_mask(
        attn_mask=attn_mask,
        q_len=q_len,
        kv_len=kv_len,
        prior_end_index=jnp.array([8]),
        prefix_length=8,
        is_ring_buffer=True,
        sliding_window_size=sw,
    )

    self.assertEqual(mask.shape, (b, q_len, kv_len))
    self.assertFalse(jnp.all(mask == 0))

  def test_write_cache_decode_linear_and_ring_buffer(self):
    b, cache_len, kh, d = 1, 8, 1, 4
    cache = {
        'k': jnp.zeros((b, cache_len, kh, d)),
        'v': jnp.zeros((b, cache_len, kh, d)),
        'end_index': jnp.array([3]),
    }
    key_proj = jnp.ones((b, 1, kh, d)) * 7.0
    value_proj = jnp.ones((b, 1, kh, d)) * 8.0
    attn_mask = jnp.array([[[1, 1, 1, 1, 0, 0, 0, 0]]], dtype=jnp.int32)

    # Linear cache branch
    new_k_lin, new_v_lin = cache_utils.write_cache_decode(
        cache, key_proj, value_proj, attn_mask, is_ring_buffer=False
    )
    np.testing.assert_allclose(new_k_lin[0, 3, 0, 0], 7.0)
    np.testing.assert_allclose(new_v_lin[0, 3, 0, 0], 8.0)

    # Ring buffer branch
    new_k_rb, new_v_rb = cache_utils.write_cache_decode(
        cache, key_proj, value_proj, attn_mask, is_ring_buffer=True
    )
    np.testing.assert_allclose(new_k_rb[0, 3, 0, 0], 7.0)
    np.testing.assert_allclose(new_v_rb[0, 3, 0, 0], 8.0)

  def test_build_sliding_window_decode_mask_both_branches(self):
    b, cache_len, sw = 1, 8, 4
    attn_mask = jnp.array([[[1, 1, 1, 1, 1, 0, 0, 0]]], dtype=jnp.int32)
    end_idx = jnp.array([4])

    # Ring buffer branch
    rb_mask = cache_utils.build_sliding_window_decode_mask(
        attn_mask,
        window_size=sw,
        is_ring_buffer=True,
        end_idx=end_idx,
        cache_len=cache_len,
    )
    self.assertEqual(rb_mask.shape, (b, 1, cache_len))

    # Linear cache branch
    lin_mask = cache_utils.build_sliding_window_decode_mask(
        attn_mask,
        window_size=sw,
        is_ring_buffer=False,
    )
    self.assertEqual(lin_mask.shape, (b, 1, cache_len))
    # Only last 4 valid tokens (indices 1..4) should remain active
    expected = jnp.array([[[0, 1, 1, 1, 1, 0, 0, 0]]], dtype=jnp.int32)
    np.testing.assert_array_equal(lin_mask, expected)

  def test_update_cache_prefill(self):
    b, seq_len, cache_len, kh, d = 1, 4, 16, 1, 4
    cache = {
        'k': jnp.ones((b, cache_len, kh, d)) * 2.0,
        'v': jnp.ones((b, cache_len, kh, d)) * 3.0,
        'end_index': jnp.array([4]),
    }
    key_proj = jnp.ones((b, seq_len, kh, d)) * 10.0
    value_proj = jnp.ones((b, seq_len, kh, d)) * 20.0

    new_cache, prefix_res, prior_end_index = cache_utils.update_cache_prefill(
        cache,
        key_proj,
        value_proj,
        seq_len,
        is_chunked_prefill=True,
        prefix_length=4,
        input_mask=None,
        is_ring_buffer_read=False,
        is_ring_buffer_write=False,
        use_split_attention=False,
    )
    self.assertEqual(prefix_res.key.shape, (b, 8, kh, d))
    self.assertEqual(prefix_res.value.shape, (b, 8, kh, d))
    self.assertIsNone(prefix_res.valid_mask)
    self.assertEqual(int(prior_end_index), 4)
    self.assertIsNone(prefix_res.split_prefix_k)
    self.assertIsNone(prefix_res.split_prefix_v)
    np.testing.assert_array_equal(new_cache['end_index'], jnp.array([8]))

  def test_build_dense_chunked_prefill_mask_shared_cache(self):
    b, q_len, prefix_length = 1, 4, 8
    kv_len = prefix_length + q_len
    attn_mask = jnp.ones((b, q_len, kv_len), dtype=jnp.bool_)
    shared_cache = {'prior_end_index': jnp.array([6])}

    mask = cache_utils.build_dense_chunked_prefill_mask(
        attn_mask=attn_mask,
        q_len=q_len,
        kv_len=kv_len,
        prior_end_index=None,
        prefix_length=prefix_length,
        kv_shared_cache=shared_cache,
        has_own_cache=False,
    )
    self.assertTrue(bool(mask[0, 0, 5]))
    self.assertFalse(bool(mask[0, 0, 6]))

    with self.assertRaises(ValueError):
      cache_utils.build_dense_chunked_prefill_mask(
          attn_mask=attn_mask,
          q_len=q_len,
          kv_len=kv_len,
          prior_end_index=None,
          prefix_length=prefix_length,
          kv_shared_cache={},
          has_own_cache=False,
      )


if __name__ == '__main__':
  absltest.main()
