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

"""Model-agnostic utilities for KV cache management and chunked prefill."""

import bisect
from collections.abc import Mapping
from flax import struct
import jax
import jax.numpy as jnp
import jaxtyping

LayerCache = dict[str, jaxtyping.Array]
Cache = Mapping[str, LayerCache]


@struct.dataclass
class PrefixKVResult:
  key: jaxtyping.Array
  value: jaxtyping.Array
  valid_mask: jaxtyping.Array | None = None
  split_prefix_k: jaxtyping.Array | None = None
  split_prefix_v: jaxtyping.Array | None = None


def pow2_buckets(max_len: int = 131072) -> tuple[int, ...]:
  """Powers-of-two ladder: (0, 128, 256, ..., max_len). Default."""
  buckets = [0]
  n = 128
  while n <= max_len:
    buckets.append(n)
    n *= 2
  return tuple(buckets)


def linear_buckets(step: int = 512, max_len: int = 131072) -> tuple[int, ...]:
  """Linear ladder: (0, step, 2*step, ..., max_len). The 'x*512' case."""
  return tuple(range(0, max_len + 1, step))


def bucket_prefix_length(
    prefix_length: int, cache_len: int, boundaries: tuple[int, ...]
) -> int:
  """Round prefix_length up to the nearest bucket for compilation stability."""
  i = bisect.bisect_left(boundaries, prefix_length)
  bucket = boundaries[i] if i < len(boundaries) else cache_len
  return min(bucket, cache_len)


def maybe_bucket_prefix_length(
    prefix_length: int,
    cache_or_len: LayerCache | int | None,
    is_chunked_prefill: bool,
    boundaries: tuple[int, ...],
) -> int:
  """Buckets prefix_length during chunked prefill; passthrough otherwise."""
  if not is_chunked_prefill or prefix_length <= 0 or not boundaries:
    return prefix_length
  if isinstance(cache_or_len, Mapping):
    if (
        'split_prefix_v' in cache_or_len
        and cache_or_len['split_prefix_v'] is not None
    ):
      effective_cap = cache_or_len['split_prefix_v'].shape[1]
    else:
      effective_cap = cache_or_len['v'].shape[1]
  elif cache_or_len is not None:
    effective_cap = cache_or_len
  else:
    effective_cap = prefix_length
  return bucket_prefix_length(prefix_length, effective_cap, boundaries)


def merge_split_attention(
    out_prefix: jaxtyping.Array,
    lse_prefix: jaxtyping.Array,
    out_suffix: jaxtyping.Array,
    lse_suffix: jaxtyping.Array,
    out_dtype: jnp.dtype,
) -> jaxtyping.Array:
  """LSE-weighted merge of two attention partitions. Pure JAX; CPU-testable.

  nan_to_num zeroes fully-masked (out=NaN / lse=-inf) partitions so they don't
  poison the residual stream. Boundary-straddling garbage rows are instead
  neutralized by weight underflow, which relies on splash's DEFAULT_MASK_VALUE
  staying large-negative.

  Args:
    out_prefix: Attention output for the prefix partition.
    lse_prefix: Log-sum-exp weights for the prefix partition.
    out_suffix: Attention output for the suffix partition.
    lse_suffix: Log-sum-exp weights for the suffix partition.
    out_dtype: Output data type for the merged attention result.

  Returns:
    Merged attention output array.
  """
  # lse shape: (B, N, T), out shape: (B, N, T, H)
  max_lse = jnp.maximum(lse_prefix, lse_suffix)
  # Guard against (-inf) - (-inf) = NaN when both partitions are fully masked.
  w_prefix = jnp.nan_to_num(jnp.exp(lse_prefix - max_lse), nan=0.0)
  w_suffix = jnp.nan_to_num(jnp.exp(lse_suffix - max_lse), nan=0.0)
  w_sum = w_prefix + w_suffix
  w_sum_safe = jnp.where(w_sum > 0, w_sum, 1.0)
  encoded = (
      w_prefix[..., None] * jnp.nan_to_num(out_prefix.astype(jnp.float32))
      + w_suffix[..., None] * jnp.nan_to_num(out_suffix.astype(jnp.float32))
  ) / w_sum_safe[..., None]
  return encoded.astype(out_dtype)


def read_prefix_kv(
    cache: LayerCache,
    key_proj: jaxtyping.Array,
    value_proj: jaxtyping.Array,
    *,
    is_chunked_prefill: bool,
    prefix_length: int,
    is_ring_buffer: bool = False,
    prior_end_index: jaxtyping.Array | None = None,
    use_split_attention: bool = False,
) -> PrefixKVResult:
  """Reads and prepares prefix KV projections from linear or ring buffer cache."""
  if not (is_chunked_prefill and prefix_length > 0):
    return PrefixKVResult(key=key_proj, value=value_proj, valid_mask=None)

  cache_len = cache['v'].shape[1]
  prefix_length = min(prefix_length, cache_len)
  if prior_end_index is None:
    prior_end_index = cache['end_index'][0]

  if is_ring_buffer:
    valid_cached = jnp.minimum(prior_end_index, cache_len)
    read_start = (prior_end_index - valid_cached) % cache_len
    i = jnp.arange(cache_len)
    kv_valid_mask = i < valid_cached
    physical_indices = (read_start + i) % cache_len
    cached_k = cache['k'][:, physical_indices, ...]
    cached_v = cache['v'][:, physical_indices, ...]
    cached_k = jnp.where(kv_valid_mask[None, :, None, None], cached_k, 0)
    cached_v = jnp.where(kv_valid_mask[None, :, None, None], cached_v, 0)
  else:
    cached_k = cache['k'][:, :prefix_length, ...]
    cached_v = cache['v'][:, :prefix_length, ...]
    valid_prefix = jnp.arange(prefix_length) < prior_end_index
    cached_k = jnp.where(valid_prefix[None, :, None, None], cached_k, 0)
    cached_v = jnp.where(valid_prefix[None, :, None, None], cached_v, 0)
    kv_valid_mask = None

  if use_split_attention:
    return PrefixKVResult(
        key=key_proj,
        value=value_proj,
        valid_mask=kv_valid_mask,
        split_prefix_k=cached_k,
        split_prefix_v=cached_v,
    )

  concat_k = jnp.concatenate([cached_k, key_proj], axis=1)
  concat_v = jnp.concatenate([cached_v, value_proj], axis=1)
  return PrefixKVResult(key=concat_k, value=concat_v, valid_mask=kv_valid_mask)


def write_cache_prefill(
    cache: LayerCache,
    key_proj: jaxtyping.Array,
    value_proj: jaxtyping.Array,
    seq_len: int,
    *,
    is_chunked_prefill: bool,
    input_mask: jaxtyping.Array | None,
    is_ring_buffer: bool = False,
) -> LayerCache:
  """Writes fresh KV projections to cache with ragged token index advance."""
  cache_len = cache['v'].shape[1]
  end_index = cache['end_index']
  b = value_proj.shape[0]

  if is_ring_buffer:
    if is_chunked_prefill and input_mask is not None:
      n_r = jnp.sum(input_mask.astype(jnp.int32), axis=-1, keepdims=True)
      c = (n_r - cache_len) + jnp.arange(cache_len)[None, :]
      valid = (c >= 0) & (c < n_r)
      slot = (end_index.reshape(b, 1) + c) % cache_len
      b_idx = jnp.arange(b)[:, None]
      c_clamped = jnp.clip(c, 0, seq_len - 1)
      new_k = (
          cache['k']
          .at[b_idx, slot]
          .set(
              jnp.where(
                  valid[..., None, None],
                  key_proj[b_idx, c_clamped],
                  cache['k'][b_idx, slot],
              )
          )
      )
      new_v = (
          cache['v']
          .at[b_idx, slot]
          .set(
              jnp.where(
                  valid[..., None, None],
                  value_proj[b_idx, c_clamped],
                  cache['v'][b_idx, slot],
              )
          )
      )
    else:
      prior_end_index = end_index[0]
      valid_len = min(seq_len, cache_len)
      latest_indices = (
          prior_end_index + (seq_len - valid_len) + jnp.arange(valid_len)
      ) % cache_len
      new_v = value_proj[:, -valid_len:, ...]
      new_k = key_proj[:, -valid_len:, ...]
      new_v = cache['v'].at[:, latest_indices, ...].set(new_v)
      new_k = cache['k'].at[:, latest_indices, ...].set(new_k)
  else:
    slice_indices = (0, end_index[0] % cache_len, 0, 0)
    new_k = jax.lax.dynamic_update_slice(cache['k'], key_proj, slice_indices)
    new_v = jax.lax.dynamic_update_slice(cache['v'], value_proj, slice_indices)

  advance_by = (
      jnp.max(jnp.sum(input_mask.astype(jnp.int32), axis=-1)).astype(jnp.int32)
      if is_chunked_prefill and input_mask is not None
      else seq_len
  )
  return {'k': new_k, 'v': new_v, 'end_index': end_index + advance_by}


def write_cache_decode(
    cache: LayerCache,
    key_proj: jaxtyping.Array,
    value_proj: jaxtyping.Array,
    attn_mask: jaxtyping.Array,
    is_ring_buffer: bool,
) -> tuple[jaxtyping.Array, jaxtyping.Array]:
  """Writes a single decode token into the KV cache and returns updated K/V buffers."""
  if is_ring_buffer:
    b = value_proj.shape[0]
    cache_len_local = cache['v'].shape[1]
    abs_slot = cache['end_index'] % cache_len_local
    logical_pos = jnp.sum((attn_mask != 0).astype(jnp.int32), axis=-1)[:, 0] - 1
    logical_slot = logical_pos % cache_len_local
    has_gap = has_physical_gap(attn_mask)[:, 0, 0]
    slot = jnp.where(has_gap, logical_slot, abs_slot)
    b_idx = jnp.arange(b)
    new_v = cache['v'].at[b_idx, slot].set(value_proj[:, 0])
    new_k = cache['k'].at[b_idx, slot].set(key_proj[:, 0])
    return new_k, new_v
  else:
    cache_len = cache['v'].shape[1]
    end_index = cache['end_index'][0]
    slice_indices = (0, end_index % cache_len, 0, 0)
    new_v = jax.lax.dynamic_update_slice(cache['v'], value_proj, slice_indices)
    new_k = jax.lax.dynamic_update_slice(cache['k'], key_proj, slice_indices)
    return new_k, new_v


def update_cache_prefill(
    cache: LayerCache,
    key_proj: jaxtyping.Array,
    value_proj: jaxtyping.Array,
    seq_len: int,
    *,
    is_chunked_prefill: bool,
    prefix_length: int,
    input_mask: jaxtyping.Array | None,
    is_ring_buffer_read: bool,
    is_ring_buffer_write: bool,
    use_split_attention: bool = False,
) -> tuple[LayerCache, PrefixKVResult, jaxtyping.Array]:
  """Orchestrates prefix read followed by prefill cache write for XLA buffer donation."""
  prior_end_index = cache['end_index'][0]
  prefix_res = read_prefix_kv(
      cache,
      key_proj,
      value_proj,
      is_chunked_prefill=is_chunked_prefill,
      prefix_length=prefix_length,
      is_ring_buffer=is_ring_buffer_read,
      prior_end_index=prior_end_index,
      use_split_attention=use_split_attention,
  )
  new_cache = write_cache_prefill(
      cache,
      key_proj,
      value_proj,
      seq_len,
      is_chunked_prefill=is_chunked_prefill,
      input_mask=input_mask,
      is_ring_buffer=is_ring_buffer_write,
  )
  return new_cache, prefix_res, prior_end_index


def find_last_one_index(attn_mask: jnp.ndarray) -> jnp.ndarray:
  """Finds the index of the last rightmost 1 from attn_mask."""
  cache_len = attn_mask.shape[-1]
  all_zeros_mask = jnp.all(attn_mask == 0, axis=-1)
  reversed_matrix = attn_mask[:, :, ::-1]
  first_one_from_right = jnp.argmax(reversed_matrix, axis=-1)
  last_one_index_original = cache_len - 1 - first_one_from_right
  final_indices = jnp.where(all_zeros_mask, 0, last_one_index_original)
  return final_indices.squeeze(axis=-1)


def create_sliding_window_mask(
    attn_mask: jnp.ndarray, sliding_window_size: int
) -> jnp.ndarray:
  """Helper function to create sliding window mask for local attention."""
  upper_index = find_last_one_index(attn_mask)
  window_start_pos = upper_index - sliding_window_size + 1
  abs_pos = jnp.arange(attn_mask.shape[-1])
  window_mask = abs_pos[None, :] >= window_start_pos[:, None]
  causal_mask = abs_pos[None, :] <= upper_index[:, None]
  final_mask = window_mask & causal_mask
  return final_mask[:, None, :]


def create_logical_sliding_window_mask(
    attn_mask: jnp.ndarray, sliding_window_size: int
) -> jnp.ndarray:
  """Sliding-window mask over logical token positions during chunked decode."""
  valid = attn_mask != 0
  valid_i = valid.astype(jnp.int32)
  logical_pos = jnp.cumsum(valid_i, axis=-1) - 1
  logical_last = jnp.sum(valid_i, axis=-1, keepdims=True) - 1
  window_mask = logical_pos > (logical_last - sliding_window_size)
  return window_mask & valid


def has_physical_gap(attn_mask: jnp.ndarray) -> jnp.ndarray:
  """Checks if the valid region in attn_mask has a physical padding gap."""
  valid = attn_mask != 0
  n = attn_mask.shape[-1]
  idx = jnp.arange(n)
  count = jnp.sum(valid.astype(jnp.int32), axis=-1, keepdims=True)
  first = jnp.min(jnp.where(valid, idx, n), axis=-1, keepdims=True)
  last = jnp.max(jnp.where(valid, idx, -1), axis=-1, keepdims=True)
  span = last - first + 1
  return count < span


def build_sliding_window_decode_mask(
    attn_mask: jaxtyping.Array,
    window_size: int,
    *,
    is_ring_buffer: bool,
    end_idx: jaxtyping.Array | None = None,
    cache_len: int | None = None,
) -> jaxtyping.Array:
  """Constructs the sliding-window attention mask during single-token decode."""
  has_gap = has_physical_gap(attn_mask)  # [B, 1, 1]
  if is_ring_buffer:
    assert end_idx is not None and cache_len is not None
    logical_end = jnp.sum((attn_mask != 0).astype(jnp.int32), axis=-1) - 1
    eff_end = jnp.where(has_gap[:, :, 0], logical_end, end_idx[:, None])
    eff_end = eff_end[:, :, None]  # [B, 1, 1]
    p = jnp.arange(cache_len)[None, None, :]
    logical_indices = eff_end - ((eff_end - p) % cache_len)
    valid_physical = logical_indices >= 0
    logical_indices = jnp.maximum(0, logical_indices)
    gathered = jnp.take_along_axis(attn_mask, logical_indices, axis=-1)
    contiguous_mask = gathered * valid_physical
    return jnp.where(
        has_gap,
        valid_physical.astype(contiguous_mask.dtype),
        contiguous_mask,
    )
  else:
    sliding_mask = create_sliding_window_mask(
        attn_mask, sliding_window_size=window_size
    )
    logical_sw = create_logical_sliding_window_mask(
        attn_mask, sliding_window_size=window_size
    )
    sliding_mask = jnp.where(has_gap, logical_sw, sliding_mask)
    return sliding_mask * attn_mask


def build_dense_chunked_prefill_mask(
    attn_mask: jaxtyping.Array,
    q_len: int,
    kv_len: int,
    prior_end_index: jaxtyping.Array | None,
    prefix_length: int,
    kv_valid_mask: jaxtyping.Array | None = None,
    is_ring_buffer: bool = False,
    sliding_window_size: int | None = None,
    kv_shared_cache: LayerCache | None = None,
    has_own_cache: bool = True,
) -> jaxtyping.Array:
  """Constructs the dense causal or sliding-window 3D attention mask (B, Q, KV)."""
  if has_own_cache:
    effective_prior_end = prior_end_index
  elif kv_shared_cache is not None:
    effective_prior_end = kv_shared_cache.get('prior_end_index', None)
    if effective_prior_end is None:
      raise ValueError(
          'shared layer missing origin prior_end_index; origin layers '
          'must propagate it via transient_kvs'
      )
  else:
    effective_prior_end = None

  prefix_kv_len = kv_len - q_len
  b = attn_mask.shape[0]

  if is_ring_buffer:
    assert sliding_window_size is not None
    if kv_valid_mask is not None:
      local_cache_mask = jnp.broadcast_to(
          kv_valid_mask[None, None, :],
          (b, q_len, prefix_kv_len),
      )
    else:
      local_cache_mask = jnp.ones((b, q_len, prefix_kv_len), dtype=jnp.bool_)

    suffix_causal = attn_mask[..., -q_len:]
    combined = jnp.concatenate([local_cache_mask, suffix_causal], axis=-1)

    position_offset = (
        effective_prior_end if effective_prior_end is not None else 0
    )
    valid_cache_len = jnp.minimum(position_offset, prefix_kv_len)
    row_pos = jnp.arange(q_len) + position_offset
    col_pos_cache = jnp.arange(prefix_kv_len) + (
        position_offset - valid_cache_len
    )
    col_pos_suffix = jnp.arange(q_len) + position_offset
    col_pos = jnp.concatenate([col_pos_cache, col_pos_suffix])

    sw_mask = (col_pos[None, :] > (row_pos[:, None] - sliding_window_size)) & (
        col_pos[None, :] <= row_pos[:, None]
    )
    return combined & sw_mask[None, :, :]
  else:
    if prefix_length > 0:
      prefix_mask = attn_mask[..., :prefix_length]
      suffix_mask = attn_mask[..., -q_len:]
      combined = jnp.concatenate([prefix_mask, suffix_mask], axis=-1)
      if effective_prior_end is not None:
        prefix_valid = jnp.arange(prefix_length) < effective_prior_end
        valid_mask = jnp.concatenate(
            [prefix_valid, jnp.ones(q_len, dtype=jnp.bool_)]
        )
        combined = combined & valid_mask[None, None, :]
      return combined
    else:
      return attn_mask[..., :kv_len]
