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

"""Gemma4 model attention."""

from functools import partial
from flax import nnx
import jax
from jax import numpy as jnp
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as splash
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask as mask_lib
from jax.experimental.shard_map import shard_map
from jax.interpreters import pxla
import jax.sharding as shd
from jax.sharding import PartitionSpec as P
import jaxtyping
import numpy as np
from tunix.models.gemma4.config import _maybe_bucket_prefix_length
from tunix.models.gemma4.config import AttentionType
from tunix.models.gemma4.config import K_MASK
from tunix.models.gemma4.config import LayerCache
from tunix.models.gemma4.config import ModelConfig
from tunix.models.gemma4.config import RematConfig
from tunix.models.gemma4.layers import apply_rope
from tunix.models.gemma4.layers import Einsum
from tunix.models.gemma4.layers import RMSNorm
from tunix.utils.sharding_utils import shard


def find_last_one_index(attn_mask: jnp.ndarray) -> jnp.ndarray:
  """Finds the index of the last (rightmost) '1' from attn_mask."""
  cache_len = attn_mask.shape[-1]

  # 1. check if the entire row is all zeros.
  all_zeros_mask = jnp.all(attn_mask == 0, axis=-1)

  # 2. reverse the rows in the attn_mask
  reversed_matrix = attn_mask[:, :, ::-1]

  # 3. find the fist 1 from the right.
  first_one_from_right = jnp.argmax(reversed_matrix, axis=-1)

  # 4. covert back to the original index
  last_one_index_original = cache_len - 1 - first_one_from_right

  # 5. return the final index, 0 for rows are all zeros.
  final_indices = jnp.where(
      all_zeros_mask,
      0,
      last_one_index_original,
  )

  return final_indices.squeeze(axis=-1)


def create_sliding_window_mask(
    attn_mask: jnp.ndarray,  # [B, seq_len, cache_len] seq_len=1 for decoding
    sliding_window_size: int,
) -> jnp.ndarray:
  """Helper function to create sliding window mask for local attention."""
  upper_index = find_last_one_index(attn_mask)

  # 1. compute the window start position
  window_start_pos = upper_index - sliding_window_size + 1

  # 2. create window mask
  abs_pos = jnp.arange(attn_mask.shape[-1])
  window_mask = abs_pos[None, :] >= window_start_pos[:, None]

  # 3. create causal mask
  causal_mask = abs_pos[None, :] <= upper_index[:, None]

  # 4. create final mask
  final_mask = window_mask & causal_mask
  return final_mask[:, None, :]  # [B, 1, cache_len]


def create_logical_sliding_window_mask(
    attn_mask: jnp.ndarray,  # [B, 1, cache_len] (decoding: seq_len == 1)
    sliding_window_size: int,
) -> jnp.ndarray:
  """Sliding-window mask over LOGICAL token positions (for chunked decode).

  Physical-slot windowing (``create_sliding_window_mask``) wrongly drops real
  prompt tokens when warm-prefix chunked prefill leaves a physical gap between
  an element's prompt KV and its generated tokens. This variant assigns each
  valid slot a contiguous logical position (cumsum of the validity mask) so the
  window follows the real tokens across the gap, and reduces exactly to the
  physical version when the valid region is contiguous.
  """
  valid = attn_mask != 0
  valid_i = valid.astype(jnp.int32)
  # Contiguous logical position of each valid slot; invalid slots are dropped by
  # the `& valid` below, so their (meaningless) values do not matter.
  logical_pos = jnp.cumsum(valid_i, axis=-1) - 1  # [B, 1, cache_len]
  logical_last = jnp.sum(valid_i, axis=-1, keepdims=True) - 1  # [B, 1, 1]
  window_mask = logical_pos > (logical_last - sliding_window_size)
  final_mask = window_mask & valid
  return final_mask  # [B, 1, cache_len]


def _has_physical_gap(attn_mask: jnp.ndarray) -> jnp.ndarray:
  """Per-row flag: is the valid region non-contiguous (a chunked-prefill gap)?

  Returns a ``[B, 1, 1]`` boolean. Standard left-pad decode has a contiguous
  valid region (count == span) so this is False everywhere, and the caller's
  ``jnp.where`` selects the original physical window unchanged -> byte-identical
  normal-decode behavior. Only genuinely gapped rows switch to the logical mask.
  """
  valid = attn_mask != 0  # [B, 1, cache_len]
  n = attn_mask.shape[-1]
  idx = jnp.arange(n)  # [cache_len]
  count = jnp.sum(valid.astype(jnp.int32), axis=-1, keepdims=True)  # [B,1,1]
  first = jnp.min(jnp.where(valid, idx, n), axis=-1, keepdims=True)  # [B,1,1]
  last = jnp.max(jnp.where(valid, idx, -1), axis=-1, keepdims=True)  # [B,1,1]
  span = last - first + 1
  return count < span  # [B, 1, 1]


def _merge_split_attention(
    out_prefix, lse_prefix, out_suffix, lse_suffix, out_dtype
):
  """LSE-weighted merge of two attention partitions. Pure JAX; CPU-testable.

  nan_to_num zeroes fully-masked (out=NaN / lse=-inf) partitions so they don't
  poison the residual stream. Boundary-straddling garbage rows are instead
  neutralized by weight underflow, which relies on splash's DEFAULT_MASK_VALUE
  staying large-negative.
  """
  # lse shape: (B, N, T), out shape: (B, N, T, H)
  max_lse = jnp.maximum(lse_prefix, lse_suffix)
  w_prefix = jnp.exp(lse_prefix - max_lse)
  w_suffix = jnp.exp(lse_suffix - max_lse)
  w_sum = w_prefix + w_suffix
  # Safe division (w_sum=0 only if both attentions are fully masked).
  w_sum_safe = jnp.where(w_sum > 0, w_sum, 1.0)
  encoded = (
      w_prefix[..., None] * jnp.nan_to_num(out_prefix.astype(jnp.float32))
      + w_suffix[..., None] * jnp.nan_to_num(out_suffix.astype(jnp.float32))
  ) / w_sum_safe[..., None]
  return encoded.astype(out_dtype)


class Attention(nnx.Module):
  """Attention module."""

  def __init__(
      self,
      config: ModelConfig,
      attn_type: AttentionType,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.rope_proportion = (
        config.global_rope_proportion
        if attn_type == AttentionType.GLOBAL
        else config.local_rope_proportion
    )
    self.attn_type = attn_type
    self.rope_base_frequency = (
        config.local_base_frequency
        if attn_type == AttentionType.LOCAL_SLIDING
        else config.global_base_frequency
    )
    self.rope_scale_factor = (
        config.local_scale_factor
        if attn_type == AttentionType.LOCAL_SLIDING
        else config.global_scale_factor
    )

    self.num_kv_heads = config.num_kv_heads
    self.head_dim = config.head_dim
    if attn_type == AttentionType.GLOBAL:
      if config.num_global_kv_heads is not None:
        self.num_kv_heads = config.num_global_kv_heads
      if config.global_key_size is not None:
        self.head_dim = config.global_key_size

    self.attn_vec_einsum = Einsum(
        einsum_str='BTNH,NHD->BTD',
        shape=(config.num_heads, self.head_dim, config.embed_dim),
        rngs=rngs,
        sharding=config.shd_config.o_weight_nhd,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    self.q_einsum = Einsum(
        einsum_str='BTD,NDH->BTNH',
        shape=(config.num_heads, config.embed_dim, self.head_dim),
        rngs=rngs,
        sharding=config.shd_config.q_weight_ndh,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )

    k_eq_v = (
        config.k_eq_v_global if attn_type == AttentionType.GLOBAL else False
    )
    if k_eq_v:
      self.k_einsum = Einsum(
          einsum_str='BSD,KDH->BSKH',
          shape=(
              self.num_kv_heads,
              config.embed_dim,
              self.head_dim,
          ),
          rngs=rngs,
          sharding=config.shd_config.q_weight_ndh,
          dtype=config.dtype,
          param_dtype=config.param_dtype,
      )
    else:
      if self.num_kv_heads == 1:
        kv_sharding = (None, None, 'fsdp', None)
      else:
        kv_sharding = config.shd_config.kv_weight_cndh

      self.kv_einsum = Einsum(
          einsum_str='BSD,CKDH->CBSKH',
          shape=(
              2,
              self.num_kv_heads,
              config.embed_dim,
              self.head_dim,
          ),
          rngs=rngs,
          sharding=kv_sharding,
          dtype=config.dtype,
          param_dtype=config.param_dtype,
      )
    self._query_norm = RMSNorm(
        self.head_dim,
        rngs=rngs,
        sharding=config.shd_config,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    self._key_norm = RMSNorm(
        self.head_dim,
        rngs=rngs,
        sharding=config.shd_config,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )

  def _compute_kv_projections(
      self,
      x: jaxtyping.Array,
      segment_pos: jaxtyping.Array,
      kv_shared_cache: LayerCache | None,
      *,
      is_chunked_prefill: bool,
      input_mask: jaxtyping.Array | None,
  ) -> tuple[jaxtyping.Array, jaxtyping.Array, jaxtyping.Array | None]:
    """Computes or retrieves key/value projections."""
    kv_valid_mask = None

    if kv_shared_cache is not None:
      key_proj = kv_shared_cache['k']
      value_proj = kv_shared_cache['v']
      kv_valid_mask = kv_shared_cache.get('valid_mask', None)
    else:
      if hasattr(self, 'k_einsum'):  # case where k_eq_v is True
        key_proj = self.k_einsum(x)
        value_proj = key_proj
      else:
        key_proj, value_proj = self.kv_einsum(x)

      key_proj = shard(key_proj, self.config.shd_config.act_btnh)
      value_proj = shard(value_proj, self.config.shd_config.act_btnh)

      # Apply norms to computed KV
      value_var = jnp.mean(jnp.square(value_proj), axis=-1, keepdims=True)
      value_proj = value_proj * jax.lax.rsqrt(value_var + 1e-06)
      key_proj = self._key_norm(key_proj)
      key_proj = apply_rope(
          key_proj,
          segment_pos,
          base_frequency=self.rope_base_frequency,
          scale_factor=self.rope_scale_factor,
          rope_proportion=self.rope_proportion,
      )

      # Zero PAD-position KVs to keep ring buffer clean.
      if is_chunked_prefill and input_mask is not None:
        mask_4d = input_mask[:, :, None, None].astype(key_proj.dtype)
        key_proj = key_proj * mask_4d
        value_proj = value_proj * mask_4d

    return key_proj, value_proj, kv_valid_mask

  def _update_cache_prefill(
      self,
      cache: LayerCache,
      key_proj: jaxtyping.Array,
      value_proj: jaxtyping.Array,
      seq_len: int,
      *,
      is_chunked_prefill: bool,
      prefix_length: int,
      input_mask: jaxtyping.Array | None,
  ) -> tuple[
      LayerCache,
      jaxtyping.Array,
      jaxtyping.Array,
      jaxtyping.Array,
      jaxtyping.Array | None,
      jaxtyping.Array | None,
      jaxtyping.Array | None,
  ]:
    """Updates KV cache and prepares KV for attention during prefill.

    Delegates to _write_cache_prefill and _read_prefix_kv; these have no data
    dependency, so XLA can overlap the cache write with attention.
    """
    prior_end_index = cache['end_index'][0]

    # Write fresh KV to cache (independent of prefix read).
    new_cache = self._write_cache_prefill(
        cache,
        key_proj,
        value_proj,
        seq_len,
        is_chunked_prefill=is_chunked_prefill,
        input_mask=input_mask,
    )

    # Read prefix KV from ORIGINAL cache (not new_cache: we want the pre-write
    # state). When use_split_attention is True, returns prefix/suffix separately
    # to avoid materializing the concatenated KV tensor in HBM.
    key_proj, value_proj, kv_valid_mask, prefix_k, prefix_v = (
        self._read_prefix_kv(
            cache,
            key_proj,
            value_proj,
            seq_len,
            is_chunked_prefill=is_chunked_prefill,
            prefix_length=prefix_length,
            prior_end_index=prior_end_index,
        )
    )

    return (
        new_cache,
        key_proj,
        value_proj,
        prior_end_index,
        kv_valid_mask,
        prefix_k,
        prefix_v,
    )

  def _write_cache_prefill(
      self,
      cache: LayerCache,
      key_proj: jaxtyping.Array,
      value_proj: jaxtyping.Array,
      seq_len: int,
      *,
      is_chunked_prefill: bool,
      input_mask: jaxtyping.Array | None,
  ) -> LayerCache:
    """Writes fresh KV projections to cache. Returns updated cache.

    Separated from prefix read so XLA can overlap cache writes with attention.
    """
    cache_len = cache['v'].shape[1]
    prior_end_index = cache['end_index'][0]

    if self.config.use_sliding_window_kv_cache:
      end_index = prior_end_index
      valid_len = min(seq_len, cache_len)
      latest_indices = (end_index + jnp.arange(valid_len)) % cache_len
      new_v = value_proj[:, -valid_len:, ...]
      new_k = key_proj[:, -valid_len:, ...]
      if is_chunked_prefill and input_mask is not None:
        # Preserve valid cache where input is padding.
        write_mask = input_mask[:, -valid_len:]
        write_mask_4d = write_mask[:, :, None, None]
        old_k = cache['k'][:, latest_indices, ...]
        old_v = cache['v'][:, latest_indices, ...]
        new_k = jnp.where(write_mask_4d, new_k, old_k)
        new_v = jnp.where(write_mask_4d, new_v, old_v)
      cache_v = cache['v'].at[:, latest_indices, ...].set(new_v)
      cache_k = cache['k'].at[:, latest_indices, ...].set(new_k)
    else:
      end_index = prior_end_index
      slice_indices = (0, end_index % cache_len, 0, 0)
      cache_v = jax.lax.dynamic_update_slice(
          cache['v'], value_proj, slice_indices
      )
      cache_k = jax.lax.dynamic_update_slice(
          cache['k'], key_proj, slice_indices
      )

    # Non-uniform (ragged) input masks are safe: PAD-position KVs are zeroed in
    # _compute_kv_projections and excluded by the attention mask.

    return {
        'v': cache_v,
        'k': cache_k,
        'end_index': (
            cache['end_index']
            + (
                # PAD-safe: advance by the batch-max real-token count
                # (elements may be ragged under PAD); PAD KVs are zeroed and
                # attention-masked, so reserving up to the max is safe.
                jnp.max(jnp.sum(input_mask, axis=-1)).astype(jnp.int32)
                if is_chunked_prefill and input_mask is not None
                else seq_len
            )
        ),
    }

  def _read_prefix_kv(
      self,
      cache: LayerCache,
      key_proj: jaxtyping.Array,
      value_proj: jaxtyping.Array,
      seq_len: int,
      *,
      is_chunked_prefill: bool,
      prefix_length: int,
      prior_end_index: jaxtyping.Array,
  ) -> tuple[
      jaxtyping.Array,
      jaxtyping.Array,
      jaxtyping.Array | None,
      jaxtyping.Array | None,
      jaxtyping.Array | None,
  ]:
    """Reads prefix KV from cache.

    When use_split_attention is True, returns prefix and suffix KV separately
    (no concatenation). Otherwise, concatenates prefix with fresh KV.

    Separated from cache write so the read is independent of the write and can
    be overlapped with attention by XLA.
    """
    kv_valid_mask = None
    cache_len = cache['v'].shape[1]

    if not (is_chunked_prefill and prefix_length > 0):
      return key_proj, value_proj, kv_valid_mask, None, None

    # Clamp prefix_length to cache_len so mask and KV slice stay consistent:
    # JAX slicing silently clamps, but the mask would use the unclamped value,
    # creating a shape mismatch.
    prefix_length = min(prefix_length, cache_len)

    if (
        self.config.use_sliding_window_kv_cache
        and self.attn_type == AttentionType.LOCAL_SLIDING
    ):
      # LOCAL: Unroll ring buffer to get chronologically-ordered prefix KV
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
      # GLOBAL: Static slice for prefix KV. Use bucketed prefix_length for
      # compilation stability; mask out padding positions dynamically.
      cached_k = cache['k'][:, :prefix_length, ...]
      cached_v = cache['v'][:, :prefix_length, ...]
      # Zero out positions beyond the actual valid prefix. The bucketed
      # prefix_length may exceed prior_end_index; those positions contain
      # uninitialized cache data that must not influence attention.
      valid_prefix = jnp.arange(prefix_length) < prior_end_index
      cached_k = jnp.where(valid_prefix[None, :, None, None], cached_k, 0)
      cached_v = jnp.where(valid_prefix[None, :, None, None], cached_v, 0)

    if self.config.use_split_attention:
      # Return prefix and suffix KV separately — no concat. Avoids
      # materializing the concatenated tensor in HBM.
      return key_proj, value_proj, kv_valid_mask, cached_k, cached_v

    # Default: Concatenate cached prefix KV with fresh suffix KV.
    key_proj = jnp.concatenate([cached_k, key_proj], axis=1)
    value_proj = jnp.concatenate([cached_v, value_proj], axis=1)

    return key_proj, value_proj, kv_valid_mask, None, None

  def _build_chunked_prefill_mask(
      self,
      attn_mask: jaxtyping.Array,
      q_len: int,
      kv_len: int,
      prior_end_index: jaxtyping.Array | None,
      kv_shared_cache: LayerCache | None,
      prefix_length: int,
      kv_valid_mask: jaxtyping.Array | None,
      has_own_cache: bool,
  ) -> jaxtyping.Array:
    """Constructs the attention mask for chunked prefill."""
    prefix_kv_len = kv_len - q_len
    if (
        self.config.use_sliding_window_kv_cache
        and self.attn_type == AttentionType.LOCAL_SLIDING
    ):
      return self._build_local_chunked_prefill_mask(
          attn_mask,
          q_len,
          prefix_kv_len,
          prior_end_index,
          kv_shared_cache,
          prefix_length,
          kv_valid_mask,
          has_own_cache,
      )
    return self._build_global_chunked_prefill_mask(
        attn_mask,
        q_len,
        kv_len,
        prior_end_index,
        kv_shared_cache,
        prefix_length,
        kv_valid_mask,
        has_own_cache,
    )

  def _build_local_chunked_prefill_mask(
      self,
      attn_mask: jaxtyping.Array,
      q_len: int,
      prefix_kv_len: int,
      prior_end_index: jaxtyping.Array | None,
      kv_shared_cache: LayerCache | None,
      prefix_length: int,
      kv_valid_mask: jaxtyping.Array | None,
      has_own_cache: bool,
  ) -> jaxtyping.Array:
    """Chunked-prefill attention mask for LOCAL_SLIDING layers."""
    # LOCAL: Build mask over [ring_buf | suffix]
    if kv_valid_mask is not None:
      local_cache_mask = jnp.broadcast_to(
          kv_valid_mask[None, None, :],
          (attn_mask.shape[0], q_len, prefix_kv_len),
      )
    else:
      local_cache_mask = jnp.ones(
          (attn_mask.shape[0], q_len, prefix_kv_len), dtype=jnp.bool_
      )
    suffix_causal = attn_mask[..., -q_len:]
    attn_mask = jnp.concatenate([local_cache_mask, suffix_causal], axis=-1)
    # Use origin layer's prior_end_index for correct window boundaries.
    if has_own_cache:
      assert prior_end_index is not None
      position_offset = prior_end_index
      valid_cache_len = jnp.minimum(position_offset, prefix_kv_len)
    elif kv_shared_cache is not None:
      # Use the origin layer's prior_end_index if available (propagated
      # via transient_kvs). Falls back to prefix_length if not present.
      origin_end_index = kv_shared_cache.get('prior_end_index', None)
      if origin_end_index is not None:
        position_offset = origin_end_index
        valid_cache_len = jnp.minimum(origin_end_index, prefix_kv_len)
      else:
        raise ValueError(
            'shared LOCAL layer missing origin prior_end_index; origin layers '
            'must propagate it via transient_kvs'
        )
    else:
      position_offset = 0
      valid_cache_len = prefix_kv_len
    row_pos = jnp.arange(q_len) + position_offset
    col_pos_cache = jnp.arange(prefix_kv_len) + (
        position_offset - valid_cache_len
    )
    col_pos_suffix = jnp.arange(q_len) + position_offset
    col_pos = jnp.concatenate([col_pos_cache, col_pos_suffix])
    window_size = self.config.sliding_window_size
    assert window_size is not None
    sw_mask = (
        col_pos[None, :] > (row_pos[:, None] - window_size)
    ) & (col_pos[None, :] <= row_pos[:, None])
    attn_mask = attn_mask & sw_mask[None, :, :]
    return attn_mask

  def _build_global_chunked_prefill_mask(
      self,
      attn_mask: jaxtyping.Array,
      q_len: int,
      kv_len: int,
      prior_end_index: jaxtyping.Array | None,
      kv_shared_cache: LayerCache | None,
      prefix_length: int,
      kv_valid_mask: jaxtyping.Array | None,
      has_own_cache: bool,
  ) -> jaxtyping.Array:
    """Chunked-prefill attention mask for GLOBAL layers."""
    # GLOBAL: Compose mask from prefix validity + suffix causal.
    if prefix_length > 0:
      prefix_mask = attn_mask[..., :prefix_length]
      suffix_mask = attn_mask[..., -q_len:]
      attn_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=-1)
      # Mask out uninitialized prefix cache positions.
      if has_own_cache:
        assert prior_end_index is not None
        prefix_valid = jnp.arange(prefix_length) < prior_end_index
        valid_mask = jnp.concatenate(
            [prefix_valid, jnp.ones(q_len, dtype=jnp.bool_)]
        )
        attn_mask = attn_mask & valid_mask[None, None, :]
      elif kv_shared_cache is not None:
        # Shared GLOBAL layers must also mask uninitialized prefix positions.
        # Use the origin layer's prior_end_index propagated through
        # kv_shared_cache.
        origin_end_index = kv_shared_cache.get('prior_end_index', None)
        if origin_end_index is None:
          raise ValueError(
              'shared GLOBAL layer missing origin prior_end_index; origin '
              'layers must propagate it via transient_kvs'
          )
        prefix_valid = jnp.arange(prefix_length) < origin_end_index
        valid_mask = jnp.concatenate(
            [prefix_valid, jnp.ones(q_len, dtype=jnp.bool_)]
        )
        attn_mask = attn_mask & valid_mask[None, None, :]
    else:
      attn_mask = attn_mask[..., :kv_len]
    return attn_mask

  def _build_flash_mask(
      self,
      q_len: int,
      kv_len: int,
      offset: int,
      is_rectangular: bool,
      kv_valid_mask: jaxtyping.Array | None,
  ):
    """Builds the single-head splash attention mask for one flash call."""
    if self.attn_type == AttentionType.LOCAL_SLIDING:
      window_size = self.config.sliding_window_size
      assert window_size is not None
      if is_rectangular and kv_valid_mask is not None:
        # Compose ring buffer validity into flash mask. LocalMask doesn't
        # support kv_valid_mask, so build a NumpyMask that combines the
        # sliding window pattern with the validity mask.
        q_ids = np.arange(q_len) + offset
        kv_ids = np.arange(kv_len)
        sw = window_size
        local_mask = (kv_ids[None, :] > (q_ids[:, None] - sw)) & (
            kv_ids[None, :] <= q_ids[:, None]
        )
        # kv_valid_mask is a JAX array; convert to numpy for NumpyMask.
        # At trace time during JIT, we use the mask shape (static) to build
        # the NumpyMask. The dynamic validity is already handled by zeroing
        # invalid KV in _read_prefix_kv.
        valid_np = np.ones(kv_len, dtype=np.bool_)
        mask_2d = local_mask & valid_np[None, :]
        mask = mask_lib.NumpyMask(mask_2d.astype(np.bool_))
      else:
        mask = mask_lib.LocalMask(
            (q_len, kv_len),
            window_size=(window_size - 1, 0),
            offset=offset,
        )
    else:
      if is_rectangular:
        # Use NumpyMask for GLOBAL rectangular attention. CausalMask creates
        # a new closure per offset → different id() → Splash recompilation;
        # NumpyMask is hashed by content. The validity mask for bucketed
        # padding is already applied to the KV tensors (zeroed in
        # _read_prefix_kv), so the NumpyMask only needs the causal pattern.
        q_ids = np.arange(q_len) + offset
        kv_ids = np.arange(kv_len)
        causal_mask_2d = q_ids[:, None] >= kv_ids[None, :]
        mask = mask_lib.NumpyMask(causal_mask_2d.astype(np.bool_))
      else:
        mask = mask_lib.CausalMask((q_len, kv_len), offset=offset)
    return mask

  def _make_block_sizes(self, is_rectangular: bool) -> splash.BlockSizes:
    """Selects splash block sizes for this attention call."""
    # Choose block sizes. block_kv must divide kv_len.
    # For LOCAL_SLIDING rectangular shapes, block_kv must divide both
    # sliding_window_size and chunk_len. Use the smaller of the two.
    block_q = self.config.flash_attention_block_size
    if is_rectangular and self.attn_type == AttentionType.LOCAL_SLIDING:
      window_size = self.config.sliding_window_size
      assert window_size is not None
      block_kv = min(
          self.config.flash_attention_block_size,
          window_size,
      )
    else:
      block_kv = self.config.flash_attention_block_size

    # Inner-loop tile size for the attention matmul; must divide block_kv.
    block_kv_compute = min(
        self.config.flash_attention_compute_block_size, block_kv
    )

    # Bwd holds Q + dO + attn_weights + grad accumulators; smaller Q block to
    # fit VMEM.
    block_bwd = min(self.config.flash_attention_bwd_block_size, block_q)
    use_fused = self.config.flash_attention_use_fused_bwd
    return splash.BlockSizes(
        block_q=block_q,
        block_kv=block_kv,
        block_kv_compute=block_kv_compute,
        block_q_dkv=block_bwd,
        block_kv_dkv=block_kv,
        block_kv_dkv_compute=block_kv_compute,
        # Fused bwd kernel computes dQ+dKV in one pass; these are ignored.
        block_q_dq=None if use_fused else block_bwd,
        block_kv_dq=None if use_fused else block_kv,
        use_fused_bwd_kernel=use_fused,
    )

  def _make_sharding_specs(self, b: int, kh: int, mesh: shd.Mesh):
    """Computes mesh/shard-axis specs for splash attention."""
    shd_b, shd_t, shd_n, shd_h = self.config.shd_config.act_btnh
    if (
        mesh is not None
        and shd_b is not None
        and shd_b in mesh.shape
        and b % mesh.shape[shd_b] != 0
    ):
      shd_b = None
    head_shards = (
        mesh.shape[shd_n] if shd_n is not None and shd_n in mesh.shape else 1
    )
    q_seq_shards = (
        mesh.shape[shd_t] if shd_t is not None and shd_t in mesh.shape else 1
    )
    shd_spec = P(shd_b, shd_n, shd_t, shd_h)
    shd_n_kv = (
        shd_n
        if mesh is not None
        and shd_n is not None
        and shd_n in mesh.shape
        and kh % mesh.shape[shd_n] == 0
        else None
    )
    unsharded_seq_kv = P(shd_b, shd_n_kv, None, shd_h)
    return (
        shd_b,
        shd_n,
        shd_t,
        shd_h,
        head_shards,
        q_seq_shards,
        shd_n_kv,
        shd_spec,
        unsharded_seq_kv,
    )

  def _make_splash_kernel(
      self,
      multi_head_mask,
      block_sizes: splash.BlockSizes,
      head_shards: int,
      q_seq_shards: int,
      mesh: shd.Mesh,
      shd_n: str | None,
      shd_t: str | None,
      save_residuals: bool = False,
  ):
    """Builds a splash MHA kernel and its manual sharding spec."""
    kernel = splash.make_splash_mha(
        multi_head_mask,
        block_sizes=block_sizes,
        head_shards=head_shards,
        q_seq_shards=q_seq_shards,
        save_residuals=save_residuals,
    )
    kernel_spec = kernel.manual_sharding_spec(
        shd.NamedSharding(mesh, P(shd_n, shd_t))
    )
    return kernel, kernel_spec

  def block(
      self,
      x: jaxtyping.Array,
      segment_pos: jaxtyping.Array,
      cache: LayerCache | None,
      attn_mask: jaxtyping.Array,
      kv_shared_cache: LayerCache | None = None,
      segment_ids: jaxtyping.Array | None = None,
      is_chunked_prefill: bool = False,
      prefix_length: int = 0,
      input_mask: jaxtyping.Array | None = None,
      force_eager: bool = False,
  ) -> tuple[
      LayerCache | None,
      jaxtyping.Array,
      tuple[
          jaxtyping.Array,
          jaxtyping.Array,
          jaxtyping.Array | None,
          jaxtyping.Array | None,
      ],
  ]:
    x = x.astype(self.config.dtype)
    seq_len = x.shape[1]
    query_proj = self.q_einsum(x)
    query_proj = shard(query_proj, self.config.shd_config.act_btnh)
    query_proj = self._query_norm(query_proj)
    query_proj = apply_rope(
        query_proj,
        segment_pos,
        base_frequency=self.rope_base_frequency,
        scale_factor=self.rope_scale_factor,
        rope_proportion=self.rope_proportion,
    )

    key_proj, value_proj, kv_valid_mask = self._compute_kv_projections(
        x,
        segment_pos,
        kv_shared_cache,
        is_chunked_prefill=is_chunked_prefill,
        input_mask=input_mask,
    )

    prior_end_index = None
    if cache is not None:
      assert kv_shared_cache is None
      cache_len = cache['v'].shape[1]
      if seq_len > 1:  # prefill
        (
            new_cache,
            key_proj,
            value_proj,
            prior_end_index,
            kv_valid_mask,
            split_prefix_k,
            split_prefix_v,
        ) = self._update_cache_prefill(
            cache,
            key_proj,
            value_proj,
            seq_len,
            is_chunked_prefill=is_chunked_prefill,
            prefix_length=prefix_length,
            input_mask=input_mask,
        )
      else:  # decode
        end_index = cache['end_index'][0]
        slice_indices = (0, end_index % cache_len, 0, 0)
        value_proj = jax.lax.dynamic_update_slice(
            cache['v'], value_proj, slice_indices
        )
        key_proj = jax.lax.dynamic_update_slice(
            cache['k'], key_proj, slice_indices
        )
        new_cache = {
            'v': value_proj,
            'k': key_proj,
            'end_index': cache['end_index'] + seq_len,
        }
        split_prefix_k = None
        split_prefix_v = None
    else:
      new_cache = {
          'v': value_proj,
          'k': key_proj,
      }
      split_prefix_k = None
      split_prefix_v = None

    b, _, qh, _ = query_proj.shape
    _, _, kh, _ = key_proj.shape

    # Determine if we can use flash attention for this call.
    q_len = query_proj.shape[1]
    kv_len = key_proj.shape[1]
    # When split attention bypasses the concat, key_proj is suffix-only.
    # Compute total kv_len from prefix + suffix for mask/guard calculations.
    if split_prefix_k is not None:
      kv_len = split_prefix_k.shape[1] + key_proj.shape[1]
    is_rectangular = kv_len > q_len
    use_flash = (
        self.config.use_flash_attention
        and seq_len > 1
        # Flash attention requires kv_len to be divisible by block_kv.
        # During early decode steps (e.g., kv_len=2 in GRPO rollout),
        # fall back to standard attention which has no such constraint.
        and kv_len >= self.config.flash_attention_block_size
        # segment_ids are incompatible with rectangular flash because
        # KV segment_ids would need to cover cached prefix positions.
        and not (is_rectangular and segment_ids is not None)
        # GLOBAL layers are flash-eligible during rectangular chunked prefill.
        # Bucketing stabilizes kv_len across chunks, preventing the
        # recompilation storm that originally motivated this exclusion.
        # Partial-cache LOCAL_SLIDING chunked prefill: flash's static relative
        # offset (kv_len - q_len) anchors the window to the padded ring length,
        # not the valid token count, so it slides past the real cached tokens
        # when the window is only partially filled. Fall back to eager here.
        # See cl/933189977. Keyed on RAW (pre-bucket) prefix_length in
        # __call__ because bucketing rounds up toward the window and hides the
        # ring gap.
        and not force_eager
    )

    if use_flash:
      query_proj = query_proj.transpose(0, 2, 1, 3)
      key_proj = key_proj.transpose(0, 2, 1, 3)
      value_proj = value_proj.transpose(0, 2, 1, 3)

      mesh = pxla.thread_resources.env.physical_mesh

      # Offset: shifts Q positions so q[0] aligns with kv[prefix_len].
      offset = kv_len - q_len if is_rectangular else 0

      mask = self._build_flash_mask(
          q_len, kv_len, offset, is_rectangular, kv_valid_mask
      )

      multi_head_mask = mask_lib.MultiHeadMask([mask for _ in range(qh)])

      block_sizes = self._make_block_sizes(is_rectangular)

      (
          shd_b,
          shd_n,
          shd_t,
          shd_h,
          head_shards,
          q_seq_shards,
          shd_n_kv,
          shd_spec,
          unsharded_seq_kv,
      ) = self._make_sharding_specs(b, kh, mesh)

      splash_attn_kernel, kernel_spec = self._make_splash_kernel(
          multi_head_mask,
          block_sizes,
          head_shards,
          q_seq_shards,
          mesh,
          shd_n,
          shd_t,
      )

      # Split attention with LSE merge for chunked prefill: attend to prefix
      # and suffix KV separately and merge via log-sum-exp, eliminating the
      # jnp.concatenate data movement cost. Guards: block sizes must divide the
      # prefix and suffix lengths; segment_ids are not supported (already
      # excluded by use_flash).
      prefix_kv_len = kv_len - q_len
      can_split = (
          self.config.use_split_attention
          and is_rectangular
          and segment_ids is None
          and prefix_kv_len % block_sizes.block_kv == 0
          and q_len % block_sizes.block_q == 0
      )
      if can_split:
        encoded, key_proj, value_proj = self._flash_attention_split(
            query_proj,
            key_proj,
            value_proj,
            split_prefix_k,
            split_prefix_v,
            q_len,
            prefix_kv_len,
            qh,
            block_sizes,
            head_shards,
            q_seq_shards,
            mesh,
            shd_b,
            shd_n,
            shd_t,
            shd_h,
            shd_n_kv,
            shd_spec,
        )

      else:
        encoded, key_proj, value_proj = self._flash_attention_single(
            query_proj,
            key_proj,
            value_proj,
            split_prefix_k,
            split_prefix_v,
            segment_ids,
            splash_attn_kernel,
            kernel_spec,
            shd_spec,
            unsharded_seq_kv,
            mesh,
            shd_b,
            shd_t,
        )

    else:
      encoded = self._eager_attention(
          query_proj,
          key_proj,
          value_proj,
          attn_mask,
          segment_pos,
          cache,
          kv_shared_cache,
          kv_valid_mask,
          prior_end_index,
          prefix_length,
          seq_len,
          is_chunked_prefill,
      )

    attn_output = self.attn_vec_einsum(encoded)
    attn_output = shard(attn_output, self.config.shd_config.act_btd)
    return (
        new_cache,
        attn_output,
        (key_proj, value_proj, kv_valid_mask, prior_end_index),
    )

  def _flash_attention_split(
      self,
      query_proj: jaxtyping.Array,
      key_proj: jaxtyping.Array,
      value_proj: jaxtyping.Array,
      split_prefix_k: jaxtyping.Array | None,
      split_prefix_v: jaxtyping.Array | None,
      q_len: int,
      prefix_kv_len: int,
      qh: int,
      block_sizes: splash.BlockSizes,
      head_shards: int,
      q_seq_shards: int,
      mesh: shd.Mesh,
      shd_b: str | None,
      shd_n: str | None,
      shd_t: str | None,
      shd_h: str | None,
      shd_n_kv: str | None,
      shd_spec: P,
  ) -> tuple[jaxtyping.Array, jaxtyping.Array, jaxtyping.Array]:
    """Split-KV flash attention with LSE merge for chunked prefill."""
    # Use separately-returned prefix KV from _read_prefix_kv. No
    # concatenation occurred — prefix comes from cache (B, S, N, H),
    # suffix is the fresh projection already transposed (B, N, S, H).
    assert split_prefix_k is not None
    assert split_prefix_v is not None
    prefix_k = split_prefix_k.transpose(0, 2, 1, 3)
    prefix_v = split_prefix_v.transpose(0, 2, 1, 3)
    suffix_k = key_proj  # already transposed above
    suffix_v = value_proj

    # Build masks for prefix and suffix
    if self.attn_type == AttentionType.LOCAL_SLIDING:
      window_size = self.config.sliding_window_size
      assert window_size is not None
      sw = window_size
      q_ids = np.arange(q_len) + prefix_kv_len
      # Prefix mask: which prefix positions are in sliding window
      prefix_kv_ids = np.arange(prefix_kv_len)
      prefix_mask_2d = (prefix_kv_ids[None, :] > (q_ids[:, None] - sw)) & (
          prefix_kv_ids[None, :] <= q_ids[:, None]
      )
      prefix_mask = mask_lib.NumpyMask(prefix_mask_2d.astype(np.bool_))
      # Suffix mask: causal within chunk
      suffix_mask = mask_lib.CausalMask((q_len, q_len))
    else:
      # GLOBAL: prefix is fully attended, suffix is causal
      prefix_mask = mask_lib.FullMask((q_len, prefix_kv_len))
      suffix_mask = mask_lib.CausalMask((q_len, q_len))

    prefix_multi_mask = mask_lib.MultiHeadMask([prefix_mask for _ in range(qh)])
    suffix_multi_mask = mask_lib.MultiHeadMask([suffix_mask for _ in range(qh)])

    # Build kernels with save_residuals=True for LSE merge.
    prefix_attn_kernel, prefix_kernel_spec = self._make_splash_kernel(
        prefix_multi_mask,
        block_sizes,
        head_shards,
        q_seq_shards,
        mesh,
        shd_n,
        shd_t,
        save_residuals=True,
    )
    suffix_attn_kernel, suffix_kernel_spec = self._make_splash_kernel(
        suffix_multi_mask,
        block_sizes,
        head_shards,
        q_seq_shards,
        mesh,
        shd_n,
        shd_t,
        save_residuals=True,
    )

    # Sharding specs for split KV and logsumexp
    prefix_kv_spec = P(shd_b, shd_n_kv, None, shd_h)
    suffix_kv_spec = P(shd_b, shd_n_kv, None, shd_h)
    lse_spec = P(shd_b, shd_n, shd_t)

    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(
            prefix_kernel_spec,
            shd_spec,
            prefix_kv_spec,
            prefix_kv_spec,
        ),
        out_specs=(shd_spec, (lse_spec,)),
        check_rep=False,
    )
    def sharded_prefix_attn(kernel, q_block, k_block, v_block):
      return jax.vmap(kernel)(q_block, k_block, v_block)

    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(
            suffix_kernel_spec,
            shd_spec,
            suffix_kv_spec,
            suffix_kv_spec,
        ),
        out_specs=(shd_spec, (lse_spec,)),
        check_rep=False,
    )
    def sharded_suffix_attn(kernel, q_block, k_block, v_block):
      return jax.vmap(kernel)(q_block, k_block, v_block)

    out_prefix, (lse_prefix,) = sharded_prefix_attn(
        prefix_attn_kernel,
        query_proj,
        prefix_k,
        prefix_v,
    )
    out_suffix, (lse_suffix,) = sharded_suffix_attn(
        suffix_attn_kernel,
        query_proj,
        suffix_k,
        suffix_v,
    )

    # Max-stabilized LSE merge (numerically stable softmax reweighting).
    # NaN-safe: fully-masked prefix partitions arrive as out=NaN/lse=-inf and
    # are gated to a zero contribution. See _merge_split_attention.
    encoded = _merge_split_attention(
        out_prefix, lse_prefix, out_suffix, lse_suffix, key_proj.dtype
    )
    # (B, N, T, H) -> (B, T, N, H)
    encoded = encoded.transpose(0, 2, 1, 3)
    # Transpose KV back for KV-sharing layers
    key_proj = key_proj.transpose(0, 2, 1, 3)
    value_proj = value_proj.transpose(0, 2, 1, 3)
    return encoded, key_proj, value_proj

  def _flash_attention_single(
      self,
      query_proj: jaxtyping.Array,
      key_proj: jaxtyping.Array,
      value_proj: jaxtyping.Array,
      split_prefix_k: jaxtyping.Array | None,
      split_prefix_v: jaxtyping.Array | None,
      segment_ids: jaxtyping.Array | None,
      splash_attn_kernel: splash.SplashAttentionKernel,
      kernel_spec: splash.SplashAttentionKernel,
      shd_spec: P,
      unsharded_seq_kv: P,
      mesh: shd.Mesh,
      shd_b: str | None,
      shd_t: str | None,
  ) -> tuple[jaxtyping.Array, jaxtyping.Array, jaxtyping.Array]:
    """Single-kernel flash attention over concatenated (or plain) KV."""
    # Fallback: if split attention was prepared but guards failed,
    # reconstruct the concatenated KV for single-kernel path.
    if split_prefix_k is not None:
      assert split_prefix_v is not None
      key_proj = jnp.concatenate(
          [split_prefix_k.transpose(0, 2, 1, 3), key_proj], axis=2
      )
      value_proj = jnp.concatenate(
          [split_prefix_v.transpose(0, 2, 1, 3), value_proj], axis=2
      )
    # Original single-kernel attention path.
    if segment_ids is not None:
      seg_spec = P(shd_b, shd_t)
      unsharded_seg_spec = P(shd_b, None)

      @partial(
          shard_map,
          mesh=mesh,
          in_specs=(
              kernel_spec,
              shd_spec,
              unsharded_seq_kv,
              unsharded_seq_kv,
              seg_spec,
              unsharded_seg_spec,
          ),
          out_specs=shd_spec,
          check_rep=False,
      )
      def sharded_splash_attn(
          kernel, q_block, k_block, v_block, q_seg_block, kv_seg_block
      ):
        seg_ids = splash.SegmentIds(q=q_seg_block, kv=kv_seg_block)
        return jax.vmap(kernel)(q_block, k_block, v_block, segment_ids=seg_ids)

      qkv: jaxtyping.Array = sharded_splash_attn(
          splash_attn_kernel,
          query_proj,
          key_proj,
          value_proj,
          segment_ids,
          segment_ids,
      )
    else:

      @partial(
          shard_map,
          mesh=mesh,
          in_specs=(
              kernel_spec,
              shd_spec,
              unsharded_seq_kv,
              unsharded_seq_kv,
          ),
          out_specs=shd_spec,
          check_rep=False,
      )
      def sharded_splash_attn(kernel, q_block, k_block, v_block):
        return jax.vmap(kernel)(q_block, k_block, v_block)

      qkv: jaxtyping.Array = sharded_splash_attn(
          splash_attn_kernel,
          query_proj,
          key_proj,
          value_proj,
      )
    encoded = qkv.transpose(0, 2, 1, 3)
    # Transpose KV back to (B, S, K, H); consumed by KV-sharing layers via
    # layers_kvs.
    key_proj = key_proj.transpose(0, 2, 1, 3)
    value_proj = value_proj.transpose(0, 2, 1, 3)
    return encoded, key_proj, value_proj

  def _eager_attention(
      self,
      query_proj: jaxtyping.Array,
      key_proj: jaxtyping.Array,
      value_proj: jaxtyping.Array,
      attn_mask: jaxtyping.Array,
      segment_pos: jaxtyping.Array,
      cache: LayerCache | None,
      kv_shared_cache: LayerCache | None,
      kv_valid_mask: jaxtyping.Array | None,
      prior_end_index: jaxtyping.Array | None,
      prefix_length: int,
      seq_len: int,
      is_chunked_prefill: bool,
  ) -> jaxtyping.Array:
    """Eager einsum attention (non-flash path)."""
    if self.use_gqa:
      b, t, kg, h = query_proj.shape
      n_groups = kg // self.num_kv_heads
      query_reshaped = query_proj.reshape(
          (b, t, self.num_kv_heads, n_groups, h)
      )
      logits = jnp.einsum('BTKGH,BSKH->BTKGS', query_reshaped, key_proj)
      b, t, k, g, s = logits.shape
      logits = logits.reshape((b, t, k * g, s))
    else:
      logits = jnp.einsum('BTNH,BSNH->BTNS', query_proj, key_proj)

    kv_len = key_proj.shape[1]
    q_len = query_proj.shape[1]

    if seq_len > 1:
      if is_chunked_prefill and kv_len > q_len:
        attn_mask = self._build_chunked_prefill_mask(
            attn_mask,
            q_len,
            kv_len,
            prior_end_index,
            kv_shared_cache,
            prefix_length,
            kv_valid_mask,
            has_own_cache=(cache is not None),
        )
      else:
        attn_mask = attn_mask[..., :seq_len]

    _skip_sliding_mask = (
        is_chunked_prefill
        and kv_len > q_len
        and self.config.use_sliding_window_kv_cache
        and self.attn_type == AttentionType.LOCAL_SLIDING
    )
    if self.attn_type == AttentionType.LOCAL_SLIDING and not _skip_sliding_mask:
      window_size = self.config.sliding_window_size
      assert window_size is not None
      if segment_pos.shape[1] == 1 and self.config.use_sliding_window_kv_cache:
        # for decoding with sliding window cache
        active_cache = cache if cache is not None else kv_shared_cache
        if active_cache is None:
          raise ValueError(
              'Cache or shared cache is required for local sliding attention'
              ' in decoding.'
          )
        cache_len = key_proj.shape[1]
        end_idx = active_cache['end_index']
        if cache is None and kv_shared_cache is not None:
          # In case of shared KV cache, the origin layer already updated the
          # end index. We need to subtract 1 to get the correct end index of
          # the previous token.
          end_idx = end_idx - 1
        end_idx = end_idx[:, None, None]
        p = jnp.arange(cache_len)[None, None, :]

        # map physical index to logical index
        logical_indices = end_idx - ((end_idx - p) % cache_len)

        # identify uninitialized slots (before the cache fills up)
        valid_physical = logical_indices >= 0
        logical_indices = jnp.maximum(0, logical_indices)

        attn_mask = jnp.take_along_axis(attn_mask, logical_indices, axis=-1)
        attn_mask = attn_mask * valid_physical
      elif segment_pos.shape[1] == 1:
        # for decoding without sliding window cache
        sliding_mask = create_sliding_window_mask(
            attn_mask,
            sliding_window_size=window_size,
        )
        # Warm-prefix chunked prefill can leave a physical gap between an
        # element's real prompt KV and its generated tokens. The physical-slot
        # window above then drops real prompt tokens once the gap >= window.
        # Recompute the window over LOGICAL positions and select it ONLY for
        # rows whose valid region is non-contiguous, so standard left-pad
        # decode (contiguous -> _has_physical_gap False) is byte-identical.
        logical_sliding_mask = create_logical_sliding_window_mask(
            attn_mask,
            sliding_window_size=window_size,
        )
        has_gap = _has_physical_gap(attn_mask)  # [B, 1, 1]
        sliding_mask = jnp.where(has_gap, logical_sliding_mask, sliding_mask)
        attn_mask = sliding_mask * attn_mask
      else:  # for prefill
        all_ones = jnp.ones_like(attn_mask)
        sliding_mask = jnp.triu(
            all_ones, -1 * window_size + 1
        ) * jnp.tril(
            all_ones, window_size - 1
        )
        attn_mask = sliding_mask * attn_mask

    attn = jnp.where((jnp.expand_dims(attn_mask, -2)), logits, K_MASK)
    attn = jax.nn.softmax(attn.astype(jnp.float32), axis=-1).astype(
        key_proj.dtype
    )

    if self.use_gqa:
      b, t, kg, s = attn.shape
      n_groups = kg // self.num_kv_heads
      probs_reshaped = attn.reshape((b, t, self.num_kv_heads, n_groups, s))
      encoded = jnp.einsum('BTKGS,BSKH->BTKGH', probs_reshaped, value_proj)
      b, t, k, g, h = encoded.shape
      encoded = encoded.reshape((b, t, k * g, h))
    else:
      encoded = jnp.einsum('BTNS,BSNH->BTNH', attn, value_proj)
    return encoded

  @property
  def use_gqa(self):
    # Include MQA (num_kv_heads=1) in the GQA path. The GQA einsum
    # correctly handles mismatched head counts via grouped reshape,
    # whereas the non-GQA einsum ('BTNH,BSNH->BTNS') requires N to
    # be equal between Q and K — which fails for MQA.
    return self.num_kv_heads != self.config.num_heads

  @jax.named_scope('attention')
  def __call__(
      self,
      x: jaxtyping.Array,
      segment_pos: jaxtyping.Array,
      cache: LayerCache | None,
      attn_mask: jaxtyping.Array,
      kv_shared_cache: LayerCache | None = None,
      segment_ids: jaxtyping.Array | None = None,
      is_chunked_prefill: bool = False,
      prefix_length: int = 0,
      input_mask: jaxtyping.Array | None = None,
      force_eager: bool = False,
  ):
    remat_config = getattr(self.config, 'remat_config', RematConfig.NONE)
    if (
        remat_config == RematConfig.BLOCK
        or remat_config == RematConfig.BLOCK.value
    ):
      # nnx.remat needs to be applied to the unbound function and take self
      # as the first argument. graph_updates=False prevents TraceContextError
      # when mutating params across jax transformation trace levels.
      # Bake static args via partial to avoid ConcretizationTypeError under remat.
      # Bucket prefix_length to prevent a recompilation storm.
      bucketed_prefix = _maybe_bucket_prefix_length(
          prefix_length,
          cache,
          is_chunked_prefill,
          self.config.prefix_bucket_boundaries,
      )
      block_fn = partial(
          self.block.__func__,
          is_chunked_prefill=is_chunked_prefill,
          prefix_length=bucketed_prefix,
          input_mask=input_mask,
          force_eager=force_eager,
      )
      policy = getattr(jax.checkpoint_policies, self.config.remat_policy)
      return nnx.remat(
          block_fn,
          graph_updates=False,
          policy=policy,
      )(self, x, segment_pos, cache, attn_mask, kv_shared_cache, segment_ids)
    else:
      # Bucket prefix_length for the non-remat path too (controls static slice
      # shapes which affect JAXPR identity).
      bucketed_prefix = _maybe_bucket_prefix_length(
          prefix_length,
          cache,
          is_chunked_prefill,
          self.config.prefix_bucket_boundaries,
      )
      return self.block(
          x,
          segment_pos,
          cache,
          attn_mask,
          kv_shared_cache=kv_shared_cache,
          segment_ids=segment_ids,
          is_chunked_prefill=is_chunked_prefill,
          prefix_length=bucketed_prefix,
          input_mask=input_mask,
          force_eager=force_eager,
      )

  def init_cache(self, batch_size, max_seq_len, dtype):
    cache_len = max_seq_len
    if (
        self.config.use_sliding_window_kv_cache
        and self.attn_type == AttentionType.LOCAL_SLIDING
        and self.config.sliding_window_size is not None
    ):
      cache_len = min(max_seq_len, self.config.sliding_window_size)

    cache_shape = (batch_size, cache_len, self.num_kv_heads, self.head_dim)
    k = shard(
        np.zeros(cache_shape, dtype),
        self.config.shd_config.act_btnh,
        eager=True,
    )
    v = shard(
        np.zeros(cache_shape, dtype),
        self.config.shd_config.act_btnh,
        eager=True,
    )
    end_index = shard(
        np.zeros((batch_size,), np.int32),
        self.config.shd_config.act_btnh[:1],
        eager=True,
    )
    return {'k': k, 'v': v, 'end_index': end_index}
