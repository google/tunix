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

"""Pure-JAX ragged paged attention over tunix's paged KV cache.

Portable fallback for backends without a kernel, and the reference the GPU
path is tested against. Same calling convention as `tpu_inference`'s v3
`ragged_paged_attention`; see `tunix/models/paged_attention.py`.
"""

from __future__ import annotations

from typing import Any

import jax
from jax import numpy as jnp


def ragged_paged_attention(
    queries: jax.Array,
    keys: jax.Array,
    values: jax.Array,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    use_causal_mask: bool = True,
    update_kv_cache: bool = True,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    out_dtype: Any = None,
    mask_value: float | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    max_q_len: int | None = None,
) -> tuple[jax.Array, jax.Array]:
  """Ragged paged attention; returns `(output, kv_cache)`.

  Args:
    queries: `[max_num_tokens, num_q_heads, head_dim]`.
    keys: `[max_num_tokens, num_kv_heads, head_dim]`.
    values: `[max_num_tokens, num_kv_heads, head_dim]`.
    kv_cache: `[pages, page_size, kv_heads_x2 // packing, packing, head_dim]`.
    kv_lens: `i32[max_num_seqs]`, including this step's tokens.
    page_indices: `i32[max_num_seqs * pages_per_seq]`.
    cu_q_lens: `i32[max_num_seqs + 1]`.
    distribution: `i32[3]`; `distribution[-1]` is the active sequence count.
    use_causal_mask: whether to apply causal masking.
    update_kv_cache: whether to write `keys`/`values` into the cache.
    sm_scale: softmax scale.
    sliding_window: a query at position `p` sees keys in `(p - window, p]`.
    soft_cap: optional tanh logit soft-cap.
    out_dtype: defaults to the query dtype.
    mask_value: masked logit value. Defaults to -1e30.
    q_scale: query dequantization scale.
    k_scale: key dequantization scale.
    v_scale: value dequantization scale.
    max_q_len: static per-sequence query bound; tokens past it are dropped.
      Defaults to `max_num_tokens`.
  """
  out_dtype = out_dtype or queries.dtype
  pages, page_size, h2_per_pack, packing, head_dim_aligned = kv_cache.shape
  cache = kv_cache.reshape(pages, page_size, h2_per_pack * packing,
                           head_dim_aligned)
  num_tokens = queries.shape[0]
  max_num_seqs = kv_lens.shape[0]
  kv_lens = kv_lens.astype(jnp.int32)
  cu_q_lens = cu_q_lens.astype(jnp.int32)
  page_table = page_indices.astype(jnp.int32).reshape(max_num_seqs, -1)

  # Map each token to its sequence and absolute KV position.
  num_seqs = distribution[-1].astype(jnp.int32)
  tokens = jnp.arange(num_tokens, dtype=jnp.int32)
  seq_idx = jnp.clip(
      jnp.searchsorted(cu_q_lens, tokens, side="right") - 1, 0, max_num_seqs - 1
  )
  q_pos = tokens - cu_q_lens[seq_idx]
  q_len = cu_q_lens[seq_idx + 1] - cu_q_lens[seq_idx]
  kv_pos = kv_lens[seq_idx] - q_len + q_pos
  valid = (tokens < cu_q_lens[num_seqs]) & (seq_idx < num_seqs)

  if update_kv_cache:
    cache = _write_kv(cache, keys, values, page_table, seq_idx, kv_pos, valid)
  out = _attention(
      queries, cache, kv_lens, page_table, seq_idx, q_pos, kv_pos, valid,
      num_kv_heads=keys.shape[1], max_q_len=max_q_len or num_tokens,
      use_causal_mask=use_causal_mask, sm_scale=sm_scale,
      sliding_window=sliding_window, soft_cap=soft_cap, out_dtype=out_dtype,
      mask_value=mask_value, q_scale=q_scale, k_scale=k_scale, v_scale=v_scale,
  )
  out = jnp.where(valid[:, None, None], out, 0).astype(out_dtype)
  return out, cache.reshape(kv_cache.shape)


def _write_kv(cache, keys, values, page_table, seq_idx, kv_pos, valid):
  """Scatters new K/V into the flattened `[P, S, kv_heads_x2, D]` cache."""
  pages, page_size, kv_heads_x2, head_dim_aligned = cache.shape
  num_tokens, num_kv_heads, head_dim = keys.shape
  kv = jnp.concatenate([keys, values], axis=-1).reshape(
      num_tokens, 2 * num_kv_heads, head_dim
  )
  kv = jnp.pad(kv, ((0, 0), (0, kv_heads_x2 - 2 * num_kv_heads),
                    (0, head_dim_aligned - head_dim)))
  page = page_table[seq_idx, kv_pos // page_size]
  # Invalid tokens are aimed past the last page and dropped.
  return cache.at[jnp.where(valid, page, pages), kv_pos % page_size].set(
      kv.astype(cache.dtype), mode="drop"
  )


def _attention(queries, cache, kv_lens, page_table, seq_idx, q_pos, kv_pos,
               valid, *, num_kv_heads, max_q_len, use_causal_mask, sm_scale,
               sliding_window, soft_cap, out_dtype, mask_value, q_scale,
               k_scale, v_scale):
  """Un-pages each sequence and attends densely."""
  num_tokens, num_q_heads, head_dim = queries.shape
  max_num_seqs, pages_per_seq = page_table.shape
  _, page_size, kv_heads_x2, head_dim_aligned = cache.shape
  max_kv_len = pages_per_seq * page_size

  seq_kv = cache[page_table].reshape(max_num_seqs, max_kv_len, kv_heads_x2,
                                     head_dim_aligned)
  k = seq_kv[:, :, 0:2 * num_kv_heads:2, :head_dim]
  v = seq_kv[:, :, 1:2 * num_kv_heads:2, :head_dim]
  if k_scale is not None:
    k = k / k_scale
  if v_scale is not None:
    v = v / v_scale
  if q_scale is not None:
    queries = queries / q_scale

  # Group queries by sequence; row `max_num_seqs` absorbs dropped tokens.
  fits = valid & (q_pos < max_q_len)
  row = jnp.where(fits, seq_idx, max_num_seqs)
  slot = jnp.where(fits, q_pos, 0)
  q = jnp.zeros((max_num_seqs + 1, max_q_len, num_q_heads, head_dim),
                queries.dtype).at[row, slot].set(queries)[:max_num_seqs]
  q_kv_pos = jnp.zeros((max_num_seqs + 1, max_q_len), jnp.int32).at[
      row, slot].set(kv_pos)[:max_num_seqs]
  q = q.reshape(max_num_seqs, max_q_len, num_kv_heads, -1, head_dim)

  s = jnp.einsum("sqkgd,slkd->sqkgl", q, k,
                 preferred_element_type=jnp.float32) * sm_scale
  if soft_cap is not None:
    s = soft_cap * jnp.tanh(s / soft_cap)
  kv_idx = jnp.arange(max_kv_len)[None, None, :]
  mask = kv_idx < kv_lens[:, None, None]
  if use_causal_mask:
    mask &= kv_idx <= q_kv_pos[:, :, None]
  if sliding_window is not None:
    mask &= kv_idx > q_kv_pos[:, :, None] - sliding_window
  s = jnp.where(mask[:, :, None, None, :], s,
                -1e30 if mask_value is None else mask_value)
  w = jax.nn.softmax(s, axis=-1).astype(out_dtype)
  o = jnp.einsum("sqkgl,slkd->sqkgd", w, v, preferred_element_type=jnp.float32)
  o = o.reshape(max_num_seqs, max_q_len, num_q_heads, head_dim)
  return jnp.where(fits[:, None, None],
                   o[seq_idx, jnp.clip(q_pos, 0, max_q_len - 1)], 0)
