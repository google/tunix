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

"""GPU ragged paged attention with vLLM's Triton kernels.

vLLM's Triton `reshape_and_cache` and unified attention kernels (vendored
verbatim under `tunix/kernels/triton/vllm`), launched with `jax_triton`. Needs
`torch`, `vllm`, `triton` and `jax_triton`. Same calling convention as
`tpu_inference`'s v3 `ragged_paged_attention`; see
`tunix/models/paged_attention.py`.

Unlike the TPU kernel, vLLM's unified attention only reads the cache, so the
new K/V are first written with `reshape_and_cache`, as vLLM's `triton_attn`
backend does.
"""

from __future__ import annotations

from typing import Any

import jax
from jax import numpy as jnp
import jax_triton as jt
import triton
import triton.language as tl
from tunix.kernels.triton.vllm import triton_reshape_and_cache_flash as vllm_cache
from tunix.kernels.triton.vllm import triton_unified_attention as vllm_attn

# From vLLM's `triton_attn.py` backend.
_NUM_SEGMENTS = 16
_MIN_LAUNCH_GRID_SIZE_2D = 128


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
    decode_only: bool = False,
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
    mask_value: ignored; vLLM's kernel masks with -inf.
    q_scale: not supported; must be None.
    k_scale: not supported; must be None.
    v_scale: not supported; must be None.
    decode_only: static promise of one query token per sequence; enables
      vLLM's split-KV decode launch.
  """
  del mask_value  # vLLM's kernel masks with -inf.
  if q_scale is not None or k_scale is not None or v_scale is not None:
    raise NotImplementedError("q/k/v scales are not supported on GPU.")
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
  q_len = cu_q_lens[seq_idx + 1] - cu_q_lens[seq_idx]
  kv_pos = kv_lens[seq_idx] - q_len + tokens - cu_q_lens[seq_idx]
  valid = (tokens < cu_q_lens[num_seqs]) & (seq_idx < num_seqs)

  if update_kv_cache:
    page = page_table[seq_idx, kv_pos // page_size]
    slots = jnp.where(valid, page * page_size + kv_pos % page_size, -1)
    cache = _write_kv(cache, keys, values, slots)
  out = _attention(
      queries, cache, kv_lens, page_table, cu_q_lens, num_seqs,
      num_kv_heads=keys.shape[1], decode_only=decode_only,
      use_causal_mask=use_causal_mask, sm_scale=sm_scale,
      sliding_window=sliding_window, soft_cap=soft_cap, out_dtype=out_dtype,
  )
  out = jnp.where(valid[:, None, None], out, 0).astype(out_dtype)
  return out, cache.reshape(kv_cache.shape)


def _write_kv(cache, keys, values, slots):
  """vLLM's Triton `reshape_and_cache`, with its PyTorch launcher's parameters.

  Updates `cache` in place (aliased); `slots` is `page * page_size + offset`,
  or -1 for tokens to skip.
  """
  _, page_size, kv_heads_x2, head_dim_aligned = cache.shape
  num_tokens, num_kv_heads, head_dim = keys.shape
  n = num_kv_heads * head_dim
  tile_size = min(2048, triton.next_power_of_2(n))
  if float(jax.devices()[0].compute_capability) < 9:
    tile_size = min(512, tile_size)
  return jt.triton_call(
      keys, values, slots, cache,
      kernel=_reshape_and_cache,
      out_shape=jax.ShapeDtypeStruct(cache.shape, cache.dtype),
      input_output_aliases={3: 0},
      grid=(num_tokens, triton.cdiv(n, tile_size)),
      num_warps=16,
      num_stages=10,
      key_stride=n,
      block_stride=page_size * kv_heads_x2 * head_dim_aligned,
      page_stride=kv_heads_x2 * head_dim_aligned,
      head_stride=2 * head_dim_aligned,
      V_OFFSET=head_dim_aligned,
      num_heads=num_kv_heads,
      head_size=head_dim,
      block_size=page_size,
      TILE_SIZE=tile_size,
  )


def _attention(queries, cache, kv_lens, page_table, cu_q_lens, num_seqs, *,
               num_kv_heads, decode_only, use_causal_mask, sm_scale,
               sliding_window, soft_cap, out_dtype):
  """vLLM's unified attention, with its PyTorch launcher's parameters."""
  num_tokens, num_q_heads, head_dim = queries.shape
  max_num_seqs, _ = page_table.shape
  _, page_size, kv_heads_x2, head_dim_aligned = cache.shape
  q_per_kv = num_q_heads // num_kv_heads

  block_m = 16 if q_per_kv <= 16 else triton.next_power_of_2(q_per_kv)
  head_dim_padded = triton.next_power_of_2(head_dim)
  window = sliding_window or 0
  use_3d = (decode_only
            and max_num_seqs <= _MIN_LAUNCH_GRID_SIZE_2D // num_kv_heads)
  tile_size = vllm_attn._get_tile_size(  # pylint: disable=protected-access
      head_dim, window, jnp.dtype(queries.dtype).itemsize, not use_3d
  )
  launch = {}
  if (head_dim == 256 and not decode_only and q_per_kv <= 16
      and vllm_attn.current_platform.is_device_capability_family(100)):
    block_m, tile_size, launch = 32, 128, dict(num_warps=8, num_stages=2)
  block_q = block_m // q_per_kv
  num_segments = _NUM_SEGMENTS if use_3d else 1

  # The kernel reads a page-table entry per tile slot before masking, so pad
  # the table for tiles wider than a page.
  block_table = jnp.pad(page_table, ((0, 0), (0, -(-tile_size // page_size))))
  # `max(.., 1)` keeps the kernel's binary search in bounds for empty batches.
  num_seqs_buf = jnp.maximum(num_seqs, 1).reshape(1)

  out_shape = jax.ShapeDtypeStruct((num_tokens, num_q_heads, head_dim),
                                   out_dtype)
  dummy = jax.ShapeDtypeStruct((1,), jnp.float32)
  segm = (num_tokens, num_q_heads, num_segments)
  out_types = (
      (dummy,
       jax.ShapeDtypeStruct((*segm, head_dim_padded), jnp.float32),
       jax.ShapeDtypeStruct(segm, jnp.float32),
       jax.ShapeDtypeStruct(segm, jnp.float32))
      if use_3d else (out_shape, dummy, dummy, dummy)
  )
  common = dict(
      num_query_heads=num_q_heads,
      output_stride_0=num_q_heads * head_dim,
      output_stride_1=head_dim,
      block_table_stride=block_table.shape[1],
      TILE_SIZE=tile_size,
      HEAD_SIZE=head_dim,
      HEAD_SIZE_PADDED=head_dim_padded,
      BLOCK_Q=block_q,
      NUM_SEGMENTS_PER_SEQ=num_segments,
  )
  out, segm_out, segm_max, segm_expsum = jt.triton_call(
      queries, cache, block_table, kv_lens, cu_q_lens, num_seqs_buf,
      kernel=_unified_attention,
      out_shape=out_types,
      grid=(num_tokens // block_q + max_num_seqs, num_kv_heads)
      + ((num_segments,) if use_3d else ()),
      **launch,
      **common,
      scale=float(sm_scale),
      softcap=float(soft_cap or 0.0),
      num_queries_per_kv=q_per_kv,
      query_stride_0=num_q_heads * head_dim,
      query_stride_1=head_dim,
      stride_kv_0=page_size * kv_heads_x2 * head_dim_aligned,
      stride_kv_1=kv_heads_x2 * head_dim_aligned,
      stride_kv_2=2 * head_dim_aligned,
      V_OFFSET=head_dim_aligned,
      BLOCK_SIZE=page_size,
      USE_SOFTCAP=bool(soft_cap),
      SLIDING_WINDOW=window,
      USE_CAUSAL=use_causal_mask,
      BLOCK_M=block_m,
      IS_3D=use_3d,
  )
  if use_3d:
    out = jt.triton_call(
        segm_out, segm_max, segm_expsum, kv_lens, cu_q_lens, num_seqs_buf,
        kernel=_reduce_segments,
        out_shape=out_shape,
        grid=(num_tokens, num_q_heads),
        **common,
    )
  return out


# Triton entry points that call the vendored vLLM kernels inline, adapting what
# `jax_triton` cannot express: the value cache is the fused cache offset by one
# head (`V_OFFSET`), and `num_seqs` must be a runtime value, so it is loaded
# from a device buffer. Outputs follow inputs because `jax_triton` appends
# output pointers.
@triton.jit
def _unified_attention(
    query_ptr, kv_cache_ptr, block_tables_ptr, seq_lens_ptr,
    query_start_len_ptr, num_seqs_ptr,
    output_ptr, segm_output_ptr, segm_max_ptr, segm_expsum_ptr,
    scale: tl.constexpr, softcap: tl.constexpr,
    num_query_heads: tl.constexpr, num_queries_per_kv: tl.constexpr,
    block_table_stride: tl.constexpr,
    query_stride_0: tl.constexpr, query_stride_1: tl.constexpr,
    output_stride_0: tl.constexpr, output_stride_1: tl.constexpr,
    stride_kv_0: tl.constexpr, stride_kv_1: tl.constexpr,
    stride_kv_2: tl.constexpr, V_OFFSET: tl.constexpr,
    BLOCK_SIZE: tl.constexpr, TILE_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr, HEAD_SIZE_PADDED: tl.constexpr,
    USE_SOFTCAP: tl.constexpr, SLIDING_WINDOW: tl.constexpr,
    USE_CAUSAL: tl.constexpr, BLOCK_Q: tl.constexpr, BLOCK_M: tl.constexpr,
    NUM_SEGMENTS_PER_SEQ: tl.constexpr, IS_3D: tl.constexpr,
):
  vllm_attn.kernel_unified_attention(
      output_ptr=output_ptr,
      query_ptr=query_ptr,
      key_cache_ptr=kv_cache_ptr,
      value_cache_ptr=kv_cache_ptr + V_OFFSET,
      sink_ptr=None,
      block_tables_ptr=block_tables_ptr,
      seq_lens_ptr=seq_lens_ptr,
      alibi_slopes_ptr=None,
      qq_bias_ptr=None,
      scale=scale,
      q_scale=None,
      k_scale=None,
      v_scale=None,
      out_scale=1.0,
      softcap=softcap,
      num_query_heads=num_query_heads,
      num_queries_per_kv=num_queries_per_kv,
      block_table_stride=block_table_stride,
      query_stride_0=query_stride_0,
      query_stride_1=query_stride_1,
      output_stride_0=output_stride_0,
      output_stride_1=output_stride_1,
      qq_bias_stride_0=0,
      BLOCK_SIZE=BLOCK_SIZE,
      TILE_SIZE=TILE_SIZE,
      HEAD_SIZE=HEAD_SIZE,
      HEAD_SIZE_PADDED=HEAD_SIZE_PADDED,
      USE_ALIBI_SLOPES=False,
      USE_ALIBI_SQRT=False,
      USE_QQ_BIAS=False,
      USE_SOFTCAP=USE_SOFTCAP,
      USE_SINKS=False,
      SLIDING_WINDOW=SLIDING_WINDOW,
      USE_CAUSAL=USE_CAUSAL,
      USE_PER_SEQ_CAUSAL=False,
      per_seq_causal_ptr=None,
      USE_MM_PREFIX=False,
      MAX_MM_RANGES=0,
      mm_prefix_range_ptr=None,
      rswa_prefix_lens_ptr=seq_lens_ptr,  # Unused; vLLM passes this too.
      R_SWA_WINDOW=0,
      USE_R_SWA=False,
      stride_k_cache_0=stride_kv_0,
      stride_k_cache_1=stride_kv_1,
      stride_k_cache_2=stride_kv_2,
      stride_k_cache_3=1,
      stride_v_cache_0=stride_kv_0,
      stride_v_cache_1=stride_kv_1,
      stride_v_cache_2=stride_kv_2,
      stride_v_cache_3=1,
      query_start_len_ptr=query_start_len_ptr,
      BLOCK_Q=BLOCK_Q,
      num_seqs=tl.load(num_seqs_ptr),
      BLOCK_M=BLOCK_M,
      NUM_SEGMENTS_PER_SEQ=NUM_SEGMENTS_PER_SEQ,
      USE_FP8=False,
      IS_3D=IS_3D,
      segm_output_ptr=segm_output_ptr,
      segm_max_ptr=segm_max_ptr,
      segm_expsum_ptr=segm_expsum_ptr,
  )


@triton.jit
def _reduce_segments(
    segm_output_ptr, segm_max_ptr, segm_expsum_ptr, seq_lens_ptr,
    query_start_len_ptr, num_seqs_ptr,
    output_ptr,
    num_query_heads: tl.constexpr,
    output_stride_0: tl.constexpr, output_stride_1: tl.constexpr,
    block_table_stride: tl.constexpr, TILE_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr, HEAD_SIZE_PADDED: tl.constexpr,
    BLOCK_Q: tl.constexpr, NUM_SEGMENTS_PER_SEQ: tl.constexpr,
):
  vllm_attn.reduce_segments(
      output_ptr=output_ptr,
      segm_output_ptr=segm_output_ptr,
      segm_max_ptr=segm_max_ptr,
      segm_expsum_ptr=segm_expsum_ptr,
      seq_lens_ptr=seq_lens_ptr,
      num_seqs=tl.load(num_seqs_ptr),
      num_query_heads=num_query_heads,
      out_scale_inv=1.0,
      output_stride_0=output_stride_0,
      output_stride_1=output_stride_1,
      block_table_stride=block_table_stride,
      TILE_SIZE=TILE_SIZE,
      HEAD_SIZE=HEAD_SIZE,
      HEAD_SIZE_PADDED=HEAD_SIZE_PADDED,
      query_start_len_ptr=query_start_len_ptr,
      BLOCK_Q=BLOCK_Q,
      NUM_SEGMENTS_PER_SEQ=NUM_SEGMENTS_PER_SEQ,
      USE_FP8=False,
  )


@triton.jit
def _reshape_and_cache(
    key_ptr, value_ptr, slot_mapping_ptr,
    kv_cache_ptr,  # Aliased to the output: written in place.
    key_stride: tl.constexpr, block_stride: tl.constexpr,
    page_stride: tl.constexpr, head_stride: tl.constexpr,
    V_OFFSET: tl.constexpr, num_heads: tl.constexpr,
    head_size: tl.constexpr, block_size: tl.constexpr,
    TILE_SIZE: tl.constexpr,
):
  vllm_cache.reshape_and_cache_kernel_flash(
      key_ptr=key_ptr,
      value_ptr=value_ptr,
      key_cache_ptr=kv_cache_ptr,
      value_cache_ptr=kv_cache_ptr + V_OFFSET,
      slot_mapping_ptr=slot_mapping_ptr,
      k_scale=None,
      v_scale=None,
      key_stride=key_stride,
      value_stride=key_stride,
      block_stride=block_stride,
      head_stride=head_stride,
      dim_stride_k=0,
      dim_stride_v=0,
      page_stride=page_stride,
      num_heads=num_heads,
      head_size=head_size,
      block_size=block_size,
      x=1,
      USE_HEAD_MAJOR_LAYOUT=False,
      FP8_KV_CACHE=False,
      TILE_SIZE=TILE_SIZE,
  )
