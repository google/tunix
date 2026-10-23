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

"""Paged attention over tunix's paged KV cache.

Dispatches by backend to one of three implementations:

* TPU: Tokamax's `ragged_paged_attention` Mosaic kernel.
* GPU: vLLM's Triton kernels (`paged_attention_vllm_gpu`).
* Otherwise: pure JAX (`paged_attention_jax`), also the reference in tests.

All share the 5D KV cache layout
`[pages, page_size, kv_heads_x2 // packing, packing, head_dim]`, with heads
interleaved `k0, v0, k1, v1, ...`.
"""

from __future__ import annotations

from typing import Any

from flax import struct
import jax
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import numpy as np


@struct.dataclass
class PagedAttnMetadata:
  """Execution metadata for paged attention.

  Rows are ordered `[decodes, chunked prefills, full prefills]`, followed by
  padding rows up to the batch's row capacity.
  """

  # `[num_rows, max_pages_per_seq]` physical page indices of each row, keyed by
  # the name of the KV cache they index.
  page_indices: dict[str, jax.Array | np.ndarray]
  # `[num_rows]` KV length of each row, including the tokens it runs this step.
  kv_lens: jax.Array | np.ndarray
  # `[num_rows]` number of tokens each row runs this step.
  query_lens: jax.Array | np.ndarray
  # `[3]` row layout `(num_decodes, num_chunked_end, num_seqs)`: the decodes
  # are rows `[0, i)`, the chunked prefills `[i, j)` and the full prefills
  # `[j, k)`.
  distribution: jax.Array | np.ndarray


def _shardable_tp_axis(
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh,
    tp_axis: str | None,
    *dim_sizes: int | None,
) -> str | None:
  """Returns `tp_axis` if every `dim_size` is divisible by its mesh size."""
  if tp_axis is None or tp_axis not in mesh.shape:
    return None
  tp_size = mesh.shape[tp_axis]
  if any(d is not None and d % tp_size != 0 for d in dim_sizes):
    return None
  return tp_axis


def paged_attention(
    queries: jax.Array,
    keys: jax.Array,
    values: jax.Array,
    kv_cache: jax.Array,
    metadata: PagedAttnMetadata,
    *,
    cache_name: str,
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh,
    tp_axis: str | None = None,
    decode_only: bool = False,
    max_q_len: int | None = None,
    backend: str | None = None,
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
) -> tuple[jax.Array, jax.Array]:
  """Paged attention over one named cache; returns `(output, kv_cache)`.

  Args:
    queries: `[max_num_tokens, num_q_heads, head_dim]`.
    keys: `[max_num_tokens, num_kv_heads, head_dim]`.
    values: `[max_num_tokens, num_kv_heads, head_dim]`.
    kv_cache: the 5D cache described in the module docstring.
    metadata: the batch's ragged layout and page tables.
    cache_name: the key of `kv_cache` in `metadata.page_indices`.
    mesh: the mesh to run the kernel over.
    tp_axis: the mesh axis KV heads are sharded across. Everything else is
      replicated, including the cache pages across data parallelism.
    decode_only: static promise of one query token per sequence. On GPU it
      enables vLLM's split-KV decode launch.
    max_q_len: static per-sequence query bound for the JAX path; tokens past it
      are dropped. Defaults to `max_num_tokens`.
    backend: `"tpu"`, `"gpu"`, or anything else for the JAX path. Defaults to
      `jax.default_backend()`.
    use_causal_mask: whether to apply causal masking.
    update_kv_cache: whether to write `keys`/`values` into the cache. Shared-KV
      layers pass False: their lender already wrote this step's K/V, and the
      kernel reads it from the cache.
    sm_scale: softmax scale.
    sliding_window: a query at position `p` sees keys in `(p - window, p]`.
    soft_cap: optional tanh logit soft-cap.
    out_dtype: defaults to the query dtype.
    mask_value: masked logit value. Ignored on GPU, which uses -inf.
    q_scale: query dequantization scale. Not supported on GPU.
    k_scale: key dequantization scale. Not supported on GPU.
    v_scale: value dequantization scale. Not supported on GPU.
  """
  def _call_paged_attention(
      q, k, v, cache, kv_lens, page_indices, query_lens, distribution
  ):
    cu_q_lens = jnp.pad(jnp.cumsum(query_lens), (1, 0))
    return _paged_attention(
        q, k, v, cache, kv_lens, page_indices, cu_q_lens, distribution,
        decode_only=decode_only,
        max_q_len=max_q_len,
        backend=backend,
        use_causal_mask=use_causal_mask,
        update_kv_cache=update_kv_cache,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        out_dtype=out_dtype,
        mask_value=mask_value,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
    )

  q_tp_axis = _shardable_tp_axis(mesh, tp_axis, queries.shape[1])
  kv_tp_axis = _shardable_tp_axis(
      mesh, q_tp_axis, keys.shape[1], kv_cache.shape[2]
  )
  if kv_tp_axis is None and keys.shape[1] != 1:
    q_tp_axis = None
  q_spec = P(None, q_tp_axis, None)
  kv_spec = P(None, kv_tp_axis, None)
  cache_spec = P(None, None, kv_tp_axis, None, None)
  return shard_map(
      _call_paged_attention,
      mesh=mesh,
      in_specs=(
          q_spec,  # queries
          kv_spec,  # keys
          kv_spec,  # values
          cache_spec,  # kv_cache
          P(),  # kv_lens
          P(),  # page_indices
          P(),  # query_lens
          P(),  # distribution
      ),
      out_specs=(q_spec, cache_spec),
      check_rep=False,
  )(
      queries,
      keys,
      values,
      kv_cache,
      metadata.kv_lens,
      metadata.page_indices[cache_name].reshape(-1),
      metadata.query_lens,
      metadata.distribution,
  )


def _paged_attention(
    queries: jax.Array,
    keys: jax.Array,
    values: jax.Array,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    decode_only: bool,
    max_q_len: int | None,
    backend: str | None,
    use_causal_mask: bool,
    update_kv_cache: bool,
    sm_scale: float,
    sliding_window: int | None,
    soft_cap: float | None,
    out_dtype: Any,
    mask_value: float | None,
    q_scale: float | None,
    k_scale: float | None,
    v_scale: float | None,
) -> tuple[jax.Array, jax.Array]:
  """Dispatches paged attention to the backend's kernel.

  Args:
    queries: `[max_num_tokens, num_q_heads, head_dim]`.
    keys: `[max_num_tokens, num_kv_heads, head_dim]`.
    values: `[max_num_tokens, num_kv_heads, head_dim]`.
    kv_cache: the 5D cache described in the module docstring.
    kv_lens: `i32[max_num_seqs]`, including this step's tokens.
    page_indices: `i32[max_num_seqs * pages_per_seq]`.
    cu_q_lens: `i32[max_num_seqs + 1]`.
    distribution: `i32[3]`; `distribution[-1]` is the active sequence count.
    decode_only: static promise of one query token per sequence. On GPU it
      enables vLLM's split-KV decode launch.
    max_q_len: static per-sequence query bound for the JAX path; tokens past it
      are dropped. Defaults to `max_num_tokens`.
    backend: `"tpu"`, `"gpu"`, or anything else for the JAX path. Defaults to
      `jax.default_backend()`.
    use_causal_mask: whether to apply causal masking.
    update_kv_cache: whether to write `keys`/`values` into the cache. Shared-KV
      layers pass False.
    sm_scale: softmax scale.
    sliding_window: a query at position `p` sees keys in `(p - window, p]`.
    soft_cap: optional tanh logit soft-cap.
    out_dtype: defaults to the query dtype.
    mask_value: masked logit value. Ignored on GPU, which uses -inf.
    q_scale: query dequantization scale. Not supported on GPU.
    k_scale: key dequantization scale. Not supported on GPU.
    v_scale: value dequantization scale. Not supported on GPU.
  """
  backend = backend or jax.default_backend()
  page_indices = jnp.maximum(page_indices, 0)
  args = (queries, keys, values, kv_cache, kv_lens, page_indices, cu_q_lens,
          distribution)
  common = dict(
      use_causal_mask=use_causal_mask,
      sm_scale=sm_scale,
      sliding_window=sliding_window,
      soft_cap=soft_cap,
      out_dtype=out_dtype or queries.dtype,
      mask_value=mask_value,
      q_scale=q_scale,
      k_scale=k_scale,
      v_scale=v_scale,
  )

  # Backends are imported lazily so the TPU path only loads Pallas/Mosaic on
  # TPU and the GPU path only loads triton and jax_triton on GPU.
  # pylint: disable=g-import-not-at-top
  if backend == "tpu":
    from tokamax import ops as tokamax_ops

    return tokamax_ops.ragged_paged_attention(
        *args,
        **common,
        skip_kv_update=not update_kv_cache,
        implementation="mosaic_tpu",
    )
  if backend == "gpu":
    from tunix.models import paged_attention_vllm_gpu

    return paged_attention_vllm_gpu.paged_attention(
        *args,
        **common,
        update_kv_cache=update_kv_cache,
        decode_only=decode_only,
    )
  from tunix.models import paged_attention_jax

  return paged_attention_jax.paged_attention(
      *args,
      **common,
      update_kv_cache=update_kv_cache,
      max_q_len=max_q_len,
  )
