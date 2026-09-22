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

"""Backend dispatch for ragged paged attention.

Tunix's paged decode path is built around `tpu_inference`'s v3
`ragged_paged_attention`, which is a Mosaic-TPU Pallas kernel and therefore
cannot lower on GPU or CPU. This module picks that kernel when it is usable and
otherwise falls back to tokamax's batched-RPA reference, which is pure
vectorized JAX and runs anywhere.

The bundled `tpu_inference` `ref_ragged_paged_attention` is deliberately *not*
used as the fallback: it loops over a traced `distribution` and slices with
traced bounds, so it raises under `jax.jit`, and tunix's model runner is jitted.
It also concatenates only the active sequences, so with padding slots it returns
fewer rows than the real kernel. tokamax's reference is `@jax.jit`-decorated,
statically shaped, and matches the real kernel's output contract.

Both backends are imported lazily. Neither `tpu-inference` nor `tokamax` is a
base install requirement of OSS tunix -- they live in `special_requirements.txt`
and `maxtext_requirements.txt` respectively -- so a module-scope import of
either would break a plain `pip install tunix`.

Two adaptations are needed to make tokamax's reference a drop-in replacement.
Both are handled here so that no change to tokamax is required.

1. KV cache layout.

       tpu_inference v3 : [pages, page_size, kv_heads_x2 // packing, packing, D]
       tokamax bRPA     : [pages, page_size, kv_heads_x2,                     D]

   Same buffer: the packing axis is contiguous with the head axis, so the
   conversion is a pure reshape with no data movement. Both use the same
   interleaved `k0, v0, k1, v1, ...` head ordering.

2. `skip_kv_update` is a no-op in tokamax's reference. The flag is declared in
   the signature and in `static_argnames` but never read in the body, so the
   cache write is unconditional. Tunix's shared-KV layers depend on the
   opposite: they pass `update_kv_cache=False` together with zero-filled keys
   and values, so an unconditional write would zero a live cache -- and would
   corrupt the attention output too, since the write happens before attention is
   computed. We compensate by writing back what is already there; see
   `_read_back_kv`.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Final

import jax
from jax import numpy as jnp


TPU_KERNEL: Final[str] = "tpu_inference_v3"
REFERENCE: Final[str] = "tokamax_reference"

_TPU_KERNEL_MODULE: Final[str] = (
    "tpu_inference.kernels.ragged_paged_attention.v3.kernel"
)
_REFERENCE_MODULE: Final[str] = (
    "tokamax._src.ops.experimental.batched_rpa.reference"
)


@functools.lru_cache(maxsize=1)
def _load_tpu_kernel() -> Callable[..., Any]:
  """Returns `tpu_inference`'s v3 `ragged_paged_attention`."""
  import importlib  # pylint: disable=g-import-not-at-top

  try:
    module = importlib.import_module(_TPU_KERNEL_MODULE)
  except ImportError as e:
    raise ImportError(
        "The TPU ragged paged attention kernel requires `tpu-inference`, which"
        " is not installed. Install it with"
        " `pip install -r requirements/special_requirements.txt`, or force the"
        f" portable backend by passing `implementation={REFERENCE!r}`."
    ) from e
  return module.ragged_paged_attention


@functools.lru_cache(maxsize=1)
def _load_reference() -> Callable[..., Any]:
  """Returns tokamax's batched-RPA reference implementation.

  This reaches into tokamax's `_src` because batched RPA is still experimental
  and is not re-exported from the top-level package yet.
  """
  import importlib  # pylint: disable=g-import-not-at-top

  try:
    module = importlib.import_module(_REFERENCE_MODULE)
  except ImportError as e:
    raise ImportError(
        "The portable ragged paged attention fallback requires `tokamax`,"
        " which is not installed. Install it with `pip install tokamax`."
    ) from e
  return module.batched_ragged_paged_attention_reference


def _default_backend_is_tpu() -> bool:
  try:
    return jax.default_backend() == "tpu"
  except RuntimeError:
    # Raised when no backend is available at all.
    return False


def select_implementation() -> str:
  """Returns the backend to use on the current platform."""
  return TPU_KERNEL if _default_backend_is_tpu() else REFERENCE


def kv_cache_5d_to_4d(kv_cache: jax.Array) -> jax.Array:
  """Converts `[P, S, H2 // packing, packing, D]` to `[P, S, H2, D]`."""
  if kv_cache.ndim != 5:
    raise ValueError(
        f"Expected a 5D tpu_inference v3 KV cache, got shape {kv_cache.shape}."
    )
  pages, page_size, h2_per_pack, packing, head_dim = kv_cache.shape
  return kv_cache.reshape(pages, page_size, h2_per_pack * packing, head_dim)


def kv_cache_4d_to_5d(kv_cache: jax.Array, packing: int) -> jax.Array:
  """Inverse of `kv_cache_5d_to_4d`."""
  pages, page_size, h2, head_dim = kv_cache.shape
  return kv_cache.reshape(pages, page_size, h2 // packing, packing, head_dim)


def _read_back_kv(
    kv_cache_4d: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    keys: jax.Array,
) -> tuple[jax.Array, jax.Array]:
  """Returns the (k, v) already stored at the slots the reference will write.

  This mirrors the index arithmetic in tokamax's reference so that its
  unconditional `.at[...].set(...)` stores values identical to what is already
  in the cache, turning the update into a no-op.

  Args:
    kv_cache_4d: the KV cache, in tokamax's 4D layout.
    kv_lens: `[max_num_seqs]` per-sequence KV lengths.
    page_indices: `[max_num_seqs * pages_per_seq]` flattened page table.
    cu_q_lens: `[max_num_seqs + 1]` cumulative query lengths.
    keys: the caller's keys, used only for shape and dtype.

  Returns:
    A `(k, v)` pair shaped and typed like `keys`.
  """
  total_q_tokens, num_kv_heads, actual_head_dim = keys.shape
  page_size = kv_cache_4d.shape[1]
  max_num_seqs = kv_lens.shape[0]
  pages_per_seq = page_indices.shape[0] // max_num_seqs

  token_indices = jnp.arange(total_q_tokens)
  seq_idx = jnp.searchsorted(cu_q_lens, token_indices, side="right") - 1
  seq_idx = jnp.clip(seq_idx, 0, max_num_seqs - 1)

  q_start = cu_q_lens[seq_idx]
  q_len = cu_q_lens[seq_idx + 1] - q_start
  token_kv_pos = kv_lens[seq_idx] - q_len + (token_indices - q_start)

  page_idx = page_indices[seq_idx * pages_per_seq + token_kv_pos // page_size]
  page_offset = token_kv_pos % page_size

  # [total_q_tokens, num_kv_heads_x2, head_dim_aligned]
  existing = kv_cache_4d[page_idx, page_offset]
  # Drop head and head_dim padding, then de-interleave k0, v0, k1, v1, ...
  real = existing[:, : num_kv_heads * 2, :actual_head_dim]
  return (
      real[:, 0::2, :].astype(keys.dtype),
      real[:, 1::2, :].astype(keys.dtype),
  )


def _reference_ragged_paged_attention(
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
) -> tuple[jax.Array, jax.Array]:
  """Runs tokamax's reference against a `tpu_inference`-shaped KV cache."""
  packing = kv_cache.shape[3]
  kv_cache_4d = kv_cache_5d_to_4d(kv_cache)

  if not update_kv_cache:
    keys, values = _read_back_kv(
        kv_cache_4d, kv_lens, page_indices, cu_q_lens, keys
    )

  out, updated_4d = _load_reference()(
      queries,
      keys,
      values,
      kv_cache_4d,
      kv_lens,
      page_indices,
      cu_q_lens,
      distribution,
      use_causal_mask=use_causal_mask,
      sm_scale=sm_scale,
      sliding_window=sliding_window,
      soft_cap=soft_cap,
      mask_value=mask_value,
      out_dtype=out_dtype,
      q_scale=q_scale,
      k_scale=k_scale,
      v_scale=v_scale,
  )

  if not update_kv_cache:
    # The write-back above is value-identical, but returning the caller's
    # original buffer keeps the no-write contract bit-exact.
    return out, kv_cache
  return out, kv_cache_4d_to_5d(updated_4d, packing)


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
    implementation: str | None = None,
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
    **kernel_kwargs: Any,
) -> tuple[jax.Array, jax.Array]:
  """Ragged paged attention, dispatched to the best available backend.

  Args:
    queries: `[max_num_tokens, num_q_heads, head_dim]`.
    keys: `[max_num_tokens, num_kv_heads, head_dim]`.
    values: `[max_num_tokens, num_kv_heads, head_dim]`.
    kv_cache: `[pages, page_size, kv_heads_x2 // packing, packing, head_dim]`.
    kv_lens: `i32[max_num_seqs]`.
    page_indices: `i32[max_num_seqs * pages_per_seq]`.
    cu_q_lens: `i32[max_num_seqs + 1]`.
    distribution: `i32[3]` decode / prefill / mixed batch split.
    implementation: `TPU_KERNEL`, `REFERENCE`, or None to choose by platform.
    use_causal_mask: whether to apply causal masking.
    update_kv_cache: whether to write the new keys and values into the cache.
      Shared-KV layers pass False along with zero-filled keys and values.
    sm_scale: softmax scale.
    sliding_window: optional local attention window.
    soft_cap: optional tanh logit soft-cap.
    out_dtype: output dtype, defaulting to the query dtype.
    mask_value: value used for masked logits.
    q_scale: optional query dequantization scale.
    k_scale: optional key dequantization scale.
    v_scale: optional value dequantization scale.
    **kernel_kwargs: TPU-kernel-only tuning parameters, such as
      `chunk_prefill_size`, `d_block_sizes` and `vmem_limit_bytes`. These are
      dropped on the reference path, where they have no meaning:
      `chunk_prefill_size` only selects a statically shaped prefill pass and
      does not change the math, and the block sizes are Mosaic tiling hints.

  Returns:
    A `(attention_output, kv_cache)` pair, with the cache in the same 5D layout
    it was passed in.
  """
  if implementation is None:
    implementation = select_implementation()

  common = dict(
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

  if implementation == TPU_KERNEL:
    return _load_tpu_kernel()(
        queries,
        keys,
        values,
        kv_cache,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        **common,
        **kernel_kwargs,
    )
  if implementation == REFERENCE:
    return _reference_ragged_paged_attention(
        queries,
        keys,
        values,
        kv_cache,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        **common,
    )
  raise ValueError(
      f"Unknown ragged paged attention implementation {implementation!r}."
      f" Expected one of ({TPU_KERNEL!r}, {REFERENCE!r})."
  )

