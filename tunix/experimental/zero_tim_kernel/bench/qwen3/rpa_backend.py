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

"""Ragged-paged-attention (RPA v3) backend of the Qwen3 bench attention.

The engine side of Zero-TIM runs its attention on tpu_inference's ragged
paged attention kernel (RPA v3) under the canonical call-site controls of
`rpa_canonical`: the pinned block sizes `(128, 512, 128, 512)` for all three
kernel passes, the forced-mixed `distribution`, the 256-token row bucket. The
bench's attention contract (`canon_attention.PALLAS_FWD`) is a different
kernel (dense KV, `128 x 128` tiles, exp2 online softmax in f32), so a bench
row and an engine row of the same query are not expected to agree bitwise.
This module lets the bench run its three programs -- KV-cached sampler
prefill + decode, cacheless scoring prefill, learner forward + backward -- on
the engine kernel itself (`canon_attention.ATTENTION_FWD = RPA_FWD`), so that
the cross-program question ("is the sampler's attention row the scoring
prefill's row and the learner's row, on the engine kernel?") and the
cross-kernel gap (RPA vs `flash_fwd_pallas` on the same dense operands) can be
measured on one TPU host without an engine.

Adapter (dense <-> paged):

* The bench keeps a dense per-layer cache `{'k': [B, S, KH, D], 'v': ...,
  'end_index': [B]}` and attends with an explicit `[B, T, S]` mask. The RPA
  kernel keeps one combined paged cache `[pages, page_size, KH, 2, D]` (bf16:
  packing 2; `[..., h, 0, :]` is `K_h`, `[..., h, 1, :]` is `V_h`), writes
  the new rows of every sequence to positions `[kv_len - q_len, kv_len)` of
  its page table and attends causally over `[0, kv_len)` on absolute
  positions. `attention_cached` converts the fresh dense cache to the paged
  layout on first use and then keeps the paged array in `LayerCache['k']`
  (`'v'` holds a placeholder so the cache pytree keeps the stock keys);
  sequence `b` owns the contiguous pages `[b * pages_per_seq, (b + 1) *
  pages_per_seq)`, `pages_per_seq = cache_size / page_size`, and its
  `kv_len` is `end_index + T` (the bench's unpadded, left-aligned sequences).
  `attention_cacheless` gives every sequence `cdiv(T, page_size)` fresh zero
  pages and `kv_len = q_len = T` (the kernel writes the rows it then attends
  to; the cache is discarded).
* Token rows: queries / keys / values are flattened to `[rows, H, D]`,
  `rows = round_up(B * T, TOKEN_BUCKET)`, `cu_q_lens = [0, T, 2T, ...]`.
  Rows past `B * T` belong to no sequence (the kernel leaves them untouched;
  they are sliced away).
* `distribution`: the natural split is `[n_decode, n_decode, B]` (`T == 1`
  sequences take the decode pass, everything else the mixed pass; the
  prefill-only pass needs `chunk_prefill_size` and is never used). With
  `FORCE_MIXED` it is rewritten to `[0, 0, B]` (`rpa_canonical`), so the
  decode step runs the same kernel pass as the prefill.
* TP: when a mesh with `tp_axis` is active the kernel runs inside a
  `jax.shard_map` over the heads (`P(None, None, 'tp', None)` for q / k / v,
  `P(None, None, 'tp', None, None)` for the paged cache, replicated
  metadata), exactly like the dense path.
* Gradient: the kernel has no JVP; the local call is wrapped in
  `rpa_diff_chunked.make_diff_rpa_chunked_ragged` (forward = kernel verbatim,
  backward = the chunked-cache replica VJP, the engine bundle's `VJP2`). Its
  backward unrolls only the first `CANON_VJP2_MAX_SEQS` (default 1) sequences;
  a learner batch of `B` sequences in one call needs `CANON_VJP2_MAX_SEQS >= B`
  (checked at trace time with a warning, not an error: the sampler's forward
  shares this code path and never differentiates).

Contract accepted by the model glue: unpadded left-aligned sequences, prefix
causal attention (`attn_mask` must be the mask implied by the positions; it is
not re-validated), no `segment_ids`, bf16 / f16 operands (the chunked VJP
assumes the packing-2 cache layout), `head_dim` a multiple of 128,
`page_size | 512` with the pinned blocks and `cache_size % page_size == 0`.

The kernel is imported lazily from tpu_inference (`load_kernel`); CPU tests
inject `reference_kernel`, a jittable pure-JAX replica of the kernel's
*contract* (page tables, positions, causal prefix, in-place cache write) in
f32 arithmetic -- not of its accumulator numerics (the real kernel keeps its
online-softmax state `m / l / acc` in `out_dtype`, which defaults to the query
dtype and must keep the query dtype's width because the output is stored
through the query block buffer -- so bf16 queries mean a bf16 state -- and it
uses `exp`, not `exp2`).
"""

from __future__ import annotations

from collections.abc import Callable
import dataclasses
import functools
import importlib
import os
from typing import Any

from absl import logging
import jax
from jax import lax
from jax import numpy as jnp
from jax.sharding import PartitionSpec as P
from tunix.experimental.zero_tim_kernel import rpa_canonical
from tunix.experimental.zero_tim_kernel import rpa_diff_chunked

# Engine page size (the TPU engine's default block size). Any multiple of the
# dtype packing that divides the pinned `bkv = 512` is admitted.
PAGE_SIZE = 16

# Token rows of one kernel call are padded to a multiple of this (the engine
# bundle's `MIN_TOKEN_BUCKET`): a `B=4` decode step and a `B=4, T=64` prefill
# call the kernel at the same `max_num_tokens`.
TOKEN_BUCKET = rpa_canonical.CANONICAL_MIN_TOKEN_BUCKET

# Pinned `(bq_sz, bkv_sz, bq_csz, bkv_csz) = (128, 512, 128, 512)` for the
# decode / prefill / mixed passes (`rpa_canonical.CANONICAL_BLOCK_SIZES`);
# False leaves the kernel's shape-dependent heuristic in charge.
PIN_BLOCK_SIZES = True

# Every sequence takes the mixed pass (`distribution = [0, 0, B]`); False keeps
# the natural split, which only differs for `T == 1` (decode) calls.
FORCE_MIXED = True

# Kernel accumulator dtype (`out_dtype`); None is the kernel default (the
# query dtype). The kernel stores the output through its query block buffer,
# so the dtype must have the query dtype's width (`_validate` rejects e.g.
# float32 with bf16 queries).
OUT_DTYPE: Any = None

# The RPA kernel entry point; None loads it from tpu_inference on first use.
# Tests inject `reference_kernel` on CPU.
KERNEL: Callable[..., Any] | None = None

KERNEL_MODULE = 'tpu_inference.kernels.ragged_paged_attention.v3.kernel'
KERNEL_NAME = 'ragged_paged_attention'

# `LayerCache` keys. The paged cache replaces the dense `'k'`; `'v'` becomes a
# placeholder so the cache pytree keeps the stock keys.
PAGED_CACHE_KEY = 'k'
PLACEHOLDER_KEY = 'v'
END_INDEX_KEY = 'end_index'

# The kernel pads the head dimension of the cache to a multiple of this.
HEAD_DIM_ALIGN = 128

_loaded_kernel: Callable[..., Any] | None = None
_warned: set[str] = set()


def _warn_once(tag: str, message: str, *args) -> None:
  if tag not in _warned:
    _warned.add(tag)
    logging.warning(message, *args)


def load_kernel() -> Callable[..., Any]:
  """Imports `tpu_inference`'s RPA v3 kernel (cached).

  Returns:
    `ragged_paged_attention(q, k, v, kv_cache, kv_lens, page_indices,
    cu_q_lens, distribution, *, sm_scale, ...) -> (out, new_kv_cache)`.

  Raises:
    ImportError: If tpu_inference is not importable (CPU test environments);
      inject `reference_kernel` through `KERNEL` or the `kernel=` argument.
  """
  global _loaded_kernel
  if _loaded_kernel is None:
    try:
      module = importlib.import_module(KERNEL_MODULE)
    except ImportError as e:
      raise ImportError(
          f'{KERNEL_MODULE} is not importable: the RPA backend needs'
          ' tpu_inference on TPU, or an injected kernel (e.g.'
          ' `rpa_backend.reference_kernel`) on CPU'
      ) from e
    _loaded_kernel = getattr(module, KERNEL_NAME)
  return _loaded_kernel


def resolve_kernel(kernel: Callable[..., Any] | None = None):
  """`kernel`, else the injected `KERNEL`, else the tpu_inference kernel."""
  if kernel is not None:
    return kernel
  if KERNEL is not None:
    return KERNEL
  return load_kernel()


def round_up(x: int, multiple: int) -> int:
  return -(-x // multiple) * multiple


def cdiv(a: int, b: int) -> int:
  return -(-a // b)


def kv_packing(dtype) -> int:
  """Elements per 32-bit word (`2` for bf16 / f16, `1` for f32)."""
  return 32 // (jnp.dtype(dtype).itemsize * 8)


def paged_cache_shape(
    num_pages: int,
    page_size: int,
    num_kv_heads: int,
    head_dim: int,
    dtype,
) -> tuple[int, int, int, int, int]:
  """The kernel's combined KV cache shape (`get_kv_cache_shape`)."""
  packing = kv_packing(dtype)
  return (
      num_pages,
      page_size,
      round_up(2 * num_kv_heads, packing) // packing,
      packing,
      round_up(head_dim, HEAD_DIM_ALIGN),
  )


def merge_kv(k: jax.Array, v: jax.Array) -> jax.Array:
  """`[..., KH, D]` keys and values -> `[..., 2KH / packing, packing, Dp]`.

  The kernel's `merge_kv` layout: flattened combined-head index `2h` is `K_h`
  and `2h + 1` is `V_h`; the head dimension is zero-padded to `Dp =
  round_up(D, 128)`.

  Args:
    k: Keys `[..., KH, D]`.
    v: Values, same shape and dtype as `k`.

  Returns:
    The combined array `[..., round_up(2KH, packing) / packing, packing, Dp]`.
  """
  if k.shape != v.shape or k.dtype != v.dtype:
    raise ValueError(f'k {k.shape}/{k.dtype} vs v {v.shape}/{v.dtype}')
  *lead, kh, d = (int(s) for s in k.shape)
  packing = kv_packing(k.dtype)
  n2 = round_up(2 * kh, packing)
  dp = round_up(d, HEAD_DIM_ALIGN)
  kv = jnp.concatenate([k, v], axis=-1).reshape(*lead, 2 * kh, d)
  pad = [(0, 0)] * len(lead) + [(0, n2 - 2 * kh), (0, dp - d)]
  kv = jnp.pad(kv, pad)
  return kv.reshape(*lead, n2 // packing, packing, dp)


def split_kv(
    kv: jax.Array, num_kv_heads: int, head_dim: int
) -> tuple[jax.Array, jax.Array]:
  """Inverse of `merge_kv`: `(k, v)` of shape `[..., KH, D]`."""
  *lead, n2p, packing, dp = (int(s) for s in kv.shape)
  flat = kv.reshape(*lead, n2p * packing, dp)[
      ..., : 2 * num_kv_heads, :head_dim
  ]
  flat = flat.reshape(*lead, num_kv_heads, 2 * head_dim)
  return flat[..., :head_dim], flat[..., head_dim:]


def dense_to_paged(
    k_dense: jax.Array, v_dense: jax.Array, page_size: int
) -> jax.Array:
  """Dense `[B, S, KH, D]` k / v -> paged `[B * S / page_size, page_size, ...]`.

  Sequence `b` owns pages `[b * S / page_size, (b + 1) * S / page_size)` in
  order, so cache slot `s` of sequence `b` is page `b * S / page_size + s //
  page_size`, row `s % page_size`.

  Args:
    k_dense: Dense keys `[B, S, KH, D]`.
    v_dense: Dense values, same shape and dtype.
    page_size: Rows per page; must divide `S`.

  Returns:
    The paged cache `[B * S / page_size, page_size, 2KH / packing, packing,
    Dp]`.
  """
  b, s = int(k_dense.shape[0]), int(k_dense.shape[1])
  if s % page_size:
    raise ValueError(
        f'cache_size {s} must be a multiple of page_size {page_size}'
    )
  kv = merge_kv(k_dense, v_dense)  # [B, S, n2p, packing, Dp]
  return kv.reshape(b * (s // page_size), page_size, *kv.shape[2:])


def paged_to_dense(
    paged: jax.Array, batch_size: int, num_kv_heads: int, head_dim: int
) -> tuple[jax.Array, jax.Array]:
  """Inverse of `dense_to_paged`: `(k, v)` of shape `[B, S, KH, D]`."""
  num_pages, page_size = int(paged.shape[0]), int(paged.shape[1])
  if num_pages % batch_size:
    raise ValueError(f'{num_pages} pages are not {batch_size} equal tables')
  s = (num_pages // batch_size) * page_size
  kv = paged.reshape(batch_size, s, *paged.shape[2:])
  return split_kv(kv, num_kv_heads, head_dim)


@dataclasses.dataclass(frozen=True)
class Metadata:
  """The kernel's int32 metadata (`max_num_seqs = B`, `pages_per_seq` pages)."""

  kv_lens: jax.Array  # [B]
  page_indices: jax.Array  # [B * pages_per_seq]
  cu_q_lens: jax.Array  # [B + 1]
  distribution: jax.Array  # [3]


def build_metadata(
    *,
    batch_size: int,
    seq_len: int,
    pages_per_seq: int,
    kv_lens: jax.Array,
    force_mixed: bool,
) -> Metadata:
  """Metadata of `B` left-aligned sequences of `T = seq_len` query rows each.

  Args:
    batch_size: `B`; also the kernel's `max_num_seqs`.
    seq_len: `T`, the query rows of every sequence (`cu_q_lens = [0, T, 2T,
      ...]`).
    pages_per_seq: Page table length of every sequence; sequence `b` owns pages
      `[b * pages_per_seq, (b + 1) * pages_per_seq)`.
    kv_lens: int32 `[B]` total KV length of every sequence (context + the `T`
      new rows); may be traced.
    force_mixed: Rewrite the natural `distribution` to `[0, 0, B]`.

  Returns:
    The `Metadata`.
  """
  kv_lens = jnp.asarray(kv_lens, jnp.int32)
  if kv_lens.shape != (batch_size,):
    raise ValueError(f'kv_lens {kv_lens.shape} must be [{batch_size}]')
  cu_q_lens = jnp.arange(batch_size + 1, dtype=jnp.int32) * seq_len
  page_indices = jnp.arange(batch_size * pages_per_seq, dtype=jnp.int32)
  n_decode = batch_size if seq_len == 1 else 0
  distribution = jnp.array([n_decode, n_decode, batch_size], jnp.int32)
  if force_mixed:
    distribution = rpa_canonical.force_mixed_distribution(distribution)
  return Metadata(kv_lens, page_indices, cu_q_lens, distribution)


def token_rows(batch_size: int, seq_len: int, token_bucket: int) -> int:
  """`max_num_tokens` of a `[B, T]` call: `B * T` rounded up to the bucket."""
  return round_up(batch_size * seq_len, token_bucket)


def flatten_tokens(x: jax.Array, rows: int) -> jax.Array:
  """`[B, T, H, D]` -> `[rows, H, D]`, sequences back to back, zero rows after."""
  b, t, h, d = (int(s) for s in x.shape)
  flat = x.reshape(b * t, h, d)
  if rows < b * t:
    raise ValueError(f'{rows} rows < B * T = {b * t}')
  if rows > b * t:
    flat = jnp.pad(flat, ((0, rows - b * t), (0, 0), (0, 0)))
  return flat


def unflatten_tokens(x: jax.Array, batch_size: int, seq_len: int) -> jax.Array:
  """Inverse of `flatten_tokens` (drops the padding rows)."""
  return x[: batch_size * seq_len].reshape(batch_size, seq_len, *x.shape[1:])


def _tp_size(mesh: jax.sharding.Mesh | None, tp_axis: str) -> int:
  if mesh is None or mesh.empty or tp_axis not in mesh.shape:
    return 1
  return int(mesh.shape[tp_axis])


def kernel_kwargs(
    *,
    scale: float,
    pin_block_sizes: bool,
    out_dtype: Any,
) -> dict[str, Any]:
  """The static keyword arguments of the kernel call."""
  kwargs: dict[str, Any] = {'sm_scale': scale}
  if pin_block_sizes:
    kwargs.update(rpa_canonical.canonical_block_size_kwargs())
  if out_dtype is not None:
    kwargs['out_dtype'] = jnp.dtype(out_dtype)
  return kwargs


def call_kernel(
    kernel: Callable[..., Any],
    q_rows: jax.Array,
    k_rows: jax.Array,
    v_rows: jax.Array,
    cache: jax.Array,
    md: Metadata,
    *,
    scale: float,
    page_size: int,
    pin_block_sizes: bool,
    out_dtype: Any,
    differentiable: bool,
    max_q_len: int | None = None,
) -> tuple[jax.Array, jax.Array]:
  """One shard-local kernel call in the engine form.

  Args:
    kernel: The RPA kernel (8 positional arrays, keyword controls).
    q_rows: Queries `[rows, QH, D]`.
    k_rows: Keys `[rows, KH, D]`.
    v_rows: Values `[rows, KH, D]`.
    cache: Paged cache `[pages, page_size, 2KH / packing, packing, Dp]`.
    md: The metadata.
    scale: Softmax scale (`sm_scale`).
    page_size: Rows per page.
    pin_block_sizes: Pass the canonical `{d,p,m}_block_sizes`.
    out_dtype: Kernel accumulator dtype, or None for the kernel default.
    differentiable: Wrap the call in the chunked replica VJP
      (`rpa_diff_chunked.make_diff_rpa_chunked_ragged`).
    max_q_len: Static per-sequence query length handed to the chunked VJP (the
      backward differentiates each sequence on a window of this many rows
      instead of the whole `rows` buffer); None keeps the full width.

  Returns:
    `(out_rows, new_cache)`; `out_rows` is `[rows, QH, D]` in the query dtype.
  """
  fn = functools.partial(
      kernel,
      **kernel_kwargs(
          scale=scale, pin_block_sizes=pin_block_sizes, out_dtype=out_dtype
      ),
  )
  if differentiable:
    fn = rpa_diff_chunked.make_diff_rpa_chunked_ragged(
        fn,
        sm_scale=scale,
        page_size=page_size,
        num_q_heads=int(q_rows.shape[1]),
        num_kv_heads=int(k_rows.shape[1]),
        max_q_len=max_q_len,
    )
  return fn(
      q_rows,
      k_rows,
      v_rows,
      cache,
      md.kv_lens,
      md.page_indices,
      md.cu_q_lens,
      md.distribution,
  )


def _check_rank(q: jax.Array, k: jax.Array, v: jax.Array) -> None:
  if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
    raise ValueError(
        f'expected [B, T, H, D]: q {q.shape} k {k.shape} v {v.shape}'
    )


def _validate(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    cache: jax.Array,
    *,
    page_size: int,
    pin_block_sizes: bool,
    out_dtype: Any,
    differentiable: bool,
) -> None:
  """Static (shape / dtype) checks of one adapter call; fail closed."""
  _check_rank(q, k, v)
  b, t, qh, d = (int(s) for s in q.shape)
  if k.shape != v.shape or tuple(k.shape[:2]) != (b, t) or k.shape[3] != d:
    raise ValueError(f'q {q.shape} vs k {k.shape} vs v {v.shape}')
  kh = int(k.shape[2])
  if qh % kh:
    raise ValueError(f'{qh} query heads are not a multiple of {kh} KV heads')
  if not (q.dtype == k.dtype == v.dtype == cache.dtype):
    raise ValueError(
        f'dtypes differ: q {q.dtype} k {k.dtype} v {v.dtype} cache'
        f' {cache.dtype}'
    )
  if (
      out_dtype is not None
      and jnp.dtype(out_dtype).itemsize != q.dtype.itemsize
  ):
    raise ValueError(
        f'out_dtype {jnp.dtype(out_dtype)} must have the width of the query'
        f' dtype {q.dtype}: the kernel stores the output through its query'
        ' block buffer'
    )
  if d % HEAD_DIM_ALIGN:
    raise ValueError(f'head_dim {d} must be a multiple of {HEAD_DIM_ALIGN}')
  packing = kv_packing(cache.dtype)
  if differentiable and packing != 2:
    raise NotImplementedError(
        'the chunked RPA VJP assumes the packing-2 (bf16 / f16) cache layout'
        f' [pages, page_size, KH, 2, D]; got {cache.dtype}'
    )
  if page_size % packing:
    raise ValueError(f'page_size {page_size} must be a multiple of {packing}')
  if pin_block_sizes and rpa_canonical.CANONICAL_BLOCK_SIZES[1] % page_size:
    raise ValueError(
        f'page_size {page_size} must divide the pinned bkv'
        f' {rpa_canonical.CANONICAL_BLOCK_SIZES[1]}'
    )
  if cache.ndim != 5 or int(cache.shape[0]) % b:
    raise ValueError(
        f'paged cache {cache.shape} is not [B * pages_per_seq, ...] for B={b}'
    )
  expected = paged_cache_shape(
      int(cache.shape[0]), page_size, kh, d, cache.dtype
  )
  if tuple(cache.shape) != expected:
    raise ValueError(f'paged cache {cache.shape} != {expected}')


def _check_vjp_unroll(batch_size: int) -> None:
  """Warns when the chunked VJP would drop the gradient of some sequences."""
  n_active = int(os.environ.get(rpa_diff_chunked.MAX_SEQS_ENV, '1'))
  if n_active < batch_size:
    _warn_once(
        f'vjp_unroll_{batch_size}',
        '%s=%d < B=%d: the chunked RPA backward unrolls only the first %d'
        ' sequences (the other sequences get zero q / k / v cotangents);'
        ' forward-only programs are unaffected.',
        rpa_diff_chunked.MAX_SEQS_ENV,
        n_active,
        batch_size,
        n_active,
    )


def _attend(
    q: jax.Array,  # [B, T, QH, D]
    k: jax.Array,  # [B, T, KH, D]
    v: jax.Array,  # [B, T, KH, D]
    cache: jax.Array,  # [B * pages_per_seq, page_size, 2KH / pk, pk, Dp]
    kv_lens: jax.Array,  # [B] int32
    *,
    scale: float,
    page_size: int,
    token_bucket: int,
    pin_block_sizes: bool,
    force_mixed: bool,
    out_dtype: Any,
    kernel: Callable[..., Any],
    differentiable: bool,
    mesh: jax.sharding.Mesh | None,
    tp_axis: str,
) -> tuple[jax.Array, jax.Array]:
  """The shared body: metadata, token rows, TP map, kernel call."""
  _validate(
      q,
      k,
      v,
      cache,
      page_size=page_size,
      pin_block_sizes=pin_block_sizes,
      out_dtype=out_dtype,
      differentiable=differentiable,
  )
  b, t, qh, _ = (int(s) for s in q.shape)
  kh = int(k.shape[2])
  pages_per_seq = int(cache.shape[0]) // b
  rows = token_rows(b, t, token_bucket)
  md = build_metadata(
      batch_size=b,
      seq_len=t,
      pages_per_seq=pages_per_seq,
      kv_lens=kv_lens,
      force_mixed=force_mixed,
  )
  if differentiable:
    _check_vjp_unroll(b)

  def local(q_l, k_l, v_l, cache_l, kv_lens_l, pages_l, cu_q_lens_l, dist_l):
    out_rows, new_cache = call_kernel(
        kernel,
        flatten_tokens(q_l, rows),
        flatten_tokens(k_l, rows),
        flatten_tokens(v_l, rows),
        cache_l,
        Metadata(kv_lens_l, pages_l, cu_q_lens_l, dist_l),
        scale=scale,
        page_size=page_size,
        pin_block_sizes=pin_block_sizes,
        out_dtype=out_dtype,
        differentiable=differentiable,
        max_q_len=t,
    )
    return unflatten_tokens(out_rows, b, t), new_cache

  tp = _tp_size(mesh, tp_axis)
  if tp > 1:
    if qh % tp or kh % tp:
      raise ValueError(
          f'QH={qh}, KH={kh} are not multiples of the TP degree {tp}; the RPA'
          ' backend has no replicated fallback'
      )
    head_spec = P(None, None, tp_axis, None)
    cache_spec = P(None, None, tp_axis, None, None)
    local = jax.shard_map(
        local,
        mesh=mesh,
        in_specs=(
            head_spec,
            head_spec,
            head_spec,
            cache_spec,
            P(),
            P(),
            P(),
            P(),
        ),
        out_specs=(head_spec, cache_spec),
        check_vma=False,
    )
  return local(
      q, k, v, cache, md.kv_lens, md.page_indices, md.cu_q_lens, md.distribution
  )


def _knobs(page_size, token_bucket, pin_block_sizes, force_mixed, out_dtype):
  """Resolves `None` arguments to the module defaults."""
  return (
      PAGE_SIZE if page_size is None else int(page_size),
      TOKEN_BUCKET if token_bucket is None else int(token_bucket),
      PIN_BLOCK_SIZES if pin_block_sizes is None else bool(pin_block_sizes),
      FORCE_MIXED if force_mixed is None else bool(force_mixed),
      OUT_DTYPE if out_dtype is None else out_dtype,
  )


def attention_cacheless(
    q: jax.Array,  # [B, T, QH, D]
    k: jax.Array,  # [B, T, KH, D]
    v: jax.Array,  # [B, T, KH, D]
    *,
    scale: float,
    mesh: jax.sharding.Mesh | None = None,
    tp_axis: str = 'tp',
    page_size: int | None = None,
    token_bucket: int | None = None,
    pin_block_sizes: bool | None = None,
    force_mixed: bool | None = None,
    out_dtype: Any = None,
    kernel: Callable[..., Any] | None = None,
    differentiable: bool = True,
) -> jax.Array:
  """Causal self-attention of `B` sequences of `T` rows on the RPA kernel.

  Every sequence gets `cdiv(T, page_size)` fresh zero pages, `kv_len = q_len
  = T` (the scoring prefill / learner program: the kernel writes the rows and
  attends to them; the cache is discarded).

  Args:
    q: Queries `[B, T, QH, D]`.
    k: Keys `[B, T, KH, D]`.
    v: Values `[B, T, KH, D]`.
    scale: Softmax scale.
    mesh: The active mesh (TP over the heads when it has `tp_axis`).
    tp_axis: The TP mesh axis.
    page_size: Rows per page (None: `PAGE_SIZE`).
    token_bucket: Token-row bucket (None: `TOKEN_BUCKET`).
    pin_block_sizes: Pin the canonical block sizes (None: `PIN_BLOCK_SIZES`).
    force_mixed: Force the mixed pass (None: `FORCE_MIXED`).
    out_dtype: Kernel accumulator dtype (None: `OUT_DTYPE`); must have the query
      dtype's width.
    kernel: The kernel (None: `KERNEL`, else tpu_inference).
    differentiable: Wrap the call in the chunked replica VJP.

  Returns:
    `out[B, T, QH, D]` in the query dtype.
  """
  page_size, token_bucket, pin, mixed, acc_dtype = _knobs(
      page_size, token_bucket, pin_block_sizes, force_mixed, out_dtype
  )
  kernel_fn = resolve_kernel(kernel)
  _check_rank(q, k, v)
  b, t, _, d = (int(s) for s in q.shape)
  kh = int(k.shape[2])
  pages_per_seq = cdiv(t, page_size)
  cache = jnp.zeros(
      paged_cache_shape(b * pages_per_seq, page_size, kh, d, k.dtype), k.dtype
  )
  kv_lens = jnp.full((b,), t, jnp.int32)
  out, _ = _attend(
      q,
      k,
      v,
      cache,
      kv_lens,
      scale=scale,
      page_size=page_size,
      token_bucket=token_bucket,
      pin_block_sizes=pin,
      force_mixed=mixed,
      out_dtype=acc_dtype,
      kernel=kernel_fn,
      differentiable=differentiable,
      mesh=mesh,
      tp_axis=tp_axis,
  )
  return out


def attention_cached(
    layer_cache: dict[str, jax.Array],
    q: jax.Array,  # [B, T, QH, D]
    k: jax.Array,  # [B, T, KH, D]
    v: jax.Array,  # [B, T, KH, D]
    *,
    scale: float,
    mesh: jax.sharding.Mesh | None = None,
    tp_axis: str = 'tp',
    page_size: int | None = None,
    token_bucket: int | None = None,
    pin_block_sizes: bool | None = None,
    force_mixed: bool | None = None,
    out_dtype: Any = None,
    kernel: Callable[..., Any] | None = None,
    differentiable: bool = True,
) -> tuple[dict[str, jax.Array], jax.Array]:
  """KV-cached attention of the sampler (prefill `T = L`, decode `T = 1`).

  The new rows of sequence `b` are written at positions `[end_index_b,
  end_index_b + T)` of its page table and attention covers `[0, end_index_b +
  T)` causally. A fresh dense cache (`init_cache`: `'k'` / `'v'` of shape
  `[B, cache_size, KH, D]`) is converted to the paged layout on first use;
  afterwards `LayerCache['k']` holds the paged cache and `'v'` a placeholder.

  Args:
    layer_cache: `{'k', 'v', 'end_index'}` (dense or already paged).
    q: Queries `[B, T, QH, D]`.
    k: New keys `[B, T, KH, D]`.
    v: New values `[B, T, KH, D]`.
    scale: Softmax scale.
    mesh: The active mesh (TP over the heads when it has `tp_axis`).
    tp_axis: The TP mesh axis.
    page_size: Rows per page (None: `PAGE_SIZE`); must divide `cache_size`.
    token_bucket: Token-row bucket (None: `TOKEN_BUCKET`).
    pin_block_sizes: Pin the canonical block sizes (None: `PIN_BLOCK_SIZES`).
    force_mixed: Force the mixed pass (None: `FORCE_MIXED`).
    out_dtype: Kernel accumulator dtype (None: `OUT_DTYPE`); must have the query
      dtype's width.
    kernel: The kernel (None: `KERNEL`, else tpu_inference).
    differentiable: Wrap the call in the chunked replica VJP (the sampler never
      differentiates; kept on for one code path with the learner).

  Returns:
    `(new_layer_cache, out[B, T, QH, D])`; `new_layer_cache['end_index']` is
    advanced by `T`.
  """
  page_size, token_bucket, pin, mixed, acc_dtype = _knobs(
      page_size, token_bucket, pin_block_sizes, force_mixed, out_dtype
  )
  kernel_fn = resolve_kernel(kernel)
  _check_rank(q, k, v)
  end_index = jnp.asarray(layer_cache[END_INDEX_KEY], jnp.int32)
  stored = layer_cache[PAGED_CACHE_KEY]
  b, t = int(q.shape[0]), int(q.shape[1])
  if end_index.shape != (b,):
    raise ValueError(f'end_index {end_index.shape} must be [{b}]')
  if stored.ndim == 4:
    dense_v = layer_cache[PLACEHOLDER_KEY]
    if dense_v.shape != stored.shape:
      raise ValueError(f"dense cache 'k' {stored.shape} vs 'v' {dense_v.shape}")
    if int(stored.shape[0]) != b:
      raise ValueError(f'cache batch {stored.shape[0]} != B={b}')
    tp = _tp_size(mesh, tp_axis)
    convert = functools.partial(dense_to_paged, page_size=page_size)
    if tp > 1:
      head_spec = P(None, None, tp_axis, None)
      convert = jax.shard_map(
          convert,
          mesh=mesh,
          in_specs=(head_spec, head_spec),
          out_specs=P(None, None, tp_axis, None, None),
          check_vma=False,
      )
    cache = convert(stored, dense_v)
  elif stored.ndim == 5:
    cache = stored
    if int(cache.shape[1]) != page_size:
      raise ValueError(f'paged cache page_size {cache.shape[1]} != {page_size}')
  else:
    raise ValueError(f'unexpected cache rank {stored.ndim}: {stored.shape}')
  out, new_cache = _attend(
      q,
      k,
      v,
      cache,
      end_index + t,
      scale=scale,
      page_size=page_size,
      token_bucket=token_bucket,
      pin_block_sizes=pin,
      force_mixed=mixed,
      out_dtype=acc_dtype,
      kernel=kernel_fn,
      differentiable=differentiable,
      mesh=mesh,
      tp_axis=tp_axis,
  )
  new_layer_cache = {
      PAGED_CACHE_KEY: new_cache,
      PLACEHOLDER_KEY: jnp.zeros((), new_cache.dtype),
      END_INDEX_KEY: end_index + t,
  }
  return new_layer_cache, out


def describe(
    *,
    page_size: int | None = None,
    token_bucket: int | None = None,
    pin_block_sizes: bool | None = None,
    force_mixed: bool | None = None,
    out_dtype: Any = None,
) -> dict[str, Any]:
  """The effective call-site controls (for reports); `None` = module default."""
  page_size, token_bucket, pin, mixed, acc_dtype = _knobs(
      page_size, token_bucket, pin_block_sizes, force_mixed, out_dtype
  )
  return {
      'page_size': page_size,
      'token_bucket': token_bucket,
      'block_sizes': (
          rpa_canonical.CANONICAL_BLOCK_SIZES if pin else 'kernel heuristic'
      ),
      'distribution': '[0, 0, B] (forced mixed)' if mixed else 'natural',
      'out_dtype': (
          'kernel default' if acc_dtype is None else str(jnp.dtype(acc_dtype))
      ),
  }


def reference_kernel(
    queries: jax.Array,  # [rows, QH, D]
    keys: jax.Array,  # [rows, KH, D]
    values: jax.Array,  # [rows, KH, D]
    kv_cache: jax.Array,  # [pages, page_size, 2KH / packing, packing, Dp]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    *,
    sm_scale: float = 1.0,
    compute_dtype: Any = jnp.float32,
    **unused_kernel_kwargs,
) -> tuple[jax.Array, jax.Array]:
  """Jittable pure-JAX replica of the RPA v3 *contract* (CPU tests).

  For every sequence `i < distribution[2]` (a static loop over the
  `max_num_seqs` slots with traced masks): the rows `[cu_q_lens[i],
  cu_q_lens[i + 1])` of `keys` / `values` are written to positions `[kv_len -
  q_len, kv_len)` of the sequence's page table, then every row attends
  causally (absolute positions) over `[0, kv_len)` of the updated cache.
  Arithmetic is `compute_dtype` (f32) throughout with one rounding to the
  query dtype at the output; this is the kernel's bookkeeping, not its
  accumulator numerics. Rows that belong to no sequence come back as zeros
  (the kernel leaves them untouched). Block-size and `out_dtype` keyword
  arguments are accepted and ignored.

  Args:
    queries: Queries `[rows, QH, D]`.
    keys: Keys `[rows, KH, D]`.
    values: Values `[rows, KH, D]`.
    kv_cache: The paged cache.
    kv_lens: Total KV length per sequence slot.
    page_indices: Flattened page tables, `pages_per_seq` entries per slot.
    cu_q_lens: Cumulative query lengths.
    distribution: `[i, j, k]`; only `k` (the number of sequences) is used.
    sm_scale: Softmax scale.
    compute_dtype: Arithmetic dtype.
    **unused_kernel_kwargs: Ignored kernel controls.

  Returns:
    `(out[rows, QH, D], new_kv_cache)`.
  """
  del unused_kernel_kwargs
  n, qh, d = (int(s) for s in queries.shape)
  kh = int(keys.shape[1])
  g = qh // kh
  max_num_seqs = int(kv_lens.shape[0])
  pages_per_seq = int(page_indices.shape[0]) // max_num_seqs
  num_pages, page_size, n2p, packing, dp = (int(s) for s in kv_cache.shape)
  s_max = pages_per_seq * page_size
  num_seqs = distribution[2]
  kv_new = merge_kv(keys, values)  # [rows, n2p, packing, Dp]
  rows = jnp.arange(n, dtype=jnp.int32)
  cols = jnp.arange(s_max, dtype=jnp.int32)
  neg = jnp.finfo(compute_dtype).min
  out = jnp.zeros((n, qh, d), queries.dtype)
  q_c = queries.astype(compute_dtype).reshape(n, kh, g, d)

  for i in range(max_num_seqs):
    active = i < num_seqs
    q0 = cu_q_lens[i]
    q_len = cu_q_lens[i + 1] - q0
    kv_len = kv_lens[i]
    table = lax.dynamic_slice(
        page_indices, (i * pages_per_seq,), (pages_per_seq,)
    )
    row_ok = active & (rows >= q0) & (rows < q0 + q_len)
    pos = kv_len - q_len + (rows - q0)  # absolute position of every row
    pos_c = jnp.clip(pos, 0, s_max - 1)
    page = jnp.where(row_ok, table[pos_c // page_size], num_pages)  # OOB: drop
    kv_cache = kv_cache.at[page, pos_c % page_size].set(kv_new, mode='drop')
    ctx = kv_cache[table].reshape(s_max, n2p * packing, dp)[:, : 2 * kh]
    ctx = ctx.reshape(s_max, kh, 2 * dp)
    ctx_k = ctx[:, :, :d].astype(compute_dtype)
    ctx_v = ctx[:, :, dp : dp + d].astype(compute_dtype)
    scores = (
        jnp.einsum(
            'nhgd,shd->hgns', q_c, ctx_k, preferred_element_type=compute_dtype
        )
        * sm_scale
    )  # [KH, G, rows, s_max]
    keep = (
        row_ok[None, None, :, None]
        & (cols < kv_len)[None, None, None, :]
        & (pos[:, None] >= cols[None, :])[None, None]
    )
    probs = jax.nn.softmax(jnp.where(keep, scores, neg), axis=-1)
    probs = jnp.where(row_ok[None, None, :, None], probs, 0.0)
    o = jnp.einsum(
        'hgns,shd->nhgd', probs, ctx_v, preferred_element_type=compute_dtype
    ).reshape(n, qh, d)
    out = jnp.where(row_ok[:, None, None], o.astype(queries.dtype), out)
  return out, kv_cache
