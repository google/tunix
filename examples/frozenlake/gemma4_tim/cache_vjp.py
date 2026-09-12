"""Gemma paged-cache pullback candidate; not registered with serving/training.

Unlike the Qwen append-only wrapper, this reference includes local windows
and shared layers which read, but never overwrite, the producer's cache.
The primal is always the supplied kernel. CPU tests certify construction of
the pullback, not equivalence with an installed TPU kernel.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np


def validate_metadata(*, token_capacity, cache_shape, kv_lens, page_tables, cu_q_lens):
  """Host admission before tracing; no device-to-host transfer in the VJP.

  Tables are [sequence, logical-page]. This first candidate deliberately
  rejects active shared physical pages; supporting serving prefix aliases
  requires a separately tested ownership contract.
  """
  lengths, tables, bounds = map(np.asarray, (kv_lens, page_tables, cu_q_lens))
  if (len(cache_shape) != 5 or cache_shape[3] != 2 or min(cache_shape) < 1
      or token_capacity < 1 or lengths.ndim != 1 or lengths.size == 0
      or tables.ndim != 2 or tables.shape[0] != lengths.size or tables.shape[1] == 0
      or bounds.shape != (lengths.size + 1,)
      or any(not np.issubdtype(a.dtype, np.integer) for a in (lengths, tables, bounds))):
    raise ValueError("invalid paged-cache metadata shapes or integer types")
  q_lengths = np.diff(bounds)
  if (bounds[0] != 0 or bounds[-1] > token_capacity or np.any(q_lengths < 0)
      or np.any(lengths < q_lengths) or np.any(lengths < 0)
      or np.any(lengths > tables.shape[1] * cache_shape[1])):
    raise ValueError("query/cache lengths exceed the admitted capacity")
  seen = set()
  for length, q_length, table in zip(lengths, q_lengths, tables, strict=True):
    # Empty scheduler slots neither read nor write their table.
    if q_length == 0:
      continue
    used = table[:(int(length) + cache_shape[1] - 1) // cache_shape[1]]
    for page in used:
      page = int(page)
      if page < 0 or page >= cache_shape[0] or page in seen:
        raise ValueError("active page is out of range or aliases another active page")
      seen.add(page)


def dense_paged_reference(q, k, v, cache, kv_lens, page_tables, cu_q_lens, *,
                          update_cache: bool, window: int | None,
                          sm_scale: float = 1.0, compute_dtype=jnp.float32):
  """Differentiable mathematical reference; never substitutes the RPA primal.

  Operands are shard-local: derive GQA replication from actual head counts,
  not a global factory constant. Caller must first validate host metadata.
  """
  kv_lens, page_tables, cu_q_lens = map(jnp.asarray, (kv_lens, page_tables, cu_q_lens))
  if (q.ndim != 3 or k.shape != v.shape or k.ndim != 3
      or cache.ndim != 5 or cache.shape[3] != 2
      or q.shape[0] != k.shape[0] or q.shape[-1] != k.shape[-1]
      or cache.shape[2] != k.shape[1] or cache.shape[-1] != q.shape[-1]
      or q.shape[1] % k.shape[1] or q.shape[0] < 1
      or page_tables.ndim != 2 or kv_lens.shape != (page_tables.shape[0],)
      or cu_q_lens.shape != (page_tables.shape[0] + 1,)
      or (window is not None and (not isinstance(window, int) or window < 1))):
    raise ValueError("invalid Gemma paged attention geometry")
  rows = jnp.arange(q.shape[0], dtype=jnp.int32)
  page_size = cache.shape[1]
  new_cache = cache
  if update_cache:
    def write_sequence(seq, current):
      start, end = cu_q_lens[seq], cu_q_lens[seq + 1]
      positions = kv_lens[seq] - (end - start) + rows - start
      live = (rows >= start) & (rows < end)
      logical_page = jnp.clip(positions // page_size, 0, page_tables.shape[1] - 1)
      physical = page_tables[seq, logical_page]
      # Out-of-bounds DROP, not -1 (which would wrap and overwrite a live page).
      physical = jnp.where(live, physical, cache.shape[0])
      values = jnp.stack((k, v), axis=2).astype(cache.dtype)
      return current.at[physical, positions % page_size].set(values, mode="drop")
    new_cache = jax.lax.fori_loop(0, kv_lens.shape[0], write_sequence, cache)

  columns = jnp.arange(page_tables.shape[1] * page_size, dtype=jnp.int32)
  group = q.shape[1] // k.shape[1]
  def attend_sequence(seq, output):
    start, end = cu_q_lens[seq], cu_q_lens[seq + 1]
    live_rows = (rows >= start) & (rows < end)
    positions = kv_lens[seq] - (end - start) + rows - start
    live_cols = columns < kv_lens[seq]
    table = jnp.clip(page_tables[seq], 0, cache.shape[0] - 1)
    packed = new_cache[table].reshape(-1, cache.shape[2], 2, cache.shape[-1])
    keys = jnp.where(live_cols[:, None, None], packed[:, :, 0], 0)
    values = jnp.where(live_cols[:, None, None], packed[:, :, 1], 0)
    keys = jnp.repeat(keys.astype(compute_dtype), group, axis=1)
    values = jnp.repeat(values.astype(compute_dtype), group, axis=1)
    queries = jnp.where(live_rows[:, None, None], q, 0).astype(compute_dtype)
    scores = jnp.einsum("thd,shd->hts", queries, keys,
                        preferred_element_type=compute_dtype) * sm_scale
    keep = live_rows[:, None] & live_cols[None, :] & (columns[None, :] <= positions[:, None])
    if window is not None:
      keep &= columns[None, :] > positions[:, None] - window
    scores = jnp.where(keep[None], scores, jnp.finfo(compute_dtype).min)
    probs = jax.nn.softmax(scores, axis=-1)
    probs = jnp.where(live_rows[None, :, None], probs, 0)
    attended = jnp.einsum("hts,shd->thd", probs, values,
                          preferred_element_type=compute_dtype)
    return output + jnp.where(live_rows[:, None, None], attended, 0).astype(q.dtype)
  out = jax.lax.fori_loop(0, kv_lens.shape[0], attend_sequence, jnp.zeros_like(q))
  return out, new_cache


def make_cache_vjp(kernel: Callable, *, update_cache: bool, window: int | None,
                   sm_scale: float = 1.0, compute_dtype=jnp.float32):
  """Wrap an explicitly bound kernel; no environment flags or global patches.

  Signature: Q,K,V,cache,kv_lengths,[sequence,page] tables,query boundaries.
  The kernel must return (attention, new_cache). No distribution metadata is
  invented here: binding to the installed ragged kernel remains a target gate.
  """
  @jax.custom_vjp
  def wrapped(q, k, v, cache, lengths, tables, bounds):
    return kernel(q, k, v, cache, lengths, tables, bounds)

  def forward(q, k, v, cache, lengths, tables, bounds):
    result = kernel(q, k, v, cache, lengths, tables, bounds)
    return result, (q, k, v, cache, lengths, tables, bounds)

  def backward(saved, cotangents):
    q, k, v, cache, lengths, tables, bounds = saved
    def reference(q_, k_, v_, cache_):
      return dense_paged_reference(
          q_, k_, v_, cache_, lengths, tables, bounds,
          update_cache=update_cache, window=window,
          sm_scale=sm_scale, compute_dtype=compute_dtype)
    _, pullback = jax.vjp(reference, q, k, v, cache)
    # Differentiate BOTH outputs. For a read-only layer this includes the
    # identity path from cache output as well as its attention contribution.
    dq, dk, dv, dc = pullback(cotangents)
    return dq, dk, dv, dc, None, None, None

  wrapped.defvjp(forward, backward)
  return wrapped
