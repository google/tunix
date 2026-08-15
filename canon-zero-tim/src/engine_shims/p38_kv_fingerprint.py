"""Bounded bit-level fingerprints for P38 KV-cache content observation.

The result is deliberately called a fingerprint, not a cryptographic hash.
It is a diagnostic observer whose collision risk must remain in the claim
ceiling. Callers validate host-visible geometry before JIT compilation.
"""

from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np


P38_SEAM_FINGERPRINT_FIELDS = (
    "xor",
    "sum",
    "weighted_sum",
    "sample_first",
    "sample_quarter",
    "sample_middle",
    "sample_three_quarters",
    "sample_last",
)


def fingerprint_tensor_rows(value):
  """Return a compact exact-integer diagnostic fingerprint per token row.

  The leading dimension is the token/program row. Every remaining dimension
  belongs to that row and is flattened. The aggregate operations are uint32
  XOR and modular addition, so the observer itself has no floating-point
  reduction freedom. Five fixed samples make the one-bit negative control
  independent of aggregate cancellation.

  This remains a diagnostic fingerprint rather than a collision-free proof.
  Callers must retain that claim ceiling and must separately prove that
  enabling the observer leaves the authoritative endpoint bitwise unchanged.
  """
  shape = tuple(int(item) for item in value.shape)
  if len(shape) < 2 or shape[0] <= 0 or any(item <= 0 for item in shape[1:]):
    raise ValueError(
        f"P38 seam fingerprint requires [token, ...] positive shape: {shape}"
    )
  dtype = jnp.dtype(value.dtype)
  if dtype in (jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float16)):
    bits = jax.lax.bitcast_convert_type(value, jnp.uint16).astype(jnp.uint32)
  elif dtype == jnp.dtype(jnp.float32):
    bits = jax.lax.bitcast_convert_type(value, jnp.uint32)
  else:
    raise ValueError(
        "P38 seam fingerprint supports bfloat16/float16/float32, "
        f"got {value.dtype}"
    )

  flat = bits.reshape((shape[0], -1))
  width = int(flat.shape[1])
  weights = jnp.arange(1, width + 1, dtype=jnp.uint32)[None, :]
  sample_indices = (0, width // 4, width // 2, (3 * width) // 4, width - 1)
  return jnp.stack(
      (
          jnp.bitwise_xor.reduce(flat, axis=1),
          jnp.sum(flat, axis=1, dtype=jnp.uint32),
          jnp.sum(flat * weights, axis=1, dtype=jnp.uint32),
          *(flat[:, index] for index in sample_indices),
      ),
      axis=1,
  )


def validate_kv_fingerprint_contract(
    cache_shape: Sequence[int], cache_dtype, valid_tokens: Sequence[int]
) -> np.ndarray:
  """Validate the static page layout and host-derived valid-token extents."""
  shape = tuple(int(value) for value in cache_shape)
  if len(shape) != 5 or any(value <= 0 for value in shape):
    raise ValueError(f"P38 KV pages require a positive rank-5 shape: {shape}")
  if shape[3] != 2:
    raise ValueError(f"P38 KV pages require a packed K/V axis of size 2: {shape}")
  if jnp.dtype(cache_dtype) != jnp.dtype(jnp.bfloat16):
    raise ValueError(f"P38 KV pages require bfloat16, got {cache_dtype}")
  extents = np.asarray(valid_tokens)
  if extents.dtype.kind not in "iu" or extents.shape != (shape[0],):
    raise ValueError(
        "P38 KV valid-token extents must be one integer per selected page"
    )
  extents = extents.astype(np.int32, copy=False)
  if np.any(extents < 1) or np.any(extents > shape[1]):
    raise ValueError(
        f"P38 KV valid-token extents exceed page size {shape[1]}: "
        f"{extents.tolist()}"
    )
  return extents


def global_page_indices(
    cache_shape: Sequence[int], dp_size: int, dp_rank: int,
    physical_pages: Sequence[int]
) -> np.ndarray:
  """Map one DP rank's local physical-page IDs into the global cache axis."""
  shape = tuple(int(value) for value in cache_shape)
  if not shape or int(dp_size) <= 0 or shape[0] % int(dp_size):
    raise ValueError(
        f"P38 KV page axis is not divisible by DP: shape={shape} dp={dp_size}"
    )
  if not 0 <= int(dp_rank) < int(dp_size):
    raise ValueError(f"P38 KV DP rank is out of range: {dp_rank}/{dp_size}")
  pages = np.asarray(physical_pages)
  if pages.ndim != 1 or pages.dtype.kind not in "iu" or pages.size == 0:
    raise ValueError("P38 KV physical pages must be a non-empty integer vector")
  pages = pages.astype(np.int32, copy=False)
  pages_per_dp = shape[0] // int(dp_size)
  if np.any(pages < 0) or np.any(pages >= pages_per_dp):
    raise ValueError(
        f"P38 KV physical page exceeds local capacity {pages_per_dp}: "
        f"{pages.tolist()}"
    )
  return pages + np.int32(int(dp_rank) * pages_per_dp)


def estimate_fingerprint_read_bytes(
    cache_shapes: Sequence[Sequence[int]], selected_pages: int
) -> int:
  """Return the upper bound of bfloat16 cache payload read by one observer."""
  if int(selected_pages) <= 0:
    raise ValueError("P38 KV observer must select at least one page")
  shapes = [tuple(int(value) for value in shape) for shape in cache_shapes]
  if not shapes:
    raise ValueError("P38 KV observer requires at least one cache layer")
  page_shapes = [shape[1:] for shape in shapes]
  if any(len(shape) != 5 or shape[3] != 2 for shape in shapes):
    raise ValueError("P38 KV observer requires rank-5 packed K/V caches")
  if len(set(page_shapes)) != 1:
    raise ValueError("P38 KV observer cache layer shapes drifted")
  return int(len(shapes) * int(selected_pages) * np.prod(page_shapes[0]) * 2)


def fingerprint_kv_pages(cache_pages, valid_tokens):
  """Return exact-integer aggregates and fixed samples for selected KV pages.

  `cache_pages` has layout `[page, token, kv_head, K_or_V, head_dim]`. Bytes
  beyond each page's logical valid-token extent are masked out, so allocator
  residue outside the request's attention range cannot create a false red.
  """
  shape = tuple(int(value) for value in cache_pages.shape)
  if len(shape) != 5 or shape[3] != 2:
    raise ValueError(f"P38 KV fingerprint received an invalid shape: {shape}")
  if jnp.dtype(cache_pages.dtype) != jnp.dtype(jnp.bfloat16):
    raise ValueError(
        f"P38 KV fingerprint received an invalid dtype: {cache_pages.dtype}"
    )
  if tuple(valid_tokens.shape) != (shape[0],):
    raise ValueError("P38 KV fingerprint valid-token shape drifted")

  bits = jax.lax.bitcast_convert_type(cache_pages, jnp.uint16)
  token_mask = jnp.arange(shape[1], dtype=jnp.int32)[None, :] < (
      valid_tokens[:, None]
  )
  token_mask = token_mask.reshape((shape[0], shape[1], 1, 1, 1))
  bits = jnp.where(token_mask, bits, jnp.uint16(0))
  flat = bits.reshape((shape[0], -1)).astype(jnp.uint32)
  weights = jnp.arange(1, flat.shape[1] + 1, dtype=jnp.uint32)[None, :]
  aggregates = jnp.stack(
      (
          jnp.sum(flat, axis=1, dtype=jnp.uint32),
          jnp.sum(flat * weights, axis=1, dtype=jnp.uint32),
          jnp.bitwise_xor.reduce(flat, axis=1),
          jnp.sum(flat != 0, axis=1, dtype=jnp.uint32),
      ),
      axis=1,
  )

  token_width = int(np.prod(shape[2:]))
  token_bits = bits.reshape((shape[0], shape[1], token_width))
  page_rows = jnp.arange(shape[0], dtype=jnp.int32)
  last_tokens = valid_tokens.astype(jnp.int32) - 1
  middle_tokens = last_tokens // 2
  sample_width = min(8, token_width)
  samples = jnp.stack(
      (
          token_bits[:, 0, :sample_width],
          token_bits[page_rows, middle_tokens, :sample_width],
          token_bits[page_rows, last_tokens, :sample_width],
      ),
      axis=1,
  )
  return aggregates, samples


def fingerprint_kv_cache_layers(
    kv_caches, global_pages, valid_tokens
):
  """Apply the same fingerprint executable to every homogeneous KV layer."""
  if not isinstance(kv_caches, (tuple, list)) or not kv_caches:
    raise ValueError("P38 KV observer requires a non-empty cache tuple")
  reference_shape = tuple(int(value) for value in kv_caches[0].shape)
  reference_dtype = jnp.dtype(kv_caches[0].dtype)
  for cache in kv_caches:
    if tuple(int(value) for value in cache.shape) != reference_shape:
      raise ValueError("P38 KV observer cache layer shapes drifted")
    if jnp.dtype(cache.dtype) != reference_dtype:
      raise ValueError("P38 KV observer cache layer dtypes drifted")
  if len(tuple(global_pages.shape)) != 1:
    raise ValueError("P38 KV observer global pages must be a vector")
  if tuple(valid_tokens.shape) != tuple(global_pages.shape):
    raise ValueError("P38 KV observer page/extent shapes differ")

  aggregates = []
  samples = []
  for cache in kv_caches:
    selected = jnp.take(cache, global_pages, axis=0, mode="fill")
    layer_aggregates, layer_samples = fingerprint_kv_pages(
        selected, valid_tokens
    )
    aggregates.append(layer_aggregates)
    samples.append(layer_samples)
  return jnp.stack(aggregates), jnp.stack(samples)


def fingerprint_kv_page_prefixes(cache_pages):
  """Fingerprint every valid prefix length of each selected physical page."""
  shape = tuple(int(value) for value in cache_pages.shape)
  if len(shape) != 5 or shape[3] != 2:
    raise ValueError(f"P38 KV prefix table received an invalid shape: {shape}")
  if jnp.dtype(cache_pages.dtype) != jnp.dtype(jnp.bfloat16):
    raise ValueError(
        f"P38 KV prefix table received an invalid dtype: {cache_pages.dtype}"
    )

  bits = jax.lax.bitcast_convert_type(cache_pages, jnp.uint16)
  token_width = int(np.prod(shape[2:]))
  token_bits = bits.reshape((shape[0], shape[1], token_width))
  token_values = token_bits.astype(jnp.uint32)
  flat_offsets = (
      jnp.arange(shape[1], dtype=jnp.uint32)[:, None]
      * jnp.uint32(token_width)
  )
  feature_offsets = jnp.arange(
      1, token_width + 1, dtype=jnp.uint32
  )[None, :]
  weights = flat_offsets + feature_offsets
  per_token = jnp.stack(
      (
          jnp.sum(token_values, axis=2, dtype=jnp.uint32),
          jnp.sum(token_values * weights[None, :, :],
                  axis=2, dtype=jnp.uint32),
          jnp.bitwise_xor.reduce(token_values, axis=2),
          jnp.sum(token_values != 0, axis=2, dtype=jnp.uint32),
      ),
      axis=2,
  )
  additive = jnp.cumsum(
      per_token[..., (0, 1, 3)], axis=1, dtype=jnp.uint32
  )
  exclusive_or = jax.lax.associative_scan(
      jnp.bitwise_xor, per_token[..., 2], axis=1
  )
  aggregates = jnp.stack(
      (additive[..., 0], additive[..., 1], exclusive_or,
       additive[..., 2]),
      axis=2,
  )

  page_rows = jnp.arange(shape[0], dtype=jnp.int32)[:, None]
  last_tokens = jnp.arange(shape[1], dtype=jnp.int32)[None, :]
  middle_tokens = last_tokens // 2
  sample_width = min(8, token_width)
  first = jnp.broadcast_to(
      token_bits[:, None, 0, :sample_width],
      (shape[0], shape[1], sample_width),
  )
  middle = token_bits[page_rows, middle_tokens, :sample_width]
  last = token_bits[page_rows, last_tokens, :sample_width]
  samples = jnp.stack((first, middle, last), axis=2)
  return aggregates, samples


def fingerprint_kv_cache_layer_prefixes(kv_caches, global_pages):
  """Build one all-prefix lookup table for every selected page and layer."""
  if not isinstance(kv_caches, (tuple, list)) or not kv_caches:
    raise ValueError("P38 KV observer requires a non-empty cache tuple")
  reference_shape = tuple(int(value) for value in kv_caches[0].shape)
  reference_dtype = jnp.dtype(kv_caches[0].dtype)
  for cache in kv_caches:
    if tuple(int(value) for value in cache.shape) != reference_shape:
      raise ValueError("P38 KV observer cache layer shapes drifted")
    if jnp.dtype(cache.dtype) != reference_dtype:
      raise ValueError("P38 KV observer cache layer dtypes drifted")
  if len(tuple(global_pages.shape)) != 1:
    raise ValueError("P38 KV observer global pages must be a vector")

  aggregates = []
  samples = []
  for cache in kv_caches:
    selected = jnp.take(cache, global_pages, axis=0, mode="fill")
    layer_aggregates, layer_samples = fingerprint_kv_page_prefixes(selected)
    aggregates.append(layer_aggregates)
    samples.append(layer_samples)
  return jnp.stack(aggregates), jnp.stack(samples)
