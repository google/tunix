# Copyright 2026 The Tunix Authors.
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

"""Model runner for token generation.

`ModelRunner` owns the transformer weights and runs the forward pass and token
sampling on device.
"""

from collections.abc import Callable, Hashable, Mapping, Sequence
import dataclasses
import functools
import math
import operator
import threading
from typing import Any, TypeVar

from flax import nnx
from flax import struct
from flax.nnx import filterlib
from flax.nnx import graph
from flax.nnx import statelib
import jax
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
import jaxtyping
import numpy as np
from tunix.experimental.generate import configs
from tunix.experimental.generate import request as request_lib
from tunix.experimental.generate import utils
from tunix.models import paged_attention
from tunix.rl import reshard
from tunix.rl import utils as rl_utils

_HIERARCHICAL_BLOCK_SIZE = 256
_HIERARCHICAL_MIN_VOCAB_SIZE = 4096
_TOPK_BLOCK_SIZE = 4096
_MIN_SAFE_TEMP = float(np.finfo(np.float32).tiny)
_MAX_SAFE_TEMP = 1.0 / _MIN_SAFE_TEMP

_StepResultT = TypeVar('_StepResultT')
_MeshType = jax.sharding.Mesh | jax.sharding.AbstractMesh
_TRACE_WARMUP_LOCK = threading.Lock()
_TRACE_KEY_LOCKS: dict[Hashable, threading.Lock] = {}
_WARMED_TRACE_KEYS: set[Hashable] = set()


def _run_with_trace_warmup_guard(
    trace_key: Hashable,
    fn: Callable[[], _StepResultT],
) -> _StepResultT:
  """Serializes cold-start tracing/lowering across threads for identical JIT keys."""
  if trace_key in _WARMED_TRACE_KEYS:
    return fn()
  with _TRACE_WARMUP_LOCK:
    if trace_key in _WARMED_TRACE_KEYS:
      return fn()
    if trace_key not in _TRACE_KEY_LOCKS:
      _TRACE_KEY_LOCKS[trace_key] = threading.Lock()
    key_lock = _TRACE_KEY_LOCKS[trace_key]
  with key_lock:
    if trace_key not in _WARMED_TRACE_KEYS:
      result = fn()
      with _TRACE_WARMUP_LOCK:
        _WARMED_TRACE_KEYS.add(trace_key)
        if trace_key in _TRACE_KEY_LOCKS:
          del _TRACE_KEY_LOCKS[trace_key]
      return result
  return fn()


@struct.dataclass
class SamplingMetadata:
  """The sampling settings of each request in a batch.

  Each array has one entry per request in the batch. Rows past the scheduled
  requests are padding, and sample greedily.
  """

  # Zero samples greedily.
  temperature: jax.Array | np.ndarray  # `[num_rows]` float32
  # Samples only from the smallest set of most likely tokens whose probability
  # reaches this value.
  top_p: jax.Array | np.ndarray  # `[num_rows]` float32, in (0, 1].
  # Samples only from the `top_k` most likely tokens. <= 0 means no limit.
  top_k: jax.Array | np.ndarray  # `[num_rows]` int32, in [1, max_top_k].
  # Whether every request samples from the full vocabulary without `top_k` or
  # `top_p` truncation.
  unconstrained: bool = struct.field(pytree_node=False, default=False)

  # TODO(yatlas): Remove sampling_params. It is being deprecated.
  @classmethod
  def from_sampling_params(
      cls,
      sampling_params: Sequence[request_lib.SamplingParams],
      num_rows: int,
      max_top_k: int,
      distribution: jax.Array | np.ndarray,
  ) -> 'SamplingMetadata':
    """Packs the scheduled requests' sampling settings into host arrays.

    Args:
      sampling_params: The sampling settings of each scheduled request, in row
        order.
      num_rows: The row capacity of the batch.
      max_top_k: The maximum `top_k` value allowed.
      distribution: `[num_groups, 3]` row layout of each DP group. Group `i`'s
        `distribution[i, 2]` requests are packed starting at row `i * (num_rows
        // num_groups)`.

    Returns:
      The metadata, padded out to `num_rows` rows.
    """
    if len(sampling_params) > num_rows:
      raise ValueError(
          f'Got {len(sampling_params)} sampling params for {num_rows} rows.'
      )

    temperature = np.zeros((num_rows,), dtype=np.float32)
    top_p = np.ones((num_rows,), dtype=np.float32)
    top_k = np.ones((num_rows,), dtype=np.int32)
    unconstrained = bool(max_top_k <= 0)

    dists = np.asarray(distribution)
    rows_per_group = num_rows // dists.shape[0]
    n_tokens = dists[:, 2]
    row_indices = []
    for i, n in enumerate(n_tokens):
      start = i * rows_per_group
      row_indices.extend(range(start, start + int(n)))

    for row, params in zip(row_indices, sampling_params):
      if (
          max_top_k > 0
          and params.top_k is not None
          and params.top_k > max_top_k
      ):
        raise ValueError(
            f'top_k={params.top_k} exceeds the engine max_top_k={max_top_k}.'
        )
      temperature[row] = params.temperature

      # No `top_p` keeps every token.
      top_p[row] = 1.0 if params.top_p is None else params.top_p
      top_k[row] = max_top_k if params.top_k is None else params.top_k
      if temperature[row] > 0.0 and (top_k[row] > 0 or top_p[row] < 1.0):
        unconstrained = False

    return cls(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        unconstrained=unconstrained,
    )


def _finite_or_zero(x: jax.Array) -> jax.Array:
  return jnp.where(jnp.isfinite(x), x, 0.0)


def _sample_hierarchical_256(
    logits: jax.Array,
    key: jax.Array,
    is_greedy: jax.Array,
    inv_temp: jax.Array,
    *,
    return_logprobs: bool = False,
) -> tuple[jax.Array, jax.Array | None]:
  """Samples tokens in two 256-wide stages without full-vocab fp32 buffers."""
  logits_3d = logits.reshape(logits.shape[0], -1, _HIERARCHICAL_BLOCK_SIZE)
  block_max = jnp.max(logits_3d, axis=-1)
  bmax_f32 = block_max.astype(jnp.float32)
  shifted = (
      logits_3d.astype(jnp.float32) - _finite_or_zero(bmax_f32)[:, :, None]
  ) * inv_temp[:, None, None]
  safe_rmax = _finite_or_zero(jnp.max(bmax_f32, axis=-1))
  block_lse = (bmax_f32 - safe_rmax[:, None]) * inv_temp[:, None] + jnp.log(
      jnp.sum(jnp.exp(shifted), axis=-1)
  )

  k_block, k_within = jax.random.split(key)
  block_idx = jnp.where(
      is_greedy,
      jnp.argmax(block_max, axis=-1),
      jax.random.categorical(k_block, block_lse, axis=-1),
  )
  chosen_raw = jnp.take_along_axis(logits_3d, block_idx[:, None, None], axis=1)[
      :, 0, :
  ]
  chosen_scaled = (
      chosen_raw.astype(jnp.float32) - safe_rmax[:, None]
  ) * inv_temp[:, None]
  within_idx = jnp.where(
      is_greedy,
      jnp.argmax(chosen_raw, axis=-1),
      jax.random.categorical(k_within, chosen_scaled, axis=-1),
  )
  next_tokens = (block_idx * _HIERARCHICAL_BLOCK_SIZE + within_idx).astype(
      jnp.int32
  )
  if not return_logprobs:
    return next_tokens, None
  chosen_logit = jnp.take_along_axis(
      chosen_scaled, within_idx[:, None], axis=-1
  )[:, 0]
  return next_tokens, chosen_logit - jax.nn.logsumexp(block_lse, axis=-1)


def _tpu_topk_custom_call_profitable(
    rows: int, vocab_size: int, k: int
) -> bool:
  """Returns True when XLA `TpuTopkRewriter` lowers `top_k` to `TopK`."""
  if k > _TOPK_BLOCK_SIZE or vocab_size <= 1:
    return False
  eff_rows = max(1, rows)
  log2_n = (vocab_size - 1).bit_length()
  sublane_tiles = (eff_rows + 7) // 8
  sort_cycles = int(
      (0.1 * vocab_size * log2_n * log2_n + 1500.0) * sublane_tiles
  )
  topk_cycles = int(
      0.0028 * vocab_size * k * eff_rows + 750.0 * k * sublane_tiles
  )
  return topk_cycles <= int(0.8 * sort_cycles)


def _blocked_local_top_k(
    logits: jax.Array,
    k: int,
) -> tuple[jax.Array, jax.Array]:
  """Exact local top-k, using direct `jax.lax.top_k` when TPU `TopK` wins."""
  rows, vocab_size = logits.shape
  if (
      vocab_size < 2 * _TOPK_BLOCK_SIZE
      or vocab_size % _TOPK_BLOCK_SIZE != 0
      or k > _TOPK_BLOCK_SIZE
      or _tpu_topk_custom_call_profitable(rows, vocab_size, k)
  ):
    return jax.lax.top_k(logits, k)
  num_blocks = vocab_size // _TOPK_BLOCK_SIZE
  blk_vals, blk_idxs = jax.lax.top_k(
      logits.reshape((rows * num_blocks, _TOPK_BLOCK_SIZE)), k
  )
  blk_offsets = jnp.arange(num_blocks, dtype=jnp.int32)[
      None, :, None
  ] * jnp.int32(_TOPK_BLOCK_SIZE)
  cand_vals = blk_vals.reshape((rows, num_blocks * k))
  cand_idxs = (
      blk_idxs.astype(jnp.int32).reshape((rows, num_blocks, k)) + blk_offsets
  ).reshape((rows, num_blocks * k))
  top_vals, pick = jax.lax.top_k(cand_vals, k)
  return top_vals, jnp.take_along_axis(cand_idxs, pick, axis=-1)


def _distributed_top_k(
    logits: jax.Array,
    k: int,
    *,
    mesh: _MeshType,
) -> tuple[jax.Array, jax.Array]:
  """Exact two-stage top-k over `'tp'`-sharded logits without full-vocab sort."""
  tp_size = int(mesh.shape['tp'])
  vocab_size = logits.shape[-1]
  local_vocab = vocab_size // tp_size
  local_k = min(k, local_vocab)
  dp_axes = utils.kv_cache_dp_axes(mesh)
  num_dp = math.prod(mesh.shape[axis] for axis in dp_axes)
  row_spec = dp_axes if (num_dp > 1 and logits.shape[0] % num_dp == 0) else None
  in_spec = jax.sharding.PartitionSpec(row_spec, 'tp')
  out_spec = jax.sharding.PartitionSpec(row_spec, None)

  def _shard_fn(local_logits: jax.Array) -> tuple[jax.Array, jax.Array]:
    local_vals, local_idxs = _blocked_local_top_k(local_logits, local_k)
    tp_rank = jax.lax.axis_index('tp').astype(jnp.int32)
    global_idxs = local_idxs.astype(jnp.int32) + tp_rank * jnp.int32(
        local_vocab
    )
    # Gather only `[rows, local_k]` per TP rank in ascending rank order so
    # equal-logit ties break identically to ascending global token IDs.
    all_vals = jax.lax.all_gather(local_vals, 'tp', axis=-1, tiled=True)
    all_idxs = jax.lax.all_gather(global_idxs, 'tp', axis=-1, tiled=True)
    top_vals, pick = jax.lax.top_k(all_vals, k)
    top_idxs = jnp.take_along_axis(all_idxs, pick, axis=-1)
    return top_vals, top_idxs

  return shard_map(
      _shard_fn,
      mesh=mesh,
      in_specs=in_spec,
      out_specs=(out_spec, out_spec),
      check_rep=False,
  )(logits)


def sample(
    logits: jax.Array,
    key: jax.Array,
    metadata: SamplingMetadata,
    max_top_k: int,
    return_logprobs: bool = False,
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh | None = None,
) -> tuple[jax.Array, jax.Array | None]:
  """Samples one token per row, under each row's own sampling settings.

  A row with zero temperature takes its most likely token. Every other row
  samples from its `top_k` most likely tokens, cut down further to the smallest
  prefix whose probability reaches its `top_p`, at its own temperature.

  Args:
    logits: The `[num_rows, vocab_size]` next-token logits.
    key: The step's PRNG key, derived from the engine seed. Every row samples
      independently under it.
    metadata: Each row's sampling settings.
    max_top_k: How many candidate tokens to draw for every row. Each row's
      `top_k` must not exceed it. -1 draws every token in the vocabulary.
    return_logprobs: Whether to return the sampled tokens' logprobs.
    mesh: Optional device mesh used for distributed top-k sharding.

  Returns:
    A tuple of
    - `next_tokens`: The `[num_rows]` sampled tokens.
    - `logprobs`: The `[num_rows]` logprobs, or None if `return_logprobs` is
      False.
  """
  # For most current use cases of Tunix, k does not vary across rows. If this
  # changes, this function should be updated to sample individually per row so
  # that performance is not dependent on max_top_k.
  temp = jnp.asarray(metadata.temperature, dtype=jnp.float32)
  is_greedy = temp <= 0.0
  temperature = jnp.where(
      is_greedy, 1.0, jnp.clip(temp, _MIN_SAFE_TEMP, _MAX_SAFE_TEMP)
  )
  vocab_size = logits.shape[-1]
  if metadata.unconstrained and max_top_k <= 0:
    inv_temp = 1.0 / temperature
    if (
        vocab_size >= _HIERARCHICAL_MIN_VOCAB_SIZE
        and vocab_size % _HIERARCHICAL_BLOCK_SIZE == 0
    ):
      return _sample_hierarchical_256(
          logits, key, is_greedy, inv_temp, return_logprobs=return_logprobs
      )
    logits_f32 = logits.astype(jnp.float32)
    rmax = _finite_or_zero(jnp.max(logits_f32, axis=-1, keepdims=True))
    scaled = (logits_f32 - rmax) * inv_temp[:, None]
    next_tokens = jnp.where(
        is_greedy,
        jnp.argmax(logits, axis=-1),
        jax.random.categorical(key, scaled, axis=-1),
    ).astype(jnp.int32)
    if not return_logprobs:
      return next_tokens, None
    chosen = jnp.take_along_axis(scaled, next_tokens[:, None], axis=-1)[:, 0]
    return next_tokens, chosen - jax.nn.logsumexp(scaled, axis=-1)

  k = vocab_size if max_top_k <= 0 else min(max_top_k, vocab_size)

  if (
      isinstance(mesh, (jax.sharding.Mesh, jax.sharding.AbstractMesh))
      and 'tp' in mesh.axis_names
      and int(mesh.shape['tp']) > 1
      and vocab_size % int(mesh.shape['tp']) == 0
      and k < vocab_size
  ):
    candidate_logits, candidate_tokens = _distributed_top_k(
        logits, k, mesh=mesh
    )
  else:
    candidate_logits, candidate_tokens = _blocked_local_top_k(logits, k)
  scaled_logits = candidate_logits.astype(jnp.float32) / temperature[:, None]

  ranks = jnp.arange(k)
  top_k = jnp.where(metadata.top_k <= 0, k, metadata.top_k)
  scaled_logits = jnp.where(
      ranks[None, :] < top_k[:, None], scaled_logits, -jnp.inf
  )

  # Keep each token whose predecessors have not yet reached `top_p`, which
  # always keeps the most likely token.
  probs = jax.nn.softmax(scaled_logits, axis=-1)
  probs_before = jnp.cumsum(probs, axis=-1) - probs
  scaled_logits = jnp.where(
      probs_before > metadata.top_p[:, None], -jnp.inf, scaled_logits
  )

  sampled = jax.random.categorical(key, scaled_logits, axis=-1)
  sampled_tokens = jnp.take_along_axis(
      candidate_tokens, sampled[:, None], axis=-1
  )[:, 0]
  # The candidates are sorted, so the first one is the most likely token.
  next_tokens = jnp.where(is_greedy, candidate_tokens[:, 0], sampled_tokens)

  if not return_logprobs:
    return next_tokens, None

  # Upscale logits before computing logprobs to preserve precision.
  scaled_full_logits = logits.astype(jnp.float32) / temperature[:, None]

  logp = jax.nn.log_softmax(scaled_full_logits, axis=-1)
  logp_sampled = jnp.take_along_axis(logp, next_tokens[:, None], axis=-1)
  return next_tokens, logp_sampled[:, 0]


Cache = dict[str, jax.Array]
_PyTree = TypeVar('_PyTree')


@struct.dataclass
class _PackedPagedMetadata:
  """Compact `PagedAttnMetadata` storing one page table per shared KV group."""

  packed_page_indices: jax.Array | np.ndarray  # `[num_groups, rows, pages]`
  kv_lens: jax.Array | np.ndarray
  query_lens: jax.Array | np.ndarray
  distribution: jax.Array | np.ndarray
  cache_group_map: tuple[tuple[str, int], ...] = struct.field(pytree_node=False)
  decode_only: bool = struct.field(pytree_node=False, default=False)

  def unpack(self) -> paged_attention.PagedAttnMetadata:
    group_tables = [
        self.packed_page_indices[idx]
        for idx in range(self.packed_page_indices.shape[0])
    ]
    return paged_attention.PagedAttnMetadata(
        page_indices={
            cache_name: group_tables[group_idx]
            for cache_name, group_idx in self.cache_group_map
        },
        kv_lens=self.kv_lens,
        query_lens=self.query_lens,
        distribution=self.distribution,
        decode_only=self.decode_only,
    )


def _pack_paged_metadata(
    metadata: paged_attention.PagedAttnMetadata,
) -> _PackedPagedMetadata:
  """Deduplicates identical page tables into a `[num_groups, rows, pages]` array."""
  unique_tables: list[jax.Array | np.ndarray] = []
  table_id_to_group: dict[int, int] = {}
  cache_group_map: list[tuple[str, int]] = []
  for cache_name, table in metadata.page_indices.items():
    obj_id = id(table)
    group_idx = table_id_to_group.get(obj_id)
    if group_idx is None:
      group_idx = len(unique_tables)
      table_id_to_group[obj_id] = group_idx
      unique_tables.append(table)
    cache_group_map.append((cache_name, group_idx))
  num_rows = metadata.query_lens.shape[0]
  if not unique_tables:
    packed: jax.Array | np.ndarray = np.zeros((0, num_rows, 0), dtype=np.int32)
  elif all(isinstance(t, np.ndarray) for t in unique_tables):
    packed = np.stack(unique_tables, axis=0)
  else:
    packed = jnp.stack(
        [jnp.asarray(t, dtype=jnp.int32) for t in unique_tables], axis=0
    )
  return _PackedPagedMetadata(
      packed_page_indices=packed,
      kv_lens=metadata.kv_lens,
      query_lens=metadata.query_lens,
      distribution=metadata.distribution,
      cache_group_map=tuple(cache_group_map),
      decode_only=metadata.decode_only,
  )


@dataclasses.dataclass(frozen=True, kw_only=True)
class _StepInputLayout:
  """Static shapes and flags for packing all step inputs into one `int32` H2D."""

  tokens_shape: tuple[int, int]
  page_indices_shape: tuple[int, int, int]
  num_rows: int
  num_groups: int
  has_target_idxs: bool
  cache_group_map: tuple[tuple[str, int], ...]
  decode_only: bool
  unconstrained: bool


def _pack_step_inputs(
    tokens: jax.Array | np.ndarray,
    packed_metadata: _PackedPagedMetadata,
    sampling_metadata: SamplingMetadata,
    target_idxs: jax.Array | np.ndarray | None,
) -> tuple[np.ndarray | jax.Array, _StepInputLayout]:
  """Packs all 32-bit step metadata and tokens into a single 1D `int32` buffer."""
  num_rows = packed_metadata.query_lens.shape[0]
  num_groups = packed_metadata.distribution.shape[0]
  page_shape = packed_metadata.packed_page_indices.shape
  layout = _StepInputLayout(
      tokens_shape=(int(tokens.shape[0]), int(tokens.shape[1])),
      page_indices_shape=(
          int(page_shape[0]),
          int(page_shape[1]),
          int(page_shape[2]),
      ),
      num_rows=int(num_rows),
      num_groups=int(num_groups),
      has_target_idxs=target_idxs is not None,
      cache_group_map=packed_metadata.cache_group_map,
      decode_only=packed_metadata.decode_only,
      unconstrained=sampling_metadata.unconstrained,
  )
  all_host_numpy = (
      isinstance(tokens, np.ndarray)
      and isinstance(packed_metadata.packed_page_indices, np.ndarray)
      and isinstance(packed_metadata.kv_lens, np.ndarray)
      and isinstance(packed_metadata.query_lens, np.ndarray)
      and isinstance(packed_metadata.distribution, np.ndarray)
      and isinstance(sampling_metadata.temperature, np.ndarray)
      and isinstance(sampling_metadata.top_p, np.ndarray)
      and isinstance(sampling_metadata.top_k, np.ndarray)
      and (target_idxs is None or isinstance(target_idxs, np.ndarray))
  )
  if all_host_numpy:
    parts_np = [
        np.asarray(tokens, dtype=np.int32).reshape(-1),
        np.asarray(packed_metadata.packed_page_indices, dtype=np.int32).reshape(
            -1
        ),
        np.asarray(packed_metadata.kv_lens, dtype=np.int32).reshape(-1),
        np.asarray(packed_metadata.query_lens, dtype=np.int32).reshape(-1),
        np.asarray(packed_metadata.distribution, dtype=np.int32).reshape(-1),
        np.ascontiguousarray(
            sampling_metadata.temperature, dtype=np.float32
        ).view(np.int32),
        np.ascontiguousarray(sampling_metadata.top_p, dtype=np.float32).view(
            np.int32
        ),
        np.asarray(sampling_metadata.top_k, dtype=np.int32).reshape(-1),
    ]
    if target_idxs is not None:
      parts_np.append(np.asarray(target_idxs, dtype=np.int32).reshape(-1))
    return np.concatenate(parts_np, axis=0), layout

  parts_jax = [
      jnp.asarray(tokens, dtype=jnp.int32).reshape(-1),
      jnp.asarray(packed_metadata.packed_page_indices, dtype=jnp.int32).reshape(
          -1
      ),
      jnp.asarray(packed_metadata.kv_lens, dtype=jnp.int32).reshape(-1),
      jnp.asarray(packed_metadata.query_lens, dtype=jnp.int32).reshape(-1),
      jnp.asarray(packed_metadata.distribution, dtype=jnp.int32).reshape(-1),
      jax.lax.bitcast_convert_type(
          jnp.asarray(sampling_metadata.temperature, dtype=jnp.float32).reshape(
              -1
          ),
          jnp.int32,
      ),
      jax.lax.bitcast_convert_type(
          jnp.asarray(sampling_metadata.top_p, dtype=jnp.float32).reshape(-1),
          jnp.int32,
      ),
      jnp.asarray(sampling_metadata.top_k, dtype=jnp.int32).reshape(-1),
  ]
  if target_idxs is not None:
    parts_jax.append(jnp.asarray(target_idxs, dtype=jnp.int32).reshape(-1))
  return jnp.concatenate(parts_jax, axis=0), layout


def _unpack_step_inputs(
    packed: jax.Array,
    layout: _StepInputLayout,
) -> tuple[jax.Array, jax.Array | None, _PackedPagedMetadata, SamplingMetadata]:
  """Slices a flat 1D `int32` device buffer back into typed step inputs."""
  off = 0
  tok_n = math.prod(layout.tokens_shape)
  tokens = packed[off : off + tok_n].reshape(layout.tokens_shape)
  off += tok_n

  page_n = math.prod(layout.page_indices_shape)
  page_indices = packed[off : off + page_n].reshape(layout.page_indices_shape)
  off += page_n

  r = layout.num_rows
  kv_lens = packed[off : off + r]
  off += r
  query_lens = packed[off : off + r]
  off += r

  dist_n = layout.num_groups * 3
  distribution = packed[off : off + dist_n].reshape((layout.num_groups, 3))
  off += dist_n

  temperature = jax.lax.bitcast_convert_type(packed[off : off + r], jnp.float32)
  off += r
  top_p = jax.lax.bitcast_convert_type(packed[off : off + r], jnp.float32)
  off += r
  top_k = packed[off : off + r]
  off += r

  target_idxs = packed[off : off + r] if layout.has_target_idxs else None
  packed_metadata = _PackedPagedMetadata(
      packed_page_indices=page_indices,
      kv_lens=kv_lens,
      query_lens=query_lens,
      distribution=distribution,
      cache_group_map=layout.cache_group_map,
      decode_only=layout.decode_only,
  )
  sampling_metadata = SamplingMetadata(
      temperature=temperature,
      top_p=top_p,
      top_k=top_k,
      unconstrained=layout.unconstrained,
  )
  return tokens, target_idxs, packed_metadata, sampling_metadata


def _pad_single_step_output(
    first_step: jax.Array,
    num_steps: int,
    repeat_first: bool | jax.Array = False,
) -> jax.Array:
  """Pads a `[num_rows, ...]` step-0 output to `[num_rows, num_steps, ...]`."""
  if num_steps == 1:
    return first_step[:, None]
  shape = (first_step.shape[0], num_steps, *first_step.shape[1:])
  if isinstance(repeat_first, bool):
    if repeat_first:
      return jnp.broadcast_to(first_step[:, None], shape)
    return jnp.zeros(shape, dtype=first_step.dtype).at[:, 0].set(first_step)
  repeated = jnp.broadcast_to(first_step[:, None], shape)
  step_mask = (jnp.arange(num_steps) == 0).reshape(
      (1, num_steps, *((1,) * (first_step.ndim - 1)))
  )
  return jnp.where(
      step_mask | repeat_first, repeated, jnp.zeros((), dtype=first_step.dtype)
  )


def _freeze_for_hash(value: Any) -> Hashable:
  """Converts nested GraphDef metadata into a hashable key."""
  if isinstance(value, (type(None), bool, int, float, str, bytes, type)):
    return value
  if isinstance(value, np.dtype):
    return value
  if isinstance(value, np.ndarray):
    return ('ndarray', value.shape, str(value.dtype), value.tobytes())
  if isinstance(value, Mapping):
    return (
        'mapping',
        tuple(
            (_freeze_for_hash(k), _freeze_for_hash(v))
            for k, v in sorted(value.items(), key=lambda kv: repr(kv[0]))
        ),
    )
  if isinstance(value, (list, tuple)):
    return (type(value), tuple(_freeze_for_hash(v) for v in value))
  if isinstance(value, (set, frozenset)):
    return (type(value), frozenset(_freeze_for_hash(v) for v in value))
  if dataclasses.is_dataclass(value) and not isinstance(value, type):
    return (
        type(value),
        tuple(
            (f.name, _freeze_for_hash(operator.attrgetter(f.name)(value)))
            for f in dataclasses.fields(value)
        ),
    )
  if isinstance(value, Hashable):
    return value
  return repr(value)


class _StaticGraphDef:
  """Hashable static wrapper around an `nnx.GraphDef` for `@jax.jit` caching."""

  __slots__ = ('graphdef', '_key')

  def __init__(self, graphdef: graph.GraphDef[nnx.Module]) -> None:
    self.graphdef = graphdef
    self._key: Hashable = _freeze_for_hash(graphdef)

  def __hash__(self) -> int:
    return hash(self._key)

  def __eq__(self, other: object) -> bool:
    if not isinstance(other, _StaticGraphDef):
      return NotImplemented
    return self._key == other._key and self.graphdef == other.graphdef


@dataclasses.dataclass(frozen=True, kw_only=True)
class _RunnerCompileKey:
  """Static compilation key shared across `ModelRunner` instances."""

  max_top_k: int
  mesh: jax.sharding.AbstractMesh
  return_logprobs: bool
  return_logits: bool


def _split_rows(x: jax.Array, mesh: _MeshType) -> jax.Array:
  """Splits row axis 0 of `x` across KV-cache DP axes when divisible."""
  dp_axes = utils.kv_cache_dp_axes(mesh)
  num_shards = math.prod(mesh.shape[axis] for axis in dp_axes)
  if num_shards == 1 or x.shape[0] % num_shards:
    return x
  return jax.lax.with_sharding_constraint(
      x,
      jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(dp_axes)),
  )


def _get_token_positions(
    tokens: jax.Array,
    metadata: paged_attention.PagedAttnMetadata,
) -> jax.Array:
  """Returns the `[num_groups, tokens_per_group]` positions of each token."""
  num_groups = metadata.distribution.shape[0]
  group_query_lens = jnp.asarray(metadata.query_lens).reshape(num_groups, -1)
  group_kv_lens = jnp.asarray(metadata.kv_lens).reshape(num_groups, -1)

  def _group_positions(toks, q_lens, kv_lens):
    total_capacity = toks.shape[0]
    n_rows = q_lens.shape[0]
    row_idxs = jnp.repeat(
        jnp.arange(n_rows),
        q_lens,
        total_repeat_length=total_capacity,
    )
    local_positions = jnp.arange(total_capacity)
    starts = jnp.zeros_like(q_lens)
    if len(q_lens) > 0:
      starts = starts.at[1:].set(jnp.cumsum(q_lens)[:-1])
    intra_offsets = local_positions - starts[row_idxs]
    curr_seq_lens = kv_lens - q_lens
    return intra_offsets + curr_seq_lens[row_idxs]

  return jax.vmap(_group_positions)(tokens, group_query_lens, group_kv_lens)


def _model_step_impl(
    graphdef: _StaticGraphDef,
    compile_key: _RunnerCompileKey,
    params: list[nnx.Variable],
    cache: Cache,
    tokens: jax.Array,
    metadata: paged_attention.PagedAttnMetadata,
    sampling_metadata: SamplingMetadata,
    rng: jax.Array,
) -> tuple[jax.Array, jax.Array | None, jax.Array | None, Cache, jax.Array]:
  """Runs one transformer forward pass over a ragged batch and samples tokens."""
  mesh = compile_key.mesh
  transformer = nnx.merge(graphdef.graphdef, params)
  positions = _get_token_positions(tokens, metadata)

  num_groups = metadata.distribution.shape[0]
  group_query_lens = jnp.asarray(metadata.query_lens).reshape(num_groups, -1)
  last_token_idxs = jnp.maximum(0, jnp.cumsum(group_query_lens, axis=-1) - 1)

  tokens, positions, last_token_idxs = (
      _split_rows(x, mesh) for x in (tokens, positions, last_token_idxs)
  )
  metadata = jax.tree.map(lambda x: _split_rows(x, mesh), metadata)
  sampling_metadata = jax.tree.map(
      lambda x: _split_rows(x, mesh), sampling_metadata
  )

  logits, updated_cache = transformer(
      tokens,
      positions,
      cache=cache,
      metadata=metadata,
      mesh=mesh,
      logits_idxs=last_token_idxs,
  )
  logits = _split_rows(logits, mesh)

  rng, step_rng = jax.random.split(rng)
  next_tokens, logp = sample(
      logits,
      step_rng,
      sampling_metadata,
      max_top_k=compile_key.max_top_k,
      return_logprobs=compile_key.return_logprobs,
      mesh=mesh,
  )

  res_logits = logits if compile_key.return_logits else None
  return next_tokens, res_logits, logp, updated_cache, rng


def _run_decode_scan(
    graphdef: _StaticGraphDef,
    compile_key: _RunnerCompileKey,
    params: list[nnx.Variable],
    cache: Cache,
    decode_tokens: jax.Array,
    decode_metadata: paged_attention.PagedAttnMetadata,
    sampling_metadata: SamplingMetadata,
    rng: jax.Array,
    num_steps: int,
    num_groups: int,
    rows_per_group: int,
):
  """Scans `num_steps` autoregressive decode steps over compacted rows."""
  query_lens = decode_metadata.query_lens

  def decode_step(carry, _):
    step_cache, tokens, kv_lens, step_rng = carry
    step_metadata = dataclasses.replace(decode_metadata, kv_lens=kv_lens)
    next_tokens, logits, logp, step_cache, step_rng = _model_step_impl(
        graphdef,
        compile_key,
        params,
        step_cache,
        tokens,
        step_metadata,
        sampling_metadata,
        step_rng,
    )
    carry = (
        step_cache,
        next_tokens.reshape(num_groups, rows_per_group),
        kv_lens + query_lens,
        step_rng,
    )
    return carry, (next_tokens, logits, logp)

  return jax.lax.scan(
      decode_step,
      (cache, decode_tokens, decode_metadata.kv_lens, rng),
      xs=None,
      length=num_steps,
  )


@functools.partial(
    jax.jit,
    static_argnames=['graphdef', 'compile_key', 'layout', 'num_steps'],
    donate_argnames=['cache'],
)
def _jit_packed_model_step(
    graphdef: _StaticGraphDef,
    compile_key: _RunnerCompileKey,
    params: list[nnx.Variable],
    cache: Cache,
    packed_step_inputs: jax.Array,
    layout: _StepInputLayout,
    rng: jax.Array,
    num_steps: int = 1,
    prev_generated_tokens: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array | None, jax.Array | None, Cache, jax.Array]:
  """Unpacks flat inputs, runs step 0, and pads outputs to `num_steps`."""
  tokens, target_idxs, packed_metadata, sampling_metadata = _unpack_step_inputs(
      packed_step_inputs, layout
  )
  if prev_generated_tokens is not None and target_idxs is not None:
    tokens = utils.scatter_last_column(
        tokens, prev_generated_tokens, target_idxs
    )
  metadata = packed_metadata.unpack()
  next_tokens, logits, logp, cache, rng = _model_step_impl(
      graphdef,
      compile_key,
      params,
      cache,
      tokens,
      metadata,
      sampling_metadata,
      rng,
  )
  mesh = compile_key.mesh
  distribution = jnp.asarray(packed_metadata.distribution, dtype=jnp.int32)
  repeat_first = (
      jnp.sum(distribution[:, 0] + distribution[:, 2] - distribution[:, 1]) > 0
  )
  out_tokens = _split_rows(
      _pad_single_step_output(next_tokens, num_steps, repeat_first), mesh
  )
  out_logits = (
      _split_rows(_pad_single_step_output(logits, num_steps), mesh)
      if logits is not None
      else None
  )
  out_logp = (
      _split_rows(_pad_single_step_output(logp, num_steps), mesh)
      if logp is not None
      else None
  )
  return out_tokens, out_logits, out_logp, cache, rng


@functools.partial(
    jax.jit,
    static_argnames=['graphdef', 'compile_key', 'layout', 'num_decode_steps'],
    donate_argnames=['cache'],
)
def _jit_prefill_and_decode_loop(
    graphdef: _StaticGraphDef,
    compile_key: _RunnerCompileKey,
    params: list[nnx.Variable],
    cache: Cache,
    packed_step_inputs: jax.Array,
    layout: _StepInputLayout,
    rng: jax.Array,
    num_decode_steps: int,
    prev_generated_tokens: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array | None, jax.Array | None, Cache, jax.Array]:
  """Runs scheduled step 0 and `num_decode_steps` survivor decodes in one JIT."""
  mesh = compile_key.mesh
  tokens, target_idxs, packed_metadata, sampling_metadata = _unpack_step_inputs(
      packed_step_inputs, layout
  )
  if prev_generated_tokens is not None and target_idxs is not None:
    tokens = utils.scatter_last_column(
        tokens, prev_generated_tokens, target_idxs
    )
  next_tokens, first_logits, first_logp, cache, rng = _model_step_impl(
      graphdef,
      compile_key,
      params,
      cache,
      tokens,
      packed_metadata.unpack(),
      sampling_metadata,
      rng,
  )

  num_rows = packed_metadata.query_lens.shape[0]
  distribution = jnp.asarray(packed_metadata.distribution, dtype=jnp.int32)
  num_groups = distribution.shape[0]
  rows_per_group = num_rows // num_groups

  group_rows = jnp.arange(rows_per_group, dtype=jnp.int32)[None, :]
  is_chunked_prefill = (group_rows >= distribution[:, 0:1]) & (
      group_rows < distribution[:, 1:2]
  )
  is_surviving = (group_rows < distribution[:, 2:3]) & ~is_chunked_prefill
  num_surviving_per_group = (
      distribution[:, 0] + distribution[:, 2] - distribution[:, 1]
  )
  local_order = jnp.argsort(~is_surviving, axis=1)
  group_row_offsets = (
      jnp.arange(num_groups, dtype=local_order.dtype)[:, None] * rows_per_group
  )
  order = (local_order + group_row_offsets).reshape(-1)
  is_active_decode = (group_rows < num_surviving_per_group[:, None]).reshape(-1)

  query_lens = jnp.where(is_active_decode, jnp.int32(1), jnp.int32(0))
  kv_lens = (
      jnp.where(
          is_active_decode,
          jnp.asarray(packed_metadata.kv_lens, dtype=jnp.int32)[order],
          jnp.int32(0),
      )
      + query_lens
  )
  compacted_tables = jnp.asarray(
      packed_metadata.packed_page_indices, dtype=jnp.int32
  )[:, order, :]
  group_slices = [
      compacted_tables[idx] for idx in range(compacted_tables.shape[0])
  ]
  decode_metadata = paged_attention.PagedAttnMetadata(
      page_indices={
          cache_name: group_slices[group_idx]
          for cache_name, group_idx in packed_metadata.cache_group_map
      },
      kv_lens=kv_lens,
      query_lens=query_lens,
      distribution=jnp.stack([num_surviving_per_group] * 3, axis=1),
      decode_only=True,
  )

  decode_sampling_metadata = SamplingMetadata(
      temperature=jnp.where(
          is_active_decode,
          jnp.asarray(sampling_metadata.temperature)[order],
          jnp.float32(0.0),
      ),
      top_p=jnp.where(
          is_active_decode,
          jnp.asarray(sampling_metadata.top_p)[order],
          jnp.float32(1.0),
      ),
      top_k=jnp.where(
          is_active_decode,
          jnp.asarray(sampling_metadata.top_k)[order],
          jnp.int32(1),
      ),
      unconstrained=sampling_metadata.unconstrained,
  )

  decode_tokens = next_tokens[order].reshape(num_groups, rows_per_group)
  inverse_order = (
      jnp.argsort(local_order, axis=1) + group_row_offsets
  ).reshape(-1)

  carry, outputs = _run_decode_scan(
      graphdef,
      compile_key,
      params,
      cache,
      decode_tokens,
      decode_metadata,
      decode_sampling_metadata,
      rng,
      num_decode_steps,
      num_groups,
      rows_per_group,
  )
  cache, _, _, rng = carry
  scan_tokens, scan_logits, scan_logp = outputs
  out_tokens = _split_rows(
      jnp.concatenate(
          [
              next_tokens[:, None],
              jnp.swapaxes(scan_tokens[:, inverse_order], 0, 1),
          ],
          axis=1,
      ),
      mesh,
  )
  if scan_logits is not None and first_logits is not None:
    out_logits = _split_rows(
        jnp.concatenate(
            [
                first_logits[:, None],
                jnp.swapaxes(scan_logits[:, inverse_order], 0, 1),
            ],
            axis=1,
        ),
        mesh,
    )
  else:
    out_logits = None
  if scan_logp is not None and first_logp is not None:
    out_logp = _split_rows(
        jnp.concatenate(
            [
                first_logp[:, None],
                jnp.swapaxes(scan_logp[:, inverse_order], 0, 1),
            ],
            axis=1,
        ),
        mesh,
    )
  else:
    out_logp = None
  return out_tokens, out_logits, out_logp, cache, rng


@functools.partial(
    jax.jit,
    static_argnames=['graphdef', 'compile_key', 'layout', 'num_steps'],
    donate_argnames=['cache'],
)
def _jit_decode_only_loop(
    graphdef: _StaticGraphDef,
    compile_key: _RunnerCompileKey,
    params: list[nnx.Variable],
    cache: Cache,
    packed_step_inputs: jax.Array,
    layout: _StepInputLayout,
    rng: jax.Array,
    num_steps: int,
    prev_generated_tokens: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array | None, jax.Array | None, Cache, jax.Array]:
  """Runs all `num_steps` pure-decode passes in a single compiled `lax.scan`."""
  tokens, target_idxs, packed_metadata, sampling_metadata = _unpack_step_inputs(
      packed_step_inputs, layout
  )
  if prev_generated_tokens is not None and target_idxs is not None:
    tokens = utils.scatter_last_column(
        tokens, prev_generated_tokens, target_idxs
    )
  num_rows = packed_metadata.query_lens.shape[0]
  num_groups = packed_metadata.distribution.shape[0]
  rows_per_group = num_rows // num_groups
  decode_metadata = packed_metadata.unpack()
  carry, outputs = _run_decode_scan(
      graphdef,
      compile_key,
      params,
      cache,
      tokens,
      decode_metadata,
      sampling_metadata,
      rng,
      num_steps,
      num_groups,
      rows_per_group,
  )
  cache, _, _, rng = carry
  out_tokens, out_logits, out_logp = outputs
  mesh = compile_key.mesh
  return (
      _split_rows(jnp.swapaxes(out_tokens, 0, 1), mesh),
      _split_rows(jnp.swapaxes(out_logits, 0, 1), mesh)
      if out_logits is not None
      else None,
      _split_rows(jnp.swapaxes(out_logp, 0, 1), mesh)
      if out_logp is not None
      else None,
      cache,
      rng,
  )


class ModelRunner:
  """Runs the transformer forward pass and samples tokens."""

  def __init__(
      self,
      transformer: nnx.Module,
      config: configs.ModelRunnerConfig,
  ):
    """Initializes the runner.

    Args:
      transformer: An instance of the transformer to run.
      config: The runner's configuration.
    """
    self._transformer_graphdef: graph.GraphDef[nnx.Module] = nnx.graphdef(
        transformer
    )
    self._static_graphdef = _StaticGraphDef(self._transformer_graphdef)
    self._transformer_state: statelib.State = nnx.variables(transformer)
    self._flattened_transformer_state: list[nnx.Variable] = jax.tree.leaves(
        self._transformer_state,
        is_leaf=lambda x: isinstance(x, nnx.Variable),
    )

    self._config = config
    self._compile_key = _RunnerCompileKey(
        max_top_k=config.max_top_k,
        mesh=config.mesh.abstract_mesh,
        return_logprobs=config.return_logprobs,
        return_logits=config.return_logits,
    )
    self._rng = self._to_device(jax.random.PRNGKey(config.seed))

  @property
  def config(self) -> configs.ModelRunnerConfig:
    return self._config

  def model_def_and_state(
      self,
  ) -> tuple[graph.GraphDef[nnx.Module], list[nnx.Variable]]:
    """Returns the transformer graphdef and state."""
    return self._transformer_graphdef, self._flattened_transformer_state

  @property
  def transformer(self) -> nnx.Module:
    return nnx.merge(
        self._transformer_graphdef, self._flattened_transformer_state
    )

  @property
  def transformer_state(self) -> statelib.State:
    return self._transformer_state

  @transformer_state.setter
  def transformer_state(self, state: statelib.State) -> None:
    def get_all_param_types(tree):
      param_types = set()
      jax.tree_util.tree_map(
          lambda x: param_types.add(type(x)),
          tree,
          is_leaf=lambda x: isinstance(x, nnx.Variable),
      )
      return param_types

    def check_tree_structure(tree1, tree2):
      if jax.tree_util.tree_structure(tree1) != jax.tree_util.tree_structure(
          tree2
      ):
        raise ValueError(
            'New state must have the same structure as the old state.'
            f' {jax.tree_util.tree_structure(tree1)} vs'
            f' {jax.tree_util.tree_structure(tree2)}'
        )

      def check_shape_dtype_sharding(x, y):

        def equivalent_sharding(x, y):
          # Lift the condition on memory_kind due to offloading.
          # Besides it seems jax.jit might change some shardings of the params
          # to equivalent representation so here we check if the specs are
          # equivalent instead of checking the identity.
          if isinstance(
              x.sharding, jax.sharding.SingleDeviceSharding
          ) and isinstance(y.sharding, jax.sharding.SingleDeviceSharding):
            return x.sharding.device_set == y.sharding.device_set
          if not (
              isinstance(x.sharding, jax.sharding.NamedSharding)
              and isinstance(y.sharding, jax.sharding.NamedSharding)
          ):
            return False
          if x.sharding.mesh != y.sharding.mesh:
            return False
          mesh = x.sharding.mesh
          diff_spec = list(set(x.sharding.spec) - set(y.sharding.spec))
          for spec in diff_spec:
            if spec and mesh.shape[spec] != 1:
              return False
          return True

        return (
            jnp.shape(x) == jnp.shape(y)
            and x.dtype == y.dtype
            and equivalent_sharding(x, y)
        )

      if not all(
          jax.tree_util.tree_leaves(
              jax.tree_util.tree_map(check_shape_dtype_sharding, tree1, tree2)
          )
      ):
        raise ValueError(
            'New state must have the same shape, dtype and sharding as the old'
            f' state. {tree1} vs {tree2}'
        )

    param_types = get_all_param_types(state)

    if nnx.Param in param_types:
      # Full state replacement.
      check_tree_structure(self._transformer_state, state)
      self._transformer_state = state
    else:
      # LoRA state replacement.
      if not (len(param_types) == 1 and nnx.LoRAParam in param_types):
        raise ValueError(
            'Only LoRAParam is supported. Received invalid `param_types`: '
            f'{param_types}'
        )
      original_lora_params = statelib.filter_state(
          self._transformer_state, nnx.LoRAParam
      )
      check_tree_structure(original_lora_params, state)
      base_state = statelib.filter_state(
          self._transformer_state, filterlib.Not(nnx.LoRAParam)
      )
      self._transformer_state = statelib.merge_state(base_state, state)

    self._flattened_transformer_state = jax.tree.leaves(
        self._transformer_state,
        is_leaf=lambda x: isinstance(x, nnx.Variable),
    )

  def update_params(
      self,
      updated_weights: jaxtyping.PyTree,
      filter_types: tuple[Any, ...] | None = None,
  ) -> None:
    """Updates model parameters in the sampler."""
    if filter_types is not None:
      dst_params = nnx.state(self.transformer, filter_types)
      dst_leaves = jax.tree.leaves(dst_params)
      if dst_leaves and all(
          isinstance(x, jax.Array)
          and isinstance(x.sharding, jax.sharding.NamedSharding)
          for x in dst_leaves
      ):
        resharded_params = reshard.reshard_pytree(updated_weights, dst_params)
      else:
        resharded_params = updated_weights
    else:
      resharded_params = updated_weights
    flat_new_params, _ = rl_utils.to_flat_dict(resharded_params)
    # TODO(linchai): Cast on rollout devices when from lower precision to
    # higher precision.
    new_params_precision = jax.tree.leaves(flat_new_params)[0].dtype
    rollout_precision = jax.tree.leaves(self.transformer_state)[0].dtype
    if new_params_precision != rollout_precision:
      flat_new_params = jax.tree.map(
          lambda x: x.astype(rollout_precision), flat_new_params
      )
    flat_old_params, tree_def = rl_utils.to_flat_dict(self.transformer_state)
    merged_params = functools.reduce(
        operator.ior, [flat_old_params, flat_new_params], {}
    )
    merged_params = jax.tree.unflatten(tree_def, merged_params.values())
    new_model = nnx.merge(self._transformer_graphdef, merged_params)  # pyrefly: ignore[no-matching-overload]
    self.transformer_state = nnx.variables(new_model, nnx.Param)

  # TODO(yatlas): dtype should have one source of truth.
  @property
  def dtype(self) -> jnp.dtype:
    if hasattr(self.transformer, 'config') and (
        hasattr(self.transformer.config, 'dtype')
    ):
      return self.transformer.config.dtype
    return self._flattened_transformer_state[0].dtype

  def _to_device(self, tree: _PyTree) -> _PyTree:
    """Moves host arrays onto the device, replicated across the mesh."""
    return jax.device_put(
        tree,
        jax.sharding.NamedSharding(
            self._config.mesh, jax.sharding.PartitionSpec()
        ),
    )

  def execute_step(
      self,
      cache: Cache,
      tokens: jax.Array | np.ndarray,
      metadata: paged_attention.PagedAttnMetadata,
      sampling_params: Sequence[request_lib.SamplingParams],
      *,
      prev_generated_tokens: jax.Array | None = None,
      target_idxs: jax.Array | np.ndarray | None = None,
      max_steps: int | None = None,
  ) -> tuple[jax.Array, jax.Array | None, jax.Array | None, Cache]:
    """Runs model forward passes on a batch of requests, and samples tokens.

    Note: This function runs up to `num_scheduler_steps` forward passes, and
    pads outputs to `num_scheduler_steps` columns.

    Args:
      cache: The physical KV cache pages to read and write.
      tokens: `[num_groups, tokens_per_group]` array of tokens for the batch,
        where each row is a DP group's padded ragged token array.
      metadata: Ragged execution metadata for the attention kernel.
      sampling_params: The sampling settings of each scheduled request, in row
        order.
      prev_generated_tokens: Optional `[num_rows, prev_steps]` device tokens
        from the previous in-flight step, whose last column is scattered into
        `tokens` at `target_idxs` inside the compiled step.
      target_idxs: Optional `[num_rows]` scatter indices into flattened `tokens`
        for `prev_generated_tokens`.
      max_steps: Optional override for forward passes to execute in this step
        (must be `None`, `1`, or `config.num_scheduler_steps`). Passing `1` runs
        a single forward pass so active chunked prefills drain without stalling
        behind multi-step decode scans.

    Returns:
      A tuple of:
        - tokens: `[num_rows, num_scheduler_steps]` array of sampled tokens.
        - logits: `[num_rows, num_scheduler_steps, vocab_size]` array of
          logits, or None.
        - logprobs: `[num_rows, num_scheduler_steps]` array of logprobs, or
          None.
        - updated cache: The updated KV cache pages.

    Raises:
      ValueError: If `sampling_params` does not hold one entry per scheduled
        request, or if `max_steps` is invalid.
    """
    config = self._config
    num_steps = config.num_scheduler_steps
    if max_steps is not None and max_steps not in (1, num_steps):
      raise ValueError(
          f'max_steps must be None, 1, or {num_steps}, got {max_steps}.'
      )
    active_steps = num_steps if max_steps is None else max_steps
    num_rows = metadata.query_lens.shape[0]
    distribution = np.asarray(metadata.distribution)
    num_scheduled = int(np.sum(distribution[:, 2]))
    if len(sampling_params) != num_scheduled:
      raise ValueError(
          f'Got {len(sampling_params)} sampling params for {num_scheduled}'
          ' scheduled requests.'
      )

    packed_metadata = _pack_paged_metadata(metadata)
    sampling_metadata = SamplingMetadata.from_sampling_params(
        sampling_params,
        num_rows,
        self._config.max_top_k,
        distribution=distribution,
    )
    packed_inputs, layout = _pack_step_inputs(
        tokens, packed_metadata, sampling_metadata, target_idxs
    )
    device_inputs = self._to_device(packed_inputs)
    # Chunked prefills drop out, but the rest of the batch carries on.
    # Each DP group's distribution is [i, j, k]:
    # i: num decodes, j: i + num chunked prefills, k: j + num full prefills.
    num_surviving = int(
        np.sum(distribution[:, 0] + distribution[:, 2] - distribution[:, 1])
    )
    num_decode_steps = active_steps - 1 if num_surviving else 0
    rows_per_group = num_rows // distribution.shape[0]
    is_pure_decode = (
        packed_metadata.decode_only
        and num_decode_steps > 0
        and int(np.sum(distribution[:, 0])) == num_scheduled
        and tokens.shape == (distribution.shape[0], rows_per_group)
        and bool(np.all(np.asarray(metadata.query_lens) <= 1))
    )
    prev_tokens_shape = (
        None
        if prev_generated_tokens is None
        else tuple(prev_generated_tokens.shape)
    )
    if is_pure_decode:
      trace_key = (
          'decode_loop',
          self._static_graphdef,
          self._compile_key,
          layout,
          num_steps,
          prev_tokens_shape,
      )
      out_tokens, out_logits, out_logp, cache, self._rng = (
          _run_with_trace_warmup_guard(
              trace_key,
              lambda: _jit_decode_only_loop(
                  self._static_graphdef,
                  self._compile_key,
                  self._flattened_transformer_state,
                  cache,
                  device_inputs,
                  layout,
                  self._rng,
                  num_steps=num_steps,
                  prev_generated_tokens=prev_generated_tokens,
              ),
          )
      )
      return out_tokens, out_logits, out_logp, cache

    if num_decode_steps:
      trace_key = (
          'prefill_decode_loop',
          self._static_graphdef,
          self._compile_key,
          layout,
          num_decode_steps,
          prev_tokens_shape,
      )
      out_tokens, out_logits, out_logp, cache, self._rng = (
          _run_with_trace_warmup_guard(
              trace_key,
              lambda: _jit_prefill_and_decode_loop(
                  self._static_graphdef,
                  self._compile_key,
                  self._flattened_transformer_state,
                  cache,
                  device_inputs,
                  layout,
                  self._rng,
                  num_decode_steps=num_decode_steps,
                  prev_generated_tokens=prev_generated_tokens,
              ),
          )
      )
      return out_tokens, out_logits, out_logp, cache

    trace_key = (
        'packed_step',
        self._static_graphdef,
        self._compile_key,
        layout,
        num_steps,
        prev_tokens_shape,
    )
    out_tokens, out_logits, out_logp, cache, self._rng = (
        _run_with_trace_warmup_guard(
            trace_key,
            lambda: _jit_packed_model_step(
                self._static_graphdef,
                self._compile_key,
                self._flattened_transformer_state,
                cache,
                device_inputs,
                layout,
                self._rng,
                num_steps=num_steps,
                prev_generated_tokens=prev_generated_tokens,
            ),
        )
    )
    return out_tokens, out_logits, out_logp, cache
