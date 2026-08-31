# Copyright 2025 Google LLC
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

"""Vanilla sampler for LLM generation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import dataclasses
import inspect
import functools
from typing import Any, Optional, Self

from absl import logging
import flax
from flax import nnx
from flax.nnx import filterlib
from flax.nnx import graph
from flax.nnx import statelib
import jax
import jax.numpy as jnp
from jax.interpreters import pxla
import jax.sharding as shd
import jaxtyping
import numpy as np
from tunix.generate import base_sampler
from tunix.generate import utils
import tunix.generate.beam_search as beam_search_lib
import tunix.generate.tokenizer_adapter as tok_adapter
from tunix.processors import audio_processor
from tunix.processors import image_processor
from jax.sharding import PartitionSpec as P


def shard(x: jnp.ndarray, s: tuple[str | None, ...]):
  mesh = pxla.thread_resources.env.physical_mesh
  if mesh.empty or jax.devices()[0].platform == 'cpu':
    return x
  return jax.lax.with_sharding_constraint(
      x, shd.NamedSharding(mesh, shd.PartitionSpec(*s))
  )


def cdiv(a: int | jax.Array, b: int | jax.Array) -> int | jax.Array:
  return (a + b - 1) // b


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class RaggedArray:
  """2D Ragged Array."""

  data: jax.Array
  lens: jax.Array

  @property
  def total_length(self) -> jax.Array:
    return jnp.sum(self.lens)

  @property
  def batch_size(self) -> int:
    return self.lens.shape[0]

  @property
  def capacity(self) -> int:
    return self.data.shape[0]

  @property
  def row_idxs(self) -> jax.Array:
    return jnp.repeat(
        jnp.arange(self.batch_size),
        self.lens,
        total_repeat_length=self.capacity,
    )

  @property
  def intra_offsets(self) -> jax.Array:
    cum_sums = jnp.pad(jnp.cumsum(self.lens), (1, 0))
    intra_offsets = jnp.arange(self.capacity) - cum_sums[self.row_idxs]
    return intra_offsets


@dataclasses.dataclass(frozen=True, kw_only=True)
class CacheConfig:
  """Unified configuration for KV cache and TokenBuffer."""

  cache_size: int
  num_layers: int
  num_kv_heads: int
  head_dim: int

  max_seq_len: int = 1028
  max_num_pages: int = 2048 
  page_size: int = 8
  num_shards: int = 1
  window_size: int | None = None

  kv_packing: int = 2
  seq_partition: str | None = None
  head_partition: str | None = None

  @property
  def tokens_per_page(self) -> int:
    return self.page_size

  @property
  def max_num_pages_per_seq_per_shard(self) -> int:
    max_pages_per_seq = cdiv(self.max_seq_len - 1, self.page_size)
    upper_bound = cdiv(max_pages_per_seq, self.num_shards)

    if self.window_size is None:
      return upper_bound

    max_window_pages_per_seq = cdiv(self.window_size, self.page_size)
    window_upper_bound = cdiv(max_window_pages_per_seq, self.num_shards)

    return min(upper_bound, window_upper_bound + 1)

jax.tree_util.register_dataclass(
    CacheConfig,
    data_fields=[],
    meta_fields=[
        'cache_size',
        'kv_packing',
        'num_layers',
        'num_kv_heads',
        'head_dim',
        'max_seq_len',
        'max_num_pages',
        'page_size',
        'num_shards',
        'window_size',
        'seq_partition',
        'head_partition',
    ],
)

LayerCacheConfig = CacheConfig


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class LayerCache:
  """Paged KV cache for a single transformer layer."""

  pages: jax.Array
  page_indices: jax.Array  # i32[batch_size, max_num_pages_per_seq_per_shard]
  available_page_indices: jax.Array  # i32[total_num_pages_per_shard]
  num_available_pages: jax.Array  # i32 scalar
  kv_lens: jax.Array
  config: CacheConfig

  @property
  def n_available_pages(self) -> jax.Array:
    return self.num_available_pages

  @property
  def page_size(self) -> int:
    return self.config.page_size

  @property
  def tokens_per_page(self) -> int:
    return self.config.page_size

  @property
  def max_seq_len(self) -> int:
    return self.config.max_seq_len

  @property
  def num_shards(self) -> int:
    return self.config.num_shards

  @property
  def window_size(self) -> int | None:
    return self.config.window_size

  @property
  def max_num_pages_per_seq_per_shard(self) -> int:
    return self.config.max_num_pages_per_seq_per_shard

  @property
  def batch_size(self) -> int:
    return self.kv_lens.shape[0]

  @property
  def total_num_pages_per_shard(self) -> int:
    return self.available_page_indices.shape[0]

  @property
  def kv_pages(self) -> jax.Array:
    return self.pages

  @property
  def num_local_pages(self) -> jax.Array:
    return cdiv(cdiv(self.kv_lens, self.page_size), self.num_shards)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class TokenBuffer:
  """Paged token buffer that manages pages and controls page tables for all layer caches."""

  pages: jax.Array  # i32[max_num_pages, page_size]
  page_indices: jax.Array  # i32[batch_size, max_num_pages_per_seq_per_shard]
  available_page_indices: jax.Array  # i32[total_num_pages_per_shard]
  num_available_pages: jax.Array  # i32 scalar
  lens: jax.Array  # i32[batch_size]
  config: CacheConfig
  dtype: Any = dataclasses.field(metadata={"static": True})

  def init_layer_caches(self) -> dict[str, Any]:
    """Initializes KV caches for all layers sharing this TokenBuffer's page tables."""
    page_spec = P('fsdp', None, 'tp', None, None)
    
    cache = {}
    for i in range(self.config.num_layers):
      pages = jax.lax.empty(
        (
            self.config.max_num_pages,
            self.config.page_size,
            2 * self.config.num_kv_heads // self.config.kv_packing,
            self.config.kv_packing,
            self.config.head_dim,
        ),
        dtype=self.dtype,
      )
      pages = jax.lax.with_sharding_constraint(pages, page_spec)
      cache[f'layer_{i}'] = LayerCache(
          pages=pages,
          page_indices=jnp.copy(self.page_indices),
          available_page_indices=jnp.copy(self.available_page_indices),
          num_available_pages=jnp.copy(self.num_available_pages),
          kv_lens=jnp.copy(self.lens),
          config=self.config,
      )
    cache['page_indices'] = self.page_indices
    cache['available_page_indices'] = self.available_page_indices
    cache['num_available_pages'] = self.num_available_pages
    cache['config'] = self.config
    return cache

  @property
  def n_available_pages(self) -> jax.Array:
    return self.num_available_pages

  @property
  def page_size(self) -> int:
    return self.config.page_size

  @property
  def max_seq_len(self) -> int:
    return self.config.max_seq_len

  @property
  def num_shards(self) -> int:
    return self.config.num_shards

  @property
  def window_size(self) -> int | None:
    return self.config.window_size

  @property
  def max_num_pages_per_seq_per_shard(self) -> int:
    return self.config.max_num_pages_per_seq_per_shard

  @property
  def batch_size(self) -> int:
    return self.lens.shape[0]

  @property
  def total_num_pages_per_shard(self) -> int:
    return self.available_page_indices.shape[0]

  @property
  def num_local_pages(self) -> jax.Array:
    return cdiv(cdiv(self.lens, self.page_size), self.num_shards)

  def load_prompt_tokens(
      self, prompt_tokens: jax.Array, lens: jax.Array
  ) -> TokenBuffer:
    """Loads packed 1D prompt tokens into allocated paged memory."""
    token_ragged = RaggedArray(data=prompt_tokens, lens=lens)
    seq_idxs = token_ragged.row_idxs
    token_offsets = token_ragged.intra_offsets

    local_page_cols = (token_offsets // self.page_size) // self.num_shards
    page_offsets = token_offsets % self.page_size
    phys_page_ids = self.page_indices[seq_idxs, local_page_cols]

    updated_pages = self.pages.at[phys_page_ids, page_offsets].set(
        prompt_tokens
    )
    return dataclasses.replace(self, pages=updated_pages)

  def append_tokens(
      self, tokens: jax.Array, cache: dict[str, Any] | None = None
  ) -> TokenBuffer | tuple[TokenBuffer, dict[str, Any]]:
    """Appends 1 new token per sequence to paged memory."""
    res = self.allocate(jnp.ones(self.batch_size, dtype=jnp.int32), cache=cache)
    if cache is not None:
      tb, updated_cache = res
    else:
      tb = res

    token_offsets = self.lens  # position of new token
    local_page_cols = (token_offsets // tb.page_size) // tb.num_shards
    page_offsets = token_offsets % tb.page_size
    seq_idxs = jnp.arange(tb.batch_size)
    phys_page_ids = tb.page_indices[seq_idxs, local_page_cols]

    updated_pages = tb.pages.at[phys_page_ids, page_offsets].set(tokens)
    updated_tb = dataclasses.replace(tb, pages=updated_pages)
    if cache is not None:
      return updated_tb, updated_tb.update_layer_caches(updated_cache)
    return updated_tb

  def update_layer_caches(self, cache: dict[str, Any]) -> dict[str, Any]:
    """Propagates updated page tables to all layer caches."""
    if cache is None:
      return cache
    new_cache = dict(cache)
    new_cache['page_indices'] = self.page_indices
    new_cache['available_page_indices'] = self.available_page_indices
    new_cache['num_available_pages'] = self.num_available_pages
    for k, v in cache.items():
      if isinstance(v, LayerCache):
        new_cache[k] = dataclasses.replace(
            v,
            page_indices=jnp.copy(self.page_indices),
            available_page_indices=jnp.copy(self.available_page_indices),
            num_available_pages=jnp.copy(self.num_available_pages),
            kv_lens=self.lens,
        )
    return new_cache

  def allocate(
      self, q_lens: jax.Array, cache: dict[str, Any] | None = None
  ) -> TokenBuffer | tuple[TokenBuffer, dict[str, Any]]:
    """Allocates pages for new tokens and updates all layer caches if cache is provided."""
    total_pages_required = cdiv(self.lens + q_lens, self.page_size)
    local_pages_required = cdiv(total_pages_required, self.num_shards)

    num_pages_to_allocate = local_pages_required - self.num_local_pages

    page_indices_to_allocate = RaggedArray(
        data=self.available_page_indices, lens=num_pages_to_allocate
    )
    page_indices_rows = page_indices_to_allocate.row_idxs
    page_indices_cols = (
        self.num_local_pages[page_indices_rows]
        + page_indices_to_allocate.intra_offsets
    )

    updated_page_indices = self.page_indices.at[
        page_indices_rows, page_indices_cols
    ].set(page_indices_to_allocate.data)

    updated_num_available_pages = (
        self.num_available_pages - page_indices_to_allocate.total_length
    )
    updated_available_page_indices = jnp.roll(
        self.available_page_indices, -page_indices_to_allocate.total_length
    )

    updated_tb = dataclasses.replace(
        self,
        lens=self.lens + q_lens,
        page_indices=updated_page_indices,
        available_page_indices=updated_available_page_indices,
        num_available_pages=updated_num_available_pages,
    )
    if cache is not None:
      return updated_tb, updated_tb.update_layer_caches(cache)
    return updated_tb

  @jax.named_call
  def release(
      self, should_release: jax.Array, cache: dict[str, Any] | None = None
  ) -> TokenBuffer | tuple[TokenBuffer, dict[str, Any]]:
    """Releases pages for completed sequences and updates all layer caches."""
    updated_lens = jnp.where(should_release, 0, self.lens)

    page_indices_to_release = RaggedArray(
        data=jax.lax.empty((self.total_num_pages_per_shard,), dtype=jnp.int32),
        lens=jnp.where(should_release, self.num_local_pages, 0),
    )
    page_indices_irows = page_indices_to_release.row_idxs
    page_indices_icols = page_indices_to_release.intra_offsets

    updated_available_page_indices = self.available_page_indices.at[
        jnp.arange(self.total_num_pages_per_shard) + self.num_available_pages
    ].set(self.page_indices[page_indices_irows, page_indices_icols])

    updated_num_available_pages = (
        self.num_available_pages + page_indices_to_release.total_length
    )
    updated_tb = dataclasses.replace(
        self,
        lens=updated_lens,
        available_page_indices=updated_available_page_indices,
        num_available_pages=updated_num_available_pages,
    )
    if cache is not None:
      return updated_tb, updated_tb.update_layer_caches(cache)
    return updated_tb

  @jax.named_call
  def release_for_window(
      self, cache: dict[str, Any] | None = None
  ) -> TokenBuffer | tuple[TokenBuffer, dict[str, Any]]:
    """Release allocations for window and updates all layer caches."""
    if self.window_size is None:
      return self if cache is None else (self, cache)

    num_pages_to_release_per_shard = (
        jnp.maximum(self.lens - self.window_size, 0)
        // self.page_size
        // self.num_shards
    )
    page_indices_irows = jnp.arange(self.batch_size)[:, None]
    page_indices_icols = (
        jnp.arange(self.max_num_pages_per_seq_per_shard)
        + num_pages_to_release_per_shard[:, None]
    )
    updated_page_indices = self.page_indices[
        page_indices_irows, page_indices_icols
    ]
    release_helper = RaggedArray(
        data=jax.lax.empty((self.total_num_pages_per_shard,), dtype=jnp.int32),
        lens=num_pages_to_release_per_shard,
    )
    released_page_indices = self.page_indices[
        release_helper.row_idxs, release_helper.intra_offsets
    ]
    updated_available_page_indices = self.available_page_indices.at[
        jnp.arange(self.total_num_pages_per_shard) + self.num_available_pages
    ].set(released_page_indices, mode='drop')
    num_pages_to_release = num_pages_to_release_per_shard * self.num_shards
    updated_tb = dataclasses.replace(
        self,
        page_indices=updated_page_indices,
        available_page_indices=updated_available_page_indices,
        num_available_pages=self.num_available_pages
        + release_helper.total_length,
        lens=self.lens - num_pages_to_release * self.page_size,
    )
    if cache is not None:
      return updated_tb, updated_tb.update_layer_caches(cache)
    return updated_tb

  def get_token_at(self, pos: int | jax.Array) -> jax.Array:
    """Directly looks up token IDs at sequence position `pos` using page_indices and modulo."""
    seq_idxs = jnp.arange(self.batch_size)
    local_page_col = (pos // self.page_size) // self.num_shards
    page_offset = pos % self.page_size
    phys_page_ids = self.page_indices[seq_idxs, local_page_col]
    return self.pages[phys_page_ids, page_offset]

  def get_slice(self, start: int | jax.Array, length: int) -> jax.Array:
    """Directly looks up a slice of token IDs [batch_size, length] starting at `start`."""
    seq_grid = jnp.arange(self.batch_size)[:, None]
    pos_grid = start + jnp.arange(length)[None, :]

    local_page_cols = (pos_grid // self.page_size) // self.num_shards
    page_offsets = pos_grid % self.page_size
    safe_cols = jnp.clip(
        local_page_cols, 0, self.max_num_pages_per_seq_per_shard - 1
    )
    phys_page_ids = self.page_indices[seq_grid, safe_cols]
    return self.pages[phys_page_ids, page_offsets]

  def to_array(self, total_num_tokens: int | None = None, is_2d: bool = True) -> jax.Array:
    """Extracts 2D array [batch_size, max_len] of token IDs from paged memory."""
    token_ragged = RaggedArray(
        data=jnp.zeros(total_num_tokens, dtype=jnp.int32),
        lens=self.lens,
    )
    seq_idxs = token_ragged.row_idxs
    token_offsets = token_ragged.intra_offsets

    local_page_cols = (token_offsets // self.page_size) // self.num_shards
    page_offsets = token_offsets % self.page_size
    phys_page_ids = self.page_indices[seq_idxs, local_page_cols]

    # Gather exact valid tokens per sequence
    packed_tokens = self.pages[phys_page_ids, page_offsets]
    return packed_tokens

Cache = dict[str, LayerCache | dict[str, jaxtyping.Array]]


@flax.struct.dataclass
class _SamplingState:
  """Internal sampling state."""

  # Decoding step.
  decoding_step: jnp.int32

  # Token buffer holding paged token IDs.
  token_buffer: TokenBuffer
  
  # Position indices, based on ignoring pad tokens.
  positions: jnp.ndarray | None

  # Model state for conditioning the model on autoregressively.
  cache: dict[str, Any]

  # Is decoding done on the given sequence?
  done: jnp.ndarray  # [B]

  # Total sampling steps (including the prompt).
  total_sampling_steps: int

  # Fixed-size buffer for accumulating the output logits.
  logits_buffer: jnp.ndarray | None

  # Fixed-size buffer for accumulating the output logprobs.
  logprobs_buffer: jnp.ndarray | None

  # List of tokens that are forbidden to be generated.
  forbidden_token_ids: tuple[int, ...] | None

  # Random seed for sampling.
  seed: jax.Array

  # The sampling mode to use, one of "greedy", "top_p" or "beam_search"
  sampling_mode: str = flax.struct.field(pytree_node=False)

  # Number of input tokens with padding.
  num_input_tokens: jnp.int32 = flax.struct.field(pytree_node=False)

  # Number of input tokens with padding.
  total_num_input_tokens: jnp.int32 = flax.struct.field(pytree_node=False)

  # Tempurature for top_p sampling.
  temperature: float = flax.struct.field(pytree_node=False)

  # Sampling parameters.
  sampling_parameters: dict[str, float | int] = flax.struct.field(
      pytree_node=False
  )

  # Only present when sampling_mode is "beam_search".
  beam_search_sampling_state: (
      beam_search_lib._BeamSearchSamplingState | None
  ) = None



def sample_top_p(
    logits: jnp.ndarray,
    key: jax.Array,
    temperature: float,
    top_p: float,
    top_k: int | None,
    return_logprobs: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
  """Sample a token using top-p sampling."""
  print("LOGITS: ", logits.shape)
  # Upcast to float32 for numerical stability of softmax and subsequent cumsum.
  next_token_logits = logits[:, -1].astype(jnp.float32) / temperature

  # top_k=0 or None both mean "no top-k filtering" — use full vocabulary.
  _no_topk = top_k is None or top_k <= 0
  # Skip softmax and sorting if top_p is 1.0 and top_k is full vocab.
  if top_p >= 1.0 and _no_topk:
    next_token = jax.random.categorical(key, logits=next_token_logits)
    if not return_logprobs:
      return next_token, None
    logp = jax.nn.log_softmax(next_token_logits, axis=-1)
    logp_sampled = jnp.take_along_axis(logp, next_token[..., None], axis=-1)
    logp_sampled = jnp.squeeze(logp_sampled, axis=-1)
    return next_token, logp_sampled

  k = next_token_logits.shape[-1] if _no_topk else top_k
  logits_sorted, indices = jax.lax.top_k(next_token_logits, k=k)  # pyrefly: ignore[bad-argument-type]

  probs_sorted = jax.nn.softmax(logits_sorted, axis=-1)
  cumsum_probs = jnp.cumsum(probs_sorted, axis=-1)
  mask = cumsum_probs - probs_sorted > top_p
  logits_sorted = jnp.where(mask, -jnp.inf, logits_sorted)

  next_token_idx = jax.random.categorical(key, logits=logits_sorted)
  next_token = jnp.take_along_axis(indices, next_token_idx[..., None], axis=-1)
  next_token = jnp.squeeze(next_token, axis=-1)

  if return_logprobs:
    logp = jax.nn.log_softmax(next_token_logits, axis=-1)
    logp_sampled = jnp.take_along_axis(logp, next_token[..., None], axis=-1)
    logp_sampled = jnp.squeeze(logp_sampled, axis=-1)
  else:
    logp_sampled = None

  return next_token, logp_sampled


def sample_best(
    logits, return_logprobs: bool = False
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
  next_token = jnp.argmax(logits[:, -1], axis=-1, keepdims=True)
  next_token = next_token[:, 0]
  if not return_logprobs:
    return next_token, None
  logp = jax.nn.log_softmax(logits[:, -1].astype(jnp.float32), axis=-1)
  logp_sampled = jnp.take_along_axis(logp, next_token[..., None], axis=-1)
  logp_sampled = jnp.squeeze(logp_sampled, axis=-1)
  return next_token, logp_sampled


def _init_cache(
    n_layers: int,
    cache_size: int,
    batch_size: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: jnp.dtype,
) -> Cache:
  """Create KV cache for the transformer."""
  config = CacheConfig(
      cache_size=cache_size,
      num_layers=n_layers,
      num_kv_heads=num_kv_heads,
      head_dim=head_dim,
      batch_size=batch_size,
      dtype=dtype,
      num_shards=2,
      page_size=8,
      max_seq_len=max(cache_size, 1028),
  )
  
  token_buffer = TokenBuffer.init(
      batch_size=batch_size,
      config=config,
  )
  return token_buffer.init_layer_caches()



class Sampler(base_sampler.BaseSampler):
  """Sampler for transformer model."""

  def __init__(
      self,
      transformer: nnx.Module,
      tokenizer: Any,
      cache_config: CacheConfig,
      image_processor: image_processor.ImageProcessor | None = None,
  ):
    """Initializes the sampler.

    Args:
      transformer: an instance of the transformer.
      tokenizer: a tokenizer for the given model.
      cache_config: configuration for the KV cache.
      image_processor: The image processor.
    """
    self.tokenizer = tokenizer
    if not isinstance(tokenizer, tok_adapter.TokenizerAdapter):
      self.tokenizer = tok_adapter.TokenizerAdapter(tokenizer)
    self.cache_config = cache_config
    self.image_processor = image_processor
    self._transformer_graphdef: graph.NodeDef = nnx.graphdef(transformer)  # pyrefly: ignore[bad-assignment]
    self._transformer_state: list[statelib.State] = nnx.variables(transformer)
    self._flattened_transformer_state: list[statelib.State] = jax.tree.leaves(
        self._transformer_state,
        is_leaf=lambda x: isinstance(x, nnx.Variable),
    )
    # We separate out state and graph def so that the state can be passed as an
    # argument to _decode_fn, resulting in it not being treated as a static
    # arg. This greatly reduces the size of the HLO and reduces compile time.
    #
    # We donate the sampling_state (argnum 1) containing the KV cache arrays.
    # JAX arrays are immutable, so updating the cache at each decoding step
    # would normally force JAX to allocate a new memory buffer and copy old
    # contents. Since the KV cache memory footprint scales with batch size and
    # prompt+decoding length (reaching gigabytes), this continuous reallocation
    # and copying triggers massive memory overhead and OOMs. Donating the input
    # state allows the XLA compiler to reuse the memory buffer in-place,
    # completely avoiding allocation/copy overhead.
    self._compiled_decode_fn = jax.jit(self._decode_fn, donate_argnums=(1,))
    self._compiled_prefill_fn = jax.jit(
        self._prefill_fn,
        donate_argnums=(1,),
        static_argnames=('echo',),
    )
    self._supports_decode_only_last_token = (
        'decode_only_last_token'
        in inspect.signature(transformer.__call__).parameters
    )

  def model_def_and_state(self) -> tuple[graph.NodeDef, statelib.State]:
    """Returns the transformer graphdef and state."""
    return self._transformer_graphdef, self._flattened_transformer_state

  @property
  def transformer(self) -> nnx.Module:
    return nnx.merge(  # pyrefly: ignore[no-matching-overload]
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

  @property
  def dtype(self) -> jnp.dtype:
    if hasattr(self.transformer, 'config') and (
        hasattr(self.transformer.config, 'dtype')
    ):
      return self.transformer.config.dtype
    return self._flattened_transformer_state[0].dtype

  def init_sample_state(
      self,
      all_input_ids: jax.Array,
      lens: jax.Array,
      total_sampling_steps: int,
      include_logits: bool = False,
      forbidden_token_ids: tuple[int, ...] | None = None,
      temperature: float = 0.0,
      top_p: Optional[float] = None,
      top_k: Optional[int] = None,
      seed: jax.Array | None = None,
      beam_size: Optional[int] = None,
      include_logprobs: bool = False,
  ) -> _SamplingState:
    """Initializes the sampling state given input prompts."""
    batch_size = lens.shape[0]
    
    num_pages_per_shard = self.cache_config.max_num_pages // self.cache_config.num_shards
    page_indices = jnp.zeros(
        (batch_size, self.cache_config.max_num_pages_per_seq_per_shard), dtype=jnp.int32
    )
    available_page_indices = jnp.arange(num_pages_per_shard, dtype=jnp.int32)
    num_available_pages = jnp.array(num_pages_per_shard, dtype=jnp.int32)
    pages = jnp.full(
        (self.cache_config.max_num_pages, self.cache_config.page_size), self.tokenizer.pad_id(), dtype=jnp.int32
    )
    # lens = jnp.zeros(batch_size, dtype=jnp.int32)
    
    total_num_input_tokens = jnp.sum(lens)
    token_buffer = TokenBuffer(
        pages=pages,
        page_indices=page_indices,
        available_page_indices=available_page_indices,
        num_available_pages=num_available_pages,
        lens=jnp.zeros_like(lens),
        config=self.cache_config,
        dtype=self.dtype
    )

    token_buffer = token_buffer.allocate(lens)
    token_buffer = token_buffer.load_prompt_tokens(all_input_ids, lens)

    cache = token_buffer.init_layer_caches()

    num_input_tokens = jnp.max(lens)
    done = jnp.zeros((batch_size,), dtype=jnp.bool_)

    if include_logits:
      logits_buffer = jnp.zeros(
          (batch_size, total_sampling_steps, self.transformer.num_embed),  # pyrefly: ignore[missing-attribute]
          dtype=jnp.float32,
      )
    else:
      logits_buffer = None

    if include_logprobs:
      logprobs_buffer = jnp.zeros(
          (batch_size, total_sampling_steps),
          dtype=jnp.float32,
      )
    else:
      logprobs_buffer = None
    sampling_parameters = {}
    sampling_mode = [None]

    if beam_size is not None:
      utils.check_sampling_mode_conflict(sampling_mode, 'beam_search')  # pyrefly: ignore[bad-argument-type]
      sampling_parameters['beam_size'] = beam_size

    if top_p is not None:
      utils.check_sampling_mode_conflict(sampling_mode, 'top_p')  # pyrefly: ignore[bad-argument-type]
      sampling_parameters['top_p'] = top_p
      sampling_parameters['top_k'] = top_k

    if sampling_mode[0] is None:
      sampling_mode[0] = 'greedy'  # pyrefly: ignore[unsupported-operation]

    logging.debug('Using sampling mode: %s', sampling_mode[0])

    return _SamplingState(
        decoding_step=num_input_tokens - 1,
        num_input_tokens=jnp.array(num_input_tokens, dtype=jnp.int32),
        total_num_input_tokens=jnp.array(total_num_input_tokens, dtype=jnp.int32),
        positions=None,
        token_buffer=token_buffer,
        logits_buffer=logits_buffer,
        logprobs_buffer=logprobs_buffer,
        cache=cache,
        done=done,
        total_sampling_steps=total_sampling_steps,
        forbidden_token_ids=forbidden_token_ids,
        temperature=temperature,
        sampling_parameters=sampling_parameters,
        seed=seed,
        sampling_mode=sampling_mode[0],
        beam_search_sampling_state=None,
    )

  def tokenize(self, input_string: str) -> np.ndarray | list[int]:
    """Tokenizes the input string."""
    input_ids = self.tokenizer.encode(input_string)
    bos_tok = [self.tokenizer.bos_id()] if self.tokenizer.bos_id() else []
    input_ids = np.array(
        self.tokenizer.dedup_bos_ids(bos_tok + input_ids), dtype=np.int32
    )
    return input_ids

  def _sample(
      self,
      logits: jnp.ndarray,
      eos: jax.Array,
      cache: dict[str, dict[str, jaxtyping.Array]],
      sampler_state: _SamplingState,
  ) -> _SamplingState:
    """Samples a token from the logits."""

    # logits = logits[:, -1][:, None, :]  # B, 1, V
    decoding_step = sampler_state.decoding_step
    token_buffer = sampler_state.token_buffer
    done = sampler_state.done
    logits_buffer = sampler_state.logits_buffer
    logprobs_buffer = sampler_state.logprobs_buffer
    beam_search_state = sampler_state.beam_search_sampling_state
    if sampler_state.forbidden_token_ids:
      logits = logits.at[:, :, sampler_state.forbidden_token_ids].set(-jnp.inf)

    if sampler_state.sampling_mode == 'beam_search':
      beam_search_state, updated_args = beam_search_lib.beam_search_step(
          logits=logits,
          done=done,
          token_buffer=token_buffer,
          cache=cache,
          logits_buffer=logits_buffer,
          state=beam_search_state,  # pyrefly: ignore[bad-argument-type]
          pad_token_id=eos[0],
          decoding_step=decoding_step,
          logprobs_buffer=logprobs_buffer,
      )
      cache = updated_args['cache']
      token_buffer = updated_args['token_buffer']
      done = updated_args['done']
      logits_buffer = updated_args['logits_buffer']
      logprobs_buffer = updated_args['logprobs_buffer']
    else:
      if sampler_state.sampling_mode == 'greedy':
        next_token_candidate, logp = sample_best(
            logits, return_logprobs=(logprobs_buffer is not None)
        )
      elif sampler_state.sampling_mode == 'top_p':
        key = jax.random.fold_in(sampler_state.seed, decoding_step)
        next_token_candidate, logp = sample_top_p(
            logits,
            key,
            sampler_state.temperature,
            sampler_state.sampling_parameters['top_p'],
            sampler_state.sampling_parameters['top_k'],  # pyrefly: ignore[bad-argument-type]
            return_logprobs=(logprobs_buffer is not None),
        )
      else:
        raise ValueError(
            'Unsupported sampling mode: %s' % sampler_state.sampling_mode
        )
      token_buffer, cache = token_buffer.append_tokens(
          next_token_candidate, cache=cache
      )
      if logprobs_buffer is not None:
        logprobs_buffer = logprobs_buffer.at[:, decoding_step + 1].set(logp)

    latest_tokens = token_buffer.get_token_at(decoding_step + 1)
    done = done | jnp.isin(latest_tokens, eos)
    return _SamplingState(
        decoding_step=sampler_state.decoding_step + 1,
        positions=None,
        num_input_tokens=sampler_state.num_input_tokens,
        total_num_input_tokens=sampler_state.total_num_input_tokens,
        token_buffer=token_buffer,
        logits_buffer=logits_buffer,
        logprobs_buffer=logprobs_buffer,
        cache=cache,
        done=done,
        total_sampling_steps=sampler_state.total_sampling_steps,
        forbidden_token_ids=sampler_state.forbidden_token_ids,
        temperature=sampler_state.temperature,
        sampling_parameters=sampler_state.sampling_parameters,
        seed=sampler_state.seed,
        sampling_mode=sampler_state.sampling_mode,
        beam_search_sampling_state=beam_search_state,
    )

  def _prefill_fn(
      self,
      params: statelib.State,
      sampler_state: _SamplingState,
      images: jnp.ndarray | None = None,
      audios: Any = None,
      echo: bool = True,
  ) -> _SamplingState:
    """Performs prefill."""
    batch_size = sampler_state.token_buffer.batch_size
    """
    tokens = sampler_state.token_buffer.get_slice(
        0, sampler_state.num_input_tokens
    )
    """

    if sampler_state.positions is not None:
      step_positions = sampler_state.positions[
          :, : sampler_state.num_input_tokens
      ]
    else:
      # TODO: This assumes positions can be infered
      # from lengths which is not always true
      pass
    
    """
    input_mask = tokens != self.tokenizer.pad_id()
    if hasattr(self.transformer, 'get_attention_mask'):
      attention_mask = self.transformer.get_attention_mask(
          tokens, inputs_mask=input_mask
      )
      seq_len = attention_mask.shape[-1]
      padding = self.cache_config.cache_size - seq_len
      attention_mask = jnp.pad(
          attention_mask,
          (*((0, 0) for _ in range(attention_mask.ndim - 1)), (0, padding)),
      )
    else:
      attention_mask = utils.make_causal_attn_mask(
          input_mask, self.cache_config.cache_size
      )
    """

    transformer = nnx.merge(self._transformer_graphdef, params)  # pyrefly: ignore[no-matching-overload]
    kwargs = {}
    if images is not None:
      kwargs['images'] = images
    if audios is not None:
      kwargs['audios'] = audios
    decode_only_last_token = self._supports_decode_only_last_token and not echo
    if decode_only_last_token:
      kwargs['decode_only_last_token'] = True

    tokens = sampler_state.token_buffer.to_array(sampler_state.total_num_input_tokens)
    position_ragged = RaggedArray(
        data=tokens,
        lens=sampler_state.token_buffer.lens,
    )
    step_positions = position_ragged.intra_offsets

    logits, cache = transformer(
        tokens,
        step_positions,
        sampler_state.cache,
        seq_lens=sampler_state.token_buffer.lens,
        **kwargs,
    )

    token_buffer = sampler_state.token_buffer
    done = sampler_state.done
    positions = sampler_state.positions
    beam_search_sampling_state = None
    if sampler_state.logits_buffer is not None:
      start_idx = (
          sampler_state.num_input_tokens if decode_only_last_token else 1
      )
      logits_buffer = jax.lax.dynamic_update_slice(
          sampler_state.logits_buffer,
          logits.astype(sampler_state.logits_buffer.dtype),
          (0, start_idx, 0),
      )
    else:
      logits_buffer = sampler_state.logits_buffer

    if sampler_state.sampling_mode == 'beam_search':
      sampling_state, updated_args = beam_search_lib.init_batched_beam_state(
          logits=logits,
          input_token_buffer=sampler_state.token_buffer,
          initial_cache=cache,
          done=sampler_state.done,
          positions=sampler_state.positions,
          logits_buffer=sampler_state.logits_buffer,
          beam_size=int(sampler_state.sampling_parameters['beam_size']),
      )
      beam_search_sampling_state = sampling_state
      logits = updated_args['logits']
      cache = updated_args['cache']
      token_buffer = updated_args['token_buffer']
      done = updated_args['done']
      positions = updated_args['positions']
      logits_buffer = updated_args['logits_buffer']

    updated_sampling_state = _SamplingState(
        decoding_step=sampler_state.decoding_step,
        num_input_tokens=sampler_state.num_input_tokens,
        total_num_input_tokens=sampler_state.total_num_input_tokens,
        token_buffer=token_buffer,
        positions=positions,
        logits_buffer=logits_buffer,
        logprobs_buffer=sampler_state.logprobs_buffer,
        cache=cache,
        done=done,
        total_sampling_steps=sampler_state.total_sampling_steps,
        forbidden_token_ids=sampler_state.forbidden_token_ids,
        temperature=sampler_state.temperature,
        sampling_parameters=sampler_state.sampling_parameters,
        seed=sampler_state.seed,
        sampling_mode=sampler_state.sampling_mode,
        beam_search_sampling_state=beam_search_sampling_state,
    )
    updated_sampler_state = self._sample(
        logits=logits,
        cache=cache,
        eos=self.eos_ids,
        sampler_state=updated_sampling_state,
    )
    return updated_sampler_state

  def _decode_fn(
      self,
      params: statelib.State,
      sampling_state: _SamplingState,
  ) -> _SamplingState:
    """Internal generating function (to be jitted)."""

    def sample_with_params(sampler_state: _SamplingState):
      return self._sample_step(params, sampler_state)

    def cond_fn(sampler_state: _SamplingState):
      return (
          sampler_state.decoding_step < sampler_state.total_sampling_steps - 1
      ) & jnp.any(jnp.logical_not(sampler_state.done))

    return jax.lax.while_loop(cond_fn, sample_with_params, sampling_state)

  def _sample_step(
      self, params: statelib.State, sampler_state: _SamplingState
  ) -> _SamplingState:
    """Performs a single sampling step."""
    batch_size = sampler_state.token_buffer.batch_size
    decoding_step = sampler_state.decoding_step

    last_token = sampler_state.token_buffer.get_token_at(decoding_step)[:, None].reshape(-1)
    if sampler_state.positions is not None:
      step_positions = jnp.expand_dims(
          sampler_state.positions[:, decoding_step], -1
      )
    else:
      step_positions = jnp.full((batch_size, 1), decoding_step, dtype=jnp.int32)
    
    """
    full_tokens = sampler_state.token_buffer.get_slice(0, decoding_step + 1)
    input_mask = full_tokens == self.tokenizer.pad_id()
    attention_mask = utils.compute_attention_masks(
        decoding_step, self.cache_config.cache_size, input_mask
    )
    """

    transformer = nnx.merge(self._transformer_graphdef, params)  # pyrefly: ignore[no-matching-overload]
    logits, cache = transformer(
        last_token,
        positions=step_positions,
        cache=sampler_state.cache,
        seq_lens=sampler_state.token_buffer.lens,
    )
    logits = jnp.expand_dims(logits, axis=1)

    updated_sampler_state = self._sample(
        logits=logits,
        cache=cache,
        eos=self.eos_ids,
        sampler_state=sampler_state,
    )

    if updated_sampler_state.logits_buffer is not None:
      next_logits = jnp.squeeze(logits, 1)
      logits_buffer = updated_sampler_state.logits_buffer.at[
          :, decoding_step + 1
      ].set(next_logits)
    else:
      logits_buffer = None

    updated_sampler_state = dataclasses.replace(
        updated_sampler_state,
        logits_buffer=logits_buffer,
    )
    return updated_sampler_state


  def __call__(
      self,
      input_strings: str | Sequence[str],
      max_generation_steps: int,
      max_prompt_length: int | None = None,
      echo: bool = False,
      return_logits: bool = False,
      return_logprobs: bool = False,
      eos_tokens: Sequence[int] | None = None,
      forbidden_tokens: Iterable[int] | None = None,
      temperature: float = 0.0,
      top_p: Optional[float] = None,
      top_k: Optional[int] = None,
      beam_size: Optional[int] = None,
      seed: int | None = None,
      pad_output: bool = False,
      images: (
          str
          | np.ndarray
          | list[str | np.ndarray | list[str | np.ndarray] | None]
          | jnp.ndarray
          | None
      ) = None,
      audios: (
          np.ndarray | list[np.ndarray | list[np.ndarray] | None] | None
      ) = None,
      max_audio_length: int | None = None,
      max_audio_clips: int | None = None,
  ) -> base_sampler.SamplerOutput:
    """Samples a completion of the input string(s)."""
    self.eos_ids = jnp.array(eos_tokens or [self.tokenizer.eos_id()])
    input_strings = (
        [input_strings] if isinstance(input_strings, str) else input_strings
    )

    forbidden_token_ids = tuple(forbidden_tokens) if forbidden_tokens else None

    tokens = [self.tokenize(x) for x in input_strings]

    is_gemma4 = self.transformer.__class__.__name__ == 'Gemma4'

    processed_images = None
    if is_gemma4 and images is not None:
      assert hasattr(self.transformer, 'vision_encoder')
      assert self.transformer.vision_encoder is not None
      processed_images, tokens = image_processor.process_gemma4_inputs(
          images,
          tokens,  # pyrefly: ignore[bad-argument-type]
          self.transformer.vision_encoder,
          self.tokenizer.pad_id(),
      )

    elif images is not None and self.image_processor is not None:
      processed_images = self.image_processor(images)  # pyrefly: ignore[bad-argument-type]
      processed_images = jnp.array(processed_images)

    processed_audios = None
    if audios is not None:
      if is_gemma4:
        assert hasattr(self.transformer, 'audio_encoder')
        assert self.transformer.audio_encoder is not None
        processed_audios, tokens = audio_processor.process_gemma4_inputs(
            audios=audios,  # pyrefly: ignore[bad-argument-type]
            tokens=tokens,  # pyrefly: ignore[bad-argument-type]
            audio_encoder=self.transformer.audio_encoder,
            max_audio_length=max_audio_length,
            max_audio_clips=max_audio_clips,
        )
      else:
        raise NotImplementedError('Audio support only implemented for Gemma4.')
    
    # Ragged Attention kernel expects 1D token buffer of conctaned seqs.
    # So we pass flat tokens here. 
    # Token Buffer: 
    # [t_{S_1, 0}, t_{S_1, 1}, ..., t_{S_1, l_1 -1}, t_{S_2, 0}, 
    # ..., t_{S_n, l_n - 1}, PAD, ..., PAD]
    # Lens:
    # [l_1, l_2, ..., l_n]

    lens = np.array([len(x) for x in tokens], dtype=np.int32)
    flat_tokens = np.concatenate(tokens)
    all_input_ids = jnp.array(flat_tokens)

    max_tokens_length = int(np.max(lens))
    if max_prompt_length is None or max_prompt_length < max_tokens_length:
      max_prompt_length = utils.next_power_of_2(max_tokens_length)

    total_sampling_steps = max_prompt_length + max_generation_steps
    if total_sampling_steps > self.cache_config.cache_size:
      raise ValueError(
          f'Total sampling steps {total_sampling_steps} must be less than the'
          f' cache size {self.cache_config.cache_size}.'
      )

    if seed is None:
      seed = jax.random.PRNGKey(0)  # pyrefly: ignore[bad-assignment]
    elif isinstance(seed, int):
      seed = jax.random.PRNGKey(seed)  # pyrefly: ignore[bad-assignment]

    sampling_state = self.init_sample_state(
        all_input_ids=all_input_ids,
        lens=jnp.array(lens),
        total_sampling_steps=total_sampling_steps,
        include_logits=return_logits,
        forbidden_token_ids=forbidden_token_ids,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,  # pyrefly: ignore[bad-argument-type]
        beam_size=beam_size,
        include_logprobs=return_logprobs,
    )
    sampling_state = self._compiled_prefill_fn(
        self._flattened_transformer_state,
        sampling_state,
        images=processed_images,
        audios=processed_audios,
        echo=echo,
    )

    sampling_state = self._compiled_decode_fn(
        self._flattened_transformer_state, sampling_state
    )
    ragged_token_buffer = sampling_state.token_buffer.to_array(total_sampling_steps)
    ragged_token_buffer = jax.device_get(ragged_token_buffer)
    seq_lens = jax.device_get(sampling_state.token_buffer.lens)
    logits_buffers = sampling_state.logits_buffer

    # Sampling state token buffer is a ragged array with no padding. 
    # We need to to revert it back to a 2D array with padding
    batch_size = len(tokens)
    """
    token_buffers = jnp.full(
        (
            batch_size,
            total_sampling_steps,
        ),
        self.tokenizer.pad_id(),
        dtype=jnp.int32,
    )
    """

    start_indices = jnp.pad(jnp.cumsum(seq_lens)[:-1], (1, 0))

    def unpack_left_padded_seq(start_idx, length):
        raw_slice = jax.lax.dynamic_slice(ragged_token_buffer, (start_idx,), (total_sampling_steps,))
        valid_mask = jnp.arange(total_sampling_steps) < length
        right_padded = jnp.where(valid_mask, raw_slice, self.tokenizer.pad_id())
        
        shift_amount = total_sampling_steps - length
        left_padded = jnp.roll(right_padded, shift_amount)
        
        return left_padded

    token_buffers = jax.vmap(unpack_left_padded_seq)(start_indices, seq_lens)
    
    # We also need to recompute all_input_ids with padding
    all_input_ids = np.array([
        utils.pad_to_length(
            x,  # pyrefly: ignore[bad-argument-type]
            target_length=max_prompt_length,
            pad_value=self.tokenizer.pad_id(),
            left=True,
        )
        for x in tokens
    ])

    """
    total_len = 0
    token_buffers = np.array(
        (
            batch_size,
            total_sampling_steps,
        ),
        self.tokenizer.pad_id(),
        dtype=jnp.int32,
    ) 
    
    # TODO: This should be batched 
    for i in range(batch_size):
      seq = ragged_token_buffers[total_len: total_len + seq_lens[i]]
      end_idx = max_token_len + seq_lens[i]
      token_buffers[i][max_token_len: end_idx] = seq
    """ 

    final_logprobs_buffer = sampling_state.logprobs_buffer

    if sampling_state.sampling_mode == 'beam_search':
      updated_args = beam_search_lib.finalize_beam_search_state(
          sampling_state.beam_search_sampling_state,
          sampling_state.token_buffer,
          sampling_state.logits_buffer,
          sampling_state.logprobs_buffer,
      )
      token_buffers = updated_args['token_buffer']
      logits_buffers = updated_args['logits_buffer']
      final_logprobs_buffer = updated_args['logprobs_buffer']
      # delete the sampling state in case the further referece
      # if need more internal states, they should be updated by
      # finalize_beam_search_state
      del sampling_state

    if pad_output:
      max_len = total_sampling_steps if echo else max_generation_steps
      lengths, out_tokens, out_logits = utils.padded_fill_tokens_and_logits(
          token_buffers,
          logits_buffers,
          return_logits,
          echo,
          self.tokenizer.pad_id(),
          self.eos_ids,
          max_prompt_length,
          max_len,
      )
      out_tokens, lengths = jax.device_get(out_tokens), jax.device_get(lengths)
      decoded_outputs = [
          self.tokenizer.decode(tokens[:length].tolist())
          for tokens, length in zip(out_tokens, lengths)
      ]
      out_logprobs = []
      if return_logprobs:
        token_buffers = jax.device_get(token_buffers)
        final_logprobs_buffer = jax.device_get(final_logprobs_buffer)
        for i in range(len(token_buffers)):
          start_idx = (
              utils.np_find_first_non_pad_idx(
                  token_buffers[i], self.tokenizer.pad_id()
              )
              if echo
              else max_prompt_length
          )
          end_idx = (
              utils.np_find_first_eos_idx(
                  token_buffers[i][max_prompt_length:], self.eos_ids
              )
              + max_prompt_length
          )
          length = end_idx - start_idx
          # Slice logprobs and pad to max_len
          sliced_logprobs = final_logprobs_buffer[i][start_idx:end_idx]
          padded_logprobs = np.pad(
              sliced_logprobs,
              (0, max_len - length),
              mode='constant',
              constant_values=0.0,
          )
          out_logprobs.append(padded_logprobs.tolist())

    else:
      out_tokens = []
      out_logits = []
      out_logprobs = []
      token_buffers = jax.device_get(token_buffers)
      if return_logprobs:
        final_logprobs_buffer = jax.device_get(final_logprobs_buffer)
      if return_logits:
        logits_buffers = jax.device_get(logits_buffers)
      for i in range(len(token_buffers)):
        token_buffer = token_buffers[i]
        start_idx = (
            utils.np_find_first_non_pad_idx(
                token_buffer, self.tokenizer.pad_id()
            )
            if echo
            else max_prompt_length
        )
        end_idx = (
            utils.np_find_first_eos_idx(
                token_buffer[max_prompt_length:], self.eos_ids
            )
            + max_prompt_length
        )
        out_tokens.append(token_buffer[start_idx:end_idx])
        if return_logits:
          out_logits.append(logits_buffers[i][start_idx:end_idx])
        if return_logprobs:
          # Extract logprobs for the generated tokens
          out_logprobs.append(
              final_logprobs_buffer[i][start_idx:end_idx].tolist()
          )

      decoded_outputs = [
          self.tokenizer.decode(tokens.tolist()) for tokens in out_tokens
      ]
    result = base_sampler.SamplerOutput(
        text=decoded_outputs,
        logits=out_logits if return_logits else [],
        tokens=out_tokens,
        padded_prompt_tokens=all_input_ids,
        logprobs=out_logprobs if return_logprobs else None,
    )
    return result
