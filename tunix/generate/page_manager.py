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

# from __future__ import annotations

from collections.abc import Iterable, Sequence
import dataclasses
import functools
from typing import Any, Optional, Self

import jax
import jax.numpy as jnp
import numpy as np
from tunix.generate import utils

def _get_dtype_packing(dtype):
  bits = jax._src.dtypes.bit_width(dtype)
  return 32 // bits

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
class PageManagerConfig:
  page_size: int
  max_seq_len: int
  num_pages: int

  num_kv_heads: int
  max_num_seqs: int
  head_dim: int
  dtype: jax.typing.DTypeLike

  dp_axis: str | None = None
  tp_axis: str | None = None

  @property
  def max_num_pages_per_seq(self) -> int:
    return int(self.max_seq_len / self.page_size)
  
  def init(self, device: jax.Device | None = None) -> page_manager_lib.PageManager:
    """Explicitly initializes physical page tensors for a PageManager pool, placing CPU caches on host memory."""
    kv_packing = _get_dtype_packing(self.dtype)

    blocks: dict[str, jax.Array] = {}

    token_block = jax.lax.empty(
        (self.num_pages, self.page_size), dtype=jnp.int32
    )
    if dp_axis is not None:
      token_block = utils.shard(token_block, (dp_axis, None))
    if device is not None:
      token_block = jax.device_put(token_block, device)
    blocks["token_buffer"] = token_block

    for i in range(self.cache_config.num_layers):
      layer_block = jax.lax.empty(
          (
              num_pages,
              page_size,
              2 * self.num_kv_heads // kv_packing,
              kv_packing,
              self.head_dim,
          ),
          dtype=self.dtype,
      )
      if dp_axis is not None or tp_axis is not None:
        layer_block = utils.shard(layer_block, (dp_axis, None, tp_axis, None, None))
      if device is not None:
        layer_block = jax.device_put(layer_block, device)
      blocks[f"layer_{i}"] = layer_block

    page_indices = jnp.zeros((self.max_num_seqs, self.max_num_pages_per_seq), dtype=jnp.int32)
    available_page_indices = jnp.arange(self.num_pages, dtype=jnp.int32)
    num_available_pages = jnp.array(self.num_pages, dtype=jnp.int32)
    seq_lens = jnp.zeros((self.max_num_seqs,), dtype=jnp.int32)

    if device is not None:
      page_indices = jax.device_put(page_indices, device)
      available_page_indices = jax.device_put(available_page_indices, device)
      num_available_pages = jax.device_put(num_available_pages, device)
      seq_lens = jax.device_put(seq_lens, device)

    return PageManager(
        pages=blocks,
        page_indices=page_indices,
        available_page_indices=available_page_indices,
        num_available_pages=num_available_pages,
        seq_lens=seq_lens,
        page_size=page_size,
        max_seq_len=max_seq_len,
        window_size=None,
    )
      

@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class PageManager:
  """Page state and data blocks for a batch of sequences."""
  pages: dict[str, jax.Array]
  page_indices: jax.Array  # i32[batch_size, max_num_pages_per_seq]
  available_page_indices: jax.Array  # i32[total_num_pages]
  num_available_pages: jax.Array  # i32 scalar
  seq_lens: jax.Array # i32[batch_size]
  
  page_size: int = dataclasses.field(
      metadata={'static': True}
  )
  max_seq_len: int = dataclasses.field(
      metadata={'static': True}
  )
  window_size: int | None = dataclasses.field(
      metadata={'static': True}
  )

  @property
  def batch_size(self) -> int:
    return self.seq_lens.shape[0]

  @property
  def max_num_pages_per_seq(self) -> int:
    return self.max_seq_len / self.page_size

  @property
  def total_num_pages(self) -> int:
    return self.available_page_indices.shape[0]

  @property
  def lens(self) -> jax.Array:
    return self.seq_lens

  @property
  def kv_lens(self) -> jax.Array:
    return self.seq_lens

  @functools.cached_property
  def num_pages(self) -> jax.Array:
    return utils.cdiv(self.seq_lens, self.page_size)

  @jax.named_call
  def allocate(self, q_lens: jax.Array) -> "PageManager":
    """Allocates pages for new tokens."""
    pages_required = utils.cdiv(self.seq_lens + q_lens, self.page_size)

    num_pages_to_allocate = pages_required - self.num_pages

    page_indices_to_allocate = RaggedArray(
        data=self.available_page_indices, lens=num_pages_to_allocate
    )
    page_indices_rows = page_indices_to_allocate.row_idxs
    page_indices_cols = (
        self.num_pages[page_indices_rows]
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

    return dataclasses.replace(
        self,
        seq_lens=self.seq_lens + q_lens,
        page_indices=updated_page_indices,
        available_page_indices=updated_available_page_indices,
        num_available_pages=updated_num_available_pages,
    )

  @jax.named_call
  def release(self, should_release: jax.Array) -> "PageManager":
    """Releases pages for completed sequences."""
    updated_lens = jnp.where(should_release, 0, self.seq_lens)

    page_indices_to_release = RaggedArray(
        data=jax.lax.empty((self.total_num_pages,), dtype=jnp.int32),
        lens=jnp.where(should_release, self.num_pages, 0),
    )
    page_indices_irows = page_indices_to_release.row_idxs
    page_indices_icols = page_indices_to_release.intra_offsets

    updated_available_page_indices = self.available_page_indices.at[
        jnp.arange(self.total_num_pages) + self.num_available_pages
    ].set(self.page_indices[page_indices_irows, page_indices_icols])

    updated_num_available_pages = (
        self.num_available_pages + page_indices_to_release.total_length
    )
    return dataclasses.replace(
        self,
        seq_lens=updated_lens,
        available_page_indices=updated_available_page_indices,
        num_available_pages=updated_num_available_pages,
    )

  @jax.named_call
  def release_for_window(self) -> "PageManager":
    """Release allocations for window."""
    if self.window_size is None:
      return self

    num_pages_to_release = (
        jnp.maximum(self.seq_lens - self.window_size, 0)
        // self.page_size
    )
    page_indices_irows = jnp.arange(self.batch_size)[:, None]
    page_indices_icols = (
        jnp.arange(self.max_num_pages_per_seq)
        + num_pages_to_release[:, None]
    )
    updated_page_indices = self.page_indices[
        page_indices_irows, page_indices_icols
    ]
    release_helper = RaggedArray(
        data=jax.lax.empty((self.total_num_pages,), dtype=jnp.int32),
        lens=num_pages_to_release,
    )
    released_page_indices = self.page_indices[
        release_helper.row_idxs, release_helper.intra_offsets
    ]
    updated_available_page_indices = self.available_page_indices.at[
        jnp.arange(self.total_num_pages) + self.num_available_pages
    ].set(released_page_indices, mode='drop')
    return dataclasses.replace(
        self,
        page_indices=updated_page_indices,
        available_page_indices=updated_available_page_indices,
        num_available_pages=self.num_available_pages
        + release_helper.total_length,
        seq_lens=self.seq_lens - num_pages_to_release * self.page_size,
    )

  def load_prompt_tokens(
      self, prompt_tokens: jax.Array, lens: jax.Array, key: str = "token_buffer"
  ) -> "PageManager":
    """Loads packed 1D prompt tokens into allocated paged memory."""
    token_ragged = RaggedArray(data=prompt_tokens, lens=lens)
    seq_idxs = token_ragged.row_idxs
    token_offsets = token_ragged.intra_offsets

    local_page_cols = (token_offsets // self.page_size) 
    page_offsets = token_offsets % self.page_size
    phys_page_ids = self.page_indices[seq_idxs, local_page_cols]
    
    updated_layer_pages = self.pages[key].at[phys_page_ids, page_offsets].set(
        prompt_tokens
    )
    new_pages = {**self.pages, key: updated_layer_pages}

    return dataclasses.replace(
          self, 
          pages=new_pages,
    )


  def append_tokens(
      self, tokens: jax.Array, valid_mask: jax.Array | None = None, key: str = "token_buffer"
  ) -> "PageManager":
    """Appends 1 new token per sequence to paged memory."""
    if valid_mask is None:
      valid_mask = jnp.ones(self.batch_size, dtype=jnp.int32)
      
    pm = self.allocate(valid_mask)

    token_offsets = pm.seq_lens - 1  # position of new token
    local_page_cols = (token_offsets // pm.page_size)
    page_offsets = token_offsets % pm.page_size
    seq_idxs = jnp.arange(pm.batch_size)
    phys_page_ids = pm.page_indices[seq_idxs, local_page_cols]
    
    updated_layer_pages = pm.pages[key].at[phys_page_ids, page_offsets].set(tokens)
    new_pages = {**pm.pages, key: updated_layer_pages}

    return dataclasses.replace(
          pm, 
          pages=new_pages,
    )

  def get_token_at(self, pos: int | jax.Array, key: str = "token_buffer") -> jax.Array:
    """Directly looks up token IDs at sequence position `pos` using page_indices."""
    seq_idxs = jnp.arange(self.batch_size)
    local_page_col = (pos // self.page_size) 
    page_offset = pos % self.page_size
    phys_page_ids = self.page_indices[seq_idxs, local_page_col]
    return self.pages[key][phys_page_ids, page_offset]

  def get_slice(self, start: int | jax.Array, length: int, key: str = "token_buffer") -> jax.Array:
    """Directly looks up a slice of token IDs [batch_size, length] starting at `start`."""
    seq_grid = jnp.arange(self.batch_size)[:, None]
    pos_grid = start + jnp.arange(length)[None, :]

    local_page_cols = (pos_grid // self.page_size)
    page_offsets = pos_grid % self.page_size
    safe_cols = jnp.clip(
        local_page_cols, 0, self.max_num_pages_per_seq - 1
    )
    phys_page_ids = self.page_indices[seq_grid, safe_cols]
    return self.pages[key][phys_page_ids, page_offsets]

  def to_array(
      self,
      total_num_tokens: int,
      block_id: str = "token_buffer",
  ) -> jax.Array:
    """Extracts array of token IDs from paged memory."""
    token_ragged = RaggedArray(
        data=jnp.zeros((total_num_tokens,), dtype=jnp.int32),
        lens=self.seq_lens,
    )
    seq_idxs = token_ragged.row_idxs
    token_offsets = token_ragged.intra_offsets

    local_page_cols = (token_offsets // self.page_size)
    page_offsets = token_offsets % self.page_size
    phys_page_ids = self.page_indices[seq_idxs, local_page_cols]

    # Gather exact valid tokens per sequence
    packed_tokens = self.pages[block_id][phys_page_ids, page_offsets]
    return packed_tokens

def batch_copy_pages(
    src_cache: PageManager,
    dst_cache: PageManager,
    src_slots: Sequence[int],
    dst_slots: Sequence[int],
    transfer_kv: bool = True,
) -> PageManager:
  """Copy pages across specified slots, optionally copying only token pages."""
  if len(src_slots) == 0:
    return dst_cache
  
  src_idxs_list = []
  dst_idxs_list = []
  for s_slot, d_slot in zip(src_slots, dst_slots):
    seq_len = int(src_cache.seq_lens[s_slot])
    if seq_len == 0:
      continue
    num_pages = (seq_len + src_cache.page_size - 1) // src_cache.page_size
    src_idxs_list.append(src_cache.page_indices[s_slot, :num_pages])
    dst_idxs_list.append(dst_cache.page_indices[d_slot, :num_pages])

  if not src_idxs_list:
    return dst_cache

  src_idxs = jnp.concatenate(src_idxs_list)
  dst_idxs = jnp.concatenate(dst_idxs_list)

  if not transfer_kv:
    src_tensor = src_cache.pages["token_buffer"]
    dst_tensor = dst_cache.pages.get("token_buffer", jnp.zeros_like(src_tensor))

    src_slice = src_tensor[src_idxs]
    src_slice = put_on_target_device(src_slice, dst_tensor)
    
    dst_cache.pages["token_buffer"] = dst_tensor.at[dst_idxs].set(src_slice)
    return dst_cache

  for key, src_tensor in src_cache.pages.items():
    dst_tensor = dst_cache.pages.get(key, jnp.zeros_like(src_tensor))

    src_slice = src_tensor[src_idxs]
    src_slice = put_on_target_device(src_slice, dst_tensor)

    dst_cache.pages[key] = dst_tensor.at[dst_idxs].set(src_slice)

  return dst_cache

def _remove_dp_spec(spec: P) -> P:
    """Replaces any 'dp' instance in a PartitionSpec with None."""
    # Convert PartitionSpec to a tuple, replace 'dp', and rebuild the spec
    dp_axis = ['dp', 'fsdp']
    new_spec = tuple(None if axis in dp_axis else axis for axis in spec)
    return P(*new_spec)

def put_on_target_device(tensor: jax.Array, target_tensor: jax.Array) -> jax.Array:
  """Safely places tensor on the same device/mesh as target_tensor."""
  if hasattr(target_tensor, "sharding") and target_tensor.sharding is not None:
    sharding = target_tensor.sharding
    if isinstance(sharding, jax.sharding.NamedSharding):
      # Replicate the tensor in the case of dp
      safe_spec = _remove_dp_spec(sharding.spec)
      target_sharding = jax.sharding.NamedSharding(sharding.mesh, safe_spec)
      return jax.device_put(tensor, target_sharding)

    elif isinstance(sharding, jax.sharding.SingleDeviceSharding):
      return jax.device_put(tensor, sharding)

  if hasattr(target_tensor, "devices") and len(target_tensor.devices()) > 0:
    return jax.device_put(tensor, list(target_tensor.devices())[0])

  return tensor
