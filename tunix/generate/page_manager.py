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
from jax.sharding import PartitionSpec as P
from tunix.generate import utils

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

@dataclasses.dataclass
class BlockSpec:
  name: str
  dtype: jnp.dtype
  subshape: tuple[int, ...] = () 
  logical_subsharding: Optional[tuple] = None  
  device: Optional[Any] = None

@dataclasses.dataclass(frozen=True, kw_only=True)
class PageManagerConfig:
  page_size: int
  max_num_seqs: int
  max_seq_len: int
  max_bytes: int
  
  block_specs: Sequence[BlockSpec]
  logical_page_sharding: str | None = None

  dp_axis: str | None = None
  tp_axis: str | None = None
  dp_size: int = 1
  tp_size: int = 1
  device: any = None

  @property
  def num_pages_per_block(self) -> int:
    """How many pages each block should have based on the max_bytes budget."""
    page_size_across_blocks = 0

    for spec in self.block_specs:
        item_size = jnp.dtype(spec.dtype).itemsize
        
        # Shape and sharding for a single page (ignoring the 0th dim: num_pages)
        page_shape = (self.page_size,) + spec.subshape
        page_sharding = self.get_logical_block_spec_sharding(spec)[1:]
        
        elements = 1
        for dim, shard in zip(page_shape, page_sharding):
            dim_size = (dim * self.dp_size) if shard == "dp_axis" else dim
            elements *= dim_size
            
        page_size_across_blocks += elements * item_size

    pages_per_block = self.max_bytes // page_size_across_blocks

    # Align down to a multiple of dp_size if the page dimension itself is sharded.
    if self.logical_page_sharding == "dp_axis":
      pages_per_block = (pages_per_block // self.dp_size) * self.dp_size

    return pages_per_block

  @property
  def max_num_pages_per_seq(self) -> int:
    return self.max_seq_len // self.page_size
  
  @property
  def logical_shard_to_physical(self) -> dict:
    return {
        "dp_axis": self.dp_axis,
        "tp_axis": self.tp_axis,
        None: None
    } 

  def get_logical_block_spec_sharding(self, spec: BlockSpec):
    logical_page_sharding = self.logical_page_sharding
    logical_subsharding = spec.logical_subsharding
    
    page_prefix = (logical_page_sharding, None)
    sub_suffix = logical_subsharding if logical_subsharding is not None else ()
    logical_sharding = page_prefix + sub_suffix 
    
    return logical_sharding


  def get_block_spec_sharding(self, spec: BlockSpec):
    logical_sharding = self.get_logical_block_spec_sharding(spec)
    
    if all(axis is None for axis in logical_sharding):
      return None
    
    mapping = self.logical_shard_to_physical

    physical_sharding = []
    for axis in logical_sharding:
      if axis not in mapping:
        raise ValueError(
          f'Invalid logical sharding axis: "{axis}" in block "{spec.name}"'
          f'Allowed axes are: {list(mapping.keys())}'
        )
      physical_sharding.append(mapping[axis])
    return physical_sharding 
      
  def init(self) -> "PageManager":
    """Initializes physical page tensors for a pool of page blocks."""
    blocks: dict[str, jax.Array] = {}

    for spec in self.block_specs:
      shape = (self.num_pages_per_block, self.page_size) + spec.subshape
      block = jax.lax.empty(shape, dtype=spec.dtype)
      sharding = self.get_block_spec_sharding(spec)
      
      if sharding is not None:
        block = utils.shard(block, sharding)
          
      if self.device is not None:
        block = jax.device_put(block, self.device)
          
      blocks[spec.name] = block

    page_indices = jnp.zeros((self.max_num_seqs, self.max_num_pages_per_seq), dtype=jnp.int32)
    available_page_indices = jnp.arange(self.num_pages_per_block, dtype=jnp.int32)
    num_available_pages = jnp.array(self.num_pages_per_block, dtype=jnp.int32)
    seq_lens = jnp.zeros((self.max_num_seqs,), dtype=jnp.int32)

    if self.device is not None:
        page_indices = jax.device_put(page_indices, self.device)
        available_page_indices = jax.device_put(available_page_indices, self.device)
        num_available_pages = jax.device_put(num_available_pages, self.device)
        seq_lens = jax.device_put(seq_lens, self.device)

    return PageManager(
        pages=blocks,
        page_indices=page_indices,
        available_page_indices=available_page_indices,
        num_available_pages=num_available_pages,
        seq_lens=seq_lens,
        page_size=self.page_size,
        max_seq_len=self.max_seq_len,
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
    return self.max_seq_len // self.page_size

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

  def load_values(
      self, prompt_tokens: jax.Array, lens: jax.Array, block_id: str = "tokens"
  ) -> "PageManager":
    """Loads packed 1D array of values into allocated paged memory of block."""
    token_ragged = RaggedArray(data=prompt_tokens, lens=lens)
    seq_idxs = token_ragged.row_idxs
    token_offsets = token_ragged.intra_offsets

    local_page_cols = (token_offsets // self.page_size) 
    page_offsets = token_offsets % self.page_size
    phys_page_ids = self.page_indices[seq_idxs, local_page_cols]
    
    updated_layer_pages = self.pages[block_id].at[phys_page_ids, page_offsets].set(
        prompt_tokens
    )
    new_pages = {**self.pages, block_id: updated_layer_pages}

    return dataclasses.replace(
          self, 
          pages=new_pages,
    )


  def insert_values(
      self, values: jax.Array, valid_mask: jax.Array | None = None, block_id: str = "tokens"
  ) -> "PageManager":
    """Insert 1 new token per sequence to the last allocated idx in paged memory."""
    if valid_mask is None:
      valid_mask = jnp.ones(self.batch_size, dtype=jnp.int32)
      
    value_offsets = self.seq_lens - 1  # position of new value 
    local_page_cols = (value_offsets // self.page_size)
    page_offsets = value_offsets % self.page_size
    seq_idxs = jnp.arange(self.batch_size)
    phys_page_ids = self.page_indices[seq_idxs, local_page_cols]
    
    # Point invalid sequences to out-of-bounds
    max_n_pages = self.max_num_pages_per_seq
    phys_page_ids = jnp.where(valid_mask, phys_page_ids, max_n_pages)
 
    updated_layer_pages = self.pages[block_id].at[phys_page_ids, page_offsets].set(
      values,
      mode="drop" 
    )
    new_pages = {**self.pages, block_id: updated_layer_pages}

    return dataclasses.replace(
          self, 
          pages=new_pages,
    )

  def to_array(
      self,
      total_num_tokens: int,
      block_id: str = "tokens",
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
    block_ids: Sequence[str],
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
  
  new_pages = {**dst_cache.pages}
  for block_id in block_ids:
    src_tensor = src_cache.pages[block_id]
    dst_tensor = dst_cache.pages.get(block_id, jnp.zeros_like(src_tensor))

    src_slice = src_tensor[src_idxs]
    src_slice = _put_on_target_device(src_slice, dst_tensor)

    dst_cache.pages[block_id] = dst_tensor.at[dst_idxs].set(src_slice)

  return dst_cache

def _remove_dp_spec(spec: P) -> P:
    """Replaces any 'dp' instance in a PartitionSpec with None."""
    # Convert PartitionSpec to a tuple, replace 'dp', and rebuild the spec
    dp_axis = ['dp', 'fsdp']
    new_spec = tuple(None if axis in dp_axis else axis for axis in spec)
    return P(*new_spec)

def _put_on_target_device(tensor: jax.Array, target_tensor: jax.Array) -> jax.Array:
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
