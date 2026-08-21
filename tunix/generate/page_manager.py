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
  logical_subsharding: tuple = () 
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
    
    if page_size_across_blocks == 0:
      return 0

    pages_per_block = self.max_bytes // page_size_across_blocks

    # Align down to a multiple of dp_size if the page dimension itself is sharded.
    if self.logical_page_sharding == "dp_axis":
      pages_per_block = (pages_per_block // self.dp_size) * self.dp_size


    return pages_per_block

  @property
  def max_num_pages_per_seq(self) -> int:
    return utils.cdiv(self.max_seq_len, self.page_size)
  
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

    logical_prefix_sharding = (logical_page_sharding, None)
    
    l_subsharding = len(logical_subsharding)
    l_subshape = len(spec.subshape)

    if l_subsharding > l_subshape:
      raise ValueError(
        f'Cannot initialize BlockSpec {spec.name}. '
        f'Block subsharding `{spec.logical_subsharding}` '
        f'cannot have length greater than block subshape '
        f'`{spec.subshape}`'
      ) 

    if l_subsharding < l_subshape:
      l_pad = l_subshape - l_subsharding
      logical_subsharding += (None,) * l_pad 

    logical_sharding = logical_prefix_sharding + logical_subsharding
    
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

  def _is_valid_shape(self, subshape) -> bool:
    for dim in subshape:
      if dim <= 0:
        return False

    return True
      
  def init(self) -> "PageManager":
    """Initializes physical page tensors for a pool of page blocks."""
    blocks: dict[str, jax.Array] = {}
    
    if not self.block_specs:
      raise ValueError(
          'Cannot initialize PageManager: `block_specs` is empty.'
          'At least one block specification must be provided.'
      )

    if self.num_pages_per_block == 0:
      raise ValueError(
          'Cannot initialize PageManager with 0 pages per block. '
          'This occurs if `max_bytes` is too small to fit a page.'
      )
    
    if self.num_pages_per_block < self.max_num_pages_per_seq:
      raise ValueError(
          f'Cannot initialize PageManager. Block capacity is too small'
          f'Each block is allocated {self.num_pages_per_block}, but a '
          f'sequence may require up to {self.max_num_pages_per_seq} pages. '
          f'A block must be able to fit at least one maximum-length sequence. '
          f'This occurs if `max_bytes` is too small or `max_seq_len` is too large.'
      )

    for spec in self.block_specs:
      if not self._is_valid_shape(spec.subshape):
        raise ValueError(
            f'Cannot initialize PageManager. Block shapes may not have 0 or negative '
            f'dimension. But block {spec.name} has an invalid subshape of {spec.subshape}.'
        )
   
      shape = (self.num_pages_per_block, self.page_size) + spec.subshape
      block = jax.lax.empty(shape, dtype=spec.dtype)
      sharding = self.get_block_spec_sharding(spec)
      
      if sharding is not None:
        block = utils.shard(block, sharding)
          
      if self.device is not None:
        block = jax.device_put(block, self.device)
          
      blocks[spec.name] = block

    page_indices = jnp.zeros((self.max_num_seqs, self.max_num_pages_per_seq), dtype=jnp.int32)
    """
    page_indices = jnp.full(
        (self.max_num_seqs, self.max_num_pages_per_seq),
        fill_value=100*self.max_num_seqs,
        dtype=jnp.int32
    )
    """
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
class BlockPool:
  """Physical data blocks."""
  pages: dict[str, jax.Array]
  available_page_indices: jax.Array  # i32[total_num_pages]
  num_available_pages: jax.Array  # i32 scalar
  
  page_size: int = dataclasses.field(
      metadata={'static': True}
  )

  @property
  def total_num_pages(self) -> int:
    return self.available_page_indices.shape[0]

  def allocate(self, num_pages_to_allocate: jax.Array) -> tuple["BlockPool", jax.Array]:
    """Allocates pages from the free pool."""
    page_indices_to_allocate = RaggedArray(
        data=self.available_page_indices, lens=num_pages_to_allocate
    )

    updated_num_available_pages = (
        self.num_available_pages - page_indices_to_allocate.total_length
    )
    updated_available_page_indices = jnp.roll(
        self.available_page_indices, -page_indices_to_allocate.total_length
    )

    return dataclasses.replace(
        self,
        available_page_indices=updated_available_page_indices,
        num_available_pages=updated_num_available_pages,
    ), page_indices_to_allocate.data

  def evict_pages(self, page_indices_to_evict: jax.Array, num_evicted: jax.Array) -> "BlockPool":
    """Releases specific physical pages back to the free pool."""
    target_indices = jnp.arange(page_indices_to_evict.shape[0])
    target_indices = jnp.where(
        target_indices < num_evicted,
        target_indices + self.num_available_pages,
        self.total_num_pages  # Out of bounds
    )
    updated_available_page_indices = self.available_page_indices.at[
        target_indices
    ].set(page_indices_to_evict, mode='drop')

    updated_num_available_pages = self.num_available_pages + num_evicted
    return dataclasses.replace(
        self,
        available_page_indices=updated_available_page_indices,
        num_available_pages=updated_num_available_pages,
    )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class PageManager(BlockPool):
  """Page state and data blocks for a batch of sequences."""
  page_indices: jax.Array  # i32[batch_size, max_num_pages_per_seq]
  seq_lens: jax.Array # i32[batch_size]

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
    return utils.cdiv(self.max_seq_len, self.page_size)


  @property
  def lens(self) -> jax.Array:
    return self.seq_lens

  @property
  def kv_lens(self) -> jax.Array:
    return self.seq_lens

  @functools.cached_property
  def num_seq_pages(self) -> jax.Array:
    return utils.cdiv(self.seq_lens, self.page_size)

  @jax.named_call
  def release(self, should_release: jax.Array) -> "PageManager":
    """Releases pages for completed sequences without freeing physical pages."""
    updated_lens = jnp.where(should_release, 0, self.seq_lens)
    return dataclasses.replace(self, seq_lens=updated_lens)

  @jax.named_call
  def assign(self, seq_idxs: jax.Array, page_indices: jax.Array, lens: jax.Array) -> "PageManager":
    """Assigns physical page indices to sequences."""
    ragged = RaggedArray(data=page_indices, lens=lens)
    
    target_rows = seq_idxs[ragged.row_idxs]
    target_cols = ragged.intra_offsets
    
    updated_page_indices = self.page_indices.at[
        target_rows, target_cols
    ].set(ragged.data, mode='drop')

    updated_lens = self.seq_lens.at[seq_idxs].set(lens * self.page_size, mode='drop')

    return dataclasses.replace(
        self,
        page_indices=updated_page_indices,
        seq_lens=updated_lens
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
      self, values: jax.Array, lens: jax.Array, block_id: str = "tokens"
  ) -> "PageManager":
    """Loads packed 1D array of values into allocated paged memory of block."""
    values_ragged = RaggedArray(data=values, lens=lens)
    seq_idxs = values_ragged.row_idxs
    value_offsets = values_ragged.intra_offsets

    local_page_cols = (value_offsets // self.page_size) 
    page_offsets = value_offsets % self.page_size
    phys_page_ids = self.page_indices[seq_idxs, local_page_cols]
    
    max_n_pages = self.pages[block_id].shape[0]
    safe_page_indices = jnp.where(
        jnp.arange(values_ragged.capacity) < values_ragged.total_length,
        phys_page_ids,
        self.batch_size,
    )

    updated_block_pages = self.pages[block_id].at[safe_page_indices, page_offsets].set(
        values,
        mode='drop', 
    )
    new_pages = {**self.pages, block_id: updated_block_pages}

    return dataclasses.replace(
          self, 
          pages=new_pages,
    )


  def insert_values(
      self, values: jax.Array, idxs: jax.Array | None = None, valid_mask: jax.Array | None = None, block_id: str = "tokens"
  ) -> "PageManager":
    """Insert 1 new token per sequence to the last allocated idx in paged memory."""
    if valid_mask is None:
      valid_mask = jnp.ones(self.batch_size, dtype=jnp.bool_)
    
    if idxs is None:
      idxs = self.seq_lens - 1
      
    local_page_cols = idxs // self.page_size
    page_offsets = idxs % self.page_size
    seq_idxs = jnp.arange(self.batch_size)
    phys_page_ids = self.page_indices[seq_idxs, local_page_cols]
    
    # Point invalid sequences to out-of-bounds
    max_n_pages = self.pages[block_id].shape[0]
    safe_phys_page_ids = jnp.where(valid_mask, phys_page_ids, max_n_pages)
    
    """
    if block_id == "logits":
      jax.debug.print("Lens: {i}", i=jax.device_get(self.seq_lens))
      jax.debug.print("Max Seq Len: {i}", i=jax.device_get(self.max_seq_len))
      jax.debug.print("Page Indices Shape: {i}", i=jax.device_get(self.page_indices.shape))
      jax.debug.print("Page Indices: {i}", i=jax.device_get(self.page_indices))
      jax.debug.print("Reg idxs: {i}", i=jax.device_get(phys_page_ids))
      jax.debug.print("Safe idxs: {i}", i=jax.device_get(safe_phys_page_ids))
      jax.debug.print("Cols: {i}", i=jax.device_get(local_page_cols))
      jax.debug.print("Offsets {i}", i=jax.device_get(page_offsets))
    """
 
    updated_layer_pages = self.pages[block_id].at[safe_phys_page_ids, page_offsets].set(
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
      total_num_elements: int,
      block_id: str = "tokens",
  ) -> jax.Array:
    """Extracts array of token IDs from paged memory."""
    elements_ragged = RaggedArray(
        data=jnp.zeros((total_num_elements,), dtype=jnp.int32),
        lens=self.seq_lens,
    )
    seq_idxs = elements_ragged.row_idxs
    element_offsets = elements_ragged.intra_offsets

    local_page_cols = (element_offsets // self.page_size)
    page_offsets = element_offsets % self.page_size
    phys_page_ids = self.page_indices[seq_idxs, local_page_cols]

    # Gather exact valid elemnts per sequence
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

@jax.jit
def copy_physical_pages(
    src_cache: PageManager,
    dst_cache: PageManager,
    src_idxs: jax.Array,
    dst_idxs: jax.Array,
    block_ids: Sequence[str],
) -> PageManager:
    """Copies raw physical page indices from one cache to another."""
    if len(src_idxs) == 0:
        return dst_cache

    for block_id in block_ids:
        src_tensor = src_cache.pages[block_id]
        dst_tensor = dst_cache.pages.get(block_id, jnp.zeros_like(src_tensor))

        src_slice = src_tensor[src_idxs]
        src_slice = _put_on_target_device(src_slice, dst_tensor)
        
        dst_tensor = dst_tensor.at[dst_idxs].set(src_slice)
        # Note: dataclass mutations within JIT are complicated if it's immutable, 
        # so we modify dict and return a new PageManager.
        # But wait, pages is a dict property?
# Removed jax.jit as per user request to let the caller decide
def copy_physical_pages(
    src_pool,
    dst_pool,
    src_idxs: jax.Array,
    dst_idxs: jax.Array,
    block_ids: Sequence[str],
):
    """Copies raw physical page indices from one BlockPool to another."""
    if len(src_idxs) == 0:
        return dst_pool

    new_pages = {**dst_pool.pages}
    for block_id in block_ids:
        src_tensor = src_pool.pages[block_id]
        dst_tensor = dst_pool.pages.get(block_id, jnp.zeros_like(src_tensor))

        src_slice = src_tensor[src_idxs]
        src_slice = _put_on_target_device(src_slice, dst_tensor)
        
        new_pages[block_id] = dst_tensor.at[dst_idxs].set(src_slice)
        
    return dataclasses.replace(dst_pool, pages=new_pages)
