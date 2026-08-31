# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# from __future__ import annotations
"""A ragged page manager for a batch of sequences."""

from collections.abc import Sequence
import dataclasses
import functools
from typing import Any, Optional

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from tunix.generate import utils


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class RaggedArray:
  """2D Ragged Array."""

  data: jax.Array  # [capacity, *subshape]
  lens: jax.Array  # [batch_size]

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
  dtype: jnp.dtype
  subshape: tuple[int, ...] = ()
  logical_subsharding: tuple[str | None, ...] = ()
  device: Optional[Any] = None


@dataclasses.dataclass(frozen=True, kw_only=True)
class PageManagerConfig:
  """Configuration for a PageManager."""

  page_size: int
  max_num_seqs: int
  max_seq_len: int
  max_bytes: int

  block_spec: BlockSpec
  logical_page_sharding: str | None = None

  dp_axis: str | None = None
  tp_axis: str | None = None
  dp_size: int = 1
  tp_size: int = 1
  device: Any = None

  @property
  def num_pages(self) -> int:
    """How many pages each block should have based on the max_bytes budget."""
    page_size_across_blocks = 0
    
    spec = self.spec
    item_size = jnp.dtype(spec.dtype).itemsize

    # Shape and sharding for a single page (ignoring the 0th dim: num_pages)
    page_shape = (self.page_size,) + spec.subshape
    page_sharding = self.get_logical_block_spec_sharding(spec)[1:]

    elements = 1
    for dim, shard in zip(page_shape, page_sharding):
      dim_size = (dim * self.dp_size) if shard == 'dp_axis' else dim
      elements *= dim_size

    page_size_across_blocks += elements * item_size

    if page_size_across_blocks == 0:
      return 0

    pages_per_block = self.max_bytes // page_size_across_blocks

    # Align down to a multiple of dp_size if the page dimension is sharded.
    if self.logical_page_sharding == 'dp_axis':
      pages_per_block = (pages_per_block // self.dp_size) * self.dp_size

    return pages_per_block

  @property
  def max_num_pages_per_seq(self) -> int:
    """Maximum number of pages required to store a single sequence."""
    return utils.cdiv(self.max_seq_len, self.page_size)

  @property
  def logical_shard_to_physical(self) -> dict:
    """Returns a mapping of logical shard names to physical shard names."""
    return {'dp_axis': self.dp_axis, 'tp_axis': self.tp_axis, None: None}
  
  @property
  def logical_block_sharding(self):
    """Returns the logical sharding for a given block spec."""
    spec = self.spec

    logical_page_sharding = self.logical_page_sharding
    logical_subsharding = spec.logical_subsharding

    logical_prefix_sharding = (logical_page_sharding, None)

    l_subsharding = len(logical_subsharding)
    l_subshape = len(spec.subshape)

    if l_subsharding > l_subshape:
      raise ValueError(
          f'Cannot initialize block. '
          f'Block subsharding `{spec.logical_subsharding}` '
          'cannot have length greater than block subshape '
          f'`{spec.subshape}`'
      )

    if l_subsharding < l_subshape:
      num_padding = l_subshape - l_subsharding
      logical_subsharding += (None,) * num_padding

    logical_sharding = logical_prefix_sharding + logical_subsharding

    return logical_sharding
  
  @property
  def block_spec_sharding(self):
    """Returns the physical sharding for a given block spec."""
    spec = self.block_spec

    logical_sharding = self.logical_block_sharding

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

  def init(self) -> 'PageManager':
    """Initializes physical page tensors for a pool of page blocks."""
    blocks: dict[str, jax.Array] = {}

    if self.num_pages == 0:
      raise ValueError(
          'Cannot initialize PageManager with 0 pages per block. '
          'This occurs if `max_bytes` is too small to fit a page.'
      )

    if self.num_pages < self.max_num_pages_per_seq:
      raise ValueError(
          'Cannot initialize PageManager. Block capacity is too small. Each '
          f'block is allocated {self.num_pages}, but a sequence may '
          f'require up to {self.max_num_pages_per_seq} pages. A block must be '
          f'able to fit at least one maximum-length sequence. This occurs if '
          f'`max_bytes` is too small or `max_seq_len` is too large. '
      )
    
    assert(all(dim > 0 for dim spec.subshape))
    shape = (self.num_pages, self.page_size) + spec.subshape
    block = jax.lax.empty(shape, dtype=spec.dtype)
    sharding = self.get_block_spec_sharding(spec)

    if sharding is not None:
      block = utils.shard(block, sharding)

    if self.device is not None:
      block = jax.device_put(block, self.device)

    page_indices = jnp.zeros(
        (self.max_num_seqs, self.max_num_pages_per_seq), dtype=jnp.int32
    )
    available_page_indices = jnp.arange(
        self.num_pages, dtype=jnp.int32
    )
    num_available_pages = jnp.array(self.num_pages, dtype=jnp.int32)
    seq_lens = jnp.zeros((self.max_num_seqs,), dtype=jnp.int32)

    if self.device is not None:
      page_indices = jax.device_put(page_indices, self.device)
      available_page_indices = jax.device_put(
          available_page_indices, self.device
      )
      num_available_pages = jax.device_put(num_available_pages, self.device)
      seq_lens = jax.device_put(seq_lens, self.device)

    return PageManager(
        pages=block,
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

  pages: dict[str, jax.Array] # dict[str, i32[num_pages, page_size, *subshape]]
  page_indices: jax.Array  # i32[batch_size, max_num_pages_per_seq]
  available_page_indices: jax.Array  # i32[total_num_pages]
  num_available_pages: jax.Array  # i32 scalar

  seq_lens: jax.Array  # i32[batch_size]

  page_size: int = dataclasses.field(metadata={'static': True})
  max_seq_len: int = dataclasses.field(metadata={'static': True})
  window_size: int | None = dataclasses.field(metadata={'static': True})

  @property
  def batch_size(self) -> int:
    return self.seq_lens.shape[0]

  @property
  def max_num_pages_per_seq(self) -> int:
    return utils.cdiv(self.max_seq_len, self.page_size)

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
  def num_pages_per_seq(self) -> jax.Array:
    return utils.cdiv(self.seq_lens, self.page_size)

  @jax.named_call
  def allocate(self, q_lens: jax.Array) -> 'PageManager':
    """Allocates pages for new tokens."""
    pages_required = utils.cdiv(self.seq_lens + q_lens, self.page_size)

    num_pages_to_allocate = pages_required - self.num_pages_per_seq

    page_indices_to_allocate = RaggedArray(
        data=self.available_page_indices, lens=num_pages_to_allocate
    )
    page_indices_rows = page_indices_to_allocate.row_idxs
    page_indices_cols = (
        self.num_pages_per_seq[page_indices_rows]
        + page_indices_to_allocate.intra_offsets
    )

    out_of_bounds_idx = self.page_indices.shape[1]
    is_real_allocation = (
        jnp.arange(page_indices_to_allocate.capacity)
        < page_indices_to_allocate.total_length
    )
    safe_page_indices_cols = jnp.where(
        is_real_allocation,
        page_indices_cols,
        out_of_bounds_idx,
    )

    updated_page_indices = self.page_indices.at[
        page_indices_rows, safe_page_indices_cols
    ].set(page_indices_to_allocate.data, mode='drop')

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
  def release(self, should_release: jax.Array) -> 'PageManager':
    """Releases pages for completed sequences."""
    updated_lens = jnp.where(should_release, 0, self.seq_lens)

    page_indices_to_release = RaggedArray(
        data=jax.lax.empty((self.total_num_pages,), dtype=jnp.int32),
        lens=jnp.where(should_release, self.num_pages_per_seq, 0),
    )
    page_indices_irows = page_indices_to_release.row_idxs
    page_indices_icols = page_indices_to_release.intra_offsets

    is_real_release = (
        jnp.arange(self.total_num_pages) < page_indices_to_release.total_length
    )
    safe_icols = jnp.where(is_real_release, page_indices_icols, 0)
    released_pages = self.page_indices[page_indices_irows, safe_icols]

    target_slots = jnp.arange(self.total_num_pages) + self.num_available_pages
    safe_target_slots = jnp.where(
        is_real_release, target_slots, self.total_num_pages
    )

    updated_available_page_indices = self.available_page_indices.at[
        safe_target_slots
    ].set(released_pages, mode='drop')

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
  def release_for_window(self) -> 'PageManager':
    """Release allocations for window."""
    if self.window_size is None:
      return self

    num_pages_to_release = (
        jnp.maximum(self.seq_lens - self.window_size, 0) // self.page_size
    )
    page_indices_irows = jnp.arange(self.batch_size)[:, None]
    page_indices_icols = (
        jnp.arange(self.max_num_pages_per_seq) + num_pages_to_release[:, None]
    )
    updated_page_indices = self.page_indices[
        page_indices_irows, page_indices_icols
    ]
    release_helper = RaggedArray(
        data=jax.lax.empty((self.total_num_pages,), dtype=jnp.int32),
        lens=num_pages_to_release,
    )

    is_real_release = (
        jnp.arange(self.total_num_pages) < release_helper.total_length
    )
    safe_icols = jnp.where(is_real_release, release_helper.intra_offsets, 0)
    self.page_manager.release(release_helper.row_idxs, safe_icols)

    target_slots = jnp.arange(self.total_num_pages) + self.num_available_pages
    safe_target_slots = (
        jnp.where(is_real_release, target_slots, self.total_num_pages)
    )

    updated_available_page_indices = self.available_page_indices.at[
        safe_target_slots
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
      self, values: jax.Array, lens: jax.Array 
  ) -> 'PageManager':
    """Loads packed 1D array of values into allocated paged memory of block."""
    values_ragged = RaggedArray(data=values, lens=lens)
    seq_idxs = values_ragged.row_idxs
    value_offsets = values_ragged.intra_offsets

    local_page_cols = value_offsets // self.page_size
    page_offsets = value_offsets % self.page_size
    phys_page_ids = self.page_indices[seq_idxs, local_page_cols]

    max_n_pages = self.pages.shape[0]
    safe_page_indices = jnp.where(
        jnp.arange(values_ragged.capacity) < values_ragged.total_length,
        phys_page_ids,
        max_n_pages,
    )
    
    self.page_manager.write(safe_page_indices, page_offsets, values)
    updated_pages = ( 
        self.pages.at[safe_page_indices, page_offsets].set(
            values,
            mode='drop',
        )
    )

    return dataclasses.replace(
        self,
        pages=updated_pages,
    )

  def insert_values(
      self,
      values: jax.Array,
      idxs: jax.Array | None = None,
      valid_mask: jax.Array | None = None,
  ) -> 'PageManager':
    """Insert 1 new token per sequence to the last allocated idx in paged memory."""
    if valid_mask is None:
      valid_mask = self.seq_lens > 0

    if idxs is None:
      idxs = self.seq_lens - 1

    local_page_cols = idxs // self.page_size
    page_offsets = idxs % self.page_size
    seq_idxs = jnp.arange(self.batch_size)
    phys_page_ids = self.page_indices[seq_idxs, local_page_cols]

    # Point invalid sequences to out-of-bounds
    max_n_pages = self.pages.shape[0]
    safe_phys_page_ids = jnp.where(valid_mask, phys_page_ids, max_n_pages)

    updated_pm = self.page_manager.write(safe_phys_page_ids, page_offsets, values)

    return dataclasses.replace(
        self,
        pages=updated_layer_pages,
    )

  def to_array(
      self,
      total_num_elements: int,
  ) -> jax.Array:
    """Extracts array of token IDs from paged memory."""
    elements_ragged = RaggedArray(
        data=jnp.zeros((total_num_elements,), dtype=jnp.int32),
        lens=self.seq_lens,
    )
    seq_idxs = elements_ragged.row_idxs
    element_offsets = elements_ragged.intra_offsets

    local_page_cols = element_offsets // self.page_size
    page_offsets = element_offsets % self.page_size
    phys_page_ids = self.page_indices[seq_idxs, local_page_cols]
    
    self.page_manager.get(phys_page_ids, page_offsets)
    return packed_tokens

def batch_copy_pages(
    src_cache: PageManager,
    dst_cache: PageManager,
    src_slots: Sequence[int],
    dst_slots: Sequence[int],
) -> PageManager:
  """Copy pages across specified slots, optionally copying only token pages."""
  if len(src_slots) == 0:
    return dst_cache

  src_idxs_list = []
  dst_idxs_list = []
  for s_slot, d_slot in zip(src_slots, dst_slots):
    seq_len = src_cache.seq_lens[s_slot]
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
  src_tensor = src_cache.pages
  dst_tensor = dst_cache.pages

  src_slice = src_tensor[src_idxs]
  src_slice = _put_on_target_device(src_slice, dst_tensor)

  new_pages = dst_tensor.at[dst_idxs].set(src_slice)

  return dataclasses.replace(dst_cache, pages=new_pages)



def _remove_dp_spec(spec: P) -> P:
  """Replaces any 'dp' instance in a PartitionSpec with None."""
  dp_axis = ['dp', 'fsdp']
  new_spec = tuple(None if axis in dp_axis else axis for axis in spec)
  return P(*new_spec)


def _put_on_target_device(
    tensor: jax.Array, target_tensor: jax.Array
) -> jax.Array:
  """Safely places tensor on the same device/mesh as target_tensor."""
  if hasattr(target_tensor, 'sharding') and target_tensor.sharding is not None:
    sharding = target_tensor.sharding
    if isinstance(sharding, jax.sharding.NamedSharding):
      # Replicate the tensor in the case of dp
      safe_spec = _remove_dp_spec(sharding.spec)
      target_sharding = jax.sharding.NamedSharding(sharding.mesh, safe_spec)
      return jax.device_put(tensor, target_sharding)

    elif isinstance(sharding, jax.sharding.SingleDeviceSharding):
      return jax.device_put(tensor, sharding)

  if hasattr(target_tensor, 'devices') and len(target_tensor.devices()) > 0:
    return jax.device_put(tensor, list(target_tensor.devices())[0])

  return tensor

