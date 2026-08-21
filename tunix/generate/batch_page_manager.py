import dataclasses
import functools
from typing import Optional, Sequence, Any
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from tunix.generate import utils

@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class RaggedArray:
  data: jax.Array
  lens: jax.Array
  
  @property
  def capacity(self) -> int:
    return self.data.shape[0]

  @property
  def total_length(self) -> int:
    return jnp.sum(self.lens)

  @property
  def row_idxs(self) -> jax.Array:
    reps = self.lens
    return jnp.repeat(jnp.arange(reps.shape[0]), reps)

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

  def init(self, num_pages: int, page_size: int, device: Any = None) -> "Block":
      dtype = self.dtype
      subshape = self.subshape
      shape = (num_pages, page_size) + subshape
      pages = jnp.zeros(shape, dtype=dtype)
      
      available_page_indices = jnp.arange(num_pages, dtype=jnp.int32)
      num_available_pages = jnp.array(num_pages, dtype=jnp.int32)
      
      if device is not None:
          pages = jax.device_put(pages, device)
          available_page_indices = jax.device_put(available_page_indices, device)
          num_available_pages = jax.device_put(num_available_pages, device)
          
      return Block(
          pages=pages,
          available_page_indices=available_page_indices,
          num_available_pages=num_available_pages,
          page_size=page_size
      )

@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class Block:
  """Physical data blocks."""
  pages: jax.Array
  available_page_indices: jax.Array  # i32[total_num_pages]
  num_available_pages: jax.Array  # i32 scalar
  
  page_size: int = dataclasses.field(metadata={'static': True})

  @property
  def total_num_pages(self) -> int:
    return self.available_page_indices.shape[0]

  def allocate(self, num_pages_to_allocate: jax.Array) -> tuple["Block", jax.Array]:
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

  def release(self, page_indices_to_evict: jax.Array, num_evicted: jax.Array) -> "Block":
    """Frees physical page indices."""
    target_indices = jnp.arange(page_indices_to_evict.shape[0])
    start_pos = self.total_num_pages - self.num_available_pages - num_evicted
    safe_indices = jnp.where(target_indices < num_evicted, start_pos + target_indices, 0)

    updated_available_page_indices = self.available_page_indices.at[
        safe_indices
    ].set(page_indices_to_evict, mode='drop')

    return dataclasses.replace(
        self,
        available_page_indices=updated_available_page_indices,
        num_available_pages=self.num_available_pages + num_evicted,
    )

@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class BatchPageManager:
  """Page state and data blocks for a batch of sequences."""
  block: Block
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
    return utils.cdiv(self.max_seq_len, self.block.page_size)

  @property
  def lens(self) -> jax.Array:
    return self.seq_lens

  @property
  def kv_lens(self) -> jax.Array:
    return self.seq_lens

  @functools.cached_property
  def num_seq_pages(self) -> jax.Array:
    return utils.cdiv(self.seq_lens, self.block.page_size)

  def allocate(self, num_pages_to_allocate: jax.Array) -> tuple["BatchPageManager", jax.Array]:
    new_block, allocated_indices = self.block.allocate(num_pages_to_allocate)
    return dataclasses.replace(self, block=new_block), allocated_indices

  def evict_pages(self, page_indices: jax.Array, count: jax.Array) -> "BatchPageManager":
    new_block = self.block.release(page_indices, count)
    return dataclasses.replace(self, block=new_block)

  @jax.named_call
  def release(self, should_release: jax.Array) -> "BatchPageManager":
    """Releases sequence tracking without freeing physical pages."""
    updated_lens = jnp.where(should_release, 0, self.seq_lens)
    return dataclasses.replace(self, seq_lens=updated_lens)

  @jax.named_call
  def assign(self, seq_idxs: jax.Array, page_indices: jax.Array, lens: jax.Array) -> "BatchPageManager":
    """Assigns physical page indices to sequences."""
    ragged = RaggedArray(data=page_indices, lens=lens)
    
    target_rows = seq_idxs[ragged.row_idxs]
    target_cols = ragged.intra_offsets
    
    updated_page_indices = self.page_indices.at[
        target_rows, target_cols
    ].set(ragged.data, mode='drop')

    updated_lens = self.seq_lens.at[seq_idxs].set(lens * self.block.page_size, mode='drop')

    return dataclasses.replace(
        self,
        page_indices=updated_page_indices,
        seq_lens=updated_lens
    )

  @jax.named_call
  def release_for_window(self) -> "BatchPageManager":
    """Release allocations for window."""
    if self.window_size is None:
      return self

    num_pages_to_release = (
        jnp.maximum(self.seq_lens - self.window_size, 0)
        // self.block.page_size
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
        data=jax.lax.empty((self.block.total_num_pages,), dtype=jnp.int32),
        lens=num_pages_to_release,
    )
    released_page_indices = self.page_indices[
        release_helper.row_idxs, release_helper.intra_offsets
    ]
    
    new_block = self.block.release(released_page_indices, jnp.array(release_helper.total_length))

    return dataclasses.replace(
        self,
        page_indices=updated_page_indices,
        block=new_block,
        seq_lens=self.seq_lens - num_pages_to_release * self.block.page_size,
    )

  def load_values(
      self, values: jax.Array, lens: jax.Array
  ) -> "BatchPageManager":
    """Loads packed 1D array of values into allocated paged memory of block."""
    values_ragged = RaggedArray(data=values, lens=lens)
    seq_idxs = values_ragged.row_idxs
    value_offsets = values_ragged.intra_offsets

    local_page_cols = (value_offsets // self.block.page_size) 
    page_offsets = value_offsets % self.block.page_size
    phys_page_ids = self.page_indices[seq_idxs, local_page_cols]
    
    max_n_pages = self.block.pages.shape[0]
    safe_page_indices = jnp.where(
        jnp.arange(values_ragged.capacity) < values_ragged.total_length,
        phys_page_ids,
        self.batch_size,
    )

    updated_block_pages = self.block.pages.at[safe_page_indices, page_offsets].set(
        values,
        mode='drop', 
    )

    return dataclasses.replace(
          self, 
          block=dataclasses.replace(self.block, pages=updated_block_pages),
    )


  def insert_values(
      self, values: jax.Array, idxs: jax.Array | None = None, valid_mask: jax.Array | None = None
  ) -> "BatchPageManager":
    """Insert 1 new token per sequence to the last allocated idx in paged memory."""
    if valid_mask is None:
      valid_mask = jnp.ones(self.batch_size, dtype=jnp.bool_)
    
    if idxs is None:
      idxs = self.seq_lens - 1
      
    local_page_cols = idxs // self.block.page_size
    page_offsets = idxs % self.block.page_size
    seq_idxs = jnp.arange(self.batch_size)
    phys_page_ids = self.page_indices[seq_idxs, local_page_cols]
    
    max_n_pages = self.block.pages.shape[0]
    safe_phys_page_ids = jnp.where(valid_mask, phys_page_ids, max_n_pages)
 
    updated_layer_pages = self.block.pages.at[safe_phys_page_ids, page_offsets].set(
      values,
      mode='drop' 
    )

    return dataclasses.replace(
          self, 
          block=dataclasses.replace(self.block, pages=updated_layer_pages),
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

    local_page_cols = (element_offsets // self.block.page_size)
    page_offsets = element_offsets % self.block.page_size
    phys_page_ids = self.page_indices[seq_idxs, local_page_cols]

    packed_tokens = self.block.pages[phys_page_ids, page_offsets]
    return packed_tokens


def _remove_dp_spec(spec: P) -> P:
    """Replaces any 'dp' instance in a PartitionSpec with None."""
    dp_axis = ['dp', 'fsdp']
    new_spec = tuple(None if axis in dp_axis else axis for axis in spec)
    return P(*new_spec)

def _put_on_target_device(tensor: jax.Array, target_tensor: jax.Array) -> jax.Array:
  """Safely places tensor on the same device/mesh as target_tensor."""
  if hasattr(target_tensor, "sharding") and target_tensor.sharding is not None:
    sharding = target_tensor.sharding
    if isinstance(sharding, jax.sharding.NamedSharding):
      safe_spec = _remove_dp_spec(sharding.spec)
      target_sharding = jax.sharding.NamedSharding(sharding.mesh, safe_spec)
      return jax.device_put(tensor, target_sharding)
    elif isinstance(sharding, jax.sharding.SingleDeviceSharding):
      return jax.device_put(tensor, sharding)

  if hasattr(target_tensor, "devices") and len(target_tensor.devices()) > 0:
    return jax.device_put(tensor, list(target_tensor.devices())[0])

  return tensor

def copy_physical_pages(
    src_pages: jax.Array,
    dst_pages: jax.Array,
    src_idxs: jax.Array,
    dst_idxs: jax.Array,
) -> jax.Array:
    """Copies raw physical page indices."""
    if len(src_idxs) == 0:
        return dst_pages

    src_slice = src_pages[src_idxs]
    src_slice = _put_on_target_device(src_slice, dst_pages)
    
    dst_pages = dst_pages.at[dst_idxs].set(src_slice)
    return dst_pages
