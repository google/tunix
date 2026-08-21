import dataclasses
from typing import Optional, Any
import jax
import jax.numpy as jnp

@dataclasses.dataclass
class BlockSpec:
  name: str
  dtype: jnp.dtype
  subshape: tuple[int, ...] = () 
  logical_subsharding: tuple = () 
  device: Optional[Any] = None

@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class Block:
  """Physical data block."""
  pages: jax.Array
  available_page_indices: jax.Array  # i32[total_num_pages]
  num_available_pages: jax.Array     # i32 scalar
  
  page_size: int = dataclasses.field(metadata={'static': True})
  
  @property
  def total_num_pages(self) -> int:
    return self.available_page_indices.shape[0]

  @classmethod
  def init(cls, num_pages: int, page_size: int, block_spec: BlockSpec, device: Any = None) -> "Block":
      dtype = block_spec.dtype
      subshape = block_spec.subshape
      shape = (num_pages, page_size) + subshape
      pages = jnp.zeros(shape, dtype=dtype)
      
      available_page_indices = jnp.arange(num_pages, dtype=jnp.int32)
      num_available_pages = jnp.array(num_pages, dtype=jnp.int32)
      
      if device is not None:
          pages = jax.device_put(pages, device)
          available_page_indices = jax.device_put(available_page_indices, device)
          num_available_pages = jax.device_put(num_available_pages, device)
          
      return cls(
          pages=pages,
          available_page_indices=available_page_indices,
          num_available_pages=num_available_pages,
          page_size=page_size
      )

  def allocate(self, num_pages_to_allocate: jax.Array) -> tuple["Block", jax.Array]:
    start = self.total_num_pages - self.num_available_pages
    allocated_indices = self.available_page_indices[:num_pages_to_allocate]
    updated_available_page_indices = jnp.roll(
        self.available_page_indices, -num_pages_to_allocate
    )
    updated_num_available_pages = self.num_available_pages - num_pages_to_allocate
    
    new_block = dataclasses.replace(
        self,
        available_page_indices=updated_available_page_indices,
        num_available_pages=updated_num_available_pages,
    )
    return new_block, allocated_indices

  def release(self, page_indices_to_evict: jax.Array, num_evicted: jax.Array) -> "Block":
    start_pos = self.total_num_pages - self.num_available_pages - num_evicted
    target_indices = jnp.arange(page_indices_to_evict.shape[0])
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
  """Manager wrapping a Block."""
  block: Block
  page_indices: jax.Array
  seq_lens: jax.Array

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
    dst_pages = dst_pages.at[dst_idxs].set(src_slice)
    return dst_pages
