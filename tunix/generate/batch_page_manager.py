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

@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class BatchPageManager:
  """Manager wrapping a Block."""
  block: Block

  @classmethod
  def init(cls, num_pages: int, page_size: int, block_spec: BlockSpec, device: Any = None) -> "BatchPageManager":
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
          
      block = Block(
          pages=pages,
          available_page_indices=available_page_indices,
          num_available_pages=num_available_pages,
          page_size=page_size
      )
      return cls(block=block)

  def allocate(self, num_pages_to_allocate: jax.Array) -> tuple["BatchPageManager", jax.Array]:
    start = self.block.total_num_pages - self.block.num_available_pages
    allocated_indices = self.block.available_page_indices[:num_pages_to_allocate]
    updated_available_page_indices = jnp.roll(
        self.block.available_page_indices, -num_pages_to_allocate
    )
    updated_num_available_pages = self.block.num_available_pages - num_pages_to_allocate
    
    new_block = dataclasses.replace(
        self.block,
        available_page_indices=updated_available_page_indices,
        num_available_pages=updated_num_available_pages,
    )
    return dataclasses.replace(self, block=new_block), allocated_indices

  def evict_pages(self, page_indices_to_evict: jax.Array, num_evicted: jax.Array) -> "BatchPageManager":
    start_pos = self.block.total_num_pages - self.block.num_available_pages - num_evicted
    target_indices = jnp.arange(page_indices_to_evict.shape[0])
    safe_indices = jnp.where(target_indices < num_evicted, start_pos + target_indices, 0)
    
    updated_available_page_indices = self.block.available_page_indices.at[
        safe_indices
    ].set(page_indices_to_evict, mode='drop')
    
    new_block = dataclasses.replace(
        self.block,
        available_page_indices=updated_available_page_indices,
        num_available_pages=self.block.num_available_pages + num_evicted,
    )
    return dataclasses.replace(self, block=new_block)

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
