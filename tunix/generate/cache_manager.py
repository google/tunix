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


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class Block:
  """A block of physical pages."""

  pages: jax.Array 
  available_page_indices: jax.Array  # i32[total_num_pages]
  num_available_pages: jax.Array  # i32 scalar

  page_size: int = dataclasses.field(metadata={'static': True})

  @property
  def total_num_pages(self) -> int:
    return self.available_page_indices.shape[0]

  @jax.named_call
  def allocate(self, num_pages: int | jax.Array) -> tuple['Block', jax.Array]:
    """Allocate `num_pages` (total scalar) pages in the block."""
    allocated_pages = self.available_page_indices

    updated_num_available_pages = (
        self.num_available_pages - num_pages
    )
    updated_available_page_indices = jnp.roll(
        self.available_page_indices, -num_pages
    )

    new_block = dataclasses.replace(
        self,
        available_page_indices=updated_available_page_indices,
        num_available_pages=updated_num_available_pages,
    )
    return new_block, allocated_pages

  @jax.named_call
  def release(self, num_pages_to_release: jax.Array, page_idxs_to_release: jax.Array) -> 'Block':
    """Releases pages."""
    target_slots = jnp.arange(self.total_num_pages) + self.num_available_pages
    
    # Map non-target slots to out of bounds indicies so that they are not modified
    safe_target_slots = jnp.where(
        jnp.arange(self.total_num_pages) < num_pages_to_release, 
        target_slots, 
        self.total_num_pages
    )

    updated_available_page_indices = self.available_page_indices.at[
        safe_target_slots
    ].set(page_idxs_to_release, mode='drop')

    updated_num_available_pages = (
        self.num_available_pages + num_total_released 
    )
    return dataclasses.replace(
        self,
        available_page_indices=updated_available_page_indices,
        num_available_pages=updated_num_available_pages,
    )

  def write_values(
      self, num_values: jax.Array, values: jax.Array, page_indices: jax.Array, page_offsets: jax.Array 
  ) -> 'Block':
    """Write packed 1D array of values into allocated paged memory of block."""
    max_n_pages = self.pages.shape[0]
    batch_size = values.shape[0]

    target_page_indices = jnp.where(
        jnp.arange(batch_size) < num_values,
        page_indices,
        max_n_pages
    )

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

@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class TpuCpuPageManager:
  """Page state and data blocks for a batch of sequences."""
  tpu_block: Block
  cpu_block: Block | None
  page_indices: jax.Array  # i32[batch_size, max_num_pages_per_seq]
  seq_lens: jax.Array  # i32[batch_size]
  
  max_seq_len: int = dataclasses.field(metadata={'static': True})
  window_size: int | None = dataclasses.field(metadata={'static': True})

  @property
  def batch_size(self) -> int:
    return self.seq_lens.shape[0]

  @property
  def page_size(self) -> int:
    return self.tpu_block.page_size

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
  def num_pages_per_seq(self) -> jax.Array:
    return utils.cdiv(self.seq_lens, self.page_size)

  @jax.named_call
  def allocate(self, q_lens: jax.Array) -> 'TpuCpuPageManager':
    """Allocates pages for new tokens."""
    pages_required = utils.cdiv(self.seq_lens + q_lens, self.page_size)
    num_pages_to_allocate = pages_required - self.num_pages_per_seq

    total_pages_to_allocate = jnp.sum(num_pages_to_allocate)
    new_tpu_block, allocated_page_data = self.tpu_block.allocate(total_pages_to_allocate)
    
    # We pad allocated_page_data back up to capacity so RaggedArray layout fits perfectly
    padded_allocated_page_data = jnp.where(
        jnp.arange(self.tpu_block.total_num_pages) < total_pages_to_allocate,
        self.tpu_block.available_page_indices,
        self.tpu_block.total_num_pages
    )
    
    page_indices_to_allocate = RaggedArray(
        data=padded_allocated_page_data, lens=num_pages_to_allocate
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
    
    new_page_manager = dataclasses.replace(
        self,
        seq_lens=self.seq_lens + q_lens,
        page_indices=updated_page_indices,
        tpu_block=new_tpu_block,
    )

    return new_page_manager, padded_allocated_page_data  

  @jax.named_call
  def assign(self, seq_idxs: jax.Array, page_indices: jax.Array, lens: jax.Array) -> 'TpuCpuPageManager':
    """Assigns physical page indices to sequences directly."""
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
  def release_sequences(self, seq_mask: jax.Array) -> 'TpuCpuPageManager':
    # ONLY resets seq_lens. Does NOT release physical pages.
    return dataclasses.replace(
        self,
        seq_lens=jnp.where(seq_mask, 0, self.seq_lens)
    )

  @jax.named_call
  def release_pages(self, num_pages: jax.Array, page_idxs: jax.Array, device: str = 'tpu') -> 'TpuCpuPageManager':
    # Actually frees the physical pages back into the block
    block = self.tpu_block if device == 'tpu' else self.cpu_block
    new_block = block.release(num_pages, page_idxs)
    if device == 'tpu':
        return dataclasses.replace(self, tpu_block=new_block)
    else:
        return dataclasses.replace(self, cpu_block=new_block)

  @jax.named_call
  def offload(self, num_pages: jax.Array, page_idxs: jax.Array) -> tuple['TpuCpuPageManager', jax.Array]:
    if self.cpu_block is None:
        raise ValueError("Cannot offload; cpu_block is None.")

    total_cpu_pages = jnp.sum(num_pages)
    new_cpu_block, cpu_allocated_pages = self.cpu_block.allocate(total_cpu_pages)
    
    padded_allocated_page_data = jnp.where(
        jnp.arange(self.cpu_block.total_num_pages) < total_cpu_pages,
        self.cpu_block.available_page_indices,
        self.cpu_block.total_num_pages
    )
    
    # 2. Copy page_idxs from TPU -> CPU
    ragged = RaggedArray(data=page_idxs, lens=num_pages)
    cpu_ragged = RaggedArray(data=padded_allocated_page_data, lens=num_pages)
    
    is_real = jnp.arange(ragged.capacity) < ragged.total_length
    
    safe_tpu_phys = jnp.where(is_real, ragged.data, 0)
    tpu_vals = self.tpu_block.pages[safe_tpu_phys]
    tpu_vals_cpu = _put_on_target_device(tpu_vals, self.cpu_block.pages)
    
    safe_cpu_phys = jnp.where(is_real, cpu_ragged.data, self.cpu_block.pages.shape[0])
    new_cpu_pages = self.cpu_block.pages.at[safe_cpu_phys].set(tpu_vals_cpu, mode='drop')
    new_cpu_block = dataclasses.replace(new_cpu_block, pages=new_cpu_pages)
    
    # 3. Release from TPU
    new_tpu_block = self.tpu_block.release(num_pages, page_idxs)
    
    new_page_manager = dataclasses.replace(
        self,
        tpu_block=new_tpu_block,
        cpu_block=new_cpu_block
    )
    return new_page_manager, cpu_allocated_pages

  @jax.named_call
  def load(self, num_pages: jax.Array, page_idxs: jax.Array) -> tuple['TpuCpuPageManager', jax.Array]:
    if self.cpu_block is None:
        raise ValueError("Cannot load; cpu_block is None.")

    total_tpu_pages = jnp.sum(num_pages)
    new_tpu_block, tpu_allocated_pages = self.tpu_block.allocate(total_tpu_pages)
    
    padded_allocated_page_data = jnp.where(
        jnp.arange(self.tpu_block.total_num_pages) < total_tpu_pages,
        self.tpu_block.available_page_indices,
        self.tpu_block.total_num_pages
    )
    
    ragged = RaggedArray(data=page_idxs, lens=num_pages)
    tpu_ragged = RaggedArray(data=padded_allocated_page_data, lens=num_pages)
    
    is_real = jnp.arange(ragged.capacity) < ragged.total_length
    
    safe_cpu_phys = jnp.where(is_real, ragged.data, 0)
    cpu_vals = self.cpu_block.pages[safe_cpu_phys]
    cpu_vals_tpu = _put_on_target_device(cpu_vals, self.tpu_block.pages)
    
    safe_tpu_phys = jnp.where(is_real, tpu_ragged.data, self.tpu_block.pages.shape[0])
    new_tpu_pages = self.tpu_block.pages.at[safe_tpu_phys].set(cpu_vals_tpu, mode='drop')
    new_tpu_block = dataclasses.replace(new_tpu_block, pages=new_tpu_pages)
    
    new_cpu_block = self.cpu_block.release(num_pages, page_idxs)
    
    new_page_manager = dataclasses.replace(
        self,
        tpu_block=new_tpu_block,
        cpu_block=new_cpu_block
    )
    return new_page_manager, tpu_allocated_pages

  @jax.named_call
  def release_for_window(self) -> 'TpuCpuPageManager':
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
        data=jax.lax.empty((self.tpu_block.total_num_pages,), dtype=jnp.int32),
        lens=num_pages_to_release,
    )

    is_real_release = (
        jnp.arange(self.tpu_block.total_num_pages) < release_helper.total_length
    )
    safe_icols = jnp.where(is_real_release, release_helper.intra_offsets, 0)
    released_page_indices = (
        self.page_indices[release_helper.row_idxs, safe_icols]
    )

    new_tpu_block = self.tpu_block.release(
        release_helper.lens, released_page_indices
    )

    return dataclasses.replace(
        self,
        page_indices=updated_page_indices,
        tpu_block=new_tpu_block,
        seq_lens=self.seq_lens - num_pages_to_release * self.page_size,
    )

  def load_values(
      self, values: jax.Array, lens: jax.Array 
  ) -> 'TpuCpuPageManager':
    """Loads packed 1D array of values into allocated paged memory of block."""
    values_ragged = RaggedArray(data=values, lens=lens)
    seq_idxs = values_ragged.row_idxs
    value_offsets = values_ragged.intra_offsets

    local_page_cols = value_offsets // self.page_size
    page_offsets = value_offsets % self.page_size
    phys_page_ids = self.page_indices[seq_idxs, local_page_cols]

    is_real = jnp.arange(values_ragged.capacity) < values_ragged.total_length
    
    new_tpu_block = self.tpu_block.write_values(
        values, phys_page_ids, page_offsets, is_real
    )

    return dataclasses.replace(self, tpu_block=new_tpu_block)

  def insert_values(
      self,
      values: jax.Array,
      idxs: jax.Array | None = None,
      valid_mask: jax.Array | None = None,
  ) -> 'TpuCpuPageManager':
    """Insert 1 new token per sequence to the last allocated idx in paged memory."""
    if valid_mask is None:
      valid_mask = self.seq_lens > 0

    if idxs is None:
      idxs = self.seq_lens - 1

    local_page_cols = idxs // self.page_size
    page_offsets = idxs % self.page_size
    seq_idxs = jnp.arange(self.batch_size)
    phys_page_ids = self.page_indices[seq_idxs, local_page_cols]

    new_tpu_block = self.tpu_block.write_values(
        values, phys_page_ids, page_offsets, valid_mask
    )

    return dataclasses.replace(self, tpu_block=new_tpu_block)

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

    packed_tokens = self.tpu_block.pages[phys_page_ids, page_offsets]
    return packed_tokens


@dataclasses.dataclass(frozen=True, kw_only=True)
class TpuCpuPageManagerConfig:
  """Configuration for a TpuCpuPageManager."""
  page_size: int
  page_subshape: tuple[int, ...] = ()
  dtype: jnp.dtype  # ADDED: to know what we are caching

  max_num_seqs: int
  max_seq_len: int
  max_tpu_bytes: int
  max_cpu_bytes: int
  
  logical_page_sharding: str | None = None
  logical_subshape_sharding: tuple[str | None, ...] = () 

  dp_axis: str | None = None
  tp_axis: str | None = None
  dp_size: int = 1
  tp_size: int = 1
  device: Any = None
    
  def _compute_num_pages(self, max_bytes: int, logical_sharding: tuple[str, ...] | None) -> int:
    item_size = jnp.dtype(self.dtype).itemsize
    page_shape = (self.page_size,) + self.page_subshape
    page_sharding = self.get_logical_block_spec_sharding()[1:]

    elements = 1
    for dim, shard in zip(page_shape, page_sharding):
      dim_size = (dim * self.dp_size) if shard == 'dp_axis' else dim
      elements *= dim_size

    page_bytes = elements * item_size
    if page_bytes == 0:
      return 0

    pages_per_block = max_bytes // page_bytes
    if self.logical_page_sharding == 'dp_axis':
      pages_per_block = (pages_per_block // self.dp_size) * self.dp_size

    return pages_per_block

  @property
  def max_num_pages_per_seq(self) -> int:
    return utils.cdiv(self.max_seq_len, self.page_size)

  @property
  def logical_shard_to_physical(self) -> dict:
    return {'dp_axis': self.dp_axis, 'tp_axis': self.tp_axis, None: None}
    
  def get_logical_block_spec_sharding(self):
    logical_sharding = (self.logical_page_sharding, None, *self.logical_subshape_sharding)
    # Ensure logical sharding matches data shape + 2 (num_pages, page_size)
    target_len = len(self.page_subshape) + 2
    sharding_list = list(logical_sharding)
    while len(sharding_list) < target_len:
        sharding_list.append(None)
    sharding_list = sharding_list[:target_len]
    return tuple(sharding_list)

  def get_physical_sharding(self):
    logical_sharding = self.get_logical_block_spec_sharding()
    if all(axis is None for axis in logical_sharding):
      return None
      
    mapping = self.logical_shard_to_physical
    return tuple(mapping[axis] for axis in logical_sharding)

  def init(self) -> 'TpuCpuPageManager':
    """Initializes physical page tensors for TPU and CPU."""
    tpu_num_pages = self._compute_num_pages(self.max_tpu_bytes)
    # TODO: cpu pages aren't sharded at all
    cpu_num_pages = self._compute_num_pages(self.max_cpu_bytes)

    if tpu_num_pages < self.max_num_pages_per_seq:
      raise ValueError(
          'TPU block capacity is too small. '
          f'Available pages: {tpu_num_pages}, Max required per seq: {self.max_num_pages_per_seq}'
      )
    
    # We do need to use device. Just get the cpu device and put them in directly.
    # If cpu_num_pages = 0, we shouldn't be making a cpu block at all
    def make_block(num_pages: int, use_device: bool = True) -> Block | None:
        if num_pages <= 0:
            return None
            
        shape = (num_pages, self.page_size) + self.page_subshape
        pages = jnp.zeros(shape, dtype=self.dtype)
        
        sharding = self.get_physical_sharding()
        if sharding is not None and use_device:
            pages = utils.shard(pages, sharding)
            
        avail_indices = jnp.arange(num_pages, dtype=jnp.int32)
        num_avail = jnp.array(num_pages, dtype=jnp.int32)
        
        if use_device and self.device is not None:
            pages = jax.device_put(pages, self.device)
            avail_indices = jax.device_put(avail_indices, self.device)
            num_avail = jax.device_put(num_avail, self.device)
            
        return Block(
            pages=pages,
            available_page_indices=avail_indices,
            num_available_pages=num_avail,
            page_size=self.page_size
        )

    tpu_block = make_block(tpu_num_pages, use_device=True)
    cpu_block = make_block(cpu_num_pages, use_device=False)
    
    # Normally, if CPU offloading is enabled it could be device=jax.devices('cpu')[0]
    # But JAX defaults to CPU anyway without sharding constraints for memory pools
    # This is redundant. It should all be handled in cpu_block. We should pass the device
    # To cpu_block
    if cpu_block is not None:
        cpu_device = jax.devices('cpu')[0]
        cpu_pages = jax.device_put(cpu_block.pages, cpu_device)
        cpu_avail_indices = jax.device_put(cpu_block.available_page_indices, cpu_device)
        cpu_num_avail = jax.device_put(cpu_block.num_available_pages, cpu_device)
        cpu_block = Block(
            pages=cpu_pages,
            available_page_indices=cpu_avail_indices,
            num_available_pages=cpu_num_avail,
            page_size=self.page_size
        )

    page_indices = jnp.full(
        (self.max_num_seqs, self.max_num_pages_per_seq), -1, dtype=jnp.int32
    )
    seq_lens = jnp.zeros((self.max_num_seqs,), dtype=jnp.int32)
    
    # TODO: Again no more self.device
    if self.device is not None:
      page_indices = jax.device_put(page_indices, self.device)
      seq_lens = jax.device_put(seq_lens, self.device)

    return TpuCpuPageManager(
        tpu_block=tpu_block,
        cpu_block=cpu_block,
        page_indices=page_indices,
        seq_lens=seq_lens,
        max_seq_len=self.max_seq_len,
        window_size=None,
    )


def _remove_dp_spec(spec: P) -> P:
  dp_axis = ['dp', 'fsdp']
  new_spec = tuple(None if axis in dp_axis else axis for axis in spec)
  return P(*new_spec)


def _put_on_target_device(
    tensor: jax.Array, target_tensor: jax.Array
) -> jax.Array:
  if hasattr(target_tensor, 'sharding') and target_tensor.sharding is not None:
    sharding = target_tensor.sharding
    if isinstance(sharding, jax.sharding.NamedSharding):
      safe_spec = _remove_dp_spec(sharding.spec)
      target_sharding = jax.sharding.NamedSharding(sharding.mesh, safe_spec)
      return jax.device_put(tensor, target_sharding)
    elif isinstance(sharding, jax.sharding.SingleDeviceSharding):
      return jax.device_put(tensor, sharding)

  if hasattr(target_tensor, 'devices') and len(target_tensor.devices()) > 0:
    return jax.device_put(tensor, list(target_tensor.devices())[0])

  return tensor

def copy_physical_pages(
    src_pages: jax.Array,
    dst_pages: jax.Array,
    src_idxs: jax.Array,
    dst_idxs: jax.Array,
) -> jax.Array:
    if len(src_idxs) == 0:
        return dst_pages

    src_slice = src_pages[src_idxs]
    src_slice = _put_on_target_device(src_slice, dst_pages)
    
    dst_pages = dst_pages.at[dst_idxs].set(src_slice)
    return dst_pages


import uuid

class CacheManager:
    """Python level cache tracker for Continuous Batching."""

    def __init__(
        self,
        transformer,
        cache_config,
        max_seq_len: int
    ):
        import jax.numpy as jnp
        import jax
        from tunix.generate import utils
        
        if hasattr(transformer, 'config'):
            dtype = transformer.config.dtype
            num_kv_heads = transformer.config.num_kv_heads
            head_dim = transformer.config.head_dim
            num_layers = transformer.config.num_layers
        else:
            dtype = jnp.float32
            num_kv_heads = 1
            head_dim = 1
            num_layers = 1
            
        page_size = cache_config.page_size
        max_num_seqs = cache_config.max_num_seqs
        
        # Discover parallel axes if present on transformer params
        dp_axis = None
        tp_axis = None
        dp_size = 1
        tp_size = 1
        try:
            shd_config = getattr(getattr(transformer, "config", None), "shd_config", None)
            if shd_config is not None:
                dp_axis = shd_config.act_btd[0]
                tp_axis = shd_config.act_btnh[2]
            
            import flax.nnx as nnx
            params = nnx.variables(transformer)
            param_0 = jax.tree.leaves(params)[0]
            if hasattr(param_0, "sharding") and hasattr(param_0.sharding, "mesh") and param_0.sharding.mesh is not None:
                mesh = param_0.sharding.mesh
                dp_size = mesh.shape.get(dp_axis, 1) if dp_axis else 1
                tp_size = mesh.shape.get(tp_axis, 1) if tp_axis else 1
        except Exception:
            pass

        config = TpuCpuPageManagerConfig(
            page_size=page_size,
            page_subshape=(num_kv_heads, head_dim),
            dtype=dtype,
            max_num_seqs=max_num_seqs,
            max_seq_len=max_seq_len,
            max_tpu_bytes=getattr(cache_config, "hbm_cache_max_bytes", 1),
            max_cpu_bytes=getattr(cache_config, "cpu_offload_bytes", 0),
            logical_page_sharding="dp_axis",
            logical_subshape_sharding=(None, "tp_axis"),
            dp_axis=dp_axis,
            tp_axis=tp_axis,
            dp_size=dp_size,
            tp_size=tp_size,
            device=None
        )
        self.hbm_pm = config.init()
        self.max_num_seqs = max_num_seqs
        self.max_num_pages_per_seq = utils.cdiv(max_seq_len, page_size)
        self.page_size = page_size

        
        self.page_locations = {} # dict of page_id (str) -> "tpu" or "cpu"
        self.page_ids_to_idxs = {} # dict of page_id (str) -> physical_idx
        
    @property
    def num_free_tpu_pages(self):
        if hasattr(self.hbm_pm, 'tpu_block') and hasattr(self.hbm_pm.tpu_block, 'num_available_pages'):
            return int(jax.device_get(self.hbm_pm.tpu_block.num_available_pages))
        return 0

    @property
    def available_hbm_pages(self):
        return self.num_free_tpu_pages
        
    def allocate(self, num_pages: int) -> list:
        # We need to allocate num_pages from hbm_pm.tpu_block
        import jax.numpy as jnp
        import jax
        
        @jax.jit
        def _alloc_tpu(pm):
            return pm.tpu_block.allocate(num_pages)
            
        new_block, allocated_idxs = _alloc_tpu(self.hbm_pm)
        import dataclasses
        self.hbm_pm = dataclasses.replace(self.hbm_pm, tpu_block=new_block)
        
        allocated_idxs_py = jax.device_get(allocated_idxs)
        new_page_ids = []
        for i in range(num_pages):
            pid = str(uuid.uuid4())
            new_page_ids.append(pid)
            self.page_locations[pid] = "tpu"
            self.page_ids_to_idxs[pid] = int(allocated_idxs_py[i])
            
        return new_page_ids

    def assign(self, new_page_ids: list):
        # new_page_ids is list of list of page_id strings [batch_size, num_pages] (ragged)
        import jax.numpy as jnp
        import jax
        import dataclasses
        
        batch_size = len(new_page_ids)
        seq_idxs = []
        page_indices = []
        lens = []
        
        for i in range(batch_size):
            pids = new_page_ids[i]
            lens.append(len(pids))
            seq_idxs.append(i)
            # padded to max_num_pages_per_seq for jax? Or flat? The RaggedArray expects flat data and lens!
            for pid in pids:
                page_indices.append(self.page_ids_to_idxs[pid])
                
        seq_idxs_j = jnp.array(seq_idxs, dtype=jnp.int32)
        page_indices_j = jnp.array(page_indices, dtype=jnp.int32)
        lens_j = jnp.array(lens, dtype=jnp.int32)
        
        @jax.jit
        def _assign(pm, seq_idxs, page_indices, lens):
            return pm.assign(seq_idxs, page_indices, lens)
            
        self.hbm_pm = _assign(self.hbm_pm, seq_idxs_j, page_indices_j, lens_j)

    def offload(self, page_ids: list):
        import jax.numpy as jnp
        import jax
        
        page_idxs = [self.page_ids_to_idxs[pid] for pid in page_ids]
        num_pages_j = jnp.array([len(page_ids)], dtype=jnp.int32)
        page_idxs_j = jnp.array(page_idxs, dtype=jnp.int32)
        
        @jax.jit
        def _offload(pm, num_pages, page_idxs):
            return pm.offload(num_pages, page_idxs)
            
        self.hbm_pm, cpu_idxs = _offload(self.hbm_pm, num_pages_j, page_idxs_j)
        cpu_idxs_py = jax.device_get(cpu_idxs)
        
        for i, pid in enumerate(page_ids):
            self.page_locations[pid] = "cpu"
            self.page_ids_to_idxs[pid] = int(cpu_idxs_py[i])

    def load(self, page_ids: list):
        import jax.numpy as jnp
        import jax
        
        page_idxs = [self.page_ids_to_idxs[pid] for pid in page_ids]
        num_pages_j = jnp.array([len(page_ids)], dtype=jnp.int32)
        page_idxs_j = jnp.array(page_idxs, dtype=jnp.int32)
        
        @jax.jit
        def _load(pm, num_pages, page_idxs):
            return pm.load(num_pages, page_idxs)
            
        self.hbm_pm, tpu_idxs = _load(self.hbm_pm, num_pages_j, page_idxs_j)
        tpu_idxs_py = jax.device_get(tpu_idxs)
        
        for i, pid in enumerate(page_ids):
            self.page_locations[pid] = "tpu"
            self.page_ids_to_idxs[pid] = int(tpu_idxs_py[i])

    def evict(self, page_ids: list):
        import jax.numpy as jnp
        import jax
        
        # Evict typically means release from whichever block they are in
        # Group by device
        tpu_pids = [pid for pid in page_ids if self.page_locations[pid] == "tpu"]
        cpu_pids = [pid for pid in page_ids if self.page_locations[pid] == "cpu"]
        
        @jax.jit
        def _release_pages(pm, num_pages, page_idxs, device):
            return pm.release_pages(num_pages, page_idxs, device)
            
        if tpu_pids:
            num_tpu = jnp.array([len(tpu_pids)], dtype=jnp.int32)
            tpu_idxs_j = jnp.array([self.page_ids_to_idxs[pid] for pid in tpu_pids], dtype=jnp.int32)
            self.hbm_pm = _release_pages(self.hbm_pm, num_tpu, tpu_idxs_j, "tpu")
            
        if cpu_pids:
            num_cpu = jnp.array([len(cpu_pids)], dtype=jnp.int32)
            cpu_idxs_j = jnp.array([self.page_ids_to_idxs[pid] for pid in cpu_pids], dtype=jnp.int32)
            self.hbm_pm = _release_pages(self.hbm_pm, num_cpu, cpu_idxs_j, "cpu")
            
        for pid in page_ids:
            del self.page_locations[pid]
            del self.page_ids_to_idxs[pid]
            
    def release_sequences(self, seq_mask: list):
        import jax.numpy as jnp
        import jax
        
        seq_mask_j = jnp.array(seq_mask, dtype=jnp.bool_)
        @jax.jit
        def _rel_seq(pm, mask):
            return pm.release_sequences(mask)
        self.hbm_pm = _rel_seq(self.hbm_pm, seq_mask_j)

