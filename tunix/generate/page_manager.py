"""A ragged page manager for a batch of sequences."""

from collections.abc import Sequence
import dataclasses
import functools
from typing import Any, Optional
import contextlib

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

  pages: dict[str, jax.Array]
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

    updated_num_available_pages = (self.num_available_pages - num_pages)
    updated_available_page_indices = jnp.roll(self.available_page_indices,
                                              -num_pages)

    new_block = dataclasses.replace(
        self,
        available_page_indices=updated_available_page_indices,
        num_available_pages=updated_num_available_pages,
    )
    return new_block, allocated_pages

  @jax.named_call
  def release(self, num_pages_to_release: int | jax.Array,
              page_idxs_to_release: jax.Array) -> 'Block':
    """Releases pages."""
    target_slots = jnp.arange(self.total_num_pages) + self.num_available_pages

    # Map non-target slots to out of bounds indicies so that they are not modified
    safe_target_slots = jnp.where(
        jnp.arange(self.total_num_pages) < num_pages_to_release, target_slots,
        self.total_num_pages)

    updated_available_page_indices = self.available_page_indices.at[
        safe_target_slots].set(page_idxs_to_release, mode='drop')

    updated_num_available_pages = (self.num_available_pages +
                                   num_pages_to_release)
    return dataclasses.replace(
        self,
        available_page_indices=updated_available_page_indices,
        num_available_pages=updated_num_available_pages,
    )

  def write_values(self, num_values: jax.Array, values: jax.Array,
                   page_indices: jax.Array, page_offsets: jax.Array, block_id: str = "tokens") -> 'Block':
    """Write packed 1D array of values into allocated paged memory of block."""
    max_n_pages = self.pages[block_id].shape[0]
    batch_size = values.shape[0]

    target_page_indices = jnp.where(
        jnp.arange(batch_size) < num_values, page_indices, max_n_pages)

    updated_arr = self.pages[block_id].at[target_page_indices, page_offsets].set(values, mode='drop')
    new_pages = dict(self.pages)
    new_pages[block_id] = updated_arr

    return dataclasses.replace(self, pages=new_pages)


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
    return self.seq_lens

  @jax.named_call
  def allocate(
      self, num_pages_to_allocate: jax.Array
  ) -> tuple['TpuCpuPageManager', jax.Array]:
    """Allocates pages for new tokens."""
    total_pages_to_allocate = jnp.sum(num_pages_to_allocate)
    new_tpu_block, allocated_page_data = self.tpu_block.allocate(
        total_pages_to_allocate)

    padded_allocated_page_data = jnp.where(
        jnp.arange(self.tpu_block.total_num_pages) < total_pages_to_allocate,
        self.tpu_block.available_page_indices, self.tpu_block.total_num_pages)

    new_page_manager = dataclasses.replace(
        self,
        tpu_block=new_tpu_block,
    )

    return new_page_manager, padded_allocated_page_data

  @jax.named_call
  def assign(self, page_indices: jax.Array,
             lens: jax.Array) -> 'TpuCpuPageManager':
    """Assigns physical page indices to sequences directly."""
    ragged = RaggedArray(data=page_indices, lens=lens)

    target_rows = ragged.row_idxs
    target_cols = self.seq_lens[target_rows] + ragged.intra_offsets

    updated_page_indices = self.page_indices.at[target_rows,
                                                target_cols].set(ragged.data,
                                                                 mode='drop')

    seq_idxs = jnp.arange(lens.shape[0])
    updated_lens = self.seq_lens.at[seq_idxs].set(self.seq_lens[seq_idxs] +
                                                  lens,
                                                  mode='drop')

    return dataclasses.replace(self,
                               page_indices=updated_page_indices,
                               seq_lens=updated_lens)

  @jax.named_call
  def release(self,
              should_release: jax.Array,
              device: str = 'tpu') -> 'TpuCpuPageManager':
    """Releases pages from block."""
    updated_lens = jnp.where(should_release, 0, self.seq_lens)

    block = self.tpu_block if device == 'tpu' else self.cpu_block

    page_indices_to_release = RaggedArray(
        data=jax.lax.empty((block.total_num_pages,), dtype=jnp.int32),
        lens=jnp.where(should_release, self.num_pages_per_seq, 0),
    )
    page_indices_irows = page_indices_to_release.row_idxs
    page_indices_icols = page_indices_to_release.intra_offsets

    is_real_release = (jnp.arange(block.total_num_pages)
                       < page_indices_to_release.total_length)
    safe_icols = jnp.where(is_real_release, page_indices_icols, 0)
    released_pages = self.page_indices[page_indices_irows, safe_icols]

    num_total_released = jnp.sum(page_indices_to_release.lens)
    new_block = block.release(num_total_released, released_pages)

    kwargs = {'seq_lens': updated_lens}
    if device == 'tpu':
      kwargs['tpu_block'] = new_block
    else:
      kwargs['cpu_block'] = new_block

    return dataclasses.replace(self, **kwargs)

  @jax.named_call
  @jax.named_call
  def release_pages(self, num_pages: int | jax.Array, page_idxs: jax.Array, device: str = 'tpu') -> 'TpuCpuPageManager':
    """Releases specific physical pages back to the block's available pool."""
    if device == 'tpu':
      new_block = self.tpu_block.release(num_pages, page_idxs)
      return dataclasses.replace(self, tpu_block=new_block)
    elif device == 'cpu':
      if self.cpu_block is None:
        raise ValueError("CPU block is not initialized.")
      new_block = self.cpu_block.release(num_pages, page_idxs)
      return dataclasses.replace(self, cpu_block=new_block)
    else:
      raise ValueError(f"Unknown device: {device}")

  def release_sequences(self, should_release: jax.Array) -> 'TpuCpuPageManager':
    """Releases sequence."""
    updated_lens = jnp.where(should_release, 0, self.seq_lens)

    block = self.tpu_block
    page_indices_to_release = RaggedArray(
        data=jax.lax.empty((block.total_num_pages,), dtype=jnp.int32),
        lens=jnp.where(should_release, self.num_pages_per_seq, 0),
    )

    page_indices_irows = page_indices_to_release.row_idxs
    page_indices_icols = page_indices_to_release.intra_offsets

    is_real_release = (jnp.arange(block.total_num_pages)
                       < page_indices_to_release.total_length)
    safe_icols = jnp.where(is_real_release, page_indices_icols, 0)
    released_pages = self.page_indices[page_indices_irows, safe_icols]

    num_total_released = jnp.sum(page_indices_to_release.lens)
    new_block = block.release(num_total_released, released_pages)

    num_released_pages = jnp.sum(self.seq_lens - updated_lens)

    new_page_manager = dataclasses.replace(self, seq_lens=updated_lens)

    return new_page_manager, num_released_pages, released_pages

  @jax.named_call
  def offload(self, num_pages: int | jax.Array,
              page_idxs: jax.Array, block_id: str = "tokens") -> tuple['TpuCpuPageManager', jax.Array]:
    """Moves physical pages from TPU block to CPU block."""
    if self.cpu_block is None:
      raise ValueError("Cannot offload; cpu_block is None.")

    new_cpu_block, cpu_allocated_pages = self.cpu_block.allocate(num_pages)

    padded_allocated_page_data = jnp.where(
        jnp.arange(self.cpu_block.total_num_pages) < num_pages,
        self.cpu_block.available_page_indices, self.cpu_block.total_num_pages)

    is_real = jnp.arange(page_idxs.shape[0]) < num_pages

    # Copy data
    safe_tpu_phys = jnp.where(is_real, page_idxs, 0)
    tpu_vals = self.tpu_block.pages[block_id][safe_tpu_phys]
    tpu_vals_cpu = _put_on_target_device(tpu_vals, self.cpu_block.pages[block_id])

    safe_cpu_phys = jnp.where(is_real,
                              padded_allocated_page_data[:page_idxs.shape[0]],
                              self.cpu_block.pages[block_id].shape[0])
    
    updated_arr = self.cpu_block.pages[block_id].at[safe_cpu_phys].set(tpu_vals_cpu, mode='drop')
    new_cpu_pages = dict(self.cpu_block.pages)
    new_cpu_pages[block_id] = updated_arr
    new_cpu_block = dataclasses.replace(new_cpu_block, pages=new_cpu_pages)

    # Release from TPU
    new_tpu_block = self.tpu_block.release(num_pages, page_idxs)

    new_pm = dataclasses.replace(
        self,
        tpu_block=new_tpu_block,
        cpu_block=new_cpu_block,
    )
    return new_pm, padded_allocated_page_data

  @jax.named_call
  def load(self, num_pages: int | jax.Array,
           page_idxs: jax.Array, block_id: str = "tokens") -> tuple['TpuCpuPageManager', jax.Array]:
    """Moves physical pages from CPU block back to TPU block."""
    if self.cpu_block is None:
      raise ValueError("Cannot load; cpu_block is None.")

    new_tpu_block, tpu_allocated_pages = self.tpu_block.allocate(num_pages)

    padded_allocated_page_data = jnp.where(
        jnp.arange(self.tpu_block.total_num_pages) < num_pages,
        self.tpu_block.available_page_indices, self.tpu_block.total_num_pages)

    is_real = jnp.arange(page_idxs.shape[0]) < num_pages

    # Copy data
    safe_cpu_phys = jnp.where(is_real, page_idxs, 0)
    cpu_vals = self.cpu_block.pages[block_id][safe_cpu_phys]
    cpu_vals_tpu = _put_on_target_device(cpu_vals, self.tpu_block.pages[block_id])

    safe_tpu_phys = jnp.where(is_real,
                              padded_allocated_page_data[:page_idxs.shape[0]],
                              self.tpu_block.pages[block_id].shape[0])
    
    updated_arr = self.tpu_block.pages[block_id].at[safe_tpu_phys].set(cpu_vals_tpu, mode='drop')
    new_tpu_pages = dict(self.tpu_block.pages)
    new_tpu_pages[block_id] = updated_arr
    new_tpu_block = dataclasses.replace(new_tpu_block, pages=new_tpu_pages)

    # Release from CPU
    new_cpu_block = self.cpu_block.release(num_pages, page_idxs)

    new_pm = dataclasses.replace(
        self,
        tpu_block=new_tpu_block,
        cpu_block=new_cpu_block,
    )
    return new_pm, padded_allocated_page_data

  @jax.named_call
  def release_for_window(self) -> 'TpuCpuPageManager':
    """Release allocations for window."""
    if self.window_size is None:
      return self

    num_pages_to_release = (jnp.maximum(
        self.seq_lens - (self.window_size // self.page_size), 0))
    page_indices_irows = jnp.arange(self.batch_size)[:, None]
    page_indices_icols = (jnp.arange(self.max_num_pages_per_seq) +
                          num_pages_to_release[:, None])
    updated_page_indices = self.page_indices[page_indices_irows,
                                             page_indices_icols]
    release_helper = RaggedArray(
        data=jax.lax.empty((self.tpu_block.total_num_pages,), dtype=jnp.int32),
        lens=num_pages_to_release,
    )

    is_real_release = (jnp.arange(self.tpu_block.total_num_pages)
                       < release_helper.total_length)
    safe_icols = jnp.where(is_real_release, release_helper.intra_offsets, 0)
    released_page_indices = (self.page_indices[release_helper.row_idxs,
                                               safe_icols])

    new_tpu_block = self.tpu_block.release(jnp.sum(release_helper.lens),
                                           released_page_indices)

    return dataclasses.replace(
        self,
        page_indices=updated_page_indices,
        tpu_block=new_tpu_block,
        seq_lens=self.seq_lens - num_pages_to_release,
    )

  def load_values(self, values: jax.Array,
                  lens: jax.Array, block_id: str = "tokens") -> 'TpuCpuPageManager':
    """Loads packed 1D array of values into allocated paged memory of block."""
    values_ragged = RaggedArray(data=values, lens=lens)
    seq_idxs = values_ragged.row_idxs
    value_offsets = values_ragged.intra_offsets

    local_page_cols = value_offsets // self.page_size
    page_offsets = value_offsets % self.page_size
    phys_page_ids = self.page_indices[seq_idxs, local_page_cols]

    is_real = jnp.arange(values_ragged.capacity) < values_ragged.total_length

    new_tpu_block = self.tpu_block.write_values(values_ragged.capacity, values,
                                                phys_page_ids, page_offsets, block_id)

    return dataclasses.replace(self, tpu_block=new_tpu_block)

  def insert_values(
      self,
      values: jax.Array,
      block_id: str = "tokens",
      idxs: jax.Array | None = None,
      valid_mask: jax.Array | None = None,
  ) -> 'TpuCpuPageManager':
    """Insert 1 new token per sequence into paged memory."""
    if valid_mask is None:
      valid_mask = self.seq_lens > 0

    n_values = jnp.sum(valid_mask)
    local_page_cols = idxs // self.page_size
    page_offsets = idxs % self.page_size
    seq_idxs = jnp.arange(self.batch_size)
    phys_page_ids = self.page_indices[seq_idxs, local_page_cols]

    new_tpu_block = self.tpu_block.write_values(n_values, values, phys_page_ids,
                                                page_offsets)

    return dataclasses.replace(self, tpu_block=new_tpu_block)

  def to_array(
      self,
      total_num_elements: int,
      seq_lens: jax.Array,
      block_id: str = "tokens",
  ) -> jax.Array:
    """Extracts array of token IDs from paged memory."""
    elements_ragged = RaggedArray(
        data=jnp.zeros((total_num_elements,), dtype=jnp.int32),
        lens=seq_lens,
    )
    seq_idxs = elements_ragged.row_idxs
    element_offsets = elements_ragged.intra_offsets

    local_page_cols = element_offsets // self.page_size
    page_offsets = element_offsets % self.page_size
    phys_page_ids = self.page_indices[seq_idxs, local_page_cols]

    packed_tokens = self.tpu_block.pages[block_id][phys_page_ids, page_offsets]
    return packed_tokens


@dataclasses.dataclass(frozen=True, kw_only=True)
class TpuCpuPageManagerConfig:
  """Configuration for a TpuCpuPageManager."""
  page_size: int
  page_subshape: tuple[int, ...] = ()
  dtype: jnp.dtype
  block_keys: tuple[str, ...] = ("tokens",)

  max_num_seqs: int
  max_seq_len: int
  max_tpu_bytes: int
  max_cpu_bytes: int = 0

  logical_page_sharding: str | None = None
  logical_subsharding: tuple[str | None, ...] = ()

  dp_axis: str | None = None
  tp_axis: str | None = None
  dp_size: int = 1
  tp_size: int = 1

  def _calculate_pages_for_capacity(self, max_bytes: int,
                                    logical_sharding: tuple) -> int:
    item_size = jnp.dtype(self.dtype).itemsize
    page_shape = (self.page_size,) + self.page_subshape

    block_subsharding = logical_sharding[1:]
    elements = 1
    for dim, shard in zip(page_shape, block_subsharding):
      dim_size = (dim * self.dp_size) if shard == 'dp_axis' else dim
      elements *= dim_size

    page_bytes = elements * item_size * len(self.block_keys)
    if page_bytes == 0:
      return 0

    num_block_pages = max_bytes // page_bytes
    page_sharding = logical_sharding[0]
    if page_sharding == 'dp_axis':
      num_block_pages = (num_block_pages // self.dp_size) * self.dp_size

    return num_block_pages

  @property
  def num_tpu_pages(self) -> int:
    return self._calculate_pages_for_capacity(
        max_bytes=self.max_tpu_bytes, logical_sharding=self.logical_sharding)

  @property
  def num_cpu_pages(self) -> int:
    # A block has shape: num_pages, page_size, *page_subshape
    sharding_len = 2 + len(self.page_subshape)

    return self._calculate_pages_for_capacity(max_bytes=self.max_cpu_bytes,
                                              logical_sharding=(None,) *
                                              sharding_len)

  @property
  def max_num_pages_per_seq(self) -> int:
    return utils.cdiv(self.max_seq_len, self.page_size)

  @property
  def logical_shard_to_physical(self) -> dict:
    return {'dp_axis': self.dp_axis, 'tp_axis': self.tp_axis, None: None}

  @property
  def logical_sharding(self):
    logical_page_sharding = self.logical_page_sharding
    logical_subsharding = self.logical_subsharding

    logical_prefix_sharding = (logical_page_sharding, None)

    l_subsharding = len(logical_subsharding)
    l_subshape = len(self.page_subshape)

    if l_subsharding > l_subshape:
      raise ValueError(f'Cannot initialize BlockSpec {spec.name}. '
                       f'Block subsharding `{spec.logical_subsharding}` '
                       'cannot have length greater than block subshape '
                       f'`{spec.subshape}`')

    if l_subsharding < l_subshape:
      num_padding = l_subshape - l_subsharding
      logical_subsharding += (None,) * num_padding

    logical_sharding = logical_prefix_sharding + logical_subsharding

    return logical_sharding

  @property
  def physical_sharding(self):
    logical_sharding = self.logical_sharding
    if all(axis is None for axis in logical_sharding):
      return None

    mapping = self.logical_shard_to_physical
    return tuple(mapping[axis] for axis in logical_sharding)

  def _make_block(self,
                  num_pages: int,
                  sharding=None,
                  device: jax.Device = None) -> Block:
    with jax.default_device(device) if device else contextlib.nullcontext():
      shape = (num_pages, self.page_size) + self.page_subshape
      
      pages_dict = {}
      for k in self.block_keys:
        arr = jnp.zeros(shape, dtype=self.dtype)
        if sharding is not None:
          arr = utils.shard(arr, sharding)
        pages_dict[k] = arr
        
      avail_indices = jnp.arange(num_pages, dtype=jnp.int32)
      num_avail = jnp.array(num_pages, dtype=jnp.int32)

    return Block(pages=pages_dict,
                 available_page_indices=avail_indices,
                 num_available_pages=num_avail,
                 page_size=self.page_size)

  def init(self) -> 'TpuCpuPageManager':
    """Initializes physical page tensors for TPU and CPU."""
    if self.num_tpu_pages < self.max_num_pages_per_seq:
      raise ValueError(
          'TPU block capacity is too small. '
          f'Available pages: {tpu_num_pages}, Max required per seq: {self.max_num_pages_per_seq}'
      )

    tpu_block = self._make_block(num_pages=self.num_tpu_pages,
                                 sharding=self.physical_sharding)

    cpu_block = None
    if self.num_cpu_pages > 0:
      cpu_block = self._make_block(num_pages=self.num_cpu_pages,
                                   device=jax.devices('cpu')[0])

    page_indices = jnp.full((self.max_num_seqs, self.max_num_pages_per_seq),
                            -1,
                            dtype=jnp.int32)
    seq_lens = jnp.zeros((self.max_num_seqs,), dtype=jnp.int32)

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


def _put_on_target_device(tensor: jax.Array,
                          target_tensor: jax.Array) -> jax.Array:
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
