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

class Block:
  """A block of physical pages."""

  layer_pages: dict[str, jax.Array]
  available_page_indices: collections.deque 
  num_available_pages: int 

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

  def num_tpu_pages(self) -> int:
    return self._calculate_pages_for_capacity(
        max_bytes=self.max_tpu_bytes, logical_sharding=self.logical_sharding)

  def num_cpu_pages(self) -> int:
    # A block has shape: num_pages, page_size, *page_subshape
    sharding_len = 2 + len(self.page_subshape)

    return self._calculate_pages_for_capacity(max_bytes=self.max_cpu_bytes,
                                              logical_sharding=(None,) *
                                              sharding_len)


