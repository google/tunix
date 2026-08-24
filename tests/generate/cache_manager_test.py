import os
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'

import dataclasses

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from tunix.generate import cache_manager as pm_lib


class CacheManagerTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.config = pm_lib.CacheManagerConfig(
        page_size=2,
        page_subshape=(2, 4),
        dtype=jnp.float32,
        max_num_seqs=5,
        max_seq_len=10,
        max_tpu_bytes=1000, 
        max_cpu_bytes=1000, 
    )

  def test_initialization_single_device(self):
    pm = self.config.init()
    
    # 1000 bytes / (2 * 2 * 4 * 4 bytes) = 1000 / 64 = 15 pages per block
    # Actually wait: page_size=2, page_subshape=(2,4). 2*2*4 = 16 elements. 16 * 4 bytes/fp32 = 64 bytes.
    # 1000 // 64 = 15 pages.
    
    self.assertIsNotNone(pm.tpu_block)
    self.assertEqual(pm.tpu_block.total_num_pages, 15)
    
    self.assertIsNotNone(pm.cpu_block)
    self.assertEqual(pm.cpu_block.total_num_pages, 15)

    self.assertEqual(pm.max_num_pages_per_seq, 5) # 10 / 2
    self.assertEqual(pm.batch_size, 5)

  def test_allocate(self):
    pm = self.config.init()
    q_lens = jnp.array([2, 5, 0, 0, 0])
    
    pm_allocated = pm.allocate(q_lens)
    
    # 2 tokens -> 1 page
    # 5 tokens -> 3 pages
    # Total allocated: 4 pages
    self.assertEqual(pm_allocated.tpu_block.num_available_pages, 11) # 15 - 4 -> 11
    
    np.testing.assert_array_equal(pm_allocated.num_pages_per_seq, jnp.array([1, 3, 0, 0, 0]))

    # page indices should have valid mappings!
    # sequence 0 offset 0 -> physical page 0
    # sequence 1 offset 0 -> physical page 1
    # sequence 1 offset 1 -> physical page 2
    # sequence 1 offset 2 -> physical page 3

    np.testing.assert_array_equal(pm_allocated.page_indices[0][:1], jnp.array([0]))
    np.testing.assert_array_equal(pm_allocated.page_indices[1][:3], jnp.array([1, 2, 3]))
    
  def test_release(self):
    pm = self.config.init()
    q_lens = jnp.array([2, 5, 0, 0, 0])
    pm = pm.allocate(q_lens)

    # Release sequence 0
    should_release = jnp.array([True, False, False, False, False])
    pm_released = pm.release(should_release)

    # Reclaimed 1 page -> 11 + 1 = 12
    self.assertEqual(pm_released.tpu_block.num_available_pages, 12)
    np.testing.assert_array_equal(pm_released.num_pages_per_seq, jnp.array([0, 3, 0, 0, 0]))

  def test_write_and_to_array(self):
    pm = self.config.init()
    q_lens = jnp.array([2])
    # Pad to batch_size=5
    q_lens = jax.numpy.pad(q_lens, (0, 4))
    
    pm = pm.allocate(q_lens)

    # Write a 2-token array of shape (2, 2, 4)
    values = jnp.ones((2, 2, 4), dtype=jnp.float32) * 5.0
    pm = pm.load_values(values, lens=q_lens)

    extracted = pm.to_array(2)
    
    # Extract the packed array to shape (2, 2, 4)
    np.testing.assert_array_equal(extracted, values)

  def test_batch_copy_pages(self):
    pm = self.config.init()
    q_lens = jnp.array([4, 0, 0, 0, 0])
    pm = pm.allocate(q_lens)

    # Write some distinct values to TPU block
    values = jnp.ones((4, 2, 4), dtype=jnp.float32) * 42.0
    pm = pm.load_values(values, lens=q_lens)

    # Copy pages to cpu Block
    new_cpu_pages = pm_lib.copy_physical_pages(
        src_pages=pm.tpu_block.pages,
        dst_pages=pm.cpu_block.pages,
        src_idxs=pm.page_indices[0, :2], # first 2 pages
        dst_idxs=jnp.array([0, 1])
    )
    
    np.testing.assert_array_equal(new_cpu_pages[0, 0, ...], jnp.ones((2, 4)) * 42.0)



  def test_offload_and_load(self):
    pm = self.config.init()
    q_lens = jnp.array([4, 2, 0, 0, 0])
    
    # Pad q_lens for allocation
    pm = pm.allocate(q_lens)

    # Write a multi-token array of shape (4+2=6, 2, 4)
    # 4 tokens -> 2 pages, 2 tokens -> 1 page. Total 3 pages.
    
    seq_mask = jnp.array([True, False, False, False, False])
    
    # 1. Offload Sequence 0 (length 4) to CPU 
    pm_offloaded = pm.offload(seq_mask)
    
    # TPU block should have reclaimed 2 pages
    self.assertEqual(pm_offloaded.tpu_block.num_available_pages, 15 - 1) # seq 1 has 1 page
    
    # CPU block should have allocated 2 pages
    self.assertEqual(pm_offloaded.cpu_block.num_available_pages, 15 - 2)
    
    # 2. Release Sequence 0 from CPU 
    pm_released = pm_offloaded.release(seq_mask, device='cpu')
    self.assertEqual(pm_released.cpu_block.num_available_pages, 15)
    
    # 3. Load Sequence 1 to CPU (offload seq 1)
    pm = pm.offload(jnp.array([False, True, False, False, False]))
    self.assertEqual(pm.tpu_block.num_available_pages, 15 - 2) # wait seq 0 had 2 pages. So if offloaded seq 1, TPU has 15-2=13.
    self.assertEqual(pm.cpu_block.num_available_pages, 15 - 1)
    
    # 4. Load Sequence 1 back to TPU
    pm = pm.load(jnp.array([False, True, False, False, False]))
    self.assertEqual(pm.tpu_block.num_available_pages, 12) # 15 - (2 + 1)
    self.assertEqual(pm.cpu_block.num_available_pages, 15)

if __name__ == '__main__':
  unittest.main()
