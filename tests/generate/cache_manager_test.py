import os

os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'

import unittest
import numpy as np
import jax.numpy as jnp

from tunix.generate import cache_manager as pm_lib
from tunix.generate import page_manager as pm_core


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

    self.assertIsNotNone(pm.page_manager.tpu_block)
    self.assertEqual(pm.page_manager.tpu_block.total_num_pages, 15)

    self.assertIsNotNone(pm.page_manager.cpu_block)
    self.assertEqual(pm.page_manager.cpu_block.total_num_pages, 15)

    self.assertEqual(pm.max_num_pages_per_seq, 5)

  def test_allocate(self):
    pm = self.config.init()

    allocated_ids = pm.allocate(2)
    self.assertEqual(len(allocated_ids), 2)
    self.assertEqual(pm.available_tpu_pages, 13)

    pm.assign([allocated_ids])
    self.assertEqual(pm.seq_lens[0], 2)

  def test_offload_and_load(self):
    pm = self.config.init()

    # allocate 3 pages
    allocated_ids = pm.allocate(3)
    self.assertEqual(pm.available_tpu_pages, 12)

    # offload
    pm.offload(allocated_ids)
    self.assertEqual(pm.available_tpu_pages, 15)
    self.assertEqual(pm.available_cpu_pages, 12)

    # load
    pm.load(allocated_ids)
    self.assertEqual(pm.available_tpu_pages, 12)
    self.assertEqual(pm.available_cpu_pages, 15)

  def test_evict(self):
    pm = self.config.init()
    # allocate 4 pages
    allocated_ids = pm.allocate(4)
    pm.evict(allocated_ids)

    self.assertEqual(pm.available_tpu_pages, 15)
    self.assertEqual(pm.available_cpu_pages, 15)


if __name__ == '__main__':
  unittest.main()
