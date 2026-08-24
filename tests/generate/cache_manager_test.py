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
    class DummyCacheConfig:
      page_size = 2
      max_num_seqs = 5
      max_prompt_length = 5
      max_tokens_to_generate = 5
      max_tpu_bytes = 1920
    
    class DummyModelConfig:
      num_layers = 1
      num_kv_heads = 2
      head_dim = 4

    self.dummy_cache_config = DummyCacheConfig()
    self.dummy_model_config = DummyModelConfig()

  def test_initialization_single_device(self):
    pm = pm_lib.init_cache_manager(self.dummy_cache_config, self.dummy_model_config, jnp.float32, max_cpu_bytes=1920)

    self.assertIsNotNone(pm.page_manager.tpu_block)
    self.assertEqual(pm.page_manager.tpu_block.total_num_pages, 15)

    self.assertIsNotNone(pm.page_manager.cpu_block)
    self.assertEqual(pm.page_manager.cpu_block.total_num_pages, 15)

    self.assertEqual(pm.max_num_pages_per_seq, 5)

  def test_allocate(self):
    pm = pm_lib.init_cache_manager(self.dummy_cache_config, self.dummy_model_config, jnp.float32, max_cpu_bytes=1920)

    allocated_ids = pm.allocate(2)
    self.assertEqual(len(allocated_ids), 2)
    self.assertEqual(pm.available_tpu_pages, 13)

    pm.assign([allocated_ids])
    self.assertEqual(pm.seq_lens[0], 2)

  def test_offload_and_load(self):
    pm = pm_lib.init_cache_manager(self.dummy_cache_config, self.dummy_model_config, jnp.float32, max_cpu_bytes=1920)

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
    pm = pm_lib.init_cache_manager(self.dummy_cache_config, self.dummy_model_config, jnp.float32, max_cpu_bytes=1920)
    # allocate 4 pages
    allocated_ids = pm.allocate(4)
    pm.evict(allocated_ids)

    self.assertEqual(pm.available_tpu_pages, 15)
    self.assertEqual(pm.available_cpu_pages, 15)


if __name__ == '__main__':
  unittest.main()
