import os

os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'

import unittest
import numpy as np
import jax.numpy as jnp

from tunix.generate import cache_manager as pm_lib


class CacheManagerTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    class DummyCacheConfig:
      page_size = 2
      max_num_seqs = 5
      max_prompt_length = 5
      max_tokens_to_generate = 5
      max_tpu_bytes = 1920
      max_cpu_bytes = 1920
    
    class DummyModelConfig:
      num_layers = 1
      num_kv_heads = 2
      head_dim = 4

    self.dummy_cache_config = DummyCacheConfig()
    self.dummy_model_config = DummyModelConfig()

  def test_initialization_single_device(self):
    pm = pm_lib.init_cache_manager(self.dummy_cache_config, self.dummy_model_config, jnp.float32)

    self.assertIsNotNone(pm.page_manager)
    self.assertEqual(pm.available_tpu_pages, 15)

    self.assertIsNotNone(pm.cpu_block)
    self.assertEqual(pm.available_cpu_pages, 15)

    self.assertEqual(pm.max_num_seqs, 5)

  def test_allocate(self):
    pm = pm_lib.init_cache_manager(self.dummy_cache_config, self.dummy_model_config, jnp.float32)

    allocated_ids = pm.allocate_tpu_pages(2)
    self.assertEqual(len(allocated_ids), 2)
    self.assertEqual(pm.available_tpu_pages, 13)

  def test_offload_and_load(self):
    pm = pm_lib.init_cache_manager(self.dummy_cache_config, self.dummy_model_config, jnp.float32)

    # allocate 3 pages
    allocated_ids = pm.allocate_tpu_pages(3)
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
    pm = pm_lib.init_cache_manager(self.dummy_cache_config, self.dummy_model_config, jnp.float32)
    # allocate 4 pages
    allocated_ids = pm.allocate_tpu_pages(4)
    pm.evict(allocated_ids)

    self.assertEqual(pm.available_tpu_pages, 15)


if __name__ == '__main__':
  unittest.main()
