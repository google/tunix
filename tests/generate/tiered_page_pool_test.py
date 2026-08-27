import os
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'

import unittest
import jax.numpy as jnp
from tunix.generate import tiered_page_pool as pm_lib

class TieredPagePoolTest(unittest.TestCase):

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
    
    page_subshape = (2 * self.dummy_model_config.num_kv_heads, 1, self.dummy_model_config.head_dim)
    self.tiered_config = pm_lib.TieredPagePoolConfig(
        page_size=self.dummy_cache_config.page_size,
        page_subshape=page_subshape,
        dtype=jnp.float32,
        partition_keys=("layer_0",),
        num_tpu_pages=15,
        num_cpu_pages=15,
    )

  def test_initialization_single_device(self):
    tpu_pool, cpu_pool = self.tiered_config.init()
    pm = pm_lib.TieredPagePoolManager(tiered_config=self.tiered_config, tpu_pool=tpu_pool, cpu_pool=cpu_pool, max_num_seqs=5)

    self.assertIsNotNone(pm.tpu_pool)
    self.assertEqual(pm.num_free_tpu_pages, 15)

    self.assertIsNotNone(pm.cpu_pool)
    self.assertEqual(pm.num_free_cpu_pages, 15)

    self.assertEqual(pm.max_num_seqs, 5)

  def test_allocate(self):
    tpu_pool, cpu_pool = self.tiered_config.init()
    pm = pm_lib.TieredPagePoolManager(tiered_config=self.tiered_config, tpu_pool=tpu_pool, cpu_pool=cpu_pool, max_num_seqs=5)

    allocated_ids = pm.allocate_tpu_pages(2)
    self.assertEqual(len(allocated_ids), 2)
    self.assertEqual(pm.num_free_tpu_pages, 13)

  def test_offload_and_load(self):
    tpu_pool, cpu_pool = self.tiered_config.init()
    pm = pm_lib.TieredPagePoolManager(tiered_config=self.tiered_config, tpu_pool=tpu_pool, cpu_pool=cpu_pool, max_num_seqs=5)

    # allocate 3 pages
    allocated_ids = pm.allocate_tpu_pages(3)
    self.assertEqual(pm.num_free_tpu_pages, 12)

    # offload
    pm.offload(allocated_ids)
    self.assertEqual(pm.num_free_tpu_pages, 15)
    self.assertEqual(pm.num_free_cpu_pages, 12)

    # load
    pm.load(allocated_ids)
    self.assertEqual(pm.num_free_tpu_pages, 12)
    self.assertEqual(pm.num_free_cpu_pages, 15)

  def test_evict(self):
    tpu_pool, cpu_pool = self.tiered_config.init()
    pm = pm_lib.TieredPagePoolManager(tiered_config=self.tiered_config, tpu_pool=tpu_pool, cpu_pool=cpu_pool, max_num_seqs=5)
    
    # allocate 4 pages
    allocated_ids = pm.allocate_tpu_pages(4)
    pm.evict(allocated_ids)

    self.assertEqual(pm.num_free_tpu_pages, 15)

if __name__ == '__main__':
  unittest.main()
