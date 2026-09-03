from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np

from tunix.experimental.generate import tiered_page_pool
from tunix.experimental.generate import utils


class PagePoolTest(parameterized.TestCase):

  @parameterized.parameters((0,), (1,), (5,))
  def test_allocate(self, num_pages):
    total_pages = 10
    pages_dict = {"layer1": jnp.zeros((total_pages, 8))}
    avail_indices = list(range(total_pages))
    pool = tiered_page_pool.PagePool(
        partition_pages=pages_dict,
        available_page_indices=avail_indices,
    )
    prev_unallocated = set(avail_indices)
    prev_len = len(avail_indices)

    allocated = pool.allocate(num_pages)

    # Check set(available page indices) does not have allocated pages
    avail_set = set(pool.available_page_indices)
    for idx in allocated:
      self.assertNotIn(idx, avail_set)

    # Check available page indices contains all unallocated pages
    expected_unallocated = prev_unallocated - set(allocated)
    self.assertEqual(avail_set, expected_unallocated)

    # Check that returned indices were previously unallocated
    for idx in allocated:
      self.assertIn(idx, prev_unallocated)

    # Check len
    self.assertLen(pool.available_page_indices, prev_len - num_pages)

  @parameterized.parameters((0,), (1,), (5,))
  def test_free(self, num_pages):
    total_pages = 10
    pages_dict = {"layer1": jnp.zeros((total_pages, 8))}
    avail_indices = list(range(total_pages))
    pool = tiered_page_pool.PagePool(
        partition_pages=pages_dict,
        available_page_indices=avail_indices,
    )
    allocated = pool.allocate(num_pages)
    prev_avail = list(pool.available_page_indices)
    prev_len = len(prev_avail)

    pool.free(allocated)

    avail_set = set(pool.available_page_indices)

    # Check available page indices contains all previous pages
    for idx in prev_avail:
      self.assertIn(idx, avail_set)

    # Check available page indices contains new freed pages
    for idx in allocated:
      self.assertIn(idx, avail_set)

    # Check len
    self.assertLen(pool.available_page_indices, prev_len + num_pages)


class TieredPagePoolManagerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if len(jax.devices()) < 4:
      self.skipTest("Requires at least 4 devices")
    mesh_shape = (2, 2)
    self.devices = np.array(jax.devices()[:4]).reshape(mesh_shape)
    self.mesh = Mesh(self.devices, axis_names=("dp", "tp"))

  def get_config(self, sharding_type, has_subshape=True):
    page_size = 16
    page_subshape = (2, 1, 5) if has_subshape else ()

    logical_page_sharding = None
    logical_subsharding = ()
    dp_axis = None
    tp_axis = None
    dp_size = 1
    tp_size = 1

    if sharding_type == "dp TPU sharding":
      logical_page_sharding = "dp"
      dp_axis = "dp"
      dp_size = 2
    elif sharding_type == "tp TPU sharding":
      logical_page_sharding = "tp"
      tp_axis = "tp"
      tp_size = 2
    elif sharding_type == "dp + tp TPU sharding":
      logical_page_sharding = "dp"
      logical_subsharding = ("tp",)
      dp_axis = "dp"
      tp_axis = "tp"
      dp_size = 2
      tp_size = 2

    return tiered_page_pool.TieredPagePoolConfig(
        page_size=page_size,
        page_subshape=page_subshape,
        dtype=jnp.float32,
        partition_keys=("layer_0", "layer_1"),
        num_tpu_pages=10,
        num_cpu_pages=10,
        logical_page_sharding=logical_page_sharding,
        logical_subsharding=logical_subsharding,
        dp_axis=dp_axis,
        tp_axis=tp_axis,
        dp_size=dp_size,
        tp_size=tp_size,
    )

  @parameterized.parameters((0,), (1,), (5,))
  def test_allocate_tpu_pages(self, num_pages):
    config = self.get_config("no TPU sharding", has_subshape=False)
    tpu_pool, cpu_pool = config.init()
    manager = tiered_page_pool.TieredPagePoolManager(config, tpu_pool, cpu_pool)

    allocated = manager.allocate_tpu_pages(num_pages)

    self.assertLen(set(allocated), num_pages)
    for pid in allocated:
      self.assertEqual(manager.get_page_location(pid), "tpu")
      phys_idx = manager.get_page_idx(pid)
      self.assertNotIn(phys_idx, manager.tpu_pool.available_page_indices)

  @parameterized.parameters(
      ("No subshape, no sharding", "no TPU sharding", False),
      ("no TPU sharding + no subshape", "no TPU sharding", False),
      ("no TPU sharding + subshape", "no TPU sharding", True),
      ("dp TPU sharding + subshape", "dp TPU sharding", True),
      ("tp TPU sharding + subshape", "tp TPU sharding", True),
      ("dp + tp TPU sharding + subshape", "dp + tp TPU sharding", True),
  )
  def test_load_offload(self, name, sharding_type, has_subshape):
    with self.mesh:
      config = self.get_config(sharding_type, has_subshape=has_subshape)
      tpu_pool, cpu_pool = config.init()
      manager = tiered_page_pool.TieredPagePoolManager(
          config, tpu_pool, cpu_pool
      )

      new_partition_pages = {}
      for k, v in tpu_pool.partition_pages.items():
        new_partition_pages[k] = jnp.ones_like(v)
      manager.update_tpu_pool(new_partition_pages)

      num_pages = 2
      tpu_pids = manager.allocate_tpu_pages(num_pages)

      prev_cpu_free = manager.num_free_cpu_pages
      prev_tpu_free = manager.num_free_tpu_pages

      manager.offload(tpu_pids)

      for pid in tpu_pids:
        self.assertEqual(manager.get_page_location(pid), "cpu")

      self.assertEqual(manager.num_free_cpu_pages, prev_cpu_free - num_pages)
      self.assertEqual(manager.num_free_tpu_pages, prev_tpu_free + num_pages)

      assert manager.cpu_pool is not None
      for layer, cpu_pages in manager.cpu_pool.partition_pages.items():
        if num_pages > 0:
          cpu_idxs = [manager.get_page_idx(pid) for pid in tpu_pids]
          try:
            # We must use jax functions or just slice directly.
            mask_arr = cpu_pages[jnp.array(cpu_idxs)]
            np.testing.assert_allclose(mask_arr, np.ones_like(mask_arr))
          except Exception as e:
            self.fail(f"Failed array verification: {e}")

      manager.load(tpu_pids)
      for pid in tpu_pids:
        self.assertEqual(manager.get_page_location(pid), "tpu")

      self.assertEqual(manager.num_free_cpu_pages, prev_cpu_free)
      self.assertEqual(manager.num_free_tpu_pages, prev_tpu_free)

      for layer, hbm_pages in manager.tpu_pool.partition_pages.items():
        if num_pages > 0:
          tpu_idxs = [manager.get_page_idx(pid) for pid in tpu_pids]
          mask_arr = hbm_pages[jnp.array(tpu_idxs)]
          np.testing.assert_allclose(mask_arr, np.ones_like(mask_arr))

  def test_evict(self):
    config = self.get_config("no TPU sharding", has_subshape=False)
    tpu_pool, _ = config.init()
    manager = tiered_page_pool.TieredPagePoolManager(config, tpu_pool, None)

    num_pages = 3
    tpu_pids = manager.allocate_tpu_pages(num_pages)
    tpu_phys_idxs = [manager.get_page_idx(pid) for pid in tpu_pids]

    for idx in tpu_phys_idxs:
      self.assertNotIn(idx, manager.tpu_pool.available_page_indices)

    manager.evict(tpu_pids)

    avail_set = set(manager.tpu_pool.available_page_indices)
    for idx in tpu_phys_idxs:
      self.assertIn(idx, avail_set)

    for pid in tpu_pids:
      self.assertIsNone(manager.get_page_location(pid))
      self.assertIsNone(manager.get_page_idx(pid))

  def test_evict_cpu(self):
    config = self.get_config("no TPU sharding", has_subshape=False)
    tpu_pool, cpu_pool = config.init()
    manager = tiered_page_pool.TieredPagePoolManager(config, tpu_pool, cpu_pool)

    num_pages = 3
    pids = manager.allocate_tpu_pages(num_pages)
    manager.offload(pids)

    cpu_phys_idxs = [manager.get_page_idx(pid) for pid in pids]

    assert manager.cpu_pool is not None
    for idx in cpu_phys_idxs:
      self.assertNotIn(idx, manager.cpu_pool.available_page_indices)

    manager.evict(pids)

    avail_set = set(manager.cpu_pool.available_page_indices)
    for idx in cpu_phys_idxs:
      self.assertIn(idx, avail_set)

    for pid in pids:
      self.assertIsNone(manager.get_page_location(pid))
      self.assertIsNone(manager.get_page_idx(pid))

  @parameterized.parameters(
      ("No subshape, no sharding", "no TPU sharding", False),
      ("no TPU sharding + no subshape", "no TPU sharding", False),
      ("no TPU sharding + subshape", "no TPU sharding", True),
      ("dp TPU sharding + subshape", "dp TPU sharding", True),
      ("tp TPU sharding + subshape", "tp TPU sharding", True),
      ("dp + tp TPU sharding + subshape", "dp + tp TPU sharding", True),
  )
  def test_put_on_target_device(self, name, sharding_type, has_subshape):
    with self.mesh:
      config = self.get_config(sharding_type, has_subshape=has_subshape)
      tpu_pool, cpu_pool = config.init()

      cpu_tensor = jnp.zeros((2, config.page_size) + config.page_subshape)
      cpu_tensor = jax.device_put(cpu_tensor, jax.devices("cpu")[0])

      tpu_target = tpu_pool.partition_pages["layer_0"]
      out_tensor = utils._put_on_target_device(cpu_tensor, tpu_target)

      self.assertEqual(out_tensor.sharding, tpu_target.sharding)

      tpu_tensor = out_tensor
      cpu_target = cpu_pool.partition_pages["layer_0"]
      out_tensor_2 = utils._put_on_target_device(tpu_tensor, cpu_target)

      self.assertEqual(out_tensor_2.sharding, cpu_target.sharding)

  @parameterized.parameters(
      ("No subshape, no sharding", "no TPU sharding", False),
      ("no TPU sharding + no subshape", "no TPU sharding", False),
      ("no TPU sharding + subshape", "no TPU sharding", True),
      ("dp TPU sharding + subshape", "dp TPU sharding", True),
      ("tp TPU sharding + subshape", "tp TPU sharding", True),
      ("dp + tp TPU sharding + subshape", "dp + tp TPU sharding", True),
  )
  def test_copy_physical_pages(self, name, sharding_type, has_subshape):
    with self.mesh:
      config = self.get_config(sharding_type, has_subshape=has_subshape)
      tpu_pool, cpu_pool = config.init()

      cpu_pages = cpu_pool.partition_pages["layer_0"]
      cpu_pages = cpu_pages.at[:2].set(1.0)

      src_idxs = jnp.array([0, 1], dtype=jnp.int32)
      dst_idxs = jnp.array([2, 3], dtype=jnp.int32)

      tpu_pages = tpu_pool.partition_pages["layer_0"]
      tpu_pages = utils.copy_physical_pages(
          src_pages=cpu_pages,
          dst_pages=tpu_pages,
          src_idxs=src_idxs,
          dst_idxs=dst_idxs,
      )

      np.testing.assert_allclose(tpu_pages[2:4], np.ones_like(tpu_pages[2:4]))
      np.testing.assert_allclose(tpu_pages[4:], np.zeros_like(tpu_pages[4:]))

      self.assertEqual(
          tpu_pages.sharding, tpu_pool.partition_pages["layer_0"].sharding
      )

      src_idxs_2 = jnp.array([2, 3], dtype=jnp.int32)
      dst_idxs_2 = jnp.array([4, 5], dtype=jnp.int32)

      cpu_pages_new = utils.copy_physical_pages(
          src_pages=tpu_pages,
          dst_pages=cpu_pages,
          src_idxs=src_idxs_2,
          dst_idxs=dst_idxs_2,
      )
      np.testing.assert_allclose(
          cpu_pages_new[4:6], np.ones_like(cpu_pages_new[4:6])
      )

      self.assertEqual(cpu_pages_new.sharding, cpu_pages.sharding)


if __name__ == "__main__":
  absltest.main()
