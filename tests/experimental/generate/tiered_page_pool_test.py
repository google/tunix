import os
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np
from tunix.experimental.generate import tiered_page_pool

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"


class PagePoolTest(parameterized.TestCase):

  def test_init_state(self):
    total_pages = 10
    pages_dict: dict[str, jax.Array | np.ndarray] = {
        "layer1": jnp.zeros((total_pages, 8))
    }
    pool = tiered_page_pool._PagePool(partition_pages=pages_dict)
    self.assertEqual(pool.num_free_pages, total_pages)
    self.assertEqual(pool._available_page_indices, list(range(total_pages)))
    self.assertEqual(pool._in_use, set())

  @parameterized.parameters((0,), (1,), (5,))
  def test_allocate(self, num_pages: int):
    total_pages = 10
    pages_dict: dict[str, jax.Array | np.ndarray] = {
        "layer1": jnp.zeros((total_pages, 8))
    }
    pool = tiered_page_pool._PagePool(partition_pages=pages_dict)
    prev_unallocated = set(range(total_pages))
    prev_len = total_pages

    allocated = pool.allocate(num_pages)

    # Check set(available page indices) does not have allocated pages
    avail_set = set(pool._available_page_indices)
    for idx in allocated:
      self.assertNotIn(idx, avail_set)

    # Check available page indices contains all unallocated pages
    expected_unallocated = prev_unallocated - set(allocated)
    self.assertEqual(avail_set, expected_unallocated)

    # Check that returned indices were previously unallocated
    for idx in allocated:
      self.assertIn(idx, prev_unallocated)

    # Check len
    self.assertLen(pool._available_page_indices, prev_len - num_pages)
    self.assertEqual(pool.num_free_pages, prev_len - num_pages)

  @parameterized.parameters((0,), (1,), (5,))
  def test_free(self, num_pages: int):
    total_pages = 10
    pages_dict: dict[str, jax.Array | np.ndarray] = {
        "layer1": jnp.zeros((total_pages, 8))
    }
    pool = tiered_page_pool._PagePool(partition_pages=pages_dict)
    allocated = pool.allocate(num_pages)
    prev_avail = list(pool._available_page_indices)
    prev_len = len(prev_avail)

    pool.free(allocated)

    avail_set = set(pool._available_page_indices)

    # Check available page indices contains all previous pages
    for idx in prev_avail:
      self.assertIn(idx, avail_set)

    # Check available page indices contains new freed pages
    for idx in allocated:
      self.assertIn(idx, avail_set)

    # Check len
    self.assertLen(pool._available_page_indices, prev_len + num_pages)
    self.assertEqual(pool.num_free_pages, prev_len + num_pages)

  def test_validations(self):
    with self.assertRaisesRegex(
        ValueError, r"Partition pages cannot be empty\."
    ):
      tiered_page_pool._PagePool(partition_pages={})

    pages_dict: dict[str, jax.Array | np.ndarray] = {
        "layer1": jnp.zeros((5, 8))
    }
    mismatched: dict[str, jax.Array | np.ndarray] = {
        "layer1": jnp.zeros((5, 8)),
        "layer2": jnp.zeros((6, 8)),
    }
    with self.assertRaisesRegex(
        ValueError,
        r"Partition 'layer2' does not match pool spec\. Expected shape=\(5, 8\),"
        r" dtype=float32; got shape=\(6, 8\), dtype=float32\.",
    ):
      tiered_page_pool._PagePool(partition_pages=mismatched)

    pool = tiered_page_pool._PagePool(partition_pages=pages_dict)

    with self.assertRaisesRegex(
        ValueError, r"Cannot allocate a negative number of pages: -1\."
    ):
      pool.allocate(-1)

    with self.assertRaisesRegex(
        ValueError, r"Cannot allocate 10 pages, only 5 available\."
    ):
      pool.allocate(10)

    allocated = pool.allocate(2)

    with self.assertRaisesRegex(
        ValueError, r"Cannot free duplicate page indices\."
    ):
      pool.free([allocated[0], allocated[0]])

    with self.assertRaisesRegex(
        ValueError, r"Cannot free pages \{0\}\. These pages are not in use\."
    ):
      pool.free([0])


class TieredPagePoolConfigTest(parameterized.TestCase):

  def test_config_validations(self):
    with self.assertRaisesRegex(
        ValueError,
        r"All dimensions of page_shape must be positive, got 0 in \(10, 0\)\.",
    ):
      tiered_page_pool.TieredPagePoolConfig(
          page_size=0,
          dtype=jnp.float32,
          partition_keys=("layer_0",),
          num_tpu_pages=10,
      )
    with self.assertRaisesRegex(
        ValueError, r"partition_keys cannot be empty\."
    ):
      tiered_page_pool.TieredPagePoolConfig(
          page_size=16,
          dtype=jnp.float32,
          partition_keys=(),
          num_tpu_pages=10,
      )
    with self.assertRaisesRegex(
        ValueError, r"num_tpu_pages must be positive, got -1\."
    ):
      tiered_page_pool.TieredPagePoolConfig(
          page_size=16,
          dtype=jnp.float32,
          partition_keys=("layer_0",),
          num_tpu_pages=-1,
      )
    with self.assertRaisesRegex(
        ValueError, r"num_tpu_pages must be positive, got 0\."
    ):
      tiered_page_pool.TieredPagePoolConfig(
          page_size=16,
          dtype=jnp.float32,
          partition_keys=("layer_0",),
          num_tpu_pages=0,
      )
    with self.assertRaisesRegex(
        ValueError, r"num_cpu_pages cannot be negative, got -1\."
    ):
      tiered_page_pool.TieredPagePoolConfig(
          page_size=16,
          dtype=jnp.float32,
          partition_keys=("layer_0",),
          num_tpu_pages=10,
          num_cpu_pages=-1,
      )
    with self.assertRaisesRegex(
        ValueError,
        r"All dimensions of page_shape must be positive, got 0 in \(10, 16, 2,"
        r" 0\)\.",
    ):
      tiered_page_pool.TieredPagePoolConfig(
          page_size=16,
          element_shape=(2, 0),
          dtype=jnp.float32,
          partition_keys=("layer_0",),
          num_tpu_pages=10,
      )
    with self.assertRaisesRegex(
        ValueError,
        r"All dimensions of page_shape must be positive, got -1 in \(10, 16,"
        r" -1\)\.",
    ):
      tiered_page_pool.TieredPagePoolConfig(
          page_size=16,
          element_shape=(-1,),
          dtype=jnp.float32,
          partition_keys=("layer_0",),
          num_tpu_pages=10,
      )
    # Valid configs with PartitionSpec succeed.
    config_1d = tiered_page_pool.TieredPagePoolConfig(
        page_size=16,
        dtype=jnp.float32,
        partition_keys=("layer_0",),
        num_tpu_pages=10,
        page_sharding=jax.sharding.PartitionSpec("dp"),
    )
    self.assertIsNotNone(config_1d)

    config_full = tiered_page_pool.TieredPagePoolConfig(
        page_size=16,
        element_shape=(2, 3),
        dtype=jnp.float32,
        partition_keys=("layer_0",),
        num_tpu_pages=10,
        page_sharding=jax.sharding.PartitionSpec("dp", None, None, None),
    )
    self.assertIsNotNone(config_full)

  def test_page_shape(self):
    config = tiered_page_pool.TieredPagePoolConfig(
        page_size=16,
        element_shape=(2, 3),
        dtype=jnp.float32,
        partition_keys=("layer_0",),
        num_tpu_pages=10,
    )
    self.assertEqual(config.page_shape(5), (5, 16, 2, 3))

    config_no_subshape = tiered_page_pool.TieredPagePoolConfig(
        page_size=16,
        dtype=jnp.float32,
        partition_keys=("layer_0",),
        num_tpu_pages=10,
    )
    self.assertEqual(config_no_subshape.page_shape(5), (5, 16))

  def test_cpu_sharding_error(self):
    config = tiered_page_pool.TieredPagePoolConfig(
        page_size=16,
        dtype=jnp.float32,
        partition_keys=("layer_0",),
        num_tpu_pages=10,
        num_cpu_pages=5,
    )
    with self.assertRaisesRegex(ValueError, r"Cannot shard pages on CPU\."):
      config._make_pool(
          num_pages=5,
          sharding=jax.sharding.PartitionSpec("dp"),
          is_cpu=True,
      )

  def test_init(self):
    config = tiered_page_pool.TieredPagePoolConfig(
        page_size=16,
        dtype=jnp.float32,
        partition_keys=("layer_0", "layer_1"),
        num_tpu_pages=10,
        num_cpu_pages=5,
    )
    tpu_pool, cpu_pool = config.init()
    self.assertIsNotNone(cpu_pool)
    self.assertEqual(tpu_pool.num_free_pages, 10)
    self.assertEqual(cpu_pool.num_free_pages, 5)
    self.assertIsInstance(cpu_pool.partition_pages["layer_0"], np.ndarray)
    self.assertIsInstance(tpu_pool.partition_pages["layer_0"], jax.Array)

    config_no_cpu = tiered_page_pool.TieredPagePoolConfig(
        page_size=16,
        dtype=jnp.float32,
        partition_keys=("layer_0",),
        num_tpu_pages=10,
        num_cpu_pages=0,
    )
    _, cpu_pool_2 = config_no_cpu.init()
    self.assertIsNone(cpu_pool_2)


class InternalHelpersTest(parameterized.TestCase):

  def test_scatter_tpu_pages(self):
    tpu_pages = {
        "layer_0": jnp.zeros((4, 8), dtype=jnp.float32),
        "layer_1": jnp.zeros((4, 8), dtype=jnp.float32),
    }
    indices = jnp.array([1, 3], dtype=jnp.int32)
    slices = {
        "layer_0": jnp.ones((2, 8), dtype=jnp.float32),
        "layer_1": jnp.full((2, 8), 2.0, dtype=jnp.float32),
    }
    updated = tiered_page_pool._scatter_tpu_pages(tpu_pages, indices, slices)
    np.testing.assert_allclose(updated["layer_0"][1], np.ones(8))
    np.testing.assert_allclose(updated["layer_0"][3], np.ones(8))
    np.testing.assert_allclose(updated["layer_0"][0], np.zeros(8))
    np.testing.assert_allclose(updated["layer_0"][2], np.zeros(8))
    np.testing.assert_allclose(updated["layer_1"][1], np.full(8, 2.0))
    np.testing.assert_allclose(updated["layer_1"][3], np.full(8, 2.0))

  def test_get_tpu_slices(self):
    layer_0 = jnp.arange(32, dtype=jnp.float32).reshape((4, 8))
    tpu_pages = {"layer_0": layer_0}
    indices = jnp.array([0, 2], dtype=jnp.int32)
    slices = tiered_page_pool._get_tpu_slices(tpu_pages, indices)
    np.testing.assert_allclose(slices["layer_0"], layer_0[indices])


class TieredPagePoolManagerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if len(jax.devices()) < 4:
      self.skipTest("Requires at least 4 devices")
    mesh_shape = (2, 2)
    self.devices = np.array(jax.devices()[:4]).reshape(mesh_shape)
    self.mesh = Mesh(self.devices, axis_names=("dp", "tp"))

  def get_config(self, sharding_type: str, has_subshape: bool = True):
    page_size = 16
    element_shape = (2, 1, 5) if has_subshape else ()
    page_sharding = None

    if has_subshape:
      if sharding_type == "dp TPU sharding":
        page_sharding = jax.sharding.PartitionSpec("dp", None, None, None, None)
      elif sharding_type == "tp TPU sharding":
        page_sharding = jax.sharding.PartitionSpec(None, None, "tp", None, None)
      elif sharding_type == "dp + tp TPU sharding":
        page_sharding = jax.sharding.PartitionSpec("dp", None, "tp", None, None)
    else:
      if sharding_type == "dp TPU sharding":
        page_sharding = jax.sharding.PartitionSpec("dp", None)
      elif sharding_type == "tp TPU sharding":
        page_sharding = jax.sharding.PartitionSpec(None, "tp")
      elif sharding_type == "dp + tp TPU sharding":
        page_sharding = jax.sharding.PartitionSpec("dp", "tp")

    return tiered_page_pool.TieredPagePoolConfig(
        page_size=page_size,
        element_shape=element_shape,
        dtype=jnp.float32,
        partition_keys=("layer_0", "layer_1"),
        num_tpu_pages=10,
        num_cpu_pages=10,
        page_sharding=page_sharding,
    )

  @parameterized.parameters((0,), (1,), (5,))
  def test_allocate_tpu_pages(self, num_pages: int):
    config = self.get_config("no TPU sharding", has_subshape=False)
    tpu_pool, cpu_pool = config.init()
    manager = tiered_page_pool.TieredPagePoolManager(config, tpu_pool, cpu_pool)

    allocated = manager.allocate_tpu_pages(num_pages)

    self.assertLen(set(allocated), num_pages)
    for pid in allocated:
      self.assertEqual(manager.get_page_location(pid), "tpu")
      phys_idx = manager.get_page_idx(pid)
      self.assertNotIn(phys_idx, manager.tpu_pool._available_page_indices)

  def test_allocate_tpu_pages_errors(self):
    config = self.get_config("no TPU sharding", has_subshape=False)
    tpu_pool, _ = config.init()
    manager = tiered_page_pool.TieredPagePoolManager(config, tpu_pool, None)

    with self.assertRaisesRegex(
        ValueError, r"Cannot allocate a negative number of pages\."
    ):
      manager.allocate_tpu_pages(-1)

    with self.assertRaisesRegex(
        ValueError, r"Cannot allocate 100 TPU pages, only 10 available\."
    ):
      manager.allocate_tpu_pages(100)

  def test_num_free_cpu_pages_when_cpu_pool_is_none(self):
    config = self.get_config("no TPU sharding", has_subshape=False)
    tpu_pool, _ = config.init()
    manager = tiered_page_pool.TieredPagePoolManager(
        config, tpu_pool, cpu_pool=None
    )
    self.assertEqual(manager.num_free_cpu_pages, 0)

    config_no_cpu = tiered_page_pool.TieredPagePoolConfig(
        page_size=16,
        dtype=jnp.float32,
        partition_keys=("layer_0",),
        num_tpu_pages=10,
        num_cpu_pages=0,
    )
    tpu_pool_no_cpu, cpu_pool_no_cpu = config_no_cpu.init()
    self.assertIsNone(cpu_pool_no_cpu)
    manager_no_cpu = tiered_page_pool.TieredPagePoolManager(
        config_no_cpu, tpu_pool_no_cpu, cpu_pool_no_cpu
    )
    self.assertEqual(manager_no_cpu.num_free_cpu_pages, 0)

  @parameterized.product(
      [
          dict(sharding_type="no TPU sharding", has_subshape=False),
          dict(sharding_type="no TPU sharding", has_subshape=True),
          dict(sharding_type="dp TPU sharding", has_subshape=True),
          dict(sharding_type="tp TPU sharding", has_subshape=True),
          dict(sharding_type="dp + tp TPU sharding", has_subshape=True),
      ],
      num_pages=[1, 2, 5],
  )
  def test_load_offload(
      self, sharding_type: str, has_subshape: bool, num_pages: int
  ):
    with jax.set_mesh(self.mesh):
      config = self.get_config(sharding_type, has_subshape=has_subshape)
      tpu_pool, cpu_pool = config.init()
      manager = tiered_page_pool.TieredPagePoolManager(
          config, tpu_pool, cpu_pool
      )

      n_layers = len(config.partition_keys)
      page_vals = np.zeros((n_layers, num_pages), dtype=np.float32)
      for l in range(n_layers):
        for p in range(num_pages):
          page_vals[l, p] = (l + 1) * 100.0 + (p + 1)

      tpu_pids = manager.allocate_tpu_pages(num_pages)
      orig_tpu_idxs: list[int] = []
      for pid in tpu_pids:
        idx = manager.get_page_idx(pid)
        self.assertIsNotNone(idx)
        assert idx is not None
        orig_tpu_idxs.append(idx)

      # Populate allocated TPU pages with distinct values from page_vals.
      new_tpu_pages = dict(manager.tpu_pool.partition_pages)
      for l_idx, layer in enumerate(config.partition_keys):
        pages = new_tpu_pages[layer]
        assert isinstance(pages, jax.Array)
        for p_idx, phys_idx in enumerate(orig_tpu_idxs):
          pages = pages.at[phys_idx].set(page_vals[l_idx, p_idx])
        new_tpu_pages[layer] = pages
      manager.update_tpu_pool(new_tpu_pages)

      prev_cpu_free = manager.num_free_cpu_pages
      prev_tpu_free = manager.num_free_tpu_pages

      manager.offload(tpu_pids)

      for pid in tpu_pids:
        self.assertEqual(manager.get_page_location(pid), "cpu")

      self.assertEqual(manager.num_free_cpu_pages, prev_cpu_free - num_pages)
      self.assertEqual(manager.num_free_tpu_pages, prev_tpu_free + num_pages)

      # Verify pages on CPU have their distinct values per page and per layer.
      self.assertIsNotNone(manager.cpu_pool)
      for l_idx, layer in enumerate(config.partition_keys):
        cpu_pages = manager.cpu_pool.partition_pages[layer]
        for p_idx, pid in enumerate(tpu_pids):
          cpu_idx = manager.get_page_idx(pid)
          assert cpu_idx is not None
          np.testing.assert_allclose(
              cpu_pages[cpu_idx], page_vals[l_idx, p_idx]
          )

      # Overwrite the freed TPU slots with sentinel values before loading to
      # guarantee that load() actively transfers data rather than reusing stale
      # TPU buffers.
      dirty_tpu_pages = {}
      for layer, pages in manager.tpu_pool.partition_pages.items():
        assert isinstance(pages, jax.Array)
        for phys_idx in orig_tpu_idxs:
          pages = pages.at[phys_idx].set(-999.0)
        dirty_tpu_pages[layer] = pages
      manager.update_tpu_pool(dirty_tpu_pages)

      for pages in manager.tpu_pool.partition_pages.values():
        for phys_idx in orig_tpu_idxs:
          np.testing.assert_allclose(pages[phys_idx], -999.0)

      manager.load(tpu_pids)
      for pid in tpu_pids:
        self.assertEqual(manager.get_page_location(pid), "tpu")

      self.assertEqual(manager.num_free_cpu_pages, prev_cpu_free)
      self.assertEqual(manager.num_free_tpu_pages, prev_tpu_free)

      # Verify TPU pages have restored their distinct per-page and per-layer values.
      for l_idx, layer in enumerate(config.partition_keys):
        hbm_pages = manager.tpu_pool.partition_pages[layer]
        for p_idx, pid in enumerate(tpu_pids):
          tpu_idx = manager.get_page_idx(pid)
          assert tpu_idx is not None
          np.testing.assert_allclose(
              hbm_pages[tpu_idx], page_vals[l_idx, p_idx]
          )
        if config.page_sharding is not None and hasattr(hbm_pages, "sharding"):
          self.assertEqual(
              hbm_pages.sharding,
              jax.sharding.NamedSharding(self.mesh, config.page_sharding),
          )

  def test_empty_load_offload(self):
    config = self.get_config("no TPU sharding", has_subshape=False)
    tpu_pool, cpu_pool = config.init()
    manager = tiered_page_pool.TieredPagePoolManager(config, tpu_pool, cpu_pool)
    # Empty operations should be no-ops
    manager.load([])
    manager.offload([])

  def test_load_offload_errors(self):
    config = self.get_config("no TPU sharding", has_subshape=False)
    tpu_pool, _ = config.init()
    manager_no_cpu = tiered_page_pool.TieredPagePoolManager(
        config, tpu_pool, None
    )
    pids = manager_no_cpu.allocate_tpu_pages(2)
    with self.assertRaisesRegex(
        ValueError,
        r"Cannot offload pages to CPU, CPU pool is not initialized\.",
    ):
      manager_no_cpu.offload(pids)

    with self.assertRaisesRegex(
        ValueError,
        r"Cannot load pages from CPU to TPU, CPU pool is not initialized\.",
    ):
      manager_no_cpu.load(pids)

    tpu_pool, cpu_pool = config.init()
    manager = tiered_page_pool.TieredPagePoolManager(config, tpu_pool, cpu_pool)
    tpu_pids = manager.allocate_tpu_pages(2)

    with self.assertRaisesRegex(
        ValueError, r"Cannot offload duplicate pages\."
    ):
      manager.offload([tpu_pids[0], tpu_pids[0]])

    with self.assertRaisesRegex(
        ValueError, r"Page ID 999 is not on TPU \(location: None\)\."
    ):
      manager.offload([999])

    config_small_cpu = tiered_page_pool.TieredPagePoolConfig(
        page_size=16,
        dtype=jnp.float32,
        partition_keys=("layer_0",),
        num_tpu_pages=10,
        num_cpu_pages=1,
    )
    tpu_p, cpu_p = config_small_cpu.init()
    mgr_small = tiered_page_pool.TieredPagePoolManager(
        config_small_cpu, tpu_p, cpu_p
    )
    more_pids = mgr_small.allocate_tpu_pages(2)
    with self.assertRaisesRegex(
        ValueError, r"Cannot offload 2 pages, only 1 available\."
    ):
      mgr_small.offload(more_pids)

    # Attempting to load a page that is already on TPU
    with self.assertRaisesRegex(
        ValueError, r"Page ID \d+ is not on CPU \(location: tpu\)\."
    ):
      manager.load(tpu_pids)

    manager.offload(tpu_pids)

    # Attempting to offload a page that is already on CPU
    with self.assertRaisesRegex(
        ValueError, r"Page ID \d+ is not on TPU \(location: cpu\)\."
    ):
      manager.offload(tpu_pids)

    with self.assertRaisesRegex(ValueError, r"Cannot load duplicate pages\."):
      manager.load([tpu_pids[0], tpu_pids[0]])

    with self.assertRaisesRegex(
        ValueError, r"Page ID 999 is not on CPU \(location: None\)\."
    ):
      manager.load([999])

    # Test load when TPU pool is full / has insufficient free pages.
    config_small_tpu = tiered_page_pool.TieredPagePoolConfig(
        page_size=16,
        dtype=jnp.float32,
        partition_keys=("layer_0",),
        num_tpu_pages=2,
        num_cpu_pages=2,
    )
    tpu_p2, cpu_p2 = config_small_tpu.init()
    mgr_small_tpu = tiered_page_pool.TieredPagePoolManager(
        config_small_tpu, tpu_p2, cpu_p2
    )
    pids_tpu = mgr_small_tpu.allocate_tpu_pages(2)
    mgr_small_tpu.offload(pids_tpu)
    # Re-allocate TPU pool to capacity so 0 free TPU pages remain
    _ = mgr_small_tpu.allocate_tpu_pages(2)
    with self.assertRaisesRegex(
        ValueError, r"Cannot load 2 pages, only 0 available\."
    ):
      mgr_small_tpu.load(pids_tpu)

  def test_free(self):
    config = self.get_config("no TPU sharding", has_subshape=False)
    tpu_pool, cpu_pool = config.init()
    manager = tiered_page_pool.TieredPagePoolManager(config, tpu_pool, cpu_pool)

    tpu_pids = manager.allocate_tpu_pages(4)
    # Offload 2 pages to CPU
    manager.offload(tpu_pids[:2])

    self.assertEqual(manager.get_page_location(tpu_pids[0]), "cpu")
    self.assertEqual(manager.get_page_location(tpu_pids[2]), "tpu")

    prev_tpu_free = manager.num_free_tpu_pages
    prev_cpu_free = manager.num_free_cpu_pages

    manager.free(tpu_pids)

    self.assertEqual(manager.num_free_tpu_pages, prev_tpu_free + 2)
    self.assertEqual(manager.num_free_cpu_pages, prev_cpu_free + 2)

    for pid in tpu_pids:
      self.assertIsNone(manager.get_page_location(pid))
      self.assertIsNone(manager.get_page_idx(pid))

    # Freeing an empty list is a safe no-op.
    manager.free([])

    # Test freeing only CPU pages
    tpu_pids_cpu_only = manager.allocate_tpu_pages(2)
    manager.offload(tpu_pids_cpu_only)
    manager.free(tpu_pids_cpu_only)
    self.assertIsNone(manager.get_page_location(tpu_pids_cpu_only[0]))

    # Test freeing only TPU pages
    tpu_pids_tpu_only = manager.allocate_tpu_pages(2)
    manager.free(tpu_pids_tpu_only)
    self.assertIsNone(manager.get_page_location(tpu_pids_tpu_only[0]))

  def test_free_errors(self):
    config = self.get_config("no TPU sharding", has_subshape=False)
    tpu_pool, cpu_pool = config.init()
    manager = tiered_page_pool.TieredPagePoolManager(config, tpu_pool, cpu_pool)
    pids = manager.allocate_tpu_pages(2)

    with self.assertRaisesRegex(
        ValueError, r"Attempting to free page 999 which is not in use\."
    ):
      manager.free([999])

    with self.assertRaisesRegex(ValueError, r"Cannot free duplicate pages\."):
      manager.free([pids[0], pids[0]])


if __name__ == "__main__":
  absltest.main()
