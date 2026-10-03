import os
from unittest import mock
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
    pool = tiered_page_pool._PartitionedPagePool(partition_pages=pages_dict)
    self.assertEqual(pool.num_free_pages, total_pages)
    self.assertEqual(pool._available_page_indices, list(range(total_pages)))
    self.assertEqual(pool._in_use, set())

  @parameterized.parameters((0,), (1,), (5,))
  def test_allocate(self, num_pages: int):
    total_pages = 10
    pages_dict: dict[str, jax.Array | np.ndarray] = {
        "layer1": jnp.zeros((total_pages, 8))
    }
    pool = tiered_page_pool._PartitionedPagePool(partition_pages=pages_dict)
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
    pool = tiered_page_pool._PartitionedPagePool(partition_pages=pages_dict)
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
      tiered_page_pool._PartitionedPagePool(partition_pages={})

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
      tiered_page_pool._PartitionedPagePool(partition_pages=mismatched)

    pool = tiered_page_pool._PartitionedPagePool(partition_pages=pages_dict)

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
          num_device_pages=10,
      )
    with self.assertRaisesRegex(
        ValueError, r"partition_keys cannot be empty\."
    ):
      tiered_page_pool.TieredPagePoolConfig(
          page_size=16,
          dtype=jnp.float32,
          partition_keys=(),
          num_device_pages=10,
      )
    with self.assertRaisesRegex(
        ValueError, r"num_device_pages must be positive, got -1\."
    ):
      tiered_page_pool.TieredPagePoolConfig(
          page_size=16,
          dtype=jnp.float32,
          partition_keys=("layer_0",),
          num_device_pages=-1,
      )
    with self.assertRaisesRegex(
        ValueError, r"num_device_pages must be positive, got 0\."
    ):
      tiered_page_pool.TieredPagePoolConfig(
          page_size=16,
          dtype=jnp.float32,
          partition_keys=("layer_0",),
          num_device_pages=0,
      )
    with self.assertRaisesRegex(
        ValueError, r"num_host_pages cannot be negative, got -1\."
    ):
      tiered_page_pool.TieredPagePoolConfig(
          page_size=16,
          dtype=jnp.float32,
          partition_keys=("layer_0",),
          num_device_pages=10,
          num_host_pages=-1,
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
          num_device_pages=10,
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
          num_device_pages=10,
      )
    # Valid configs with a concrete Sharding succeed.
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ("dp",))
    config_1d = tiered_page_pool.TieredPagePoolConfig(
        page_size=16,
        dtype=jnp.float32,
        partition_keys=("layer_0",),
        num_device_pages=10,
        sharding=jax.sharding.NamedSharding(
            mesh, jax.sharding.PartitionSpec("dp")
        ),
    )
    self.assertIsNotNone(config_1d)

    config_full = tiered_page_pool.TieredPagePoolConfig(
        page_size=16,
        element_shape=(2, 3),
        dtype=jnp.float32,
        partition_keys=("layer_0",),
        num_device_pages=10,
        sharding=jax.sharding.NamedSharding(
            mesh, jax.sharding.PartitionSpec("dp", None, None, None)
        ),
    )
    self.assertIsNotNone(config_full)

  def test_page_shape(self):
    config = tiered_page_pool.TieredPagePoolConfig(
        page_size=16,
        element_shape=(2, 3),
        dtype=jnp.float32,
        partition_keys=("layer_0",),
        num_device_pages=10,
    )
    self.assertEqual(config._page_shape(5), (5, 16, 2, 3))

    config_no_subshape = tiered_page_pool.TieredPagePoolConfig(
        page_size=16,
        dtype=jnp.float32,
        partition_keys=("layer_0",),
        num_device_pages=10,
    )
    self.assertEqual(config_no_subshape._page_shape(5), (5, 16))

  def test_host_sharding_error(self):
    config = tiered_page_pool.TieredPagePoolConfig(
        page_size=16,
        dtype=jnp.float32,
        partition_keys=("layer_0",),
        num_device_pages=10,
        num_host_pages=5,
    )
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ("dp",))
    with self.assertRaisesRegex(ValueError, r"Cannot shard pages on host\."):
      config._make_pool(
          num_pages=5,
          sharding=jax.sharding.NamedSharding(
              mesh, jax.sharding.PartitionSpec("dp")
          ),
          is_host=True,
      )

  def test_create_manager(self):
    config = tiered_page_pool.TieredPagePoolConfig(
        page_size=16,
        dtype=jnp.float32,
        partition_keys=("layer_0", "layer_1"),
        num_device_pages=10,
        num_host_pages=5,
    )
    manager = config.create_manager()
    self.assertIsNotNone(manager._host_pool)
    self.assertEqual(manager.num_free_device_pages, 10)
    self.assertEqual(manager.num_free_host_pages, 5)
    assert manager._host_pool is not None
    self.assertIsInstance(
        manager._host_pool.partition_pages["layer_0"], np.ndarray
    )
    self.assertIsInstance(manager._device_pool.partition_pages["layer_0"], jax.Array)

    config_no_host = tiered_page_pool.TieredPagePoolConfig(
        page_size=16,
        dtype=jnp.float32,
        partition_keys=("layer_0",),
        num_device_pages=10,
        num_host_pages=0,
    )
    manager_no_host = config_no_host.create_manager()
    self.assertIsNone(manager_no_host._host_pool)
    self.assertEqual(manager_no_host.num_free_host_pages, 0)
    self.assertEqual(manager_no_host.num_free_device_pages, 10)


class InternalHelpersTest(parameterized.TestCase):

  def test_scatter_device_pages(self):
    device_pages = {
        "layer_0": jnp.zeros((4, 8), dtype=jnp.float32),
        "layer_1": jnp.zeros((4, 8), dtype=jnp.float32),
    }
    indices = jnp.array([1, 3], dtype=jnp.int32)
    slices = {
        "layer_0": jnp.ones((2, 8), dtype=jnp.float32),
        "layer_1": jnp.full((2, 8), 2.0, dtype=jnp.float32),
    }
    updated = tiered_page_pool._scatter_device_pages(
        device_pages, indices, slices
    )
    np.testing.assert_allclose(updated["layer_0"][1], np.ones(8))
    np.testing.assert_allclose(updated["layer_0"][3], np.ones(8))
    np.testing.assert_allclose(updated["layer_0"][0], np.zeros(8))
    np.testing.assert_allclose(updated["layer_0"][2], np.zeros(8))
    np.testing.assert_allclose(updated["layer_1"][1], np.full(8, 2.0))
    np.testing.assert_allclose(updated["layer_1"][3], np.full(8, 2.0))

  def test_get_device_slices(self):
    layer_0 = jnp.arange(32, dtype=jnp.float32).reshape((4, 8))
    device_pages = {"layer_0": layer_0}
    indices = jnp.array([0, 2], dtype=jnp.int32)
    slices = tiered_page_pool._get_device_slices(device_pages, indices)
    np.testing.assert_allclose(slices["layer_0"], layer_0[indices])

  @parameterized.parameters(
      ([3], 10, [3]),
      ([3, 5], 10, [3, 5]),
      ([3, 5, 7], 10, [3, 5, 7, 3]),
      ([3, 5, 7, 1], 10, [3, 5, 7, 1]),
      ([3, 5, 7, 1, 2], 10, [3, 5, 7, 1, 2, 3, 3, 3]),
      ([0, 1, 2, 3, 4, 5, 6, 7, 8], 10, [0, 1, 2, 3, 4, 5, 6, 7, 8, 0]),
  )
  def test_pad_indices(
      self, indices: list[int], max_length: int, expected: list[int]
  ):
    self.assertEqual(
        tiered_page_pool._pad_indices(indices, max_length), expected
    )


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
    sharding = None

    def named(*spec):
      return jax.sharding.NamedSharding(
          self.mesh, jax.sharding.PartitionSpec(*spec)
      )

    if has_subshape:
      if sharding_type == "dp device sharding":
        sharding = named("dp", None, None, None, None)
      elif sharding_type == "tp device sharding":
        sharding = named(None, None, "tp", None, None)
      elif sharding_type == "dp + tp device sharding":
        sharding = named("dp", None, "tp", None, None)
    else:
      if sharding_type == "dp device sharding":
        sharding = named("dp", None)
      elif sharding_type == "tp device sharding":
        sharding = named(None, "tp")
      elif sharding_type == "dp + tp device sharding":
        sharding = named("dp", "tp")

    return tiered_page_pool.TieredPagePoolConfig(
        page_size=page_size,
        element_shape=element_shape,
        dtype=jnp.float32,
        partition_keys=("layer_0", "layer_1"),
        num_device_pages=10,
        num_host_pages=10,
        sharding=sharding,
    )

  @parameterized.parameters((0,), (1,), (5,))
  def test_allocate_device_pages(self, num_pages: int):
    config = self.get_config("no device sharding", has_subshape=False)
    manager = config.create_manager()

    allocated = manager.allocate_device_pages(num_pages)

    self.assertLen(set(allocated), num_pages)
    for pid in allocated:
      self.assertEqual(manager.page_location(pid), "device")
      phys_idx = manager.page_idx(pid)
      self.assertNotIn(phys_idx, manager._device_pool._available_page_indices)

  def test_allocate_device_pages_errors(self):
    config = self.get_config("no device sharding", has_subshape=False)
    manager = config.create_manager()

    with self.assertRaisesRegex(
        ValueError, r"Cannot allocate a negative number of pages\."
    ):
      manager.allocate_device_pages(-1)

    with self.assertRaisesRegex(
        ValueError, r"Cannot allocate 100 device pages, only 10 available\."
    ):
      manager.allocate_device_pages(100)

  def test_num_free_host_pages_when_host_pool_is_none(self):
    config_no_host = tiered_page_pool.TieredPagePoolConfig(
        page_size=16,
        dtype=jnp.float32,
        partition_keys=("layer_0",),
        num_device_pages=10,
        num_host_pages=0,
    )
    manager_no_host = config_no_host.create_manager()
    self.assertEqual(manager_no_host.num_free_host_pages, 0)

  @parameterized.product(
      [
          dict(sharding_type="no device sharding", has_subshape=False),
          dict(sharding_type="no device sharding", has_subshape=True),
          dict(sharding_type="dp device sharding", has_subshape=True),
          dict(sharding_type="tp device sharding", has_subshape=True),
          dict(sharding_type="dp + tp device sharding", has_subshape=True),
      ],
      num_pages=[1, 2, 5],
  )
  def test_load_offload(
      self, sharding_type: str, has_subshape: bool, num_pages: int
  ):
    with jax.set_mesh(self.mesh):
      config = self.get_config(sharding_type, has_subshape=has_subshape)
      manager = config.create_manager()

      n_layers = len(config.partition_keys)
      page_vals = np.zeros((n_layers, num_pages), dtype=np.float32)
      for l in range(n_layers):
        for p in range(num_pages):
          page_vals[l, p] = (l + 1) * 100.0 + (p + 1)

      device_pids = manager.allocate_device_pages(num_pages)
      orig_device_idxs: list[int] = []
      for pid in device_pids:
        idx = manager.page_idx(pid)
        self.assertIsNotNone(idx)
        assert idx is not None
        orig_device_idxs.append(idx)

      # Populate allocated device pages with distinct values from page_vals.
      new_device_pages = dict(manager._device_pool.partition_pages)
      for l_idx, layer in enumerate(config.partition_keys):
        pages = new_device_pages[layer]
        assert isinstance(pages, jax.Array)
        for p_idx, phys_idx in enumerate(orig_device_idxs):
          pages = pages.at[phys_idx].set(page_vals[l_idx, p_idx])
        new_device_pages[layer] = pages
      manager.update_device_pool(new_device_pages)

      prev_host_free = manager.num_free_host_pages
      prev_device_free = manager.num_free_device_pages

      manager.offload(device_pids)

      for pid in device_pids:
        self.assertEqual(manager.page_location(pid), "host")

      self.assertEqual(manager.num_free_host_pages, prev_host_free - num_pages)
      self.assertEqual(
          manager.num_free_device_pages, prev_device_free + num_pages
      )

      # Verify pages on host have their distinct values per page and per layer.
      self.assertIsNotNone(manager._host_pool)
      assert manager._host_pool is not None
      for l_idx, layer in enumerate(config.partition_keys):
        host_pages = manager._host_pool.partition_pages[layer]
        for p_idx, pid in enumerate(device_pids):
          host_idx = manager.page_idx(pid)
          assert host_idx is not None
          np.testing.assert_allclose(
              host_pages[host_idx], page_vals[l_idx, p_idx]
          )

      # Overwrite the freed device slots with sentinel values before loading to
      # guarantee that load() actively transfers data rather than reusing stale
      # device buffers.
      dirty_device_pages = {}
      for layer, pages in manager._device_pool.partition_pages.items():
        assert isinstance(pages, jax.Array)
        for phys_idx in orig_device_idxs:
          pages = pages.at[phys_idx].set(-999.0)
        dirty_device_pages[layer] = pages
      manager.update_device_pool(dirty_device_pages)

      for pages in manager._device_pool.partition_pages.values():
        for phys_idx in orig_device_idxs:
          np.testing.assert_allclose(pages[phys_idx], -999.0)

      manager.load(device_pids)
      for pid in device_pids:
        self.assertEqual(manager.page_location(pid), "device")

      self.assertEqual(manager.num_free_host_pages, prev_host_free)
      self.assertEqual(manager.num_free_device_pages, prev_device_free)

      # Verify device pages have restored their distinct per-page and per-layer values.
      for l_idx, layer in enumerate(config.partition_keys):
        hbm_pages = manager._device_pool.partition_pages[layer]
        for p_idx, pid in enumerate(device_pids):
          device_idx = manager.page_idx(pid)
          assert device_idx is not None
          np.testing.assert_allclose(
              hbm_pages[device_idx], page_vals[l_idx, p_idx]
          )
        if config.sharding is not None and hasattr(hbm_pages, "sharding"):
          self.assertEqual(hbm_pages.sharding, config.sharding)

  def test_empty_load_offload(self):
    config = self.get_config("no device sharding", has_subshape=False)
    manager = config.create_manager()
    # Empty operations should be no-ops
    manager.load([])
    manager.offload([])

  @parameterized.parameters((3, 4), (9, 10))
  def test_load_offload_pads_transfers(
      self, num_pages: int, expected_padded_length: int
  ):
    config = self.get_config("no device sharding", has_subshape=False)
    manager = config.create_manager()
    pids = manager.allocate_device_pages(num_pages)
    manager.update_device_pool({
        k: jnp.arange(config.num_device_pages, dtype=jnp.float32)[:, None]
        * jnp.ones((1, config.page_size))
        for k in config.partition_keys
    })
    orig_vals = [float(manager.page_idx(pid)) for pid in pids]

    with mock.patch.object(
        tiered_page_pool,
        "_get_device_slices",
        wraps=tiered_page_pool._get_device_slices,
    ) as get_slices:
      manager.offload(pids)
    self.assertEqual(
        get_slices.call_args.args[1].shape, (expected_padded_length,)
    )

    manager.update_device_pool({
        k: jnp.full_like(v, -1.0)
        for k, v in manager.physical_device_pages.items()
    })
    with mock.patch.object(
        tiered_page_pool,
        "_scatter_device_pages",
        wraps=tiered_page_pool._scatter_device_pages,
    ) as scatter:
      manager.load(pids)
    self.assertEqual(scatter.call_args.args[1].shape, (expected_padded_length,))

    for pid, val in zip(pids, orig_vals):
      for pages in manager.physical_device_pages.values():
        np.testing.assert_allclose(pages[manager.page_idx(pid)], val)

  def test_load_offload_errors(self):
    config_no_host = tiered_page_pool.TieredPagePoolConfig(
        page_size=16,
        dtype=jnp.float32,
        partition_keys=("layer_0", "layer_1"),
        num_device_pages=10,
        num_host_pages=0,
    )
    manager_no_host = config_no_host.create_manager()
    pids = manager_no_host.allocate_device_pages(2)
    with self.assertRaisesRegex(
        ValueError,
        r"Cannot offload pages to host, host pool is not initialized\.",
    ):
      manager_no_host.offload(pids)

    with self.assertRaisesRegex(
        ValueError,
        r"Cannot load pages from host to device, host pool is not"
        r" initialized\.",
    ):
      manager_no_host.load(pids)

    config = self.get_config("no device sharding", has_subshape=False)
    manager = config.create_manager()
    device_pids = manager.allocate_device_pages(2)

    with self.assertRaisesRegex(
        ValueError, r"Cannot offload duplicate pages\."
    ):
      manager.offload([device_pids[0], device_pids[0]])

    with self.assertRaisesRegex(
        ValueError, r"Page ID 999 is not on device \(location: None\)\."
    ):
      manager.offload([999])

    config_small_host = tiered_page_pool.TieredPagePoolConfig(
        page_size=16,
        dtype=jnp.float32,
        partition_keys=("layer_0",),
        num_device_pages=10,
        num_host_pages=1,
    )
    mgr_small = config_small_host.create_manager()
    more_pids = mgr_small.allocate_device_pages(2)
    with self.assertRaisesRegex(
        ValueError, r"Cannot offload 2 pages, only 1 available\."
    ):
      mgr_small.offload(more_pids)

    # Attempting to load a page that is already on device
    with self.assertRaisesRegex(
        ValueError, r"Page ID \d+ is not on host \(location: device\)\."
    ):
      manager.load(device_pids)

    manager.offload(device_pids)

    # Attempting to offload a page that is already on host
    with self.assertRaisesRegex(
        ValueError, r"Page ID \d+ is not on device \(location: host\)\."
    ):
      manager.offload(device_pids)

    with self.assertRaisesRegex(ValueError, r"Cannot load duplicate pages\."):
      manager.load([device_pids[0], device_pids[0]])

    with self.assertRaisesRegex(
        ValueError, r"Page ID 999 is not on host \(location: None\)\."
    ):
      manager.load([999])

    # Test load when device pool is full / has insufficient free pages.
    config_small_device = tiered_page_pool.TieredPagePoolConfig(
        page_size=16,
        dtype=jnp.float32,
        partition_keys=("layer_0",),
        num_device_pages=2,
        num_host_pages=2,
    )
    mgr_small_device = config_small_device.create_manager()
    pids_device = mgr_small_device.allocate_device_pages(2)
    mgr_small_device.offload(pids_device)
    # Re-allocate device pool to capacity so 0 free device pages remain
    _ = mgr_small_device.allocate_device_pages(2)
    with self.assertRaisesRegex(
        ValueError, r"Cannot load 2 pages, only 0 available\."
    ):
      mgr_small_device.load(pids_device)

  def test_free(self):
    config = self.get_config("no device sharding", has_subshape=False)
    manager = config.create_manager()

    device_pids = manager.allocate_device_pages(4)
    # Offload 2 pages to host
    manager.offload(device_pids[:2])

    self.assertEqual(manager.page_location(device_pids[0]), "host")
    self.assertEqual(manager.page_location(device_pids[2]), "device")

    prev_device_free = manager.num_free_device_pages
    prev_host_free = manager.num_free_host_pages

    manager.free(device_pids)

    self.assertEqual(manager.num_free_device_pages, prev_device_free + 2)
    self.assertEqual(manager.num_free_host_pages, prev_host_free + 2)

    for pid in device_pids:
      self.assertIsNone(manager.page_location(pid))
      self.assertIsNone(manager.page_idx(pid))

    # Freeing an empty list is a safe no-op.
    manager.free([])

    # Test freeing only host pages
    device_pids_host_only = manager.allocate_device_pages(2)
    manager.offload(device_pids_host_only)
    manager.free(device_pids_host_only)
    self.assertIsNone(manager.page_location(device_pids_host_only[0]))

    # Test freeing only device pages
    device_pids_device_only = manager.allocate_device_pages(2)
    manager.free(device_pids_device_only)
    self.assertIsNone(manager.page_location(device_pids_device_only[0]))

  def test_free_errors(self):
    config = self.get_config("no device sharding", has_subshape=False)
    manager = config.create_manager()
    pids = manager.allocate_device_pages(2)

    with self.assertRaisesRegex(
        ValueError, r"Attempting to free page 999 which is not in use\."
    ):
      manager.free([999])

    with self.assertRaisesRegex(ValueError, r"Cannot free duplicate pages\."):
      manager.free([pids[0], pids[0]])


if __name__ == "__main__":
  absltest.main()
