# Copyright 2026 The Tunix Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for KVCacheManager."""

from __future__ import annotations

import os

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tunix.experimental.generate import kv_cache_manager
from tunix.experimental.generate import request as request_lib
from tunix.experimental.generate import single_type_kv_cache_manager

# Sharding tests need up to 4 devices.
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"


def _create_mesh() -> jax.sharding.Mesh:
  return jax.sharding.Mesh(
      np.array(jax.devices()[:4]).reshape((2, 2)), ("dp", "tp")
  )


def _create_cache_config(
    page_size: int = 4,
    num_device_pages: int = 10,
    num_host_pages: int = 10,
    dp_axis: str | None = None,
    tp_axis: str | None = None,
    dp_size: int = 1,
    num_caches: int = 1,
    num_kv_heads: int = 2,
    head_dim: int = 16,
    mesh: jax.sharding.Mesh | None = None,
) -> kv_cache_manager.CacheConfig:
  bytes_per_cache_page = page_size * (2 * num_kv_heads) * head_dim * 4
  total_bytes_per_page = bytes_per_cache_page * num_caches
  if mesh is None and (dp_axis or tp_axis):
    mesh = _create_mesh()
  return kv_cache_manager.CacheConfig(
      max_device_bytes=total_bytes_per_page * num_device_pages,
      max_host_bytes=total_bytes_per_page * num_host_pages,
      page_size=page_size,
      dtype=jnp.float32,
      dp_axis=dp_axis,
      tp_axis=tp_axis,
      dp_size=dp_size,
      mesh=mesh,
  )


def _create_cache_geometries(
    num_caches: int = 1,
    num_kv_heads: int = 2,
    head_dim: int = 16,
    cache_to_window_size: dict[str, int | None] | None = None,
) -> dict[str, kv_cache_manager.CacheGeometry]:
  if cache_to_window_size is None:
    cache_to_window_size = {f"cache_{i}": None for i in range(num_caches)}

  return {
      cache_name: kv_cache_manager.CacheGeometry(
          num_kv_heads=num_kv_heads,
          head_dim=head_dim,
          window_size=window_size,
      )
      for cache_name, window_size in cache_to_window_size.items()
  }


def _create_manager(
    num_caches: int = 1,
    cache_to_window_size: dict[str, int | None] | None = None,
    **config_kwargs,
) -> kv_cache_manager.KVCacheManager:
  cache_geometries = _create_cache_geometries(
      num_caches=num_caches, cache_to_window_size=cache_to_window_size
  )
  return kv_cache_manager.KVCacheManager(
      config=_create_cache_config(
          num_caches=len(cache_geometries), **config_kwargs
      ),
      cache_geometries=cache_geometries,
  )


def _create_two_group_manager(
    page_size: int = 4,
    num_device_pages: int = 10,
    window_size: int = 8,
) -> tuple[
    kv_cache_manager.KVCacheManager,
    single_type_kv_cache_manager.SingleTypeKVCacheManager,
    single_type_kv_cache_manager.SingleTypeKVCacheManager,
]:
  """Creates a manager with a full attention and a local attention group.

  Args:
    page_size: The number of tokens per page.
    num_device_pages: The number of device pages in each group.
    window_size: The sliding window size of the local attention group.

  Returns:
    The manager, its full attention group (cache_0, cache_1) and its local
    attention group (cache_2, cache_3).
  """
  mgr = _create_manager(
      cache_to_window_size={
          "cache_0": None,
          "cache_1": None,
          "cache_2": window_size,
          "cache_3": window_size,
      },
      page_size=page_size,
      num_device_pages=num_device_pages,
  )
  full, local = mgr._kv_cache_group_managers
  return mgr, full, local


class CacheConfigTest(parameterized.TestCase):

  def test_valid_config(self):
    cfg = kv_cache_manager.CacheConfig(
        max_device_bytes=1024 * 1024,
        max_host_bytes=512 * 1024,
        page_size=16,
        dtype=jnp.float32,
        dp_axis="dp",
        tp_axis="tp",
        dp_size=2,
        mesh=_create_mesh(),
    )
    self.assertEqual(cfg.page_size, 16)
    self.assertEqual(cfg.dtype, jnp.float32)
    self.assertEqual(cfg.dp_axis, "dp")
    self.assertEqual(cfg.tp_axis, "tp")
    self.assertEqual(cfg.dp_size, 2)

  def test_dtype_required(self):
    with self.assertRaises(TypeError):
      kv_cache_manager.CacheConfig(  # pytype: disable=missing-parameter
          max_device_bytes=1024 * 1024,
      )

  def test_sharding_axes_without_mesh_raises(self):
    with self.assertRaisesRegex(ValueError, r"mesh is required"):
      kv_cache_manager.CacheConfig(
          max_device_bytes=1024 * 1024,
          page_size=16,
          dtype=jnp.float32,
          dp_axis="dp",
      )

  def test_invalid_page_size_raises(self):
    with self.assertRaisesRegex(ValueError, r"page_size must be positive"):
      kv_cache_manager.CacheConfig(
          page_size=0,
          max_device_bytes=1024,
          dtype=jnp.float32,
      )

  def test_invalid_dp_size_raises(self):
    with self.assertRaisesRegex(ValueError, r"dp_size must be positive"):
      kv_cache_manager.CacheConfig(
          page_size=16,
          max_device_bytes=1024,
          dtype=jnp.float32,
          dp_size=0,
      )

  def test_invalid_negative_bytes_raises(self):
    with self.assertRaisesRegex(
        ValueError, r"max_device_bytes must be positive"
    ):
      kv_cache_manager.CacheConfig(
          page_size=16,
          max_device_bytes=-100,
          dtype=jnp.float32,
      )
    with self.assertRaisesRegex(
        ValueError, r"max_host_bytes cannot be negative"
    ):
      kv_cache_manager.CacheConfig(
          page_size=16,
          max_device_bytes=1024,
          max_host_bytes=-100,
          dtype=jnp.float32,
      )

  def test_zero_device_bytes_raises(self):
    with self.assertRaisesRegex(
        ValueError, r"max_device_bytes must be positive"
    ):
      kv_cache_manager.CacheConfig(
          page_size=16,
          max_device_bytes=0,
          dtype=jnp.float32,
      )


class CacheSizingTest(parameterized.TestCase):

  def test_derive_kv_geometry(self):
    cfg = kv_cache_manager.CacheConfig(
        max_device_bytes=1024 * 1024,
        dtype=jnp.bfloat16,
        page_size=16,
    )
    element_shape = kv_cache_manager._derive_kv_geometry(
        cfg, num_kv_heads=8, head_dim=64
    )
    self.assertEqual(element_shape, (8, 2, 64))

  @parameterized.named_parameters(
      dict(testcase_name="dp_and_tp", dp_axis="dp", tp_axis="tp", dp_size=2),
      dict(testcase_name="dp_only", dp_axis="dp", tp_axis=None, dp_size=2),
      dict(testcase_name="tp_only", dp_axis=None, tp_axis="tp", dp_size=1),
  )
  def test_derive_cache_sharding(self, dp_axis, tp_axis, dp_size):
    mesh = _create_mesh()
    cfg = kv_cache_manager.CacheConfig(
        max_device_bytes=1024 * 1024,
        dtype=jnp.bfloat16,
        page_size=16,
        dp_axis=dp_axis,
        tp_axis=tp_axis,
        dp_size=dp_size,
        mesh=mesh,
    )
    self.assertEqual(
        kv_cache_manager._derive_cache_sharding(cfg),
        jax.sharding.NamedSharding(
            mesh,
            jax.sharding.PartitionSpec(dp_axis, None, tp_axis, None, None),
        ),
    )

  def test_derive_cache_sharding_no_sharding(self):
    cfg = kv_cache_manager.CacheConfig(
        max_device_bytes=1024 * 1024,
        dtype=jnp.bfloat16,
        page_size=16,
    )
    self.assertIsNone(kv_cache_manager._derive_cache_sharding(cfg))

  def test_compute_page_limits(self):
    cfg = kv_cache_manager.CacheConfig(
        max_device_bytes=1024 * 1024,
        max_host_bytes=512 * 1024,
        dtype=jnp.bfloat16,
        page_size=16,
    )
    element_shape = kv_cache_manager._derive_kv_geometry(
        cfg, num_kv_heads=8, head_dim=64
    )
    num_device_pages, num_host_pages = kv_cache_manager._compute_page_limits(
        config=cfg,
        element_shapes=(element_shape,) * 4,
        sharding=None,
    )
    self.assertEqual(num_device_pages, 8)
    self.assertEqual(num_host_pages, 4)

  # One page is 16 tokens * (16 KV heads * 64 head dim) * 2 bytes = 32 KiB.
  # The budget fits 8 unsharded pages on a device, and 16 pages sharded over
  # TP. DP places one page of each group of dp_size on every device.
  @parameterized.named_parameters(
      dict(
          testcase_name="no_sharding",
          dp_axis=None,
          tp_axis=None,
          dp_size=1,
          expected_device_pages=8,
          expected_device_pages_one_byte_short=7,
      ),
      dict(
          testcase_name="dp_only",
          dp_axis="dp",
          tp_axis=None,
          dp_size=2,
          expected_device_pages=16,
          expected_device_pages_one_byte_short=14,
      ),
      dict(
          testcase_name="tp_only",
          dp_axis=None,
          tp_axis="tp",
          dp_size=1,
          expected_device_pages=16,
          expected_device_pages_one_byte_short=15,
      ),
      dict(
          testcase_name="dp_and_tp",
          dp_axis="dp",
          tp_axis="tp",
          dp_size=2,
          expected_device_pages=32,
          expected_device_pages_one_byte_short=30,
      ),
  )
  def test_compute_page_limits_budget_boundary(
      self,
      dp_axis,
      tp_axis,
      dp_size,
      expected_device_pages,
      expected_device_pages_one_byte_short,
  ):
    page_bytes = 32 * 1024
    for budget_offset, expected_device, expected_host in (
        (0, expected_device_pages, 3),
        (-1, expected_device_pages_one_byte_short, 2),
    ):
      with self.subTest(budget_offset=budget_offset):
        cfg = kv_cache_manager.CacheConfig(
            max_device_bytes=8 * page_bytes + budget_offset,
            max_host_bytes=3 * page_bytes + budget_offset,
            dtype=jnp.bfloat16,
            page_size=16,
            dp_axis=dp_axis,
            tp_axis=tp_axis,
            dp_size=dp_size,
            mesh=_create_mesh(),
        )
        element_shape = kv_cache_manager._derive_kv_geometry(
            cfg, num_kv_heads=8, head_dim=64
        )
        num_device_pages, num_host_pages = (
            kv_cache_manager._compute_page_limits(
                config=cfg,
                element_shapes=(element_shape,),
                sharding=kv_cache_manager._derive_cache_sharding(cfg),
            )
        )
        self.assertEqual(num_device_pages, expected_device)
        self.assertEqual(num_host_pages, expected_host)

  def test_compute_page_limits_heterogeneous_geometry(self):
    # Weighting each group's byte budget by its own per-page cost makes the
    # weight cancel, so both groups end up with the same page count.
    cfg = kv_cache_manager.CacheConfig(
        max_device_bytes=3 * (65536 + 16384),
        dtype=jnp.bfloat16,
        page_size=16,
    )
    geometries = (
        kv_cache_manager.CacheGeometry(num_kv_heads=8, head_dim=64),
        kv_cache_manager.CacheGeometry(num_kv_heads=8, head_dim=64),
        kv_cache_manager.CacheGeometry(
            num_kv_heads=4, head_dim=32, window_size=128
        ),
        kv_cache_manager.CacheGeometry(
            num_kv_heads=4, head_dim=32, window_size=128
        ),
    )
    element_shapes = tuple(
        kv_cache_manager._derive_kv_geometry(cfg, g.num_kv_heads, g.head_dim)
        for g in geometries
    )
    self.assertEqual(
        element_shapes, ((8, 2, 64), (8, 2, 64), (4, 2, 32), (4, 2, 32))
    )

    num_device_pages, _ = kv_cache_manager._compute_page_limits(
        config=cfg,
        element_shapes=element_shapes,
        sharding=None,
    )
    self.assertEqual(num_device_pages, 3)

  def test_compute_page_limits_is_multiple_of_dp_size(self):
    # Axis 0 of the pool is partitioned over dp, so a count that isn't a
    # multiple of dp_size would fail allocation with IndivisibleError.
    cfg = kv_cache_manager.CacheConfig(
        max_device_bytes=37 * 1024,
        dtype=jnp.bfloat16,
        page_size=16,
        dp_axis="dp",
        dp_size=4,
        mesh=jax.sharding.Mesh(np.array(jax.devices()[:4]), ("dp",)),
    )
    element_shape = kv_cache_manager._derive_kv_geometry(
        cfg, num_kv_heads=2, head_dim=8
    )
    num_device_pages, _ = kv_cache_manager._compute_page_limits(
        config=cfg,
        element_shapes=(element_shape,),
        sharding=kv_cache_manager._derive_cache_sharding(cfg),
    )
    # One page is 1 KiB, so each of the 4 devices holds 37 pages.
    self.assertEqual(num_device_pages, 4 * 37)


class KVCacheManagerInitTest(parameterized.TestCase):

  def test_init_pools_caches_of_equal_geometry(self):
    # Two caches, same geometry: one pool, so they share an allocator and a
    # single set of page indices.
    mgr = _create_manager(num_caches=2)

    self.assertEqual(mgr.page_size, 4)
    self.assertEqual(mgr.cache_names, ("cache_0", "cache_1"))
    self.assertLen(mgr._kv_cache_group_managers, 1)
    self.assertEqual(
        mgr._kv_cache_group_managers[0].cache_names, ("cache_0", "cache_1")
    )

  def test_init_multiple_windows(self):
    mgr = _create_manager(
        cache_to_window_size={
            "cache_0": None,
            "cache_1": 8,
            "cache_2": None,
            "cache_3": 8,
        }
    )

    full, local = mgr._kv_cache_group_managers
    self.assertIsNone(full.window_size)
    self.assertEqual(full.cache_names, ("cache_0", "cache_2"))
    self.assertEqual(local.window_size, 8)
    self.assertEqual(local.cache_names, ("cache_1", "cache_3"))

  def test_init_per_group_element_shape(self):
    # Gemma 4's global layers project fewer, wider KV heads than its local
    # ones, so each group's pool must carry its own element shape.
    cfg = kv_cache_manager.CacheConfig(
        max_device_bytes=4 * (4 * 4 * 8 * 4 + 4 * 1 * 16 * 4),
        page_size=4,
        dtype=jnp.float32,
    )
    mgr = kv_cache_manager.KVCacheManager(
        config=cfg,
        cache_geometries={
            "cache_0": kv_cache_manager.CacheGeometry(
                num_kv_heads=2, head_dim=8, window_size=8
            ),
            "cache_1": kv_cache_manager.CacheGeometry(
                num_kv_heads=1, head_dim=16
            ),
        },
    )

    pages = mgr.get_physical_pages()
    # (num_pages, page_size, *element_shape); element_shape is
    # (2 * num_kv_heads // kv_packing, kv_packing, head_dim), and float32
    # packs 1 element per word.
    self.assertEqual(pages["cache_0"].shape[2:], (4, 1, 8))
    self.assertEqual(pages["cache_1"].shape[2:], (2, 1, 16))

  def test_init_no_caches_raises(self):
    with self.assertRaisesRegex(ValueError, r"(?i)at least one cache"):
      kv_cache_manager.KVCacheManager(
          config=_create_cache_config(), cache_geometries={}
      )

  def test_init_sharding(self):
    # Pages are allocated outside any mesh context, so the mesh travels on the
    # config rather than through `jax.set_mesh`.
    mesh = _create_mesh()
    mgr = _create_manager(
        num_caches=1, dp_axis="dp", tp_axis="tp", dp_size=2, mesh=mesh
    )

    arr = mgr.get_physical_pages()["cache_0"]
    assert isinstance(arr, jax.Array)
    self.assertEqual(
        arr.sharding,
        jax.sharding.NamedSharding(
            mesh, jax.sharding.PartitionSpec("dp", None, "tp", None, None)
        ),
    )

  def test_init_missing_geometry_raises(self):
    # `CacheGeometry` is typed, so an incomplete entry fails at construction
    # rather than deep inside pool setup.
    with self.assertRaises(TypeError):
      kv_cache_manager.CacheGeometry(num_kv_heads=2)  # pytype: disable=missing-parameter

  def test_init_insufficient_device_bytes_raises(self):
    cfg = kv_cache_manager.CacheConfig(
        page_size=4,
        dtype=jnp.float32,
        max_device_bytes=10,
    )
    with self.assertRaisesRegex(ValueError, r"Cannot allocate 0 device pages"):
      kv_cache_manager.KVCacheManager(
          config=cfg,
          cache_geometries=_create_cache_geometries(num_caches=1),
      )


class AllocateSlotsTest(parameterized.TestCase):

  def test_allocate_slots_success(self):
    mgr = _create_manager()
    req = request_lib.Request(req_id="r1", prompt_token_ids=list(range(8)))
    self.assertTrue(mgr.allocate_slots(req, num_new_tokens=8))
    self.assertLen(mgr.get_page_idxs("r1")["cache_0"], 2)

  def test_allocate_slots_insufficient_space_returns_false(self):
    mgr = _create_manager(num_device_pages=1)
    req = request_lib.Request(req_id="r1", prompt_token_ids=list(range(8)))
    # 8 tokens requires 2 pages, but only 1 page capacity exists
    self.assertFalse(mgr.allocate_slots(req, num_new_tokens=8))

  def test_allocate_slots_zero_tokens_returns_true(self):
    mgr = _create_manager()
    req = request_lib.Request(req_id="r1", prompt_token_ids=list(range(8)))
    self.assertTrue(mgr.allocate_slots(req, num_new_tokens=0))

  def test_allocate_slots_wrong_computed_pages_raises(self):
    mgr, _, _ = _create_two_group_manager()
    req = request_lib.Request(req_id="r1", prompt_token_ids=list(range(8)))
    with self.assertRaisesRegex(ValueError, r"returned by get_computed_pages"):
      mgr.allocate_slots(req, num_new_tokens=8, new_computed_pages=((),))

  def test_allocate_slots_insufficient_space_in_one_group_allocates_none(self):
    mgr, full, local = _create_two_group_manager(
        page_size=4, num_device_pages=10
    )
    # Leave 1 free page in the local group only.
    for i in range(9):
      local.allocate_slots("other", num_tokens=4, num_completed_tokens=4 * i)

    req = request_lib.Request(req_id="r1", prompt_token_ids=list(range(8)))
    self.assertFalse(mgr.allocate_slots(req, num_new_tokens=8))
    self.assertNotIn("r1", full._request_to_pages)
    self.assertNotIn("r1", local._request_to_pages)
    self.assertEqual(full._page_manager.num_free_device_pages, 10)


class ReleaseRequestTest(parameterized.TestCase):

  def test_release_request_releases_every_group(self):
    mgr, full, local = _create_two_group_manager()
    req = request_lib.Request(req_id="r1", prompt_token_ids=list(range(8)))
    mgr.allocate_slots(req, num_new_tokens=8)
    self.assertIn("r1", full._request_to_pages)
    self.assertIn("r1", local._request_to_pages)

    mgr.release_request(req)
    self.assertNotIn("r1", full._request_to_pages)
    self.assertNotIn("r1", local._request_to_pages)
    self.assertEqual(
        mgr.get_page_idxs("r1"), {f"cache_{i}": () for i in range(4)}
    )

  def test_release_unknown_request_is_noop(self):
    mgr = _create_manager()
    req = request_lib.Request(req_id="unknown", prompt_token_ids=[0])
    mgr.release_request(req)
    mgr.release_request(req)

  def test_reset_kv_caches_frees_all_pages(self):
    mgr, full, local = _create_two_group_manager(page_size=4)
    req = request_lib.Request(req_id="r1", prompt_token_ids=list(range(8)))
    mgr.allocate_slots(req, num_new_tokens=8)

    mgr.reset_kv_caches()

    for manager in (full, local):
      self.assertEmpty(manager._request_to_pages)
      self.assertEqual(manager._page_manager.num_free_device_pages, 10)


class InspectionTest(parameterized.TestCase):

  def test_get_page_idxs_maps_every_cache_to_its_group(self):
    mgr, full, local = _create_two_group_manager()
    req = request_lib.Request(req_id="r1", prompt_token_ids=list(range(8)))
    mgr.allocate_slots(req, num_new_tokens=8)

    page_idxs = mgr.get_page_idxs("r1")

    full_page_idxs = tuple(full.get_page_idxs("r1"))
    local_page_idxs = tuple(local.get_page_idxs("r1"))
    self.assertLen(full_page_idxs, 2)
    self.assertLen(local_page_idxs, 2)
    self.assertEqual(
        page_idxs,
        {
            "cache_0": full_page_idxs,
            "cache_1": full_page_idxs,
            "cache_2": local_page_idxs,
            "cache_3": local_page_idxs,
        },
    )

  def test_get_physical_pages(self):
    mgr, _, _ = _create_two_group_manager()
    phys = mgr.get_physical_pages()
    self.assertEqual(
        set(phys.keys()), {"cache_0", "cache_1", "cache_2", "cache_3"}
    )
    for cache_name in ("cache_0", "cache_1", "cache_2", "cache_3"):
      self.assertIsInstance(phys[cache_name], (jax.Array, np.ndarray))

  def test_update_device_pool(self):
    mgr, _, _ = _create_two_group_manager()
    old_pages = mgr.get_physical_pages()
    new_pages = {k: np.ones_like(v) * 99.0 for k, v in old_pages.items()}
    mgr.update_device_pool(new_pages)
    updated_pages = mgr.get_physical_pages()
    for k in ("cache_0", "cache_1", "cache_2", "cache_3"):
      np.testing.assert_array_equal(updated_pages[k], new_pages[k])


if __name__ == "__main__":
  absltest.main()
