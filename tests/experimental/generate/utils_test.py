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

"""Unit tests for utils."""

import math
import os
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tunix.common import configs as common_configs
from tunix.common import datatypes
from tunix.experimental.generate import configs
from tunix.experimental.generate import request as request_lib
from tunix.experimental.generate import utils

# Sharding tests need up to 4 devices.
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

_GIB = 1 << 30


class _FakeDevice:

  def __init__(
      self,
      stats: dict[str, int] | None = None,
      *,
      device_kind: str = "CPU",
      raise_on_stats: bool = False,
  ):
    self._stats = stats
    self.device_kind = device_kind
    self._raise_on_stats = raise_on_stats

  def memory_stats(self) -> dict[str, int] | None:
    if self._raise_on_stats:
      raise jax.errors.JaxRuntimeError("memory_stats not implemented")
    return self._stats


def _create_mesh(
    shape: tuple[int, ...] = (2, 2),
    axis_names: tuple[str, ...] = ("dp", "tp"),
) -> jax.sharding.Mesh:
  devices = jax.devices()[: math.prod(shape)]
  return jax.sharding.Mesh(np.array(devices).reshape(shape), axis_names)


class UtilsTest(parameterized.TestCase):

  @parameterized.parameters(
      (0, 4, 0),
      (1, 4, 1),
      (4, 4, 1),
      (5, 4, 2),
      (8, 4, 2),
      (7, 1, 7),
  )
  def test_cdiv(self, a, b, expected):
    self.assertEqual(utils.cdiv(a, b), expected)

  @parameterized.parameters(
      (jnp.float32, 1),
      (jnp.bfloat16, 2),
      (jnp.int8, 4),
      (jnp.int4, 8),
  )
  def test_get_dtype_packing(self, dtype, expected):
    self.assertEqual(utils.get_dtype_packing(dtype), expected)

  @parameterized.parameters(
      (0, 1),
      (1, 1),
      (2, 2),
      (3, 4),
      (4, 4),
      (5, 8),
  )
  def test_next_power_of_2(self, x, expected):
    self.assertEqual(utils.next_power_of_2(x), expected)

  def test_jit_scatter_last_column(self):
    target = jnp.zeros((5,), dtype=jnp.int32)
    source = jnp.array([[10, 11], [20, 21], [30, 31]], dtype=jnp.int32)
    target_idxs = jnp.array([3, 5, 1], dtype=jnp.int32)
    result = utils.jit_scatter_last_column(target, source, target_idxs)
    np.testing.assert_array_equal(result, [0, 31, 0, 11, 0])


class CacheSizingTest(parameterized.TestCase):

  def test_derive_kv_geometry_without_active_mesh_raises(self):
    cfg = configs.CacheConfig(
        max_device_size_gib=1.0,
        dtype=jnp.bfloat16,
        page_size=16,
    )
    with self.assertRaisesRegex(ValueError, r"An active JAX mesh is required"):
      utils.derive_kv_geometry(cfg, num_kv_heads=8, head_dim=64)

  def test_derive_kv_geometry(self):
    cfg = configs.CacheConfig(
        max_device_size_gib=1.0,
        dtype=jnp.bfloat16,
        page_size=16,
    )
    with jax.set_mesh(_create_mesh()):
      element_shape = utils.derive_kv_geometry(cfg, num_kv_heads=8, head_dim=64)
    self.assertEqual(element_shape, (8, 2, 64))

  def test_derive_kv_geometry_pads_kv_heads_to_tp_size(self):
    cfg = configs.CacheConfig(
        max_device_size_gib=1.0,
        dtype=jnp.bfloat16,
        page_size=16,
    )
    with jax.set_mesh(_create_mesh()):
      element_shape = utils.derive_kv_geometry(cfg, num_kv_heads=1, head_dim=64)
    self.assertEqual(element_shape, (2, 2, 64))

  def test_derive_kv_geometry_invalid_tp_divisibility_raises(self):
    cfg = configs.CacheConfig(
        max_device_size_gib=1.0,
        dtype=jnp.bfloat16,
        page_size=16,
    )
    with jax.set_mesh(_create_mesh((4,), ("tp",))):
      with self.assertRaisesRegex(
          ValueError, r"tp_size \(4\) must be divisible"
      ):
        utils.derive_kv_geometry(cfg, num_kv_heads=3, head_dim=64)
      with self.assertRaisesRegex(
          ValueError, r"num_kv_heads \(6\) must be divisible"
      ):
        utils.derive_kv_geometry(cfg, num_kv_heads=6, head_dim=64)

  def test_derive_kv_geometry_invalid_kv_packing_divisibility_raises(self):
    cfg = configs.CacheConfig(
        max_device_size_gib=1.0,
        dtype=jnp.int8,
        page_size=16,
    )
    with jax.set_mesh(_create_mesh()):
      with self.assertRaisesRegex(
          ValueError,
          r"2 \* \(num_kv_heads // tp_size\) \(2\) must be divisible by "
          r"kv_packing \(4\)\.",
      ):
        utils.derive_kv_geometry(cfg, num_kv_heads=2, head_dim=64)

  @parameterized.named_parameters(
      dict(
          testcase_name="dp_and_tp",
          shape=(2, 2),
          axis_names=("dp", "tp"),
          dp_axis=("dp",),
          tp_axis="tp",
      ),
      dict(
          testcase_name="dp_only",
          shape=(2,),
          axis_names=("dp",),
          dp_axis=("dp",),
          tp_axis=None,
      ),
      dict(
          testcase_name="fsdp_and_dp",
          shape=(2, 2),
          axis_names=("fsdp", "dp"),
          dp_axis=("fsdp", "dp"),
          tp_axis=None,
      ),
      dict(
          testcase_name="tp_only",
          shape=(2,),
          axis_names=("tp",),
          dp_axis=None,
          tp_axis="tp",
      ),
      dict(
          testcase_name="single_dp_replica",
          shape=(1,),
          axis_names=("dp",),
          dp_axis=None,
          tp_axis=None,
      ),
  )
  def test_derive_cache_sharding(self, shape, axis_names, dp_axis, tp_axis):
    mesh = _create_mesh(shape, axis_names)
    with jax.set_mesh(mesh):
      sharding = utils.derive_cache_sharding()
    self.assertEqual(
        sharding,
        jax.sharding.NamedSharding(
            mesh,
            jax.sharding.PartitionSpec(dp_axis, None, tp_axis, None, None),
        ),
    )

  def test_derive_cache_sharding_without_active_mesh_raises(self):
    with self.assertRaisesRegex(ValueError, r"An active JAX mesh is required"):
      utils.derive_cache_sharding()

  def test_compute_page_limits(self):
    cfg = configs.CacheConfig(
        max_device_size_gib=(1024 * 1024) / _GIB,
        max_host_size_gib=(512 * 1024) / _GIB,
        dtype=jnp.bfloat16,
        page_size=16,
    )
    with jax.set_mesh(_create_mesh((1,), ("tp",))):
      element_shape = utils.derive_kv_geometry(cfg, num_kv_heads=8, head_dim=64)
      num_device_pages, num_host_pages = utils.compute_page_limits(
          config=cfg,
          element_shapes=(element_shape,) * 4,
          sharding=utils.derive_cache_sharding(),
      )
    self.assertEqual(num_device_pages, 8)
    self.assertEqual(num_host_pages, 4)

  def test_compute_page_limits_empty_element_shapes_raises(self):
    cfg = configs.CacheConfig(
        max_device_size_gib=1.0,
        dtype=jnp.bfloat16,
        page_size=16,
    )
    with jax.set_mesh(_create_mesh()):
      with self.assertRaisesRegex(
          ValueError, r"element_shapes must not be empty\."
      ):
        utils.compute_page_limits(
            config=cfg,
            element_shapes=(),
            sharding=utils.derive_cache_sharding(),
        )

  # One page is 16 tokens * (16 KV heads * 64 head dim) * 2 bytes = 32 KiB.
  # The budget fits 8 unsharded pages on a device, and 16 pages sharded over
  # TP. DP places one page of each group of dp_size on every device.
  @parameterized.named_parameters(
      dict(
          testcase_name="single_device",
          shape=(1,),
          axis_names=("tp",),
          expected_device_pages=8,
          expected_device_pages_one_byte_short=7,
      ),
      dict(
          testcase_name="dp_only",
          shape=(2,),
          axis_names=("dp",),
          expected_device_pages=16,
          expected_device_pages_one_byte_short=14,
      ),
      dict(
          testcase_name="tp_only",
          shape=(2,),
          axis_names=("tp",),
          expected_device_pages=16,
          expected_device_pages_one_byte_short=15,
      ),
      dict(
          testcase_name="dp_and_tp",
          shape=(2, 2),
          axis_names=("dp", "tp"),
          expected_device_pages=32,
          expected_device_pages_one_byte_short=30,
      ),
  )
  def test_compute_page_limits_budget_boundary(
      self,
      shape,
      axis_names,
      expected_device_pages,
      expected_device_pages_one_byte_short,
  ):
    page_bytes = 32 * 1024
    mesh = _create_mesh(shape, axis_names)
    for budget_offset, expected_device, expected_host in (
        (0, expected_device_pages, 3),
        (-1, expected_device_pages_one_byte_short, 2),
        (-0.2, expected_device_pages_one_byte_short, 2),
    ):
      with self.subTest(budget_offset=budget_offset), jax.set_mesh(mesh):
        cfg = configs.CacheConfig(
            max_device_size_gib=(8 * page_bytes + budget_offset) / _GIB,
            max_host_size_gib=(3 * page_bytes + budget_offset) / _GIB,
            dtype=jnp.bfloat16,
            page_size=16,
        )
        element_shape = utils.derive_kv_geometry(
            cfg, num_kv_heads=8, head_dim=64
        )
        num_device_pages, num_host_pages = utils.compute_page_limits(
            config=cfg,
            element_shapes=(element_shape,),
            sharding=utils.derive_cache_sharding(),
        )
        self.assertEqual(num_device_pages, expected_device)
        self.assertEqual(num_host_pages, expected_host)

  def test_compute_page_limits_heterogeneous_geometry(self):
    # Weighting each group's byte budget by its own per-page cost makes the
    # weight cancel, so both groups end up with the same page count.
    cfg = configs.CacheConfig(
        max_device_size_gib=(3 * (65536 + 16384)) / _GIB,
        dtype=jnp.bfloat16,
        page_size=16,
    )
    with jax.set_mesh(_create_mesh((1,), ("tp",))):
      element_shapes = tuple(
          utils.derive_kv_geometry(cfg, num_kv_heads, head_dim)
          for num_kv_heads, head_dim in ((8, 64), (8, 64), (4, 32), (4, 32))
      )
      self.assertEqual(
          element_shapes, ((8, 2, 64), (8, 2, 64), (4, 2, 32), (4, 2, 32))
      )

      num_device_pages, _ = utils.compute_page_limits(
          config=cfg,
          element_shapes=element_shapes,
          sharding=utils.derive_cache_sharding(),
      )
    self.assertEqual(num_device_pages, 3)

  def test_compute_page_limits_is_multiple_of_dp_size(self):
    # Axis 0 of the pool is partitioned over dp, so a count that isn't a
    # multiple of dp_size would fail allocation with IndivisibleError.
    cfg = configs.CacheConfig(
        max_device_size_gib=(37 * 1024) / _GIB,
        dtype=jnp.bfloat16,
        page_size=16,
    )
    with jax.set_mesh(_create_mesh((4,), ("dp",))):
      element_shape = utils.derive_kv_geometry(cfg, num_kv_heads=2, head_dim=8)
      num_device_pages, _ = utils.compute_page_limits(
          config=cfg,
          element_shapes=(element_shape,),
          sharding=utils.derive_cache_sharding(),
      )
    # One page is 1 KiB, so each of the 4 devices holds 37 pages.
    self.assertEqual(num_device_pages, 4 * 37)

  @parameterized.parameters(0.0, -0.1, 1.5)
  def test_resolve_max_device_size_gib_invalid_hbm_utilization_raises(
      self, hbm_utilization
  ):
    with self.assertRaisesRegex(
        ValueError, "rollout_vanilla_v2_hbm_utilization"
    ):
      utils.resolve_max_device_size_gib(hbm_utilization, _create_mesh())

  def test_resolve_max_device_size_gib_takes_min_across_devices(self):
    mesh = _create_mesh((1,), ("tp",))
    with mock.patch.object(
        jax.sharding.Mesh,
        "local_devices",
        new_callable=mock.PropertyMock,
        return_value=[
            _FakeDevice({"bytes_limit": 1000, "bytes_in_use": 100}),
            _FakeDevice({"bytes_limit": 1000, "bytes_in_use": 300}),
        ],
    ):
      size_gib = utils.resolve_max_device_size_gib(0.5, mesh)
    self.assertEqual(size_gib, (500 - 300) / _GIB)

  def test_resolve_max_device_size_gib_insufficient_hbm_raises(self):
    mesh = _create_mesh((1,), ("tp",))
    with mock.patch.object(
        jax.sharding.Mesh,
        "local_devices",
        new_callable=mock.PropertyMock,
        return_value=[_FakeDevice({"bytes_limit": 1000, "bytes_in_use": 600})],
    ):
      with self.assertRaisesRegex(ValueError, "Insufficient HBM"):
        utils.resolve_max_device_size_gib(0.5, mesh)

  def test_resolve_max_device_size_gib_missing_memory_stats_raises(self):
    mesh = _create_mesh((1,), ("tp",))
    with mock.patch.object(
        jax.sharding.Mesh,
        "local_devices",
        new_callable=mock.PropertyMock,
        return_value=[_FakeDevice(None)],
    ):
      with self.assertRaisesRegex(ValueError, "memory_stats"):
        utils.resolve_max_device_size_gib(0.5, mesh)

  @parameterized.parameters(
      ("TPU v5p", 95),
      ("TPU v5", 95),
      ("TPU v5e", 16),
      ("TPU v5 lite", 16),
      ("TPU v6e", 32),
      ("TPU v6 lite", 32),
      ("TPU v4", 32),
      ("TPU7x", 96),
      ("TPU v7", 96),
      ("TPU8x", 144),
      ("TPU v8i", 144),
  )
  def test_device_hbm_limit_bytes_returns_tpu_capacity(
      self, device_kind, expected_gib
  ):
    device = _FakeDevice(device_kind=device_kind)
    self.assertEqual(
        utils._device_hbm_limit_bytes(device),
        expected_gib * _GIB,
    )

  def test_device_hbm_limit_bytes_rejects_non_tpu_and_unknown_tpu(self):
    with self.assertRaisesRegex(ValueError, "unsupported device_kind"):
      utils._device_hbm_limit_bytes(_FakeDevice(device_kind="CPU"))
    with self.assertRaisesRegex(ValueError, "Unknown TPU device_kind"):
      utils._device_hbm_limit_bytes(_FakeDevice(device_kind="TPU v99"))

  @parameterized.parameters(True, False)
  def test_resolve_max_device_size_gib_falls_back_to_live_arrays(
      self, raise_on_stats
  ):
    mesh = _create_mesh((1,), ("tp",))
    device = _FakeDevice(
        stats=None, device_kind="TPU v6e", raise_on_stats=raise_on_stats
    )
    fake_data = mock.Mock(device=device, nbytes=4 * _GIB)
    fake_shard = mock.Mock(data=fake_data)
    fake_live_array = mock.Mock(
        is_deleted=lambda: False, addressable_shards=[fake_shard, fake_shard]
    )
    deleted_array = mock.Mock(
        is_deleted=lambda: True, addressable_shards=[fake_shard]
    )
    with (
        mock.patch.object(
            jax.sharding.Mesh,
            "local_devices",
            new_callable=mock.PropertyMock,
            return_value=[device],
        ),
        mock.patch.object(
            jax, "live_arrays", return_value=[fake_live_array, deleted_array]
        ),
    ):
      size_gib = utils.resolve_max_device_size_gib(0.5, mesh)
    # 0.5 * 32 GiB - 4 GiB (deduplicated across shards sharing buf_id) = 12 GiB.
    self.assertEqual(size_gib, 12.0)

  def test_compute_group_page_limits_asymmetric_local_and_global(self):
    budget_bytes = 59_592_687_616
    cfg = configs.CacheConfig(
        max_device_size_gib=budget_bytes / _GIB,
        max_host_size_gib=0.0,
        dtype=jnp.bfloat16,
        page_size=256,
        enable_prefix_caching=True,
    )
    with jax.set_mesh(_create_mesh((4,), ("tp",))):
      local_shape = utils.derive_kv_geometry(cfg, num_kv_heads=16, head_dim=256)
      global_shape = utils.derive_kv_geometry(cfg, num_kv_heads=4, head_dim=512)
      sharding = utils.derive_cache_sharding()
      specs = (
          utils.CacheGroupSpec(
              element_shape=local_shape, num_caches=50, window_size=1024
          ),
          utils.CacheGroupSpec(
              element_shape=global_shape, num_caches=10, window_size=None
          ),
      )
      (local_dev, local_host), (global_dev, global_host) = (
          utils.compute_group_page_limits(
              config=cfg,
              group_specs=specs,
              sharding=sharding,
              max_num_seqs=32,
              max_chunked_prefill_length=1024,
              num_scheduler_steps=8,
          )
      )
    # Per-sequence active bound = cdiv(1024 + 1024 + 7, 256) + 1 = 10 pages ->
    # 32 * 10 = 320 local pages, leaving remaining HBM for global attention.
    self.assertEqual(local_dev, 320)
    self.assertEqual(local_host, 0)
    self.assertEqual(global_dev, 8166)
    self.assertEqual(global_host, 0)
    total_used_bytes = (
        local_dev * 50 * 256 * (2 * 16 // 4) * 256 * 2
        + global_dev * 10 * 256 * (2 * 4 // 4) * 512 * 2
    )
    self.assertLessEqual(total_used_bytes, budget_bytes)

  def test_compute_group_page_limits_batched_token_bound(self):
    cfg = configs.CacheConfig(
        max_device_size_gib=64.0,
        max_host_size_gib=0.0,
        dtype=jnp.bfloat16,
        page_size=512,
    )
    with jax.set_mesh(_create_mesh((4,), ("tp",))):
      local_shape = utils.derive_kv_geometry(cfg, num_kv_heads=16, head_dim=256)
      global_shape = utils.derive_kv_geometry(cfg, num_kv_heads=4, head_dim=512)
      sharding = utils.derive_cache_sharding()
      specs = (
          utils.CacheGroupSpec(
              element_shape=local_shape, num_caches=50, window_size=1024
          ),
          utils.CacheGroupSpec(
              element_shape=global_shape, num_caches=10, window_size=None
          ),
      )
      (local_tight, _), (global_tight, _) = utils.compute_group_page_limits(
          config=cfg,
          group_specs=specs,
          sharding=sharding,
          max_num_seqs=64,
          max_chunked_prefill_length=1024,
          num_scheduler_steps=16,
          max_num_batched_tokens=4096,
      )
      (local_unbounded, _), (global_unbounded, _) = (
          utils.compute_group_page_limits(
              config=cfg,
              group_specs=specs,
              sharding=sharding,
              max_num_seqs=64,
              max_chunked_prefill_length=1024,
              num_scheduler_steps=16,
          )
      )
    self.assertEqual(local_tight, 264)
    self.assertEqual(local_unbounded, 384)
    self.assertGreater(global_tight, global_unbounded)

  def test_local_window_active_pages_zero_extra_and_subthreshold_budget(self):
    # decode_pages = cdiv(65, 256) + 1 = 2; prefill_pages = cdiv(127, 256) + 1
    # = 2 -> max_extra == 0.
    self.assertEqual(
        utils.local_window_active_pages(
            window_size=64,
            page_size=256,
            max_num_seqs=4,
            max_chunked_prefill_length=64,
            num_scheduler_steps=1,
            max_num_batched_tokens=128,
        ),
        8,
    )
    # prefill_pages = cdiv(320, 256) + 1 = 3 (max_extra == 1), while
    # first_extra_tokens = 193 > max_num_batched_tokens = 64 -> k_max == 0.
    self.assertEqual(
        utils.local_window_active_pages(
            window_size=64,
            page_size=256,
            max_num_seqs=4,
            max_chunked_prefill_length=256,
            num_scheduler_steps=1,
            max_num_batched_tokens=64,
        ),
        8,
    )
    self.assertEqual(
        utils.local_window_active_pages(
            window_size=64,
            page_size=256,
            max_num_seqs=1,
            max_chunked_prefill_length=256,
            num_scheduler_steps=1,
            max_num_batched_tokens=64,
        ),
        3,
    )
    # Boundary case where per_seq (6) exceeds decode_pages (5) at chunk size 2.
    self.assertEqual(
        utils.local_window_active_pages(
            window_size=15,
            page_size=4,
            max_num_seqs=1,
            max_chunked_prefill_length=2,
            num_scheduler_steps=1,
            max_num_batched_tokens=100,
        ),
        utils.local_window_active_pages_per_seq(
            window_size=15,
            page_size=4,
            max_chunked_prefill_length=2,
            num_scheduler_steps=1,
        ),
    )
    # Exact threshold where two 2-token prefills fit in a 4-token budget.
    self.assertEqual(
        utils.local_window_active_pages(
            window_size=15,
            page_size=4,
            max_num_seqs=2,
            max_chunked_prefill_length=2,
            num_scheduler_steps=1,
            max_num_batched_tokens=4,
        ),
        12,
    )
    # Multi-step decode dominates when chunked prefill is disabled (7 > 3).
    self.assertEqual(
        utils.local_window_active_pages_per_seq(
            window_size=8,
            page_size=4,
            max_chunked_prefill_length=0,
            num_scheduler_steps=4,
        ),
        5,
    )

  def test_compute_group_page_limits_uniform_fallbacks_and_dp_alignment(self):
    page_bytes = 32 * 1024
    dp_group_bytes = 16 * 1024
    with jax.set_mesh(_create_mesh((2, 2), ("dp", "tp"))):
      cfg = configs.CacheConfig(
          max_device_size_gib=(40 * page_bytes) / _GIB,
          max_host_size_gib=(20 * page_bytes) / _GIB,
          dtype=jnp.bfloat16,
          page_size=16,
      )
      elem = utils.derive_kv_geometry(cfg, num_kv_heads=8, head_dim=64)
      sharding = utils.derive_cache_sharding()
      specs = (
          utils.CacheGroupSpec(
              element_shape=elem, num_caches=1, window_size=16
          ),
          utils.CacheGroupSpec(
              element_shape=elem, num_caches=1, window_size=None
          ),
      )
      tight_budget_bytes = 40 * dp_group_bytes - 1
      tight_cfg = configs.CacheConfig(
          max_device_size_gib=tight_budget_bytes / _GIB,
          max_host_size_gib=0.0,
          dtype=jnp.bfloat16,
          page_size=16,
      )
      (tight_local, _), (tight_global, _) = utils.compute_group_page_limits(
          config=tight_cfg,
          group_specs=specs,
          sharding=sharding,
          max_num_seqs=2,
          max_chunked_prefill_length=16,
          num_scheduler_steps=1,
      )
      self.assertEqual((tight_local + tight_global) // 2, 39)
      self.assertLessEqual(
          ((tight_local + tight_global) // 2) * dp_group_bytes,
          tight_budget_bytes,
      )
      uniform_dev, uniform_host = utils.compute_page_limits(
          config=cfg, element_shapes=(elem, elem), sharding=sharding
      )
      self.assertEqual(
          utils.compute_group_page_limits(
              config=cfg, group_specs=specs, sharding=sharding
          ),
          [(uniform_dev, uniform_host), (uniform_dev, uniform_host)],
      )
      (local_dev, local_host), (global_dev, global_host) = (
          utils.compute_group_page_limits(
              config=cfg,
              group_specs=specs,
              sharding=sharding,
              max_num_seqs=2,
              max_chunked_prefill_length=16,
              num_scheduler_steps=1,
          )
      )
      self.assertEqual(local_dev % 2, 0)
      self.assertEqual(global_dev % 2, 0)
      self.assertLess(local_dev, uniform_dev)
      self.assertGreater(global_dev, uniform_dev)
      self.assertLessEqual(local_host, uniform_host)
      self.assertGreaterEqual(global_host, uniform_host)
      self.assertEqual(
          utils.compute_group_page_limits(
              config=cfg,
              group_specs=specs,
              sharding=sharding,
              max_num_seqs=1000,
              max_chunked_prefill_length=16,
              num_scheduler_steps=1,
          ),
          [(uniform_dev, uniform_host), (uniform_dev, uniform_host)],
      )

  def test_compute_group_page_limits_invalid_inputs_raises(self):
    cfg = configs.CacheConfig(
        max_device_size_gib=1.0, dtype=jnp.bfloat16, page_size=16
    )
    with jax.set_mesh(_create_mesh()):
      elem = utils.derive_kv_geometry(cfg, num_kv_heads=8, head_dim=64)
      sharding = utils.derive_cache_sharding()
      with self.assertRaisesRegex(ValueError, "group_specs must not be empty"):
        utils.compute_group_page_limits(
            config=cfg, group_specs=(), sharding=sharding
        )
      with self.assertRaisesRegex(ValueError, "num_caches must be positive"):
        utils.compute_group_page_limits(
            config=cfg,
            group_specs=(
                utils.CacheGroupSpec(element_shape=elem, num_caches=0),
            ),
            sharding=sharding,
        )
      with self.assertRaisesRegex(ValueError, "window_size must be positive"):
        utils.compute_group_page_limits(
            config=cfg,
            group_specs=(
                utils.CacheGroupSpec(
                    element_shape=elem, num_caches=1, window_size=0
                ),
            ),
            sharding=sharding,
        )
      with self.assertRaisesRegex(ValueError, "max_num_seqs must be positive"):
        utils.compute_group_page_limits(
            config=cfg,
            group_specs=(
                utils.CacheGroupSpec(element_shape=elem, num_caches=1),
            ),
            sharding=sharding,
            max_num_seqs=0,
        )
      with self.assertRaisesRegex(
          ValueError, "max_num_batched_tokens must be positive"
      ):
        utils.compute_group_page_limits(
            config=cfg,
            group_specs=(
                utils.CacheGroupSpec(element_shape=elem, num_caches=1),
            ),
            sharding=sharding,
            max_num_seqs=4,
            max_num_batched_tokens=0,
        )


class ConfigAndSamplingParamsTest(parameterized.TestCase):

  def test_to_cache_config_wires_host_size_gib(self):
    mesh = _create_mesh((1,), ("tp",))
    rollout_config = common_configs.RolloutConfig(
        rollout_vanilla_v2_host_size_gib=2.5
    )
    with mock.patch.object(
        utils, "resolve_max_device_size_gib", return_value=8.0
    ):
      cache_config = utils.to_cache_config(
          rollout_config, jnp.bfloat16, mesh
      )

    self.assertEqual(cache_config.max_device_size_gib, 8.0)
    self.assertEqual(cache_config.max_host_size_gib, 2.5)
    self.assertEqual(cache_config.dtype, jnp.bfloat16)

  def test_to_cache_config_splits_host_size_gib_across_dp_ranks(self):
    mesh = _create_mesh((4, 1), ("dp", "tp"))
    rollout_config = common_configs.RolloutConfig(
        rollout_vanilla_v2_host_size_gib=10.0
    )
    with mock.patch.object(
        utils, "resolve_max_device_size_gib", return_value=8.0
    ):
      cache_config = utils.to_cache_config(
          rollout_config, jnp.bfloat16, mesh
      )

    self.assertEqual(cache_config.max_device_size_gib, 8.0)
    self.assertEqual(cache_config.max_host_size_gib, 2.5)

  def test_to_scheduler_config_truncates_chunked_prefill_to_sliding_window(self):
    rollout_config = common_configs.RolloutConfig(
        rollout_vanilla_v2_chunked_prefill_length=32
    )
    geometries = {
        "full": datatypes.CacheGeometry(num_kv_heads=1, head_dim=1),
        "sliding": datatypes.CacheGeometry(
            num_kv_heads=1, head_dim=1, window_size=12
        ),
    }

    scheduler_config = utils.to_scheduler_config(rollout_config, geometries)

    self.assertEqual(scheduler_config.max_chunked_prefill_length, 8)

  def test_verify_sampling_params_top_k_over_max_top_k_raises(self):
    rollout_config = common_configs.RolloutConfig(top_k=4)
    utils.verify_sampling_params(
        "fits",
        request_lib.SamplingParams(max_tokens=8, top_k=4),
        rollout_config,
    )

    with self.assertRaisesRegex(ValueError, "max_top_k"):
      utils.verify_sampling_params(
          "0",
          request_lib.SamplingParams(max_tokens=8, top_k=5),
          rollout_config,
      )

  def test_verify_sampling_params_any_top_k_allowed_without_max_top_k(self):
    rollout_config = common_configs.RolloutConfig(top_k=None)
    utils.verify_sampling_params(
        "0",
        request_lib.SamplingParams(max_tokens=8, top_k=1000),
        rollout_config,
    )

  @parameterized.named_parameters(
      ("zero_max_tokens", dict(max_tokens=0), "max_tokens"),
      (
          "negative_temperature",
          dict(max_tokens=8, temperature=-0.1),
          "temperature",
      ),
      ("zero_top_p", dict(max_tokens=8, top_p=0.0), "top_p"),
      ("top_p_over_one", dict(max_tokens=8, top_p=1.5), "top_p"),
      ("beam_size", dict(max_tokens=8, beam_size=2), "beam_size"),
      (
          "routed_experts",
          dict(max_tokens=8, return_routed_experts=True),
          "routed_experts",
      ),
      (
          "routed_experts_prompt_start",
          dict(max_tokens=8, routed_experts_prompt_start=1),
          "routed_experts_prompt_start",
      ),
      ("logprobs", dict(max_tokens=8, return_logprobs=True), "return_logprobs"),
      ("logits", dict(max_tokens=8, return_logits=True), "return_logits"),
  )
  def test_verify_sampling_params_unsupported_raises(
      self, sampling_kwargs, message
  ):
    with self.assertRaisesRegex(ValueError, message):
      utils.verify_sampling_params(
          "0",
          request_lib.SamplingParams(**sampling_kwargs),
          common_configs.RolloutConfig(),
      )

  def test_verify_sampling_params_must_match_configured_outputs(self):
    rollout_config = common_configs.RolloutConfig(
        return_logprobs=True, return_logits=True
    )
    utils.verify_sampling_params(
        "fits",
        request_lib.SamplingParams(
            max_tokens=8, return_logprobs=True, return_logits=True
        ),
        rollout_config,
    )

    with self.assertRaisesRegex(ValueError, "return_logprobs"):
      utils.verify_sampling_params(
          "0",
          request_lib.SamplingParams(max_tokens=8, return_logits=True),
          rollout_config,
      )


class DPRanksTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="dp_and_tp",
          shape=(2, 2),
          axis_names=("dp", "tp"),
          expected_device_ids=((0, 1), (2, 3)),
      ),
      # A rank's devices run along "tp", wherever it is in the mesh.
      dict(
          testcase_name="tp_first",
          shape=(2, 2),
          axis_names=("tp", "dp"),
          expected_device_ids=((0, 2), (1, 3)),
      ),
      # The DP axes ("dp" and "fsdp") index the ranks, in mesh order.
      dict(
          testcase_name="two_dp_axes",
          shape=(2, 2),
          axis_names=("dp", "fsdp"),
          expected_device_ids=((0,), (1,), (2,), (3,)),
      ),
      dict(
          testcase_name="tp_only",
          shape=(4,),
          axis_names=("tp",),
          expected_device_ids=((0, 1, 2, 3),),
      ),
  )
  def test_dp_ranks(self, shape, axis_names, expected_device_ids):
    devices = jax.devices()
    ranks = utils.dp_ranks(_create_mesh(shape, axis_names))

    self.assertEqual(
        tuple(tuple(r.devices.flat) for r in ranks),
        tuple(tuple(devices[i] for i in ids) for ids in expected_device_ids),
    )
    for r in ranks:
      self.assertEqual(r.axis_names, ("tp",))

  def test_stitch_and_split_physical_pages_single_rank(self):
    mesh = _create_mesh((1,), ("tp",))
    pages = {"layer_0": jnp.zeros((2, 4, 2, 1, 8), dtype=jnp.float32)}
    stitched = utils.stitch_physical_pages(mesh, [pages])
    np.testing.assert_array_equal(stitched["layer_0"], pages["layer_0"])
    (split_rank,) = utils.split_physical_pages(stitched, [pages])
    np.testing.assert_array_equal(split_rank["layer_0"], pages["layer_0"])

  def test_stitch_and_split_physical_pages_multiple_ranks(self):
    mesh = _create_mesh((2,), ("dp",))
    rank_pages = []
    for r, rank_mesh in enumerate(utils.dp_ranks(mesh)):
      sharding = jax.sharding.NamedSharding(
          rank_mesh,
          jax.sharding.PartitionSpec(),
      )
      arr = jax.device_put(
          jnp.full((2, 4, 2, 1, 8), r + 1, dtype=jnp.float32), sharding
      )
      rank_pages.append({"layer_0": arr})

    stitched = utils.stitch_physical_pages(mesh, rank_pages)
    self.assertEqual(stitched["layer_0"].shape, (4, 4, 2, 1, 8))
    np.testing.assert_array_equal(stitched["layer_0"][:2], 1.0)
    np.testing.assert_array_equal(stitched["layer_0"][2:], 2.0)

    split_ranks = utils.split_physical_pages(stitched, rank_pages)
    self.assertLen(split_ranks, 2)
    for r, (split_rank, orig_rank) in enumerate(
        zip(split_ranks, rank_pages)
    ):
      self.assertEqual(
          split_rank["layer_0"].shape, orig_rank["layer_0"].shape
      )
      self.assertEqual(
          split_rank["layer_0"].sharding, orig_rank["layer_0"].sharding
      )
      np.testing.assert_array_equal(split_rank["layer_0"], r + 1)
if __name__ == "__main__":
  absltest.main()
