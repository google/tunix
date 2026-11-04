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

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tunix.experimental.generate import configs
from tunix.experimental.generate import utils

# Sharding tests need up to 4 devices.
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

_GIB = 1 << 30


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

  def test_derive_kv_geometry(self):
    cfg = configs.CacheConfig(
        max_device_size_gib=1.0,
        dtype=jnp.bfloat16,
        page_size=16,
        mesh=_create_mesh(),
    )
    element_shape = utils.derive_kv_geometry(cfg, num_kv_heads=8, head_dim=64)
    self.assertEqual(element_shape, (8, 2, 64))

  def test_derive_kv_geometry_pads_kv_heads_to_tp_size(self):
    cfg = configs.CacheConfig(
        max_device_size_gib=1.0,
        dtype=jnp.bfloat16,
        page_size=16,
        mesh=_create_mesh(),
    )
    element_shape = utils.derive_kv_geometry(cfg, num_kv_heads=1, head_dim=64)
    self.assertEqual(element_shape, (2, 2, 64))

  def test_derive_kv_geometry_invalid_tp_divisibility_raises(self):
    cfg = configs.CacheConfig(
        max_device_size_gib=1.0,
        dtype=jnp.bfloat16,
        page_size=16,
        mesh=_create_mesh((4,), ("tp",)),
    )
    with self.assertRaisesRegex(ValueError, r"tp_size \(4\) must be divisible"):
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
        mesh=_create_mesh(),
    )
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
    cfg = configs.CacheConfig(
        max_device_size_gib=1.0,
        dtype=jnp.bfloat16,
        page_size=16,
        mesh=mesh,
    )
    self.assertEqual(
        utils.derive_cache_sharding(cfg),
        jax.sharding.NamedSharding(
            mesh,
            jax.sharding.PartitionSpec(dp_axis, None, tp_axis, None, None),
        ),
    )

  def test_compute_page_limits(self):
    cfg = configs.CacheConfig(
        max_device_size_gib=(1024 * 1024) / _GIB,
        max_host_size_gib=(512 * 1024) / _GIB,
        dtype=jnp.bfloat16,
        page_size=16,
        mesh=_create_mesh((1,), ("tp",)),
    )
    element_shape = utils.derive_kv_geometry(cfg, num_kv_heads=8, head_dim=64)
    num_device_pages, num_host_pages = utils.compute_page_limits(
        config=cfg,
        element_shapes=(element_shape,) * 4,
        sharding=utils.derive_cache_sharding(cfg),
    )
    self.assertEqual(num_device_pages, 8)
    self.assertEqual(num_host_pages, 4)

  def test_compute_page_limits_empty_element_shapes_raises(self):
    cfg = configs.CacheConfig(
        max_device_size_gib=1.0,
        dtype=jnp.bfloat16,
        page_size=16,
        mesh=_create_mesh(),
    )
    with self.assertRaisesRegex(
        ValueError, r"element_shapes must not be empty\."
    ):
      utils.compute_page_limits(
          config=cfg,
          element_shapes=(),
          sharding=utils.derive_cache_sharding(cfg),
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
      with self.subTest(budget_offset=budget_offset):
        cfg = configs.CacheConfig(
            max_device_size_gib=(8 * page_bytes + budget_offset) / _GIB,
            max_host_size_gib=(3 * page_bytes + budget_offset) / _GIB,
            dtype=jnp.bfloat16,
            page_size=16,
            mesh=mesh,
        )
        element_shape = utils.derive_kv_geometry(
            cfg, num_kv_heads=8, head_dim=64
        )
        num_device_pages, num_host_pages = utils.compute_page_limits(
            config=cfg,
            element_shapes=(element_shape,),
            sharding=utils.derive_cache_sharding(cfg),
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
        mesh=_create_mesh((1,), ("tp",)),
    )
    geometries = (
        configs.CacheGeometry(num_kv_heads=8, head_dim=64),
        configs.CacheGeometry(num_kv_heads=8, head_dim=64),
        configs.CacheGeometry(num_kv_heads=4, head_dim=32, window_size=128),
        configs.CacheGeometry(num_kv_heads=4, head_dim=32, window_size=128),
    )
    element_shapes = tuple(
        utils.derive_kv_geometry(cfg, g.num_kv_heads, g.head_dim)
        for g in geometries
    )
    self.assertEqual(
        element_shapes, ((8, 2, 64), (8, 2, 64), (4, 2, 32), (4, 2, 32))
    )

    num_device_pages, _ = utils.compute_page_limits(
        config=cfg,
        element_shapes=element_shapes,
        sharding=utils.derive_cache_sharding(cfg),
    )
    self.assertEqual(num_device_pages, 3)

  def test_compute_page_limits_is_multiple_of_dp_size(self):
    # Axis 0 of the pool is partitioned over dp, so a count that isn't a
    # multiple of dp_size would fail allocation with IndivisibleError.
    cfg = configs.CacheConfig(
        max_device_size_gib=(37 * 1024) / _GIB,
        dtype=jnp.bfloat16,
        page_size=16,
        mesh=_create_mesh((4,), ("dp",)),
    )
    element_shape = utils.derive_kv_geometry(cfg, num_kv_heads=2, head_dim=8)
    num_device_pages, _ = utils.compute_page_limits(
        config=cfg,
        element_shapes=(element_shape,),
        sharding=utils.derive_cache_sharding(cfg),
    )
    # One page is 1 KiB, so each of the 4 devices holds 37 pages.
    self.assertEqual(num_device_pages, 4 * 37)


if __name__ == "__main__":
  absltest.main()
