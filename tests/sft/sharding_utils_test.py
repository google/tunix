# Copyright 2025 Google LLC
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

import os
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import jax.sharding as shd
import numpy as np
from tunix.sft import checkpoint_manager
from tunix.sft import sharding_utils
from tunix.utils import sharding_utils as core_sharding_utils

# CPU environment setup to simulate multi device env.
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'


class ShardingUtilsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='happy_path',
          batch_size=8,
          expected_pspec=shd.PartitionSpec(*('fsdp',)),
      ),
      dict(
          testcase_name='not_evenly_divisible',
          batch_size=9,
          expected_pspec=shd.PartitionSpec(),
      ),
      dict(
          testcase_name='not_enough_size',
          batch_size=7,
          expected_pspec=shd.PartitionSpec(),
      ),
  )
  def test_simple_reshard(self, batch_size, expected_pspec):
    device_cnt = jax.device_count()
    mesh = shd.Mesh(
        np.array(jax.devices()).reshape(device_cnt, 1),
        axis_names=('fsdp', 'tp'),
    )
    x = jnp.ones((batch_size, 4, 8))
    self.assertIsInstance(x.sharding, shd.SingleDeviceSharding)
    with mesh:
      x_sharded = sharding_utils.shard_input(x, ('fsdp',))
    assert isinstance(x_sharded.sharding, shd.NamedSharding)
    self.assertEqual(x_sharded.sharding.mesh, mesh)
    self.assertEqual(x_sharded.sharding.spec, expected_pspec)

  def test_reshard_across_meshes(self):
    device_cnt = jax.device_count()
    split_idx = device_cnt // 2
    mesh1 = shd.Mesh(
        np.array(jax.devices()[:split_idx]).reshape(1, split_idx),
        axis_names=('fsdp', 'tp'),
    )
    mesh2 = shd.Mesh(
        np.array(jax.devices()[split_idx:]).reshape(split_idx, 1),
        axis_names=('fsdp', 'tp'),
    )
    pspec = shd.PartitionSpec(*('tp',))
    x = jnp.ones((device_cnt, 4, 8))
    sharding = shd.NamedSharding(mesh1, pspec)
    x_sharded = jax.device_put(x, sharding)
    assert isinstance(x_sharded.sharding, shd.NamedSharding)
    self.assertEqual(x_sharded.sharding.mesh, mesh1)
    self.assertEqual(x_sharded.sharding.spec, pspec)

    with mesh2:
      x_resharded = sharding_utils.shard_input(x_sharded, ('fsdp',))
    assert isinstance(x_resharded.sharding, shd.NamedSharding)
    self.assertEqual(x_resharded.sharding.mesh, mesh2)
    self.assertEqual(
        x_resharded.sharding.spec,
        shd.PartitionSpec(*('fsdp',)),
    )
    self.assertEqual(
        x_resharded.addressable_data(0).shape, (device_cnt / split_idx, 4, 8)
    )

  def test_reshard_same_mesh_different_pspec(self):
    device_cnt = jax.device_count()
    mesh = shd.Mesh(
        np.array(jax.devices()).reshape(1, device_cnt),
        axis_names=('fsdp', 'tp'),
    )
    pspec1 = shd.PartitionSpec(*('tp',))
    pspec2 = shd.PartitionSpec(*('fsdp',))
    x = jnp.ones((device_cnt, 4, 8))
    sharding = shd.NamedSharding(mesh, pspec1)
    x_sharded = jax.device_put(x, sharding)
    assert isinstance(x_sharded.sharding, shd.NamedSharding)
    self.assertEqual(x_sharded.sharding.mesh, mesh)
    self.assertEqual(x_sharded.sharding.spec, pspec1)

    with mesh:
      x_resharded = sharding_utils.shard_input(x_sharded, ('fsdp',))
    assert isinstance(x_resharded.sharding, shd.NamedSharding)
    self.assertEqual(x_resharded.sharding.mesh, mesh)
    self.assertEqual(x_resharded.sharding.spec, pspec2)

  def test_noop_on_already_sharded_input(self):
    device_cnt = jax.device_count()
    mesh = shd.Mesh(
        np.array(jax.devices()).reshape(1, device_cnt),
        axis_names=('fsdp', 'tp'),
    )
    pspec = shd.PartitionSpec('fsdp')
    x = jnp.ones((device_cnt, 4, 8))
    sharding = shd.NamedSharding(mesh, pspec)
    x_sharded = jax.device_put(x, sharding)
    with mesh:
      x_resharded = sharding_utils.shard_input(x_sharded, ('fsdp',))
    self.assertIs(x_sharded, x_resharded)

    # If resharding happened, a new obj will be created
    with mesh:
      x_resharded = sharding_utils.shard_input(x_sharded, ('tp',))
    self.assertIsNot(x_sharded, x_resharded)

  def test_get_sharding_replicates_unknown_mesh_axis(self):
    device_cnt = jax.device_count()
    mesh = shd.Mesh(
        np.array(jax.devices()).reshape(device_cnt, 1),
        axis_names=('fsdp', 'tp'),
    )
    x = jnp.ones((device_cnt, 4, 8))
    sharding = sharding_utils.get_sharding(
        x, mesh, shd.PartitionSpec('norm')
    )
    self.assertEqual(sharding.mesh, mesh)
    self.assertEqual(sharding.spec, shd.PartitionSpec())

  @mock.patch('tunix.sft.sharding_utils.jax.make_array_from_process_local_data')
  def test_global_array_noop(self, mock_make_array):
    mock_make_array.side_effect = ValueError(
        "device_put's first argument must be a fully addressable array, but got value with devices {TpuDevice(id=0..."
    )
    device_cnt = jax.device_count()
    mesh = shd.Mesh(
        np.array(jax.devices()).reshape(1, device_cnt),
        axis_names=('fsdp', 'tp'),
    )

    x = mock.MagicMock(spec=jax.Array)
    x.is_fully_addressable = False
    x.ndim = 3

    with mesh:
      x_resharded = sharding_utils.shard_input(x, ('fsdp',))

    self.assertIs(x, x_resharded)
    mock_make_array.assert_not_called()

  def test_place_array_on_target_local_and_noop(self):
    self.assertIs(
        sharding_utils.place_array_on_target,
        core_sharding_utils.place_array_on_target,
    )
    self.assertIs(
        sharding_utils.place_arrays_on_targets,
        core_sharding_utils.place_arrays_on_targets,
    )
    device_cnt = jax.device_count()
    mesh = shd.Mesh(
        np.array(jax.devices()).reshape(device_cnt, 1),
        axis_names=('fsdp', 'tp'),
    )
    sharding = shd.NamedSharding(mesh, shd.PartitionSpec('fsdp'))
    host_arr = np.ones((device_cnt, 4), dtype=np.float32)
    placed = sharding_utils.place_array_on_target(host_arr, sharding)
    self.assertEqual(placed.sharding, sharding)
    self.assertIs(
        sharding_utils.place_array_on_target(placed, sharding), placed
    )

  @mock.patch('tunix.utils.sharding_utils.jax.make_array_from_callback')
  def test_place_array_on_target_multihost_and_non_addressable_guard(
      self, mock_from_callback
  ):
    sentinel = jnp.ones((4, 4), dtype=jnp.float32)
    mock_from_callback.return_value = sentinel

    multihost_sharding = mock.MagicMock(spec=shd.NamedSharding)
    multihost_sharding.is_fully_addressable = False

    host_arr = np.arange(16, dtype=np.float32).reshape(4, 4)
    out = sharding_utils.place_array_on_target(host_arr, multihost_sharding)
    self.assertIs(out, sentinel)
    mock_from_callback.assert_called_once()
    shape_arg, sharding_arg, cb_arg = mock_from_callback.call_args[0]
    self.assertEqual(shape_arg, (4, 4))
    self.assertIs(sharding_arg, multihost_sharding)
    np.testing.assert_array_equal(
        cb_arg((slice(0, 2), slice(None))), host_arr[:2]
    )

    non_addressable_arr = mock.MagicMock(spec=jax.Array)
    non_addressable_arr.sharding = mock.MagicMock(spec=shd.NamedSharding)
    non_addressable_arr.is_fully_addressable = False
    with self.assertRaisesRegex(
        ValueError, 'Cannot re-place a non-fully-addressable'
    ):
      sharding_utils.place_array_on_target(
          non_addressable_arr, multihost_sharding
      )

  @mock.patch('tunix.utils.sharding_utils.jax.make_array_from_callback')
  @mock.patch('tunix.utils.sharding_utils.shd.NamedSharding')
  @mock.patch('tunix.utils.sharding_utils.jax.devices')
  def test_utils_shard_eager_multihost_uses_callback(
      self, mock_devices, mock_named_sharding, mock_from_callback
  ):
    fake_gpu = mock.MagicMock()
    fake_gpu.platform = 'gpu'
    mock_devices.return_value = [fake_gpu]
    multihost_sharding = mock.MagicMock(spec=shd.Sharding)
    multihost_sharding.is_fully_addressable = False
    mock_named_sharding.return_value = multihost_sharding
    sentinel = jnp.ones((8, 4), dtype=jnp.float32)
    mock_from_callback.return_value = sentinel

    device_cnt = len(jax.local_devices())
    mesh = shd.Mesh(
        np.array(jax.local_devices()).reshape(device_cnt, 1),
        axis_names=('fsdp', 'tp'),
    )
    host_arr = np.arange(32, dtype=np.float32).reshape(8, 4)
    with mesh:
      out = core_sharding_utils.shard(host_arr, ('fsdp', None), eager=True)
    self.assertIs(out, sentinel)
    mock_from_callback.assert_called_once()

    already_sharded = mock.MagicMock(spec=jax.Array)
    already_sharded.sharding = multihost_sharding
    already_sharded.is_fully_addressable = False
    with mesh:
      self.assertIs(
          core_sharding_utils.shard(
              already_sharded, ('fsdp', None), eager=True
          ),
          already_sharded,
      )

    mismatched_non_addressable = mock.MagicMock(spec=jax.Array)
    mismatched_non_addressable.sharding = mock.MagicMock(spec=shd.Sharding)
    mismatched_non_addressable.is_fully_addressable = False
    with mesh, self.assertRaisesRegex(
        ValueError, 'Cannot re-place a non-fully-addressable'
    ):
      core_sharding_utils.shard(
          mismatched_non_addressable, ('fsdp', None), eager=True
      )

  @mock.patch(
      'tunix.sft.checkpoint_manager.sharding_utils.place_array_on_target'
  )
  @mock.patch(
      'tunix.sft.checkpoint_manager.ocp_utils.fully_replicated_host_local_array_to_global_array'
  )
  def test_convert_host_local_array_uses_place_array_on_target_fallback(
      self, mock_ocp_to_global, mock_place
  ):
    mock_ocp_to_global.side_effect = ValueError('non-contiguous devices')
    sentinel = jnp.ones((4,), dtype=jnp.float32)
    mock_place.return_value = sentinel

    local_arr = jax.device_put(
        np.ones((4,), dtype=np.float32), jax.local_devices()[0]
    )
    out = checkpoint_manager._convert_host_local_array(local_arr)
    self.assertIs(out, sentinel)
    mock_place.assert_called_once()

  def test_place_arrays_on_targets_batches_local_and_handles_multihost(self):
    self.assertEqual(sharding_utils.place_arrays_on_targets([], []), [])
    with self.assertRaisesRegex(ValueError, 'Length mismatch'):
      sharding_utils.place_arrays_on_targets(
          [np.ones((2,), dtype=np.float32)], []
      )

    local_dev = jax.local_devices()[0]
    arrs = [
        np.arange(4, dtype=np.float32),
        np.arange(6, dtype=np.float32).reshape(2, 3),
    ]
    with mock.patch.object(
        jax, 'device_put', wraps=jax.device_put
    ) as mock_device_put:
      placed = sharding_utils.place_arrays_on_targets(
          arrs, [local_dev, local_dev]
      )
    self.assertEqual(mock_device_put.call_count, 1)
    for out, expected in zip(placed, arrs, strict=True):
      np.testing.assert_array_equal(out, expected)

    multihost_sharding = mock.MagicMock(spec=shd.NamedSharding)
    multihost_sharding.is_fully_addressable = False
    with (
        mock.patch.object(
            jax, 'device_put', wraps=jax.device_put
        ) as mock_device_put,
        mock.patch.object(
            jax,
            'make_array_from_callback',
            side_effect=lambda shape, s, cb: jnp.asarray(
                cb(tuple(slice(0, d) for d in shape))
            ),
        ) as mock_from_callback,
    ):
      placed_mh = sharding_utils.place_arrays_on_targets(
          arrs, [multihost_sharding, multihost_sharding]
      )
    mock_device_put.assert_not_called()
    self.assertEqual(mock_from_callback.call_count, 2)
    for out, expected in zip(placed_mh, arrs, strict=True):
      np.testing.assert_array_equal(out, expected)


if __name__ == '__main__':
  absltest.main()
