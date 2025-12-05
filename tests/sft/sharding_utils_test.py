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
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import jax.sharding as shd
import numpy as np
from tunix.sft import sharding_utils

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

if __name__ == '__main__':
  absltest.main()
