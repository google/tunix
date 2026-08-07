# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Abstract-state inventory gate for Qwen3-8B DP16xTP4 training."""

from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from tunix.models.qwen3 import model as model_lib
from tunix.rl import dp_training


class Qwen8BDPAbstractInventoryTest(absltest.TestCase):

  def _physical_bytes_per_device(self, state):
    totals = {int(device.id): 0 for device in jax.devices()}
    for value in jax.tree.leaves(state):
      if not isinstance(value, jax.ShapeDtypeStruct):
        continue
      for device, index in value.sharding.devices_indices_map(
          value.shape
      ).items():
        shard_shape = tuple(
            len(range(
                0 if item.start is None else item.start,
                dimension if item.stop is None else item.stop,
                1 if item.step is None else item.step,
            ))
            for item, dimension in zip(index, value.shape)
            if isinstance(item, slice)
        )
        totals[int(device.id)] += int(
            np.prod(shard_shape, dtype=np.int64) * value.dtype.itemsize
        )
    self.assertLen(set(totals.values()), 1)
    return next(iter(totals.values()))

  def test_model_optimizer_and_accumulator_are_dp_replicated(self):
    if len(jax.devices()) != 64:
      self.skipTest('requires exactly 64 forced CPU or accelerator devices')
    mesh = jax.sharding.Mesh(
        np.asarray(jax.devices()).reshape(16, 4), ('dp', 'tp')
    )
    config = model_lib.ModelConfig.qwen3_8b()
    config.dtype = jnp.bfloat16
    config.param_dtype = jnp.float32
    config.shd_config = model_lib.ShardingConfig.get_data_parallel_sharding()
    abstract_model = nnx.eval_shape(
        lambda: model_lib.Qwen3(config, rngs=nnx.Rngs(params=0))
    )
    model_state = nnx.state(abstract_model, nnx.Param)
    named_shardings = nnx.get_named_sharding(model_state, mesh)
    abstract_params = jax.tree.map(
        lambda value, sharding: jax.ShapeDtypeStruct(
            value.shape, value.dtype, sharding=sharding
        ),
        model_state,
        named_shardings,
    )
    optimizer = optax.adamw(learning_rate=1.0e-6)
    abstract_optimizer = jax.eval_shape(optimizer.init, abstract_params)
    abstract_optimizer = dp_training.attach_adam_state_shardings(
        abstract_optimizer, params=abstract_params, mesh=mesh
    )
    abstract_accumulator = jax.tree.map(
        lambda value: jax.ShapeDtypeStruct(
            value.shape, jnp.float32, sharding=value.sharding
        ),
        abstract_params,
    )

    inventory = dp_training.inspect_abstract_training_state_inventories(
        model=abstract_params,
        optimizer=abstract_optimizer,
        accumulator=abstract_accumulator,
    )
    print(f'[P32.2B] ABSTRACT_INVENTORY {inventory}', flush=True)
    self.assertEqual(set(inventory), {'model', 'optimizer', 'accumulator'})
    for summary in inventory.values():
      self.assertEqual(summary['dp_partitioned_leaves'], 0)
      self.assertGreater(summary['tp_partitioned_leaves'], 0)
      self.assertGreater(summary['logical_bytes'], 0)
    self.assertGreater(inventory['model']['leaves'], 300)
    self.assertEqual(
        inventory['accumulator']['logical_bytes'],
        inventory['model']['logical_bytes'],
    )
    self.assertGreater(
        inventory['optimizer']['logical_bytes'],
        2 * inventory['model']['logical_bytes'],
    )
    self.assertEqual(
        self._physical_bytes_per_device(abstract_params), 8_190_735_360
    )
    self.assertEqual(
        self._physical_bytes_per_device(abstract_optimizer), 16_381_470_724
    )
    self.assertEqual(
        self._physical_bytes_per_device(abstract_accumulator), 8_190_735_360
    )


if __name__ == '__main__':
  absltest.main()
