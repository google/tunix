import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

import asyncio
from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
from tunix.experimental.weight_sync import weight_sync
from tunix.experimental.weight_sync.raiden_handler import RaidenHandler
from tunix.experimental.weight_sync.raiden_synchronizer import RaidenSynchronizer
from tunix.tests.test_common import ModelConfig, ToyTransformer


class RaidenIntegrationTest(absltest.TestCase):

  def test_toy_transformer_raiden_sync(self):
    config = ModelConfig(vocab_size=128, num_layers=2)

    rng_s = nnx.Rngs(0)
    sender_model = ToyTransformer(config, rngs=rng_s)

    rng_r = nnx.Rngs(1)
    receiver_model = ToyTransformer(config, rngs=rng_r)

    sender_state = nnx.state(sender_model)
    receiver_state = nnx.state(receiver_model)

    import numpy as np
    devices = np.array(jax.devices())
    print("devices: ", devices)
    mesh1 = jax.sharding.Mesh(devices, ('x',))
    mesh2 = jax.sharding.Mesh(devices.reshape((2, 2)), ('x', 'y'))

    def shard1(x):
      # Shard along the first dimension if available
      if x.ndim >= 1 and x.shape[0] % 4 == 0:
        spec = jax.sharding.PartitionSpec('x')
      else:
        spec = jax.sharding.PartitionSpec()
      return jax.device_put(x, jax.sharding.NamedSharding(mesh1, spec))

    def shard2(x):
      # Shard along a different dimension if available
      if x.ndim >= 2 and x.shape[1] % 2 == 0 and x.shape[0] % 2 == 0:
        spec = jax.sharding.PartitionSpec('x', 'y')
      elif x.ndim >= 1 and x.shape[0] % 2 == 0:
        spec = jax.sharding.PartitionSpec('x')
      else:
        spec = jax.sharding.PartitionSpec()
      return jax.device_put(x, jax.sharding.NamedSharding(mesh2, spec))

    sender_state = jax.tree_util.tree_map(shard1, sender_state)
    receiver_state = jax.tree_util.tree_map(shard2, receiver_state)

    sender_state = jax.tree_util.tree_map(lambda x: x + 1.0, sender_state)

    handler = RaidenHandler(port=0, transfer_parallelism=2)

    async def run_sync():
      trainer_sync = RaidenSynchronizer("trainer")
      trainer_sync.bind(sender_state)

      sampler_sync = RaidenSynchronizer("sampler")
      sampler_sync.bind(receiver_state)

      handler.register_work_unit(trainer_sync.work_unit_metadata())
      handler.register_work_unit(sampler_sync.work_unit_metadata())

      trainer_sync.d2h()

      await asyncio.to_thread(
          handler.transfer,
          req_id="manual_test_1",
          src_units=[trainer_sync.work_unit_metadata().unit],
          dst_units=[sampler_sync.work_unit_metadata().unit],
      )

      sampler_sync.h2d()

    asyncio.run(run_sync())

    leaves_s, _ = jax.tree_util.tree_flatten(sender_state)
    leaves_r, _ = jax.tree_util.tree_flatten(receiver_state)
    for i, (s, r) in enumerate(zip(leaves_s, leaves_r)):
      self.assertTrue(jnp.allclose(s, r), f"Mismatch at leaf {i}")


if __name__ == "__main__":
  absltest.main()
