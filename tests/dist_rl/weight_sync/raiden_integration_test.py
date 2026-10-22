import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

import asyncio
import types
from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
from tunix.dist_rl.weight_sync import raiden_synchronizer
from tunix.dist_rl.weight_sync import weight_sync
from tunix.dist_rl.weight_sync.raiden_handler import RaidenHandler
from tunix.dist_rl.weight_sync.raiden_synchronizer import RaidenSynchronizer
from tunix.tests.test_common import ModelConfig, ToyTransformer
from tunix.utils import env_utils


def _ffi_runtime_available() -> bool:
  """True when the runtime exposes the Host FFI registry for device_host calls.

  Internal builds link the FFI handlers into the same XLA registry as JAX, so
  they run on both CPU and TPU. The standalone OSS wheel registers them in its
  own copy of the registry, which JAX cannot see outside Pathways proxy.
  """
  if "proxy" in os.environ.get("JAX_PLATFORMS", ""):
    return True
  if os.environ.get("TUNIX_RUN_FFI_TPU_TEST") == "1":
    return True
  return env_utils.is_internal_env()


_skip_unless_ffi_runtime = absltest.skipUnless(
    _ffi_runtime_available(),
    "FFI device_host custom calls require Pathways proxy, an internal build,"
    " or TUNIX_RUN_FFI_TPU_TEST=1 (the standalone OSS wheel registers its FFI"
    " handlers outside JAX's XLA registry).",
)


class RaidenIntegrationTest(absltest.TestCase):

  def _build_sharded_states(self):
    config = ModelConfig(vocab_size=128, num_layers=2)

    rng_s = nnx.Rngs(0)
    sender_model = ToyTransformer(config, rngs=rng_s)

    rng_r = nnx.Rngs(1)
    receiver_model = ToyTransformer(config, rngs=rng_r)

    sender_state = nnx.state(sender_model)
    receiver_state = nnx.state(receiver_model)

    import numpy as np
    devices = np.array(jax.devices()[:4])
    mesh1 = jax.sharding.Mesh(devices, ('x',))
    mesh2 = jax.sharding.Mesh(devices.reshape((2, 2)), ('x', 'y'))

    def shard1(x):
      if x.ndim >= 1 and x.shape[0] % 4 == 0:
        spec = jax.sharding.PartitionSpec('x')
      else:
        spec = jax.sharding.PartitionSpec()
      return jax.device_put(x, jax.sharding.NamedSharding(mesh1, spec))

    def shard2(x):
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
    return sender_state, receiver_state

  def _run_sync(
      self, *, trainer_ffi: bool = False, sampler_ffi: bool = False
  ):
    sender_state, receiver_state = self._build_sharded_states()
    handler = RaidenHandler(port=0, transfer_parallelism=2)

    async def run_sync():
      nonlocal receiver_state
      trainer_sync = RaidenSynchronizer("trainer")
      if trainer_ffi:
        trainer_sync._is_proxy = True
      trainer_sync.bind(sender_state)

      sampler_sync = RaidenSynchronizer("sampler", auto_h2d=True)
      if sampler_ffi:
        sampler_sync._is_proxy = True
      sampler_sync.bind(receiver_state)

      trainer_sync.d2h()

      handler.register_work_unit(trainer_sync.work_unit_metadata())
      handler.register_work_unit(sampler_sync.work_unit_metadata())

      await asyncio.to_thread(
          handler.transfer,
          req_id="manual_test_1",
          src_units=[trainer_sync.work_unit_metadata().unit],
          dst_units=[sampler_sync.work_unit_metadata().unit],
      )

      sampler_sync.h2d()
      if sampler_ffi:
        runner = types.SimpleNamespace(
            state=receiver_state,
            state_leaves=tuple(jax.tree_util.tree_leaves(receiver_state)),
        )
        sampler_sync.apply_to_runner(runner)
        receiver_state = runner.state

    try:
      asyncio.run(run_sync())
    finally:
      handler.close()
      if (
          trainer_ffi or sampler_ffi
      ) and raiden_synchronizer._raiden_ffi is not None:
        raiden_synchronizer._raiden_ffi.destroy_weight_synchronizer()

    return sender_state, receiver_state

  @absltest.skipIf(
      raiden_synchronizer._ws_lib is None,
      "Requires TPU Raiden C++ extension (_ws_lib)",
  )
  @absltest.skipIf(
      "proxy" in os.environ.get("JAX_PLATFORMS", ""),
      "Native trainer-side sync is only the default outside Pathways proxy"
      " runtimes.",
  )
  def test_toy_transformer_raiden_sync(self):
    sender_state, receiver_state = self._run_sync()

    leaves_s, _ = jax.tree_util.tree_flatten(sender_state)
    leaves_r, _ = jax.tree_util.tree_flatten(receiver_state)
    for i, (s, r) in enumerate(zip(leaves_s, leaves_r)):
      self.assertTrue(jnp.allclose(s, r), f"Mismatch at leaf {i}")

  @absltest.skipIf(
      raiden_synchronizer._raiden_ffi is None,
      "Requires Pathways FFI extension (_raiden_ffi)",
  )
  @_skip_unless_ffi_runtime
  def test_toy_transformer_raiden_sync_trainer_ffi(self):
    sender_state, receiver_state = self._run_sync(trainer_ffi=True)

    leaves_s, _ = jax.tree_util.tree_flatten(sender_state)
    leaves_r, _ = jax.tree_util.tree_flatten(receiver_state)
    for i, (s, r) in enumerate(zip(leaves_s, leaves_r)):
      self.assertTrue(jnp.allclose(s, r), f"Mismatch at leaf {i}")

  @absltest.skipIf(
      raiden_synchronizer._raiden_ffi is None,
      "Requires Pathways FFI extension (_raiden_ffi)",
  )
  @_skip_unless_ffi_runtime
  def test_toy_transformer_raiden_sync_sampler_ffi(self):
    sender_state, receiver_state = self._run_sync(sampler_ffi=True)

    leaves_s, _ = jax.tree_util.tree_flatten(sender_state)
    leaves_r, _ = jax.tree_util.tree_flatten(receiver_state)
    for i, (s, r) in enumerate(zip(leaves_s, leaves_r)):
      self.assertTrue(jnp.allclose(s, r), f"Mismatch at leaf {i}")


if __name__ == "__main__":
  absltest.main()
