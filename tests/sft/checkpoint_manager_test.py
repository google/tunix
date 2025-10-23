# Copyright 2025 Google LLC
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

"""Peft Checkpoint manager unittest."""

import os
import tempfile
from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
from flax import config as flax_config
from flax import nnx
import jax
import jax.numpy as jnp
import jax.sharding as shd
import numpy as np
import qwix
from tunix.sft import checkpoint_manager

os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'


if hasattr(flax_config, 'flax_always_shard_variable'):
  flax_config.update('flax_always_shard_variable', False)


def assert_close(path, x, y, atol=1e-5, rtol=1e-5):
  np.testing.assert_allclose(
      x, y, atol, rtol, err_msg=f'Mismatch at path: {path}'
  )


def assert_not_equal(path, x, y):
  np.testing.assert_(
      np.any(np.not_equal(x, y)), msg=f'Unexpected match at path: {path}'
  )


class TestModel(nnx.Module):

  def __init__(self, rngs: nnx.Rngs):
    kernel_init_fn = nnx.initializers.lecun_normal()
    self.w1 = nnx.Linear(
        in_features=2,
        out_features=4,
        rngs=rngs,
        kernel_init=nnx.with_partitioning(kernel_init_fn, ('fsdp', 'tp')),
    )
    self.w2 = nnx.Linear(
        in_features=4,
        out_features=2,
        rngs=rngs,
        kernel_init=nnx.with_partitioning(kernel_init_fn, ('tp', 'fsdp')),
    )

  def __call__(self, x):
    h = nnx.relu(self.w1(x))
    h = self.w2(h) + x
    return h


def create_sharded_model(model_ctor, rngs, mesh):
  @nnx.jit(static_argnums=(0,))
  def _create_sharded_model(model_ctor, rngs):
    model = model_ctor(rngs)
    state = nnx.state(model)
    pspecs = nnx.get_partition_spec(state)
    sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
    nnx.update(model, sharded_state)
    return model, state

  with mesh:
    model, state = _create_sharded_model(model_ctor, rngs)
  state_sharding = nnx.get_named_sharding(state, mesh)
  return model, state_sharding


class CheckpointManagerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    try:
      self.temp_path = self.create_tempdir().full_path
    except Exception:
      self.temp_path = tempfile.TemporaryDirectory().name
    self.device_count = jax.device_count()
    self.mesh = jax.sharding.Mesh(
        devices=np.array(jax.devices()).reshape(2, self.device_count // 2),
        axis_names=('fsdp', 'tp'),
    )

  def test_empty_root_directory(self):
    peft_checkpoint_manager = checkpoint_manager.CheckpointManager(
        root_directory=None
    )
    self.assertIsNone(peft_checkpoint_manager.latest_step())
    self.assertFalse(peft_checkpoint_manager.save(1, None))
    self.assertEqual(peft_checkpoint_manager.maybe_restore(None), (0, {}))

  def test_checkpoint_manager_options_none_sets_default(self):
    peft_checkpoint_manager = checkpoint_manager.CheckpointManager(
        self.temp_path, options=None
    )
    self.assertIsNotNone(peft_checkpoint_manager._checkpoint_manager)
    self.assertEqual(
        peft_checkpoint_manager._checkpoint_manager._options,  # pytype: disable=attribute-error
        checkpoint_manager._DEFAULT_CHECKPOINTING_OPTIONS,
    )

  def test_save(self):
    peft_checkpoint_manager = checkpoint_manager.CheckpointManager(
        self.temp_path
    )
    model, _ = create_sharded_model(TestModel, nnx.Rngs(0), self.mesh)

    # Save the model state.
    self.assertTrue(peft_checkpoint_manager.save(1, model))
    self.assertEqual(peft_checkpoint_manager.latest_step(), 1)

    peft_checkpoint_manager.close()
    model_param_path = epath.Path(self.temp_path) / '1' / 'model_params'
    # Verify the model params are saved.
    self.assertTrue(model_param_path.exists())

  def test_restore(self):
    peft_checkpoint_manager = checkpoint_manager.CheckpointManager(
        self.temp_path
    )
    model, _ = create_sharded_model(TestModel, nnx.Rngs(0), self.mesh)
    expected_state = nnx.state(model)

    # Save the model params.
    self.assertTrue(peft_checkpoint_manager.save(1, model))

    # Change the model state.
    changed_state = jax.tree.map(lambda x: x + 1, nnx.state(model))
    nnx.update(model, changed_state)

    # Restore the model params.
    self.assertEqual(peft_checkpoint_manager.maybe_restore(model), (1, {}))
    # Check the model params are restored correctly.
    jax.tree.map_with_path(
        assert_close,
        expected_state,
        nnx.state(model),
    )

  def test_restore_different_sharding(self):
    peft_checkpoint_manager = checkpoint_manager.CheckpointManager(
        self.temp_path
    )
    unsharded_model = TestModel(nnx.Rngs(0))
    model, _ = create_sharded_model(TestModel, nnx.Rngs(0), self.mesh)

    # Save the model params.
    self.assertTrue(peft_checkpoint_manager.save(1, unsharded_model))

    # Restore the model without shardings.
    self.assertEqual(
        peft_checkpoint_manager.maybe_restore(unsharded_model), (1, {})
    )
    unsharded_variables = nnx.state(unsharded_model, nnx.Param)
    # Check the model shardings are restored correctly.
    self.assertIsInstance(
        unsharded_variables.w1.kernel.value.sharding,
        jax._src.lib.xla_client.SingleDeviceSharding,
    )
    self.assertIsInstance(
        unsharded_variables.w2.kernel.value.sharding,
        jax._src.lib.xla_client.SingleDeviceSharding,
    )

    # Restore the model with shardings.
    self.assertEqual(peft_checkpoint_manager.maybe_restore(model), (1, {}))
    # Check the model shardings are restored correctly.
    variables = nnx.state(model, nnx.Param)

    self.assertEqual(
        variables.w1.kernel.value.sharding.spec,
        shd.PartitionSpec('fsdp', 'tp'),
    )
    self.assertEqual(
        variables.w2.kernel.value.sharding.spec,
        shd.PartitionSpec('tp', 'fsdp'),
    )

  def test_restore_with_lora(self):
    peft_checkpoint_manager = checkpoint_manager.CheckpointManager(
        self.temp_path
    )
    model, _ = create_sharded_model(TestModel, nnx.Rngs(0), self.mesh)
    lora_provider = qwix.LoraProvider(
        module_path='.*w1',
        rank=4,
        alpha=2.0,
    )
    dummy_model_input = {
        'x': jnp.ones(2, dtype=jnp.int32),
    }
    model = qwix.apply_lora_to_model(model, lora_provider, **dummy_model_input)
    expected_lora_state = nnx.clone(nnx.state(model, nnx.LoRAParam))
    old_non_lora_state = nnx.clone(
        nnx.state(model, (nnx.filterlib.Not(nnx.LoRAParam)))
    )

    # Save the model params.
    self.assertTrue(
        peft_checkpoint_manager.save(1, model, save_only_lora_params=True)
    )

    # Change the model state.
    changed_state = jax.tree.map(lambda x: x + 1, nnx.state(model))
    nnx.update(model, changed_state)

    # Restore the model lora params.
    self.assertEqual(
        peft_checkpoint_manager.maybe_restore(
            model, restore_only_lora_params=True
        ),
        (1, {}),
    )
    # Check the model lora params are restored correctly.
    jax.tree.map_with_path(
        assert_close,
        expected_lora_state,
        nnx.state(model, nnx.LoRAParam),
    )
    # Check the rest of the params are not restored.
    jax.tree.map_with_path(
        assert_not_equal,
        old_non_lora_state,
        nnx.state(model, nnx.filterlib.Not(nnx.LoRAParam)),
    )

  def test_save_and_restore_with_custom_metadata(self):
    ckpt_manager = checkpoint_manager.CheckpointManager(self.temp_path)
    model, _ = create_sharded_model(TestModel, nnx.Rngs(0), self.mesh)
    custom_metadata = {'foo': 1, 'bar': 2}
    ckpt_manager.save(1, model, custom_metadata=custom_metadata)
    restored_step, restored_metadata = ckpt_manager.maybe_restore(model)
    self.assertEqual(restored_step, 1)
    self.assertEqual(restored_metadata, custom_metadata)

  @parameterized.parameters(['test_data/checkpoints'])
  def test_restore_with_backward_compatibility(self, ckpt_path):
    # The checkpoints in test_data is saved with StandardSave. The test is to
    # verify the checkpoint manager with PyTreeRestore can still restore the
    # checkpoints saved with StandardSave.
    ckpt_manager = checkpoint_manager.CheckpointManager(
        os.path.join(os.path.dirname(__file__), ckpt_path)
    )
    model, _ = create_sharded_model(TestModel, nnx.Rngs(0), self.mesh)
    expected_state = nnx.state(model)
    # Change the model state.
    changed_state = jax.tree.map(lambda x: x + 1, nnx.state(model))
    nnx.update(model, changed_state)

    # Restore the model params.
    self.assertEqual(ckpt_manager.maybe_restore(model), (1, {}))
    # Check the model params are restored correctly.
    jax.tree.map_with_path(
        assert_close,
        expected_state,
        nnx.state(model),
    )


if __name__ == '__main__':
  absltest.main()
