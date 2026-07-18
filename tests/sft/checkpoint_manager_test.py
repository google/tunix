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
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
from flax import config as flax_config
from flax import nnx
import jax
import jax.numpy as jnp
import jax.sharding as shd
import numpy as np
import optax
import qwix
from tunix.sft import checkpoint_manager
from tunix.sft import checkpoint_options

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
    cp_manager = checkpoint_manager.CheckpointManager(root_directory=None)
    self.assertIsNone(cp_manager.latest_step())
    self.assertFalse(cp_manager.save(1, None))  # pyrefly: ignore[bad-argument-type]
    self.assertEqual(cp_manager.maybe_restore(None), (0, {}))  # pyrefly: ignore[bad-argument-type]

  def test_checkpoint_manager_options_none_sets_default(self):
    cp_path = f'{self.temp_path}/{self.id()}'
    cp_manager = checkpoint_manager.CheckpointManager(cp_path, options=None)
    self.assertIsNotNone(cp_manager._checkpointer)
    self.assertEqual(
        cp_manager._options,
        checkpoint_options.DEFAULT_CHECKPOINTING_OPTIONS,
    )

  def test_context_property(self):
    cp_path = f'{self.temp_path}/{self.id()}'
    cp_manager = checkpoint_manager.CheckpointManager(cp_path)
    self.assertIsNotNone(cp_manager._context)

  def test_context_property_with_pathways(self):
    with mock.patch.dict(os.environ, {'JAX_PLATFORMS': 'proxy'}):
      cp_path = f'{self.temp_path}/{self.id()}'
      cp_manager = checkpoint_manager.CheckpointManager(cp_path)
      self.assertIsNotNone(cp_manager._context)
      self.assertFalse(cp_manager._context.array_options.saving.use_ocdbt)
      self.assertFalse(cp_manager._context.array_options.saving.use_zarr3)

  def test_save(self):
    cp_path = f'{self.temp_path}/{self.id()}'
    cp_manager = checkpoint_manager.CheckpointManager(cp_path)
    model, _ = create_sharded_model(TestModel, nnx.Rngs(0), self.mesh)

    # Save the model state.
    self.assertTrue(cp_manager.save(1, model))
    assert cp_manager._checkpointer is not None
    cp_manager._checkpointer.wait()
    self.assertEqual(cp_manager.latest_step(), 1)

    cp_manager.close()
    model_param_path = epath.Path(cp_path) / '1' / 'model_params'
    # Verify the model params are saved.
    self.assertTrue(model_param_path.exists())

  def test_restore(self):
    cp_path = f'{self.temp_path}/{self.id()}'
    cp_manager = checkpoint_manager.CheckpointManager(cp_path)
    model, _ = create_sharded_model(TestModel, nnx.Rngs(0), self.mesh)
    expected_state = nnx.state(model)

    # Save the model params.
    self.assertTrue(cp_manager.save(1, model))
    assert cp_manager._checkpointer is not None
    cp_manager._checkpointer.wait()

    # Change the model state.
    changed_state = jax.tree.map(lambda x: x + 1, nnx.state(model))
    nnx.update(model, changed_state)

    # Restore the model params.
    self.assertEqual(cp_manager.maybe_restore(model), (1, {}))
    # Check the model params are restored correctly.
    jax.tree.map_with_path(
        assert_close,
        expected_state,
        nnx.state(model),
    )

  def test_restore_different_sharding(self):
    cp_path = f'{self.temp_path}/{self.id()}'
    cp_manager = checkpoint_manager.CheckpointManager(cp_path)
    unsharded_model = TestModel(nnx.Rngs(0))
    model, _ = create_sharded_model(TestModel, nnx.Rngs(0), self.mesh)

    # Save the model params.
    self.assertTrue(cp_manager.save(1, unsharded_model))
    assert cp_manager._checkpointer is not None
    cp_manager._checkpointer.wait()

    # Restore the model without shardings.
    self.assertEqual(cp_manager.maybe_restore(unsharded_model), (1, {}))
    unsharded_variables = nnx.state(unsharded_model, nnx.Param)
    # Check the model shardings are restored correctly.
    self.assertIsInstance(
        unsharded_variables.w1.kernel.value.sharding,
        jax.sharding.SingleDeviceSharding,
    )
    self.assertIsInstance(
        unsharded_variables.w2.kernel.value.sharding,
        jax.sharding.SingleDeviceSharding,
    )

    # Restore the model with shardings.
    self.assertEqual(cp_manager.maybe_restore(model), (1, {}))
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
    cp_path = f'{self.temp_path}/{self.id()}'
    cp_manager = checkpoint_manager.CheckpointManager(cp_path)
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
    self.assertTrue(cp_manager.save(1, model, save_only_lora_params=True))  # pyrefly: ignore[bad-argument-type]
    assert cp_manager._checkpointer is not None
    cp_manager._checkpointer.wait()

    # Change the model state.
    changed_state = jax.tree.map(lambda x: x + 1, nnx.state(model))
    nnx.update(model, changed_state)

    # Restore the model lora params.
    self.assertEqual(
        cp_manager.maybe_restore(model, restore_only_lora_params=True),  # pyrefly: ignore[bad-argument-type]
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

  def test_restore_only_lora_params(self):
    cp_path = f'{self.temp_path}/{self.id()}'
    cp_manager = checkpoint_manager.CheckpointManager(cp_path)
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
    changed_non_lora_state = jax.tree.map(
        lambda x: x + 2, nnx.state(model, (nnx.filterlib.Not(nnx.LoRAParam)))
    )

    # Save the model params (entire model).
    self.assertTrue(cp_manager.save(1, model, save_only_lora_params=False))  # pyrefly: ignore[bad-argument-type]
    assert cp_manager._checkpointer is not None
    cp_manager._checkpointer.wait()

    # Change the model state.
    nnx.update(
        model, jax.tree.map(lambda x: x + 1, nnx.state(model, nnx.LoRAParam))
    )
    nnx.update(model, changed_non_lora_state)

    # Restore only the model lora params.
    self.assertEqual(
        cp_manager.maybe_restore(model, restore_only_lora_params=True),  # pyrefly: ignore[bad-argument-type]
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
        assert_close,
        changed_non_lora_state,
        nnx.state(model, nnx.filterlib.Not(nnx.LoRAParam)),
    )

  def test_restore_full_from_lora_only_checkpoint_fails(self):
    cp_path = f'{self.temp_path}/{self.id()}'
    cp_manager = checkpoint_manager.CheckpointManager(cp_path)
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

    # Save only the lora params.
    self.assertTrue(cp_manager.save(1, model, save_only_lora_params=True))  # pyrefly: ignore[bad-argument-type]
    assert cp_manager._checkpointer is not None
    cp_manager._checkpointer.wait()

    # Try to restore full model, expect failure.
    with self.assertRaisesRegex(
        ValueError, 'If this checkpoint only contains LoRA parameters'
    ):
      cp_manager.maybe_restore(model, restore_only_lora_params=False)  # pyrefly: ignore[bad-argument-type]

  def test_save_and_restore_with_custom_metadata(self):
    cp_path = f'{self.temp_path}/{self.id()}'
    ckpt_manager = checkpoint_manager.CheckpointManager(cp_path)
    model, _ = create_sharded_model(TestModel, nnx.Rngs(0), self.mesh)
    custom_metadata = {'foo': 1, 'bar': 2}
    ckpt_manager.save(1, model, custom_metadata=custom_metadata)
    assert ckpt_manager._checkpointer is not None
    ckpt_manager._checkpointer.wait()
    restored_step, restored_metadata = ckpt_manager.maybe_restore(model)
    self.assertEqual(restored_step, 1)
    self.assertEqual(restored_metadata, custom_metadata)

  def test_save_and_restore_with_optimizer_state(self):
    cp_path = f'{self.temp_path}/{self.id()}'
    ckpt_manager = checkpoint_manager.CheckpointManager(cp_path)
    model, _ = create_sharded_model(TestModel, nnx.Rngs(0), self.mesh)
    optimizer = nnx.Optimizer(
        model,
        optax.inject_hyperparams(optax.adamw)(learning_rate=1e-3),
        wrt=nnx.Param,
    )
    custom_metadata = {'foo': 1, 'bar': 2}
    ckpt_manager.save(1, model, optimizer, custom_metadata=custom_metadata)
    assert ckpt_manager._checkpointer is not None
    ckpt_manager._checkpointer.wait()

    new_optimizer = nnx.Optimizer(
        model,
        optax.inject_hyperparams(optax.adamw)(learning_rate=1e-5),
        wrt=nnx.Param,
    )
    self.assertEqual(
        new_optimizer.opt_state.hyperparams['learning_rate'].value, 1e-5
    )
    restored_step, restored_metadata = ckpt_manager.maybe_restore(
        model, new_optimizer
    )
    self.assertEqual(restored_step, 1)
    self.assertEqual(restored_metadata, custom_metadata)
    jax.tree.map_with_path(
        assert_close,
        nnx.state(new_optimizer, nnx.optimizer.OptState),
        nnx.state(optimizer, nnx.optimizer.OptState),
    )
    self.assertEqual(
        new_optimizer.opt_state.hyperparams['learning_rate'].value, 1e-3
    )

  def test_save_and_restore_with_forced_single_device_sharding(self):
    cp_path = f'{self.temp_path}/{self.id()}'
    ckpt_manager = checkpoint_manager.CheckpointManager(cp_path)
    model, _ = create_sharded_model(TestModel, nnx.Rngs(0), self.mesh)
    optimizer = nnx.Optimizer(
        model,
        optax.inject_hyperparams(optax.adamw)(learning_rate=1e-3),
        wrt=nnx.Param,
    )
    custom_metadata = {'foo': 1, 'bar': 2}
    ckpt_manager.save(1, model, optimizer, custom_metadata=custom_metadata)
    assert ckpt_manager._checkpointer is not None
    ckpt_manager._checkpointer.wait()

    new_optimizer = nnx.Optimizer(
        model,
        optax.inject_hyperparams(optax.adamw)(learning_rate=1e-5),
        wrt=nnx.Param,
    )

    new_optimizer.opt_state.hyperparams['learning_rate'].value = jax.device_put(
        new_optimizer.opt_state.hyperparams['learning_rate'].value,
        jax.devices()[0],
    )

    self.assertIsInstance(
        new_optimizer.opt_state.hyperparams['learning_rate'].value.sharding,
        jax.sharding.SingleDeviceSharding,
    )

    restored_step, _ = ckpt_manager.maybe_restore(
        model, new_optimizer
    )
    self.assertEqual(restored_step, 1)

    errors = []
    def assert_named_sharding(path, x):
      if hasattr(x, 'sharding'):
        try:
          self.assertIsInstance(
              x.sharding,
              jax.sharding.NamedSharding,
              f'Variable at {path} is not NamedSharding',
          )
        except AssertionError as e:
          errors.append(str(e))
          return

        path_str = str(path)
        if 'hyperparams' in path_str:
          try:
            self.assertEqual(x.sharding.spec, jax.sharding.PartitionSpec())
          except AssertionError as e:
            errors.append(str(e))

    jax.tree.map_with_path(
        assert_named_sharding,
        nnx.state(new_optimizer, nnx.optimizer.OptState),
    )
    if errors:
      self.fail(f'Found sharding mismatches:\n{"\n".join(errors)}')

  def test_restore_without_optimizer(self):
    cp_path = f'{self.temp_path}/{self.id()}'
    ckpt_manager = checkpoint_manager.CheckpointManager(cp_path)
    model, _ = create_sharded_model(TestModel, nnx.Rngs(0), self.mesh)
    optimizer = nnx.Optimizer(
        model,
        optax.inject_hyperparams(optax.adamw)(learning_rate=1e-3),
        wrt=nnx.Param,
    )
    ckpt_manager.save(1, model, optimizer)
    assert ckpt_manager._checkpointer is not None
    ckpt_manager._checkpointer.wait()
    ckpt_manager.maybe_restore(model)

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

  @parameterized.parameters(True, False)
  def test_save_aligns_with_policy(self, enable_async):
    cp_path = f'{self.temp_path}/{self.id()}_{enable_async}'
    options = checkpoint_options.TunixCheckpointingOptions(
        save_decision_policy=(
            checkpoint_manager.ocp.training.save_decision_policies.FixedIntervalPolicy(
                2
            )
        ),
        enable_async_checkpointing=enable_async,
    )
    cp_manager = checkpoint_manager.CheckpointManager(cp_path, options=options)
    model, _ = create_sharded_model(TestModel, nnx.Rngs(0), self.mesh)

    # Step 1 should be skipped by FixedIntervalPolicy(2).
    self.assertFalse(cp_manager.save(1, model))

    # Step 2 should be saved.
    self.assertTrue(cp_manager.save(2, model))
    assert cp_manager._checkpointer is not None
    cp_manager._checkpointer.wait()
    self.assertEqual(cp_manager.latest_step(), 2)

  def test_save_force_true_overrides_policy(self):
    cp_path = f'{self.temp_path}/{self.id()}'
    options = checkpoint_options.TunixCheckpointingOptions(
        save_decision_policy=(
            checkpoint_manager.ocp.training.save_decision_policies.FixedIntervalPolicy(
                2
            )
        ),
        enable_async_checkpointing=True,
    )
    cp_manager = checkpoint_manager.CheckpointManager(cp_path, options=options)
    model, _ = create_sharded_model(TestModel, nnx.Rngs(0), self.mesh)

    # Step 1 would normally be skipped by FixedIntervalPolicy(2), but force=True
    # should force the save.
    self.assertTrue(cp_manager.save(1, model, force=True))
    assert cp_manager._checkpointer is not None
    cp_manager._checkpointer.wait()
    self.assertEqual(cp_manager.latest_step(), 1)

  def test_context_timeout_secs(self):
    options = checkpoint_options.TunixCheckpointingOptions(
        async_options=checkpoint_manager.ocp.options.AsyncOptions(
            timeout_secs=42
        )
    )
    cp_manager = checkpoint_manager.CheckpointManager(
        self.temp_path, options=options
    )
    self.assertEqual(cp_manager._context.asynchronous.timeout_secs, 42)

  @parameterized.parameters(True, False)
  def test_checkpointing_method_selection(self, enable_async):
    cp_path = f'{self.temp_path}/{self.id()}_{enable_async}'
    options = checkpoint_options.TunixCheckpointingOptions(
        enable_async_checkpointing=enable_async,
    )
    cp_manager = checkpoint_manager.CheckpointManager(cp_path, options=options)
    model, _ = create_sharded_model(TestModel, nnx.Rngs(0), self.mesh)

    with mock.patch.object(
        cp_manager._checkpointer,
        'save_checkpointables_async',
        return_value=mock.MagicMock(),
    ) as mock_async, mock.patch.object(
        cp_manager._checkpointer, 'save_checkpointables', return_value=True
    ) as mock_sync:
      cp_manager.save(1, model)
      if enable_async:
        mock_async.assert_called_once()
        mock_sync.assert_not_called()
      else:
        mock_sync.assert_called_once()
        mock_async.assert_not_called()


if __name__ == '__main__':
  absltest.main()
