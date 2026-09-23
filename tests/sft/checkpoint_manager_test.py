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
  if jax.dtypes.issubdtype(getattr(x, 'dtype', None), jax.dtypes.prng_key):
    np.testing.assert_array_equal(
        jax.random.key_data(x),
        jax.random.key_data(y),
        err_msg=f'Mismatch at path: {path}',
    )
  else:
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


class TestModelWithRngs(TestModel):

  def __init__(self, rngs: nnx.Rngs):
    super().__init__(rngs)
    self.dropout = nnx.Dropout(rate=0.1, rngs=rngs)

  def __call__(self, x):
    h = nnx.relu(self.w1(x))
    h = self.dropout(h)
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

  def test_restore_from_separate_directory_and_save_to_new_directory(self):
    restore_dir = f'{self.temp_path}/{self.id()}_restore'
    save_dir = f'{self.temp_path}/{self.id()}_save'
    options = checkpoint_options.checkpointing_options_from_dict({
        'save_interval_steps': 2,
        'max_to_keep': 5,
        'enable_async_checkpointing': False,
    })
    source_manager = checkpoint_manager.CheckpointManager(
        restore_dir, options=options
    )
    model, _ = create_sharded_model(TestModel, nnx.Rngs(0), self.mesh)
    step2_state = jax.tree.map(lambda x: x + 2, nnx.state(model))
    nnx.update(model, step2_state)
    self.assertTrue(
        source_manager.save(2, model, force=True, custom_metadata={'s': 2})
    )
    step4_state = jax.tree.map(lambda x: x + 4, nnx.state(model))
    nnx.update(model, step4_state)
    self.assertTrue(
        source_manager.save(4, model, force=True, custom_metadata={'s': 4})
    )
    source_manager.close()

    # Create a new CheckpointManager writing to `save_dir`, but restore step 2
    # from `restore_dir`. Subsequent saves at steps 2 and 4 go to `save_dir`
    # without colliding with or mutating `restore_dir`.
    target_manager = checkpoint_manager.CheckpointManager(
        save_dir, options=options
    )
    restored_model, _ = create_sharded_model(TestModel, nnx.Rngs(1), self.mesh)
    with self.assertLogs(level='INFO') as logs:
      self.assertEqual(
          target_manager.maybe_restore(
              restored_model, step=2, directory=restore_dir
          ),
          (2, {'s': 2}),
      )
    self.assertTrue(
        any(
            f'Restoring step 2 from restore directory {restore_dir}; future'
            f' checkpoints will be written to root directory {save_dir}.'
            in line
            for line in logs.output
        ),
        logs.output,
    )
    jax.tree.map_with_path(
        assert_close,
        step2_state,
        nnx.state(restored_model),
    )

    new_step4_state = jax.tree.map(lambda x: x + 10, nnx.state(restored_model))
    nnx.update(restored_model, new_step4_state)
    self.assertTrue(
        target_manager.save(
            4, restored_model, force=False, custom_metadata={'s': 4, 'new': 1}
        )
    )
    self.assertEqual(target_manager.latest_step(), 4)
    target_manager.close()

    # Verify original checkpoint in `restore_dir` was untouched.
    verify_source = checkpoint_manager.CheckpointManager(
        restore_dir, options=options
    )
    check_model, _ = create_sharded_model(TestModel, nnx.Rngs(2), self.mesh)
    self.assertEqual(
        verify_source.maybe_restore(check_model, step=4),
        (4, {'s': 4}),
    )
    jax.tree.map_with_path(
        assert_close,
        step4_state,
        nnx.state(check_model),
    )
    verify_source.close()

  def test_restore_prefers_root_directory_after_preemption_restart(self):
    restore_dir = f'{self.temp_path}/{self.id()}_restore'
    save_dir = f'{self.temp_path}/{self.id()}_save'
    options = checkpoint_options.checkpointing_options_from_dict({
        'save_interval_steps': 2,
        'max_to_keep': 5,
        'enable_async_checkpointing': False,
    })
    source_manager = checkpoint_manager.CheckpointManager(
        restore_dir, options=options
    )
    model, _ = create_sharded_model(TestModel, nnx.Rngs(0), self.mesh)
    step2_state = jax.tree.map(lambda x: x + 2, nnx.state(model))
    nnx.update(model, step2_state)
    self.assertTrue(
        source_manager.save(2, model, force=True, custom_metadata={'s': 2})
    )
    source_manager.close()

    # Initial run restores step 2 from `restore_dir` and saves step 6 to
    # `save_dir` before being preempted.
    target_manager = checkpoint_manager.CheckpointManager(
        save_dir, options=options
    )
    run_model, _ = create_sharded_model(TestModel, nnx.Rngs(1), self.mesh)
    self.assertEqual(
        target_manager.maybe_restore(run_model, step=2, directory=restore_dir),
        (2, {'s': 2}),
    )
    step6_state = jax.tree.map(lambda x: x + 6, nnx.state(run_model))
    nnx.update(run_model, step6_state)
    self.assertTrue(
        target_manager.save(
            6, run_model, force=True, custom_metadata={'s': 6, 'resumed': 1}
        )
    )
    target_manager.close()

    # Simulate preemption restart with the same CLI arguments (`step=2`,
    # `directory=restore_dir`): because `save_dir` now has step 6, it must
    # resume from `save_dir` at step 6 rather than re-reading `restore_dir`
    # at step 2.
    restarted_manager = checkpoint_manager.CheckpointManager(
        save_dir, options=options
    )
    restarted_model, _ = create_sharded_model(TestModel, nnx.Rngs(2), self.mesh)
    self.assertEqual(
        restarted_manager.maybe_restore(
            restarted_model, step=2, directory=restore_dir
        ),
        (6, {'s': 6, 'resumed': 1}),
    )
    jax.tree.map_with_path(
        assert_close,
        step6_state,
        nnx.state(restarted_model),
    )
    restarted_manager.close()

  def test_restore_latest_from_directory_into_empty_root_directory(self):
    restore_dir = f'{self.temp_path}/{self.id()}_restore'
    save_dir = f'{self.temp_path}/{self.id()}_save'
    source_manager = checkpoint_manager.CheckpointManager(restore_dir)
    model, _ = create_sharded_model(TestModel, nnx.Rngs(0), self.mesh)
    expected_state = nnx.state(model)
    self.assertTrue(
        source_manager.save(3, model, force=True, custom_metadata={'s': 3})
    )
    assert source_manager._checkpointer is not None
    source_manager._checkpointer.wait()
    source_manager.close()

    target_manager = checkpoint_manager.CheckpointManager(save_dir)
    restored_model, _ = create_sharded_model(TestModel, nnx.Rngs(1), self.mesh)
    with self.assertLogs(level='INFO') as logs:
      self.assertEqual(
          target_manager.maybe_restore(restored_model, directory=restore_dir),
          (3, {'s': 3}),
      )
    self.assertTrue(
        any(
            'Restoring the latest checkpoint from restore directory'
            f' {restore_dir}; future checkpoints will be written to root'
            f' directory {save_dir}.' in line
            for line in logs.output
        ),
        logs.output,
    )
    jax.tree.map_with_path(
        assert_close,
        expected_state,
        nnx.state(restored_model),
    )
    self.assertIsNone(target_manager.latest_step())
    target_manager.close()

  def test_restore_directory_same_as_root_rejects_non_latest_step(self):
    cp_path = f'{self.temp_path}/{self.id()}'
    options = checkpoint_options.checkpointing_options_from_dict({
        'save_interval_steps': 2,
        'max_to_keep': 5,
        'enable_async_checkpointing': False,
    })
    cp_manager = checkpoint_manager.CheckpointManager(cp_path, options=options)
    model, _ = create_sharded_model(TestModel, nnx.Rngs(0), self.mesh)
    self.assertTrue(cp_manager.save(2, model, force=True))
    self.assertTrue(cp_manager.save(4, model, force=True))

    # Checkpoints saved after restoring step 2 would be mixed with step 4.
    with self.assertRaisesRegex(ValueError, 'whose latest step is 4'):
      cp_manager.maybe_restore(model, step=2, directory=cp_path)
    cp_manager.close()

  @parameterized.named_parameters(
      dict(testcase_name='unset_step', step=None),
      dict(testcase_name='latest_step', step=4),
  )
  def test_restore_directory_same_as_root_resumes_from_latest_step(self, step):
    cp_path = f'{self.temp_path}/{self.id()}'
    options = checkpoint_options.checkpointing_options_from_dict({
        'save_interval_steps': 2,
        'max_to_keep': 5,
        'enable_async_checkpointing': False,
    })
    cp_manager = checkpoint_manager.CheckpointManager(cp_path, options=options)
    model, _ = create_sharded_model(TestModel, nnx.Rngs(0), self.mesh)
    self.assertTrue(
        cp_manager.save(2, model, force=True, custom_metadata={'s': 2})
    )
    step4_state = jax.tree.map(lambda x: x + 4, nnx.state(model))
    nnx.update(model, step4_state)
    self.assertTrue(
        cp_manager.save(4, model, force=True, custom_metadata={'s': 4})
    )

    restored_model, _ = create_sharded_model(TestModel, nnx.Rngs(1), self.mesh)
    with self.assertLogs(level='INFO') as logs:
      # A trailing slash still refers to the root directory.
      self.assertEqual(
          cp_manager.maybe_restore(
              restored_model, step=step, directory=f'{cp_path}/'
          ),
          (4, {'s': 4}),
      )
    self.assertTrue(
        any(
            'is the same as the root directory; resuming from its latest step'
            ' (4).' in line
            for line in logs.output
        ),
        logs.output,
    )
    jax.tree.map_with_path(
        assert_close,
        step4_state,
        nnx.state(restored_model),
    )
    cp_manager.close()

  def test_restore_from_directory_when_root_directory_is_none(self):
    restore_dir = f'{self.temp_path}/{self.id()}_restore'
    source_manager = checkpoint_manager.CheckpointManager(restore_dir)
    model, _ = create_sharded_model(TestModel, nnx.Rngs(0), self.mesh)
    expected_state = nnx.state(model)
    self.assertTrue(
        source_manager.save(3, model, force=True, custom_metadata={'s': 3})
    )
    assert source_manager._checkpointer is not None
    source_manager._checkpointer.wait()
    source_manager.close()

    cp_manager = checkpoint_manager.CheckpointManager(root_directory=None)
    restored_model, _ = create_sharded_model(TestModel, nnx.Rngs(1), self.mesh)
    with self.assertLogs(level='INFO') as logs:
      self.assertEqual(
          cp_manager.maybe_restore(restored_model, directory=restore_dir),
          (3, {'s': 3}),
      )
    self.assertTrue(
        any(
            'Restoring the latest checkpoint from restore directory'
            f' {restore_dir}; no root directory is configured, so checkpoint'
            ' saving is disabled.' in line
            for line in logs.output
        ),
        logs.output,
    )
    jax.tree.map_with_path(
        assert_close,
        expected_state,
        nnx.state(restored_model),
    )

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

  def test_restore_with_pathways_persistence(self):
    with mock.patch.dict(
        os.environ,
        {'JAX_PLATFORMS': 'proxy', 'ENABLE_PATHWAYS_PERSISTENCE': '1'},
    ):
      cp_path = f'{self.temp_path}/{self.id()}'
      cp_manager = checkpoint_manager.CheckpointManager(cp_path)
      model, _ = create_sharded_model(TestModelWithRngs, nnx.Rngs(0), self.mesh)
      optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)

      # Verify model state contains PRNG key states (key<fry>).
      prng_key_leaves = [
          x
          for x in jax.tree.leaves(nnx.state(model))
          if jax.dtypes.issubdtype(x.dtype, jax.dtypes.prng_key)
      ]
      self.assertNotEmpty(prng_key_leaves)

      # Save checkpoint at step 25.
      self.assertTrue(
          cp_manager.save(25, model, optimizer=optimizer, force=True)
      )
      assert cp_manager._checkpointer is not None
      cp_manager._checkpointer.wait()

      # Mutate state before saving step 50.
      changed_model_state = jax.tree.map(
          lambda x: x
          if jax.dtypes.issubdtype(x.dtype, jax.dtypes.prng_key)
          else x + 1,
          nnx.state(model),
      )
      nnx.update(model, changed_model_state)

      # Save checkpoint at step 50.
      expected_opt_state = nnx.state(optimizer, nnx.optimizer.OptState)
      self.assertTrue(
          cp_manager.save(50, model, optimizer=optimizer, force=True)
      )
      cp_manager._checkpointer.wait()
      cp_manager.close()

      # Simulate workload restart by creating a new CheckpointManager instance.
      restart_cp_manager = checkpoint_manager.CheckpointManager(cp_path)
      restore_model, _ = create_sharded_model(
          TestModelWithRngs, nnx.Rngs(1), self.mesh
      )
      restore_optimizer = nnx.Optimizer(
          restore_model, optax.adam(1e-3), wrt=nnx.Param
      )

      # Restore latest step (50) automatically upon restart.
      restored_step, _ = restart_cp_manager.maybe_restore(
          restore_model, optimizer=restore_optimizer
      )
      self.assertEqual(restored_step, 50)

      # Verify model and optimizer states match expected step 50 state.
      jax.tree.map_with_path(
          assert_close,
          changed_model_state,
          nnx.state(restore_model),
      )
      jax.tree.map_with_path(
          assert_close,
          expected_opt_state,
          nnx.state(restore_optimizer, nnx.optimizer.OptState),
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
      error_msg = '\n'.join(errors)
      self.fail(f'Found sharding mismatches:\n{error_msg}')

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

  def test_convert_host_local_scalar_array(self):
    scalar = jax.device_put(jnp.array(42), jax.devices()[0])
    self.assertEqual(len(scalar.devices()), 1)
    self.assertEqual(scalar.shape, ())
    converted = checkpoint_manager._convert_host_local_array(scalar)
    self.assertIsInstance(converted.sharding, jax.sharding.NamedSharding)
    self.assertEqual(int(converted), 42)

  def test_convert_host_local_1d_array(self):
    arr = jax.device_put(jnp.array([1.0, 2.0, 3.0]), jax.devices()[0])
    self.assertEqual(len(arr.devices()), 1)
    converted = checkpoint_manager._convert_host_local_array(arr)
    self.assertIsInstance(converted, jax.Array)
    np.testing.assert_allclose(converted, np.array([1.0, 2.0, 3.0]))

  def test_unknown_partition_spec_axis_is_replicated_for_restore(self):
    pspec = shd.PartitionSpec('norm')
    fixed_pspec = checkpoint_manager._replicate_if_pspec_uses_unknown_mesh_axis(
        pspec, self.mesh
    )
    self.assertEqual(fixed_pspec, shd.PartitionSpec())

  def test_save_with_host_local_optimizer_state(self):
    cp_path = f'{self.temp_path}/{self.id()}'
    cp_manager = checkpoint_manager.CheckpointManager(cp_path)
    model, _ = create_sharded_model(TestModel, nnx.Rngs(0), self.mesh)
    optimizer = nnx.Optimizer(
        model,
        optax.inject_hyperparams(optax.adamw)(learning_rate=1e-3),
        wrt=nnx.Param,
    )
    self.assertTrue(cp_manager.save(1, model, optimizer=optimizer))
    assert cp_manager._checkpointer is not None
    cp_manager._checkpointer.wait()
    self.assertEqual(cp_manager.latest_step(), 1)


if __name__ == '__main__':
  absltest.main()
