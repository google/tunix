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

"""Peft trainer unittest."""

import contextlib
import functools
import os
import tempfile
from typing import Any, Tuple
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
import chex
from flax import nnx
import jax
import jax.numpy as jnp
import jax.sharding as shd
import numpy as np
import optax
from tunix.rl import common
from tunix.sft import checkpoint_manager
from tunix.sft import checkpoint_options
from tunix.sft import hooks
from tunix.sft import peft_trainer
from tunix.sft import profiler
from tunix.sft import utils
from tunix.tests import test_common as tc
from tunix.utils import compat

TEST_LEARNING_RATE = 1e-3

# CPU environment setup to simulate multi device env.
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'

# Set Precision to highest for numeric stability across different hardware.
jax.config.update('jax_default_matmul_precision', 'highest')

def create_sharded_model(model_ctor, rngs, mesh):
  @nnx.jit(static_argnums=(0,))
  def _create_sharded_model(model_ctor, rngs):
    model = model_ctor(config=tc.ModelConfig(), rngs=rngs)
    state = nnx.state(model)
    pspecs = nnx.get_partition_spec(state)
    sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
    nnx.update(model, sharded_state)
    return model, state

  with compat.set_mesh(mesh):
    model, state = _create_sharded_model(model_ctor, rngs)
  state_sharding = nnx.get_named_sharding(state, mesh)
  return model, state_sharding


def dummy_gen_model_input_fn(x: peft_trainer.TrainingInput):
  return {
      'input_tokens': x.input_tokens,
      'input_mask': x.input_mask,
      'positions': jnp.arange(x.input_tokens.shape[1]),
      'attention_mask': jnp.ones_like(x.input_tokens),
  }


def dummy_datasets(batch_size: int, repeat: int = 1):
  # (num_batch, batch_size, seq_len)
  dummy_input = np.arange(128).reshape((-1, batch_size, 16))
  return [
      peft_trainer.TrainingInput(
          input_tokens=x, input_mask=jnp.ones(x.shape, dtype=jnp.int32)
      )
      for x in dummy_input
  ] * repeat


global_counter = 0


class PeftTrainerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    try:
      self.temp_path = self.create_tempdir().full_path
    except Exception:
      self.temp_path = tempfile.TemporaryDirectory().name

    # CPU env setup to simulate multi device env. Won't affect TPU env. But
    # need to be careful not to use self.num_cpus in TPU env.
    self.num_cpus = 4
    chex.set_n_cpu_devices(self.num_cpus)

    self.eval_ds = self.train_ds = dummy_datasets(batch_size=4)
    total_devices = jax.device_count()
    self.mesh = shd.Mesh(
        devices=np.array(jax.devices()).reshape(2, total_devices // 2),
        axis_names=('fsdp', 'tp'),
    )

    self.eval_ds = self.train_ds = dummy_datasets(batch_size=4)

  def test_compile_once(self):
    class CountCompiledTimesTrainer(peft_trainer.PeftTrainer):

      def _train_step(
          self, model, optimizer, grad_accumulator, inputs, is_update_step
      ):
        global global_counter
        global_counter += 1
        return super()._train_step(
            model, optimizer, grad_accumulator, inputs, is_update_step
        )

    config = peft_trainer.TrainingConfig(eval_every_n_steps=2, max_steps=100)
    rngs = nnx.Rngs(0)
    model = tc.get_lora_model(
        tc.ToyTransformer(config=tc.ModelConfig(), rngs=rngs), mesh=self.mesh
    )
    trainer = CountCompiledTimesTrainer(model, optax.sgd(1e-3), config)
    trainer = trainer.with_gen_model_input_fn(dummy_gen_model_input_fn)
    global global_counter
    global_counter = 0  # make mypy happy
    with self.mesh:
      trainer.train(self.train_ds, self.eval_ds)
    self.assertEqual(global_counter, 1)

  @parameterized.named_parameters(
      ('cache_nnx_graph', True),
      ('no_cache_nnx_graph', False),
  )
  def test_basic_training(self, cache_nnx_graph: bool):
    config = peft_trainer.TrainingConfig(eval_every_n_steps=2, max_steps=100)
    rngs = nnx.Rngs(0)
    model = tc.ToyTransformer(config=tc.ModelConfig(), rngs=rngs)
    original_variables = jax.tree.map(jnp.copy, nnx.state(model, nnx.Param))
    optimizer = optax.inject_hyperparams(optax.sgd)(
        learning_rate=optax.constant_schedule(TEST_LEARNING_RATE)
    )
    trainer = peft_trainer.PeftTrainer(model, optimizer, config)
    trainer = trainer.with_gen_model_input_fn(dummy_gen_model_input_fn)

    trainer.train(self.train_ds, self.eval_ds, cache_nnx_graph=cache_nnx_graph)
    variables = nnx.state(model, nnx.Param)

    jax.tree.map_with_path(tc.assert_not_equal, original_variables, variables)

    self.assertGreater(
        trainer.metrics_logger.get_metric('', 'perplexity', 'train'), 0
    )
    self.assertEqual(
        trainer.metrics_logger.get_metric('', 'learning_rate', 'train'),
        TEST_LEARNING_RATE,
    )
    self.assertGreater(
        trainer.metrics_logger.get_metric('', 'perplexity', 'eval'), 0
    )
    self.assertGreater(trainer._train_steps, 0)

    self.assertLen(
        trainer.metrics_logger.get_metric_history('', 'perplexity', 'train'),
        trainer._train_steps,
    )

    trainer.train(self.train_ds)  # No eval dataset.

  @parameterized.named_parameters(
      ('lora_disabled_distributed', False, True),
      ('lora_disabled_single_device', False, False),
      ('lora_enabled_distributed', True, True),
      ('lora_enabled_single_device', True, False),
  )
  def test_checkpoint_save_and_restore(
      self, enable_lora: bool, distributed: bool
  ):
    def create_model_and_optimizer():
      rngs = nnx.Rngs(0)
      if distributed:
        model, _ = create_sharded_model(tc.ToyTransformer, rngs, self.mesh)
      else:
        model = tc.ToyTransformer(config=tc.ModelConfig(), rngs=rngs)
      if enable_lora:
        model = tc.get_lora_model(model)

      optimizer = optax.inject_hyperparams(optax.adamw)(
          learning_rate=optax.constant_schedule(TEST_LEARNING_RATE)
      )
      return model, optimizer

    config = peft_trainer.TrainingConfig(
        eval_every_n_steps=2,
        max_steps=100,
        checkpoint_root_directory=f'{self.temp_path}/{self.id()}/checkpoints',
    )

    model, optimizer = create_model_and_optimizer()
    original_model_state = jax.tree.map(
        jnp.copy, nnx.state(model, nnx.LoRAParam if enable_lora else nnx.Param)
    )

    trainer = peft_trainer.PeftTrainer(model, optimizer, config)
    trainer = trainer.with_gen_model_input_fn(dummy_gen_model_input_fn)

    ctx = self.mesh if distributed else contextlib.nullcontext()

    with ctx:
      trainer.train(self.train_ds, self.eval_ds, cache_nnx_graph=True)
    trained_model_state = nnx.state(
        model, nnx.LoRAParam if enable_lora else nnx.Param
    )
    trained_opt_state = nnx.state(trainer.optimizer, nnx.optimizer.OptState)

    jax.tree.map_with_path(
        tc.assert_not_equal, original_model_state, trained_model_state
    )

    # Resume from checkpoint with a new model and optimizer, and check that
    # the model and optimizer states are the same as the trained ones.
    new_model, new_optimizer = create_model_and_optimizer()

    resumed_trainer = peft_trainer.PeftTrainer(new_model, new_optimizer, config)
    resumed_model_state = nnx.state(
        resumed_trainer.model, nnx.LoRAParam if enable_lora else nnx.Param
    )
    resumed_opt_state = nnx.state(
        resumed_trainer.optimizer, nnx.optimizer.OptState
    )

    jax.tree.map_with_path(
        tc.assert_equal, trained_model_state, resumed_model_state
    )
    jax.tree.map_with_path(
        tc.assert_equal, trained_opt_state, resumed_opt_state
    )

    resumed_trainer = resumed_trainer.with_gen_model_input_fn(
        dummy_gen_model_input_fn
    )
    with ctx:
      resumed_trainer.train(self.train_ds, self.eval_ds, cache_nnx_graph=True)

    resumed_opt_state = nnx.state(
        resumed_trainer.optimizer, nnx.optimizer.OptState
    )

    jax.tree.map(
        lambda x, y: self.assertTrue(
            x.sharding.is_equivalent_to(y.sharding, ndim=x.ndim)
        ),
        trained_opt_state,
        resumed_opt_state,
    )

  def test_basic_training_with_hooks(self):
    train_ds = dummy_datasets(batch_size=4, repeat=2)
    config = peft_trainer.TrainingConfig(eval_every_n_steps=2, max_steps=100)
    rngs = nnx.Rngs(0)
    model = tc.ToyTransformer(config=tc.ModelConfig(), rngs=rngs)

    mock_training_hooks_instance = mock.create_autospec(hooks.TrainingHooks)
    trainer = peft_trainer.PeftTrainer(
        model,
        optax.sgd(1e-3),
        config,
    )
    trainer.with_training_hooks(mock_training_hooks_instance)
    trainer = trainer.with_gen_model_input_fn(dummy_gen_model_input_fn)
    trainer.train(train_ds, self.eval_ds)

    expected_training_hooks_calls = (
        [mock.call.on_train_start(trainer)]
        + [mock.call.on_train_step_start(trainer) for _ in range(4)]
        + [
            mock.call.on_train_step_end(trainer, mock.ANY, mock.ANY)
            for _ in range(4)
        ]
        + [mock.call.on_eval_step_start(trainer) for _ in range(4)]
        + [mock.call.on_eval_step_end(trainer, mock.ANY) for _ in range(2)]
        + [mock.call.on_train_end(trainer)]
    )
    mock_training_hooks_instance.assert_has_calls(
        expected_training_hooks_calls,
        any_order=True,
    )

  def test_reusing_trainer(self):
    config = peft_trainer.TrainingConfig(eval_every_n_steps=2, max_steps=100)
    rngs = nnx.Rngs(0)
    model = tc.ToyTransformer(config=tc.ModelConfig(), rngs=rngs)

    trainer = peft_trainer.PeftTrainer(model, optax.sgd(1e-3), config)
    trainer = trainer.with_gen_model_input_fn(dummy_gen_model_input_fn)
    trainer.train(self.train_ds, None)

    previous_jit_func = trainer._jitted_train_step_fn
    self.assertIsNotNone(previous_jit_func)

    trainer = trainer.with_gen_model_input_fn(dummy_gen_model_input_fn)
    trainer.train(self.train_ds, None)
    curr_jit_func = trainer._jitted_train_step_fn
    self.assertIsNotNone(curr_jit_func)
    self.assertIsNot(previous_jit_func, curr_jit_func)

  @mock.patch.object(profiler, 'Profiler')
  def test_basic_training_with_profiler(self, mock_profiler_init):
    mock_profiler_instance = mock.MagicMock()
    mock_profiler_init.return_value = mock_profiler_instance
    mock_profiler_instance.should_activate.side_effect = (
        lambda step: step == profiler_options.skip_first_n_steps
    )
    mock_profiler_instance.should_deactivate.side_effect = (
        lambda step: step
        == (
            profiler_options.skip_first_n_steps
            + profiler_options.profiler_steps
        )
    )
    profiler_options = profiler.ProfilerOptions(
        '/tmp/profiler', skip_first_n_steps=2, profiler_steps=3
    )
    config = peft_trainer.TrainingConfig(
        eval_every_n_steps=2, max_steps=100, profiler_options=profiler_options
    )
    rngs = nnx.Rngs(0)
    model = tc.ToyTransformer(config=tc.ModelConfig(), rngs=rngs)

    trainer = peft_trainer.PeftTrainer(model, optax.sgd(1e-3), config)
    trainer = trainer.with_gen_model_input_fn(dummy_gen_model_input_fn)

    train_ds = dummy_datasets(batch_size=4, repeat=4)
    trainer.train(train_ds)  # No eval dataset.

    mock_profiler_init.assert_called_once_with(
        initial_step=0,
        max_step=config.max_steps,
        profiler_options=profiler_options,
    )
    expected_calls = (
        # steps 0 through 8.
        [mock.call.maybe_activate(step) for step in range(8)]
        # steps 1 through 9 as step number is incremented during each step.
        + [mock.call.maybe_deactivate(step) for step in range(1, 9)]
    )
    mock_profiler_instance.assert_has_calls(
        expected_calls,
        any_order=True,
    )

  def test_dist_training(self):
    rngs = nnx.Rngs(0)
    model, _ = create_sharded_model(tc.ToyTransformer, rngs, self.mesh)
    original_variables = jax.tree.map(jnp.copy, nnx.state(model, nnx.Param))

    config = peft_trainer.TrainingConfig(eval_every_n_steps=2, max_steps=100)
    trainer = peft_trainer.PeftTrainer(model, optax.sgd(1e-3), config)
    trainer = trainer.with_gen_model_input_fn(dummy_gen_model_input_fn)

    with self.mesh:
      trainer.train(self.train_ds, self.eval_ds)
    variables = nnx.state(model, nnx.Param)

    self.assertEqual(
        variables.layers[0].w1.kernel.value.sharding.spec,
        shd.PartitionSpec('fsdp', 'tp'),
    )
    self.assertEqual(
        variables.layers[0].w2.kernel.value.sharding.spec,
        shd.PartitionSpec('tp', 'fsdp'),
    )

    jax.tree.map_with_path(tc.assert_not_equal, original_variables, variables)

    # compare with unsharded model
    rngs = nnx.Rngs(0)
    unsharded_model = tc.ToyTransformer(config=tc.ModelConfig(), rngs=rngs)
    trainer = peft_trainer.PeftTrainer(unsharded_model, optax.sgd(1e-3), config)
    trainer = trainer.with_gen_model_input_fn(dummy_gen_model_input_fn)
    trainer.train(self.train_ds, self.eval_ds)
    unsharded_variables = nnx.state(unsharded_model, nnx.Param)
    self.assertIsInstance(
        unsharded_variables.layers[0].w1.kernel.value.sharding,
        jax.sharding.SingleDeviceSharding,
    )
    jax.tree.map_with_path(tc.assert_close, variables, unsharded_variables)

  def test_custom_loss_fn(self):
    def custom_loss_fn(
        model: nnx.Module,
        input_tokens: jax.Array,
        input_mask: jax.Array,
        positions: jax.Array,
        attention_mask: jax.Array,
    ) -> jax.Array:
      logits, _ = model(input_tokens, positions, None, attention_mask)
      logits = logits[:, :-1, :]
      target_tokens = input_tokens[:, 1:]
      target_mask = input_mask[:, 1:]
      one_hot = jax.nn.one_hot(target_tokens, logits.shape[-1])
      one_hot = one_hot * target_mask.astype(one_hot.dtype)[..., None]
      return optax.softmax_cross_entropy(logits, one_hot).mean()

    config = peft_trainer.TrainingConfig(eval_every_n_steps=2, max_steps=100)
    rngs = nnx.Rngs(0)
    model = tc.ToyTransformer(config=tc.ModelConfig(), rngs=rngs)
    original_variables = jax.tree.map(jnp.copy, nnx.state(model, nnx.Param))

    trainer = peft_trainer.PeftTrainer(model, optax.sgd(1e-3), config)
    trainer = trainer.with_gen_model_input_fn(
        dummy_gen_model_input_fn
    ).with_loss_fn(custom_loss_fn)
    trainer.train(self.train_ds, self.eval_ds)
    variables = nnx.state(model, nnx.Param)

    jax.tree.map_with_path(tc.assert_not_equal, original_variables, variables)

  @parameterized.named_parameters(
      ('scalar', TEST_LEARNING_RATE),
      ('constant_schedule', optax.constant_schedule(TEST_LEARNING_RATE)),
  )
  def test_lora_training(self, learning_rate_scheduler):
    config = peft_trainer.TrainingConfig(eval_every_n_steps=2, max_steps=100)
    rngs = nnx.Rngs(0)
    model = tc.get_lora_model(
        tc.ToyTransformer(config=tc.ModelConfig(), rngs=rngs)
    )

    original_params = jax.tree.map(
        jnp.copy, nnx.state(model, (nnx.filterlib.Not(nnx.LoRAParam)))
    )
    original_lora_params = jax.tree.map(
        jnp.copy, nnx.state(model, nnx.LoRAParam)
    )
    optimizer = optax.inject_hyperparams(optax.sgd)(
        learning_rate=learning_rate_scheduler
    )
    trainer = peft_trainer.PeftTrainer(model, optimizer, config)
    trainer = trainer.with_gen_model_input_fn(dummy_gen_model_input_fn)

    trainer.train(self.train_ds, self.eval_ds)
    params = nnx.state(model, (nnx.filterlib.Not(nnx.LoRAParam)))
    lora_params = nnx.state(model, nnx.LoRAParam)

    jax.tree.map_with_path(tc.assert_equal, original_params, params)
    jax.tree.map_with_path(
        tc.assert_not_equal, original_lora_params, lora_params
    )
    self.assertEqual(
        trainer.metrics_logger.get_metric('', 'learning_rate', 'train'),
        TEST_LEARNING_RATE,
    )

  @parameterized.named_parameters(
      ('scalar', TEST_LEARNING_RATE),
      ('constant_schedule', optax.constant_schedule(TEST_LEARNING_RATE)),
  )
  def test_gradient_accumulation(self, learning_rate_schedule):
    def train(
        train_ds,
        gradient_accumulation_steps: int | None,
        learning_rate_schedule: int | optax.Schedule,
    ):
      config = peft_trainer.TrainingConfig(
          eval_every_n_steps=2,
          max_steps=100,
          gradient_accumulation_steps=gradient_accumulation_steps,
      )
      rngs = nnx.Rngs(0)
      model = tc.ToyTransformer(config=tc.ModelConfig(), rngs=rngs)

      optimizer = optax.inject_hyperparams(optax.sgd)(
          learning_rate=learning_rate_schedule
      )
      trainer = peft_trainer.PeftTrainer(model, optimizer, config)
      trainer = trainer.with_gen_model_input_fn(dummy_gen_model_input_fn)

      trainer.train(train_ds, self.eval_ds)
      self.assertEqual(
          trainer.metrics_logger.get_metric('', 'learning_rate', 'train'),
          TEST_LEARNING_RATE,
      )
      return nnx.state(model, nnx.Param), trainer

    train_ds = dummy_datasets(batch_size=4, repeat=4)
    params, trainer = train(
        train_ds,
        gradient_accumulation_steps=None,
        learning_rate_schedule=learning_rate_schedule,
    )
    params_with_grad_accumulation, grad_accu_trainer = train(
        dummy_datasets(batch_size=2, repeat=4),
        gradient_accumulation_steps=2,
        learning_rate_schedule=learning_rate_schedule,
    )
    jax.tree.map_with_path(
        functools.partial(tc.assert_close, atol=1e-7, rtol=1e-7),
        params,
        params_with_grad_accumulation,
    )
    self.assertEqual(trainer.train_steps, grad_accu_trainer.train_steps)
    self.assertEqual(trainer.iter_steps * 2, grad_accu_trainer.iter_steps)
    np.testing.assert_allclose(
        trainer.metrics_logger.get_metric('', 'loss', 'train'),
        grad_accu_trainer.metrics_logger.get_metric('', 'loss', 'train'),
        atol=1e-5,
        rtol=1e-5,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='without_grad_accu',
          grad_accu=1,
          resume_step=0,
          expected_save_steps=[1, 2, 3, 4],
      ),
      dict(
          testcase_name='grad_accu',
          grad_accu=2,
          resume_step=0,
          expected_save_steps=[1, 2],
      ),
      dict(
          testcase_name='with_resume',
          grad_accu=1,
          resume_step=1,
          expected_save_steps=[2, 3, 4],
      ),
      dict(
          testcase_name='with_resume_and_grad_accu',
          grad_accu=2,
          resume_step=1,
          expected_save_steps=[2],
      ),
  )
  @mock.patch.object(checkpoint_manager, 'CheckpointManager')
  def test_checkpointing(
      self,
      mock_checkpoint_manager_init,
      grad_accu,
      resume_step,
      expected_save_steps,
  ):
    mock_checkpoint_manager = mock.MagicMock()
    mock_checkpoint_manager_init.return_value = mock_checkpoint_manager
    mock_checkpoint_manager.maybe_restore.return_value = (resume_step, {})
    mock_checkpoint_manager.save.return_value = True
    mock_checkpoint_manager.latest_step.return_value = (
        expected_save_steps[-1] - 1
    )  # force save at close
    checkpointing_options = checkpoint_options.create_checkpointing_options()
    config = peft_trainer.TrainingConfig(
        eval_every_n_steps=2,
        max_steps=100,
        gradient_accumulation_steps=grad_accu,
        checkpoint_root_directory='/tmp/checkpoint',
        checkpointing_options=checkpointing_options,
    )
    rngs = nnx.Rngs(0)
    model = tc.get_lora_model(
        tc.ToyTransformer(config=tc.ModelConfig(), rngs=rngs)
    )
    trainer = peft_trainer.PeftTrainer(model, optax.sgd(1e-3), config)
    trainer = trainer.with_gen_model_input_fn(dummy_gen_model_input_fn)

    train_ds = eval_ds = dummy_datasets(batch_size=2, repeat=1)  # 4 batches
    trainer.train(train_ds, eval_ds)

    mock_checkpoint_manager_init.assert_called_once_with(
        root_directory='/tmp/checkpoint', options=checkpointing_options
    )
    # Assert that the checkpoint manager is called with the correct arguments
    # and does not have any unexpected calls.
    mock_checkpoint_manager.assert_has_calls(
        [
            mock.call.maybe_restore(
                mock.ANY, mock.ANY, restore_only_lora_params=True
            ),
            *[
                mock.call.save(
                    i,
                    mock.ANY,
                    mock.ANY,
                    save_only_lora_params=True,
                    custom_metadata={},
                )
                for i in expected_save_steps
            ],
            mock.call.latest_step(),
            mock.call.save(
                expected_save_steps[-1],
                mock.ANY,
                mock.ANY,
                save_only_lora_params=True,
                force=True,
            ),
            mock.call.close(),
        ],
        any_order=False,
    )

  def test_loss_fn_with_aux(self):
    def custom_loss_fn(
        model: nnx.Module,
        input_tokens: jax.Array,
        input_mask: jax.Array,
        positions: jax.Array,
        attention_mask: jax.Array,
    ) -> Tuple[jax.Array, Any]:
      del model, input_tokens, input_mask, positions, attention_mask
      return jnp.array(1.0), {'foo': 1, 'bar': 2}

    train_invoke = {'foo': 0, 'bar': 0}
    eval_invoke = {'foo': 1, 'bar': 1}

    class CustomTrainer(peft_trainer.PeftTrainer):

      def _post_process_train_step(self, aux):
        train_invoke['foo'] += aux['foo']
        train_invoke['bar'] += aux['bar']

      def _post_process_eval_step(self, aux):
        eval_invoke['foo'] *= aux['foo']
        eval_invoke['bar'] *= aux['bar']

    config = peft_trainer.TrainingConfig(eval_every_n_steps=2, max_steps=100)
    model = tc.ToyTransformer(config=tc.ModelConfig(), rngs=nnx.Rngs(0))

    trainer = CustomTrainer(model, optax.sgd(1e-3), config)
    trainer = trainer.with_gen_model_input_fn(
        dummy_gen_model_input_fn
    ).with_loss_fn(custom_loss_fn, has_aux=True)

    trainer.train(self.train_ds, self.eval_ds)
    self.assertEqual(train_invoke, {'foo': 2, 'bar': 4})
    self.assertEqual(eval_invoke, {'foo': 1, 'bar': 16})

  def test_loss_output_format(self):
    def custom_loss_fn(
        model: nnx.Module,
        input_tokens: jax.Array,
        input_mask: jax.Array,
        positions: jax.Array,
        attention_mask: jax.Array,
        images: jax.Array | None = None,
    ) -> utils.LossOutput:
      del model, input_tokens, input_mask, positions, attention_mask, images
      return utils.LossOutput(
          primary_loss=utils.WeightedMetric(
              jnp.array(2.0, dtype=jnp.float32),
              jnp.array(2.0, dtype=jnp.float32),
          ),
          aux_metrics={
              'foo': utils.WeightedMetric(
                  jnp.array(10.0, dtype=jnp.float32),
                  jnp.array(5.0, dtype=jnp.float32),
              ),
              'bar': utils.WeightedMetric(
                  jnp.array(6.0, dtype=jnp.float32),
                  jnp.array(2.0, dtype=jnp.float32),
              ),
          },
      )

    train_invoke = {'foo': 0.0, 'bar': 0.0}
    eval_invoke = {'foo': 0.0, 'bar': 0.0}

    class CustomTrainer(peft_trainer.PeftTrainer):

      def _post_process_train_step(self, aux):
        # aux values are now raw WeightedMetric (no legacy pre-compute).
        train_invoke['foo'] += aux['foo'].compute()
        train_invoke['bar'] += aux['bar'].compute()

      def _post_process_eval_step(self, aux):
        eval_invoke['foo'] += aux['foo'].compute()
        eval_invoke['bar'] += aux['bar'].compute()

    config = peft_trainer.TrainingConfig(eval_every_n_steps=2, max_steps=100)
    model = tc.ToyTransformer(config=tc.ModelConfig(), rngs=nnx.Rngs(0))

    trainer = CustomTrainer(model, optax.sgd(1e-3), config)
    trainer = trainer.with_gen_model_input_fn(
        dummy_gen_model_input_fn
    ).with_loss_fn(
        custom_loss_fn
    )  # Note: has_aux=False is default but LossOutput returns aux natively

    trainer.train(self.train_ds, self.eval_ds)
    # The dataset provides 2 training steps.
    # foo = 10.0 / 5.0 = 2.0 per step.
    # bar = 6.0 / 2.0 = 3.0 per step.
    self.assertEqual(train_invoke, {'foo': 4.0, 'bar': 6.0})

    # Since eval_ds is length 2, it evaluates at step 2.
    self.assertEqual(eval_invoke, {'foo': 8.0, 'bar': 12.0})

  def test_loss_output_gradient_scaling(self):
    # _train_step accumulates grad(unreduced_sum) with the metric's denominator
    # and defers the division into the accumulator (Sum grads / Sum denom), so a
    # LossOutput with denominator d must yield the same update as the reduced
    # scalar loss sum/d. With a constant d (=2 here) the global-weighted and the
    # old mean-of-means recipes coincide, so this equivalence is stable across
    # that change. Uses a parameter-dependent loss because the original test's
    # constant loss has zero gradient and can't exercise it.
    def param_dependent_sum(model, input_tokens, positions):
      logits, _ = model(input_tokens, positions)
      return jnp.sum(logits)  # depends on the parameters

    def unreduced_loss_fn(
        model, input_tokens, input_mask, positions, attention_mask, images=None
    ):
      del input_mask, attention_mask, images
      s = param_dependent_sum(model, input_tokens, positions)
      return utils.LossOutput(
          primary_loss=utils.WeightedMetric(
              s, jnp.array(2.0, dtype=jnp.float32)
          ),
          aux_metrics={},
      )

    def make_reduced_loss_fn(denominator):
      def reduced_loss_fn(
          model, input_tokens, input_mask, positions, attention_mask,
          images=None,
      ):
        del input_mask, attention_mask, images
        s = param_dependent_sum(model, input_tokens, positions)
        return s / jnp.array(denominator, dtype=jnp.float32)

      return reduced_loss_fn

    def train_and_get_params(loss_fn):
      model = tc.ToyTransformer(config=tc.ModelConfig(), rngs=nnx.Rngs(0))
      config = peft_trainer.TrainingConfig(
          eval_every_n_steps=100, max_steps=100
      )
      trainer = peft_trainer.PeftTrainer(model, optax.sgd(1e-3), config)
      trainer = trainer.with_gen_model_input_fn(
          dummy_gen_model_input_fn
      ).with_loss_fn(loss_fn)
      init = [
          jnp.copy(x)
          for x in jax.tree_util.tree_leaves(nnx.state(model, nnx.Param))
      ]
      trainer.train(self.train_ds, self.eval_ds)
      trained = jax.tree_util.tree_leaves(nnx.state(model, nnx.Param))
      return init, trained

    def max_abs_diff(a, b):
      return max(float(jnp.max(jnp.abs(x - y))) for x, y in zip(a, b))

    # A: unreduced loss -> grad(sum) * (1/2). B: reduced ref sum/2. B1: sum/1.
    init_a, params_a = train_and_get_params(unreduced_loss_fn)
    _, params_b = train_and_get_params(make_reduced_loss_fn(2.0))
    _, params_b1 = train_and_get_params(make_reduced_loss_fn(1.0))

    # The scaled path must actually move the weights (non-zero gradient), else
    # the equivalence below would hold trivially.
    self.assertGreater(max_abs_diff(params_a, init_a), 1e-8)

    # Correct scaling: grad(sum) * (1/2) == grad(sum / 2).
    chex.assert_trees_all_close(params_a, params_b, atol=1e-6)

    # Non-trivial scaling: had the 1/denominator factor been dropped, the
    # update would instead match the sum/1 reference. It does not, which proves
    # the scale is applied.
    self.assertGreater(max_abs_diff(params_a, params_b1), 1e-6)

  def test_stream_train_step_returns_raw_weighted_metric_aux(self):
    # Since _compute_legacy_aux was dropped, the (stream) _train_step returns
    # the raw aux with WeightedMetric preserved, so the metric ops can reduce
    # sum/denom themselves (mean_of_means / global_weighted_mean) instead of
    # receiving pre-divided scalars.
    def loss_fn(
        model, input_tokens, input_mask, positions, attention_mask, images=None
    ):
      del input_mask, attention_mask, images
      logits, _ = model(input_tokens, positions)
      s = jnp.sum(logits)
      return utils.LossOutput(
          primary_loss=utils.WeightedMetric(s, jnp.array(2.0, jnp.float32)),
          aux_metrics={
              'foo': utils.WeightedMetric(jnp.array(10.0), jnp.array(5.0)),
              'bar': utils.WeightedMetric(jnp.array(6.0), jnp.array(2.0)),
          },
      )

    config = peft_trainer.TrainingConfig(eval_every_n_steps=100, max_steps=100)
    model = tc.ToyTransformer(config=tc.ModelConfig(), rngs=nnx.Rngs(0))
    trainer = peft_trainer.PeftTrainer(model, optax.sgd(1e-3), config)
    trainer = trainer.with_gen_model_input_fn(
        dummy_gen_model_input_fn
    ).with_loss_fn(loss_fn)
    acc = peft_trainer.GradientAccumulator(trainer.model, nnx.Param)

    _, aux, _ = trainer._train_step(
        trainer.model, trainer.optimizer, acc, self.train_ds[0], jnp.array(True)
    )

    # aux values stay raw WeightedMetric (not pre-divided to scalars).
    self.assertIsInstance(aux['foo'], utils.WeightedMetric)
    self.assertIsInstance(aux['bar'], utils.WeightedMetric)
    # Metric ops reduce the raw WeightedMetric correctly across micro-batches.
    self.assertAlmostEqual(
        float(common.mean_of_means([aux['foo'], aux['foo']])), 2.0, places=5
    )  # mean(10/5, 10/5)
    self.assertAlmostEqual(
        float(common.global_weighted_mean([aux['bar'], aux['bar']])), 3.0,
        places=5,
    )  # (6+6)/(2+2)

  def test_train_step_denom_is_global_weighted(self):
    # Two micro-batches with UNEQUAL denominators on the SAME input (so both
    # gradients equal g = d(sum_logits)/dparam) accumulate into one accumulator
    # WITHOUT applying (is_update=False, so the model is unchanged between the
    # two calls). The fixed recipe adds the raw grad(sum) with the real denom,
    # so acc.get() == (g + g) / (d1 + d2) [global weighted], NOT
    # (g/d1 + g/d2) / 2 [the old mean-of-means, which pre-scaled by 1/denom and
    # accumulated with denom=1]. Fails on the pre-fix code.
    d1, d2 = 2.0, 6.0

    def make_loss(denom):
      def loss_fn(
          model, input_tokens, input_mask, positions, attention_mask,
          images=None,
      ):
        del input_mask, attention_mask, images
        logits, _ = model(input_tokens, positions)
        return utils.LossOutput(
            primary_loss=utils.WeightedMetric(
                jnp.sum(logits), jnp.asarray(denom, jnp.float32)
            ),
            aux_metrics={},
        )
      return loss_fn

    config = peft_trainer.TrainingConfig(eval_every_n_steps=100, max_steps=100)
    model = tc.ToyTransformer(config=tc.ModelConfig(), rngs=nnx.Rngs(0))
    trainer = peft_trainer.PeftTrainer(
        model, optax.sgd(1e-3), config
    ).with_gen_model_input_fn(dummy_gen_model_input_fn)
    acc = peft_trainer.GradientAccumulator(trainer.model, nnx.Param)
    inp = self.train_ds[0]

    # `with_loss_fn` mutates the trainer (returns self) and `_train_step` is
    # eager, so swapping the loss between the two accumulate-only calls is safe.
    trainer.with_loss_fn(make_loss(d1))._train_step(
        trainer.model, trainer.optimizer, acc, inp, jnp.array(False)
    )
    trainer.with_loss_fn(make_loss(d2))._train_step(
        trainer.model, trainer.optimizer, acc, inp, jnp.array(False)
    )
    result = _unwrap(acc.get())

    # Reference g = grad of sum(logits) on the (unchanged) model, matching
    # diff_fn which differentiates primary_loss.unreduced_sum.
    processed = dummy_gen_model_input_fn(inp)

    def loss_sum(m):
      logits, _ = m(processed['input_tokens'], processed['positions'])
      return jnp.sum(logits)

    _, g = nnx.value_and_grad(loss_sum)(trainer.model)
    g = _unwrap(g)
    global_weighted = jax.tree_util.tree_map(lambda a: (a + a) / (d1 + d2), g)
    mean_of_means = jax.tree_util.tree_map(lambda a: (a / d1 + a / d2) / 2.0, g)

    # Now equals the global weighting ...
    jax.tree_util.tree_map(
        lambda a, e: np.testing.assert_allclose(a, e, rtol=1e-5, atol=1e-5),
        result,
        global_weighted,
    )
    # ... and is NOT the old mean-of-means (uneven denoms).
    max_diff = jax.tree_util.tree_reduce(
        jnp.maximum,
        jax.tree_util.tree_map(
            lambda a, b: jnp.max(jnp.abs(a - b)), result, mean_of_means
        ),
        initializer=jnp.asarray(0.0),
    )
    self.assertGreater(float(max_diff), 1e-4)

  def test_injected_params(self):

    config = peft_trainer.TrainingConfig(eval_every_n_steps=2, max_steps=100)
    model = tc.ToyTransformer(config=tc.ModelConfig(), rngs=nnx.Rngs(0))

    learning_rate_scheduler = optax.constant_schedule(TEST_LEARNING_RATE)
    optimizer = optax.inject_hyperparams(optax.sgd)(
        momentum=0.001,
        learning_rate=learning_rate_scheduler,
    )

    trainer = peft_trainer.PeftTrainer(model, optimizer, config)
    trainer = trainer.with_gen_model_input_fn(dummy_gen_model_input_fn)
    trainer.train(self.train_ds, self.eval_ds)
    self.assertEqual(
        trainer.metrics_logger.get_metric('', 'learning_rate', 'train'),
        TEST_LEARNING_RATE,
    )


def _unwrap(state):
  """Unwrap a `State` of `Variable` leaves to raw arrays for numeric checks."""
  return jax.tree_util.tree_map(
      lambda v: v[...] if isinstance(v, nnx.Variable) else v,
      state,
      is_leaf=lambda x: isinstance(x, nnx.Variable),
  )


class GradientAccumulatorTest(parameterized.TestCase):
  """Unit tests for the GradientAccumulator module.

  Covers the unified `add(grads, denom=None)` contract:

  * default (`denom=None`): each call contributes 1.0 to the denominator,
    so `get()` returns the mean of the per-micro-step gradients — the
    `optax.MultiSteps` semantics expected by callers using a per-batch
    scalar (mean) loss.
  * explicit `denom`: caller supplies the unreduced-loss denominator
    (e.g. token count). `get()` returns `Σg / Σd`, which is the gradient
    of a single step on the concatenated batch — required when
    micro-batches have varying effective batch sizes (sequence packing).
  """

  def _make_accumulator(self):
    rngs = nnx.Rngs(0)
    model = nnx.Linear(in_features=4, out_features=2, rngs=rngs)
    return model, peft_trainer.GradientAccumulator(model, nnx.Param)

  def _ones_like_params(self, model, scale: float = 1.0):
    """Creates a dummy copy of model parameters filled entirely with the `scale` value."""
    return jax.tree_util.tree_map(
        lambda x: jnp.asarray(scale, dtype=x.dtype) * jnp.ones_like(x),
        nnx.state(model, nnx.Param),
    )

  def test_default_mode_averages_grads(self):
    """Default add() returns the mean of micro-step grads.

    Matches ``optax.MultiSteps`` semantics: K micro-steps of size B/K are
    equivalent to a single step on a batch of size B when the loss
    function returns a per-batch scalar (mean) value. ``get()`` returns
    ``(Σ_i grads_i) / max(Σ_i 1, 1)``; here K=2 and the per-step grads
    have scale 1.0 and 2.0, so the mean is 1.5.
    """
    model, acc = self._make_accumulator()
    acc.add(self._ones_like_params(model, scale=1.0))
    acc.add(self._ones_like_params(model, scale=2.0))
    out = _unwrap(acc.get())
    jax.tree_util.tree_map(
        lambda v: np.testing.assert_allclose(v, 1.5 * jnp.ones_like(v)),
        out,
    )

  def test_single_add_weighted_is_grads_over_denom(self):
    """A single add(grads, denom=d) then get() returns grads / max(d, 1).

    The weighted (denom != 1) case of a one-micro-batch update: the global-
    weighted mean ``Sum grads / Sum denom`` collapses to ``grads / max(denom, 1)``
    for a single add. This is the premise the depth-1 fast path relies on when
    ``gradient_accumulation_steps == 1`` -- updating directly from
    ``grads / max(denom, 1)`` (skipping the accumulator + nnx.cond) must equal
    the accumulator path. Here grads=6.0 and denom=4.0, so get()=6/4=1.5;
    the "pre-scale by 1/d then divide again" bug would give a different value.
    """
    model, acc = self._make_accumulator()
    acc.add(
        self._ones_like_params(model, scale=6.0),
        denom=jnp.asarray(4.0, dtype=jnp.float32),
    )
    out = _unwrap(acc.get())
    jax.tree_util.tree_map(
        lambda v: np.testing.assert_allclose(v, 1.5 * jnp.ones_like(v)),
        out,
    )

  @parameterized.named_parameters(
      dict(testcase_name='equal_denoms', denoms=(4.0, 4.0, 4.0, 4.0)),
      dict(testcase_name='varying_denoms', denoms=(1.0, 7.0, 3.0, 5.0)),
      dict(testcase_name='extreme_variance', denoms=(1.0, 1.0, 100.0, 1.0)),
  )
  def test_explicit_denom_matches_single_step_baseline(self, denoms):
    """Passing explicit denom matches the equivalent single-step batch.

    Setup: K micro-batches with denominator d_i and unreduced-sum
    gradient g_i. The accumulator computes ``Σ_i g_i / Σ_i d_i``, which
    is ``grad(Σ_i loss_unreduced_i) / Σ_i d_i`` — i.e., a single step on
    the concatenated batch — for any choice of d_i. The "pre-scale grads
    by 1/d_i then mean over K" pattern fails this equality when d_i are
    unequal; this test guards against that regression.

    Args:
      denoms: A tuple of floats representing the denominators for each
        micro-batch.
    """
    model, acc = self._make_accumulator()

    keys = jax.random.split(jax.random.PRNGKey(0), len(denoms))
    grads = [
        jax.tree_util.tree_map(
            lambda x, k=k: jax.random.normal(k, x.shape, dtype=x.dtype),
            nnx.state(model, nnx.Param),
        )
        for k in keys
    ]

    for g_i, d_i in zip(grads, denoms):
      acc.add(g_i, denom=jnp.asarray(d_i, dtype=jnp.float32))
    accumulated = _unwrap(acc.get())

    total_denom = sum(denoms)
    expected = jax.tree_util.tree_map(lambda *gs: sum(gs) / total_denom, *grads)
    jax.tree_util.tree_map(
        lambda a, e: np.testing.assert_allclose(a, e, rtol=1e-6, atol=1e-6),
        accumulated,
        expected,
    )

    if len(set(denoms)) > 1:
      naive_mean = jax.tree_util.tree_map(
          lambda *gs: sum(g / d for g, d in zip(gs, denoms)) / len(gs),
          *grads,
      )
      diff_tree = jax.tree_util.tree_map(
          lambda a, b: jnp.max(jnp.abs(a - b)), accumulated, naive_mean
      )
      max_naive_diff = jax.tree_util.tree_reduce(
          jnp.maximum, diff_tree, initializer=jnp.asarray(0.0, jnp.float32)
      )
      self.assertGreater(
          float(max_naive_diff),
          1e-3,
          msg=(
              'naive pre-scale-then-mean and Sigma g / Sigma d should '
              'disagree when denominators vary; if they agree the test setup '
              'is degenerate.'
          ),
      )

  def test_reset_clears_denom(self):
    model, acc = self._make_accumulator()
    acc.add(self._ones_like_params(model), denom=jnp.asarray(7.0, jnp.float32))
    acc.reset()
    self.assertEqual(float(acc.denom[...]), 0.0)
    jax.tree_util.tree_map(
        lambda v: np.testing.assert_array_equal(v[...], jnp.zeros_like(v[...])),
        acc.grads,
        is_leaf=lambda x: isinstance(x, nnx.Variable),
    )

  # ---------------------------------------------------------------------
  # End-to-end numerical equivalence tests against `nnx.value_and_grad`.
  #
  # The tests above exercise the accumulator with hand-rolled arrays; the
  # tests below thread the *real* differentiation path (`nnx.value_and_grad`
  # on a small model) so the assertions hold for the exact pytree shape /
  # Variable wrappers the production `_train_step` produces.
  # ---------------------------------------------------------------------

  def _make_model_and_data(self, total_examples: int, seed: int = 42):
    rngs = nnx.Rngs(seed)
    model = nnx.Linear(in_features=4, out_features=2, rngs=rngs)
    keys = jax.random.split(jax.random.PRNGKey(seed), 2)
    x = jax.random.normal(keys[0], (total_examples, 4))
    y = jax.random.normal(keys[1], (total_examples, 2))
    return model, x, y

  @staticmethod
  def _loss_mean(model, x, y):
    # Mean over the batch / sequence axis only (sum over feature axis)
    # so `sum_loss == batch_size * mean_loss`. The full-tensor `jnp.mean`
    # would divide by `batch_size * feature_dim`, which would only match
    # the denom-aware path if `denom` were passed as `size * feature_dim`
    # — pinning the contract to a model-architecture quirk we don't want
    # the test to rely on.
    per_example = jnp.sum((model(x) - y) ** 2, axis=-1)
    return jnp.mean(per_example)

  @staticmethod
  def _loss_sum(model, x, y):
    # Matches the reduction of `_loss_mean` modulo division by batch size:
    # sum over both batch and feature axes.
    return jnp.sum((model(x) - y) ** 2)

  @parameterized.named_parameters(
      dict(testcase_name='K1', K=1),
      dict(testcase_name='K2', K=2),
      dict(testcase_name='K4', K=4),
      dict(testcase_name='K8', K=8),
  )
  def test_default_mode_K_micro_batches_match_full_batch(self, K):
    """Default mode: K equal-size micro-batches ≡ one full batch.

    Mean-of-means equals mean-over-all when the micro-batches partition
    the full batch into equal-size chunks. This is the
    `optax.MultiSteps`-equivalent contract the unpacked grad-accumulation
    path relies on.
    """
    B = 16
    self.assertEqual(B % K, 0)
    micro = B // K
    model, x, y = self._make_model_and_data(B)

    grad_fn = nnx.value_and_grad(self._loss_mean)
    _, expected = grad_fn(model, x, y)

    acc = peft_trainer.GradientAccumulator(model, nnx.Param)
    for i in range(K):
      _, g = grad_fn(
          model, x[i * micro : (i + 1) * micro], y[i * micro : (i + 1) * micro]
      )
      acc.add(g)

    accumulated = _unwrap(acc.get())
    expected_arrays = _unwrap(expected)
    jax.tree_util.tree_map(
        lambda a, e: np.testing.assert_allclose(a, e, rtol=1e-6, atol=1e-6),
        accumulated,
        expected_arrays,
    )

  def test_default_mode_K_micro_batches_match_concatenated_baseline_under_jit(
      self,
  ):
    """Same as above but with the accumulator's mutations under `nnx.jit`.

    The unpacked `_train_step` calls `acc.add()` from inside a jit; this
    test exercises the same trace path so any nnx.Variable / pytree
    breakage in jitted mutation surfaces here (rather than only at the
    full trainer integration level).
    """
    B = 12
    K = 3
    micro = B // K
    model, x, y = self._make_model_and_data(B, seed=7)
    acc = peft_trainer.GradientAccumulator(model, nnx.Param)

    @nnx.jit
    def _add_step(model, acc, x_b, y_b):
      _, g = nnx.value_and_grad(self._loss_mean)(model, x_b, y_b)
      acc.add(g)

    for i in range(K):
      _add_step(
          model,
          acc,
          x[i * micro : (i + 1) * micro],
          y[i * micro : (i + 1) * micro],
      )

    accumulated = _unwrap(acc.get())
    _, expected = nnx.value_and_grad(self._loss_mean)(model, x, y)
    expected_arrays = _unwrap(expected)
    jax.tree_util.tree_map(
        lambda a, e: np.testing.assert_allclose(a, e, rtol=1e-6, atol=1e-6),
        accumulated,
        expected_arrays,
    )

  @parameterized.named_parameters(
      dict(testcase_name='small_pack', sizes=(3, 5, 1, 7)),
      dict(testcase_name='single_dominant_pack', sizes=(1, 1, 28, 2)),
      dict(testcase_name='single_pack', sizes=(8,)),
      dict(testcase_name='many_small_packs', sizes=(1, 1, 1, 1, 1, 1, 1, 1)),
  )
  def test_explicit_denom_packed_micro_batches_match_full_batch(self, sizes):
    """Sequence packing: varying-size micro-batches with denom=size.

    Under sequence packing each yielded micro-batch carries a different
    number of training examples (varying pack sizes). The denom-aware
    path computes Σ_i grad(sum_loss_i) / Σ_i size_i, which is the
    gradient of mean(loss_over_all_examples) for *any* partition. Tests
    span uniform, dominantly-one-pack, single-pack, and
    many-small-packs partitions to catch regressions where the divisor
    drifts off-by-one.

    Args:
      sizes: A tuple of integers representing the sizes of each micro-batch.
    """
    total = sum(sizes)
    model, x, y = self._make_model_and_data(total, seed=13)

    _, expected = nnx.value_and_grad(self._loss_mean)(model, x, y)

    grad_sum = nnx.value_and_grad(self._loss_sum)
    acc = peft_trainer.GradientAccumulator(model, nnx.Param)
    start = 0
    for size in sizes:
      end = start + size
      _, g = grad_sum(model, x[start:end], y[start:end])
      acc.add(g, denom=jnp.asarray(float(size)))
      start = end

    accumulated = _unwrap(acc.get())
    expected_arrays = _unwrap(expected)
    jax.tree_util.tree_map(
        lambda a, e: np.testing.assert_allclose(a, e, rtol=1e-6, atol=1e-6),
        accumulated,
        expected_arrays,
    )

  def test_explicit_denom_packed_matches_unpacked_concatenation_under_jit(self):
    """Packed + denom-aware path under `nnx.jit`, against unpacked baseline.

    Mirrors the production sequence-packing flow: each "pack" is a
    micro-batch of varying size, fed through a jitted grad-sum step and
    accumulated with `denom=size`. The expected value is computed *on
    the same model* via the mean-loss path, so any mismatch isolates the
    accumulation math (not data setup).
    """
    sizes = (2, 4, 1, 3, 6)
    total = sum(sizes)
    model, x, y = self._make_model_and_data(total, seed=21)
    acc = peft_trainer.GradientAccumulator(model, nnx.Param)

    @nnx.jit
    def _packed_add(model, acc, x_b, y_b, denom):
      _, g = nnx.value_and_grad(self._loss_sum)(model, x_b, y_b)
      acc.add(g, denom=denom)

    start = 0
    for size in sizes:
      end = start + size
      _packed_add(
          model,
          acc,
          x[start:end],
          y[start:end],
          jnp.asarray(float(size), jnp.float32),
      )
      start = end

    accumulated = _unwrap(acc.get())
    _, expected = nnx.value_and_grad(self._loss_mean)(model, x, y)
    expected_arrays = _unwrap(expected)
    jax.tree_util.tree_map(
        lambda a, e: np.testing.assert_allclose(a, e, rtol=1e-6, atol=1e-6),
        accumulated,
        expected_arrays,
    )

  def test_default_and_explicit_denom_agree_when_micro_batches_uniform(self):
    """Sanity bridge: explicit denom with uniform sizes ≡ default mode.

    When every micro-batch has the same size, the default (mean-of-means)
    path and the denom-aware (sum-of-sums / sum-of-sizes) path must
    produce the same gradient. This sanity-checks that the unification
    of `count` and `denom` into a single field hasn't introduced a
    silent off-by-N (e.g. summing K vs K+1 in one of the branches).
    """
    sizes = (4, 4, 4, 4)
    total = sum(sizes)
    model, x, y = self._make_model_and_data(total, seed=99)

    # Default (mean) path.
    acc_default = peft_trainer.GradientAccumulator(model, nnx.Param)
    grad_mean = nnx.value_and_grad(self._loss_mean)
    for i, size in enumerate(sizes):
      s, e = i * size, (i + 1) * size
      _, g = grad_mean(model, x[s:e], y[s:e])
      acc_default.add(g)
    default_out = _unwrap(acc_default.get())

    # Explicit-denom path with uniform sizes.
    acc_denom = peft_trainer.GradientAccumulator(model, nnx.Param)
    grad_sum = nnx.value_and_grad(self._loss_sum)
    start = 0
    for size in sizes:
      end = start + size
      _, g = grad_sum(model, x[start:end], y[start:end])
      acc_denom.add(g, denom=jnp.asarray(float(size)))
      start = end
    denom_out = _unwrap(acc_denom.get())

    jax.tree_util.tree_map(
        lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6),
        default_out,
        denom_out,
    )

  def test_reset_then_reuse_does_not_leak_state(self):
    """After `reset()`, a second accumulation cycle must match a fresh acc.

    Guards against state leaking across reset boundaries — e.g. the
    denom counter not zeroing, or `grads` keeping a residual that would
    silently bias subsequent updates.
    """
    sizes = (4, 4)
    total = sum(sizes)
    model, x, y = self._make_model_and_data(total, seed=33)
    grad_mean = nnx.value_and_grad(self._loss_mean)

    acc = peft_trainer.GradientAccumulator(model, nnx.Param)
    # First cycle on unrelated data — must be erased by reset.
    junk_x = jax.random.normal(jax.random.PRNGKey(101), (8, 4))
    junk_y = jax.random.normal(jax.random.PRNGKey(102), (8, 2))
    for i in range(2):
      _, g = grad_mean(
          model, junk_x[i * 4 : (i + 1) * 4], junk_y[i * 4 : (i + 1) * 4]
      )
      acc.add(g)
    acc.reset()

    # Second cycle on the real data after reset.
    for i, size in enumerate(sizes):
      s, e = i * size, (i + 1) * size
      _, g = grad_mean(model, x[s:e], y[s:e])
      acc.add(g)
    after_reset = _unwrap(acc.get())

    # Reference: fresh accumulator on the same real data.
    acc_fresh = peft_trainer.GradientAccumulator(model, nnx.Param)
    for i, size in enumerate(sizes):
      s, e = i * size, (i + 1) * size
      _, g = grad_mean(model, x[s:e], y[s:e])
      acc_fresh.add(g)
    fresh = _unwrap(acc_fresh.get())

    jax.tree_util.tree_map(
        lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6),
        after_reset,
        fresh,
    )

  @parameterized.named_parameters(
      dict(testcase_name='bfloat16', dtype=jnp.bfloat16),
      dict(testcase_name='float16', dtype=jnp.float16),
      dict(testcase_name='float32', dtype=jnp.float32),
  )
  def test_get_preserves_grad_dtype(self, dtype: jnp.dtype):
    rngs = nnx.Rngs(0)
    model = nnx.Linear(
        in_features=4, out_features=2, rngs=rngs, param_dtype=dtype
    )
    acc = peft_trainer.GradientAccumulator(model, nnx.Param)

    grads = jax.tree_util.tree_map(
        lambda v: type(v)(jnp.ones_like(v[...])),
        nnx.state(model, nnx.Param),
        is_leaf=lambda x: isinstance(x, nnx.Variable),
    )
    acc.add(grads, denom=jnp.asarray(3.0, dtype=jnp.float32))
    out = acc.get()

    jax.tree_util.tree_map(
        lambda v: self.assertEqual(v[...].dtype, dtype),
        out,
        is_leaf=lambda x: isinstance(x, nnx.Variable),
    )

  def test_cond_apply_vs_skip_branches_have_matching_dtypes_in_bf16(self):
    rngs = nnx.Rngs(0)
    model = nnx.Linear(
        in_features=4, out_features=2, rngs=rngs, param_dtype=jnp.bfloat16
    )
    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
    acc = peft_trainer.GradientAccumulator(model, nnx.Param)

    x = jnp.ones((2, 4), dtype=jnp.bfloat16)
    y = jnp.ones((2, 2), dtype=jnp.bfloat16)
    _, grads = nnx.value_and_grad(lambda m, x, y: jnp.sum((m(x) - y) ** 2))(
        model, x, y
    )
    acc.add(grads, denom=jnp.asarray(1.0, dtype=jnp.float32))

    def apply_updates(model, optimizer, acc):
      acc_grads = acc.get()
      optimizer.update(model, acc_grads)
      acc.reset()
      return jnp.asarray(0.0, dtype=jnp.float32)

    def skip_updates(model, optimizer, acc):
      return jnp.asarray(0.0, dtype=jnp.float32)

    @nnx.jit
    def step(model, optimizer, acc, is_update_step):
      return nnx.cond(
          is_update_step, apply_updates, skip_updates, model, optimizer, acc
      )

    step(model, optimizer, acc, jnp.asarray(False))
    step(model, optimizer, acc, jnp.asarray(True))

    opt_state_dtypes = jax.tree_util.tree_leaves(
        jax.tree_util.tree_map(
            lambda v: v[...].dtype,
            nnx.state(optimizer, nnx.optimizer.OptState),
            is_leaf=lambda x: isinstance(x, nnx.Variable),
        )
    )
    float_dtypes = [
        d for d in opt_state_dtypes if jnp.issubdtype(d, jnp.floating)
    ]
    self.assertNotEmpty(float_dtypes)
    for d in float_dtypes:
      self.assertEqual(d, jnp.bfloat16)

  def test_peft_trainer_promotes_bf16_opt_state_floats_to_float32(self):
    """`PeftTrainer.__init__` casts float opt_state leaves to float32.

    `optax.adam` / `optax.adamw` promote their floating-point moments
    (`mu`, `nu`) to float32 inside `update` whenever the learning rate is
    a float32 tracer (as produced by `optax.inject_hyperparams`). This test
    verifies that the trainer casts these to float32 in-place during init.
    """
    rngs = nnx.Rngs(0)
    model = tc.ToyTransformer(config=tc.ModelConfig(), rngs=rngs)
    bf16_state = jax.tree.map(
        lambda x: x.astype(jnp.bfloat16)
        if jnp.issubdtype(x.dtype, jnp.floating)
        else x,
        nnx.state(model, nnx.Param),
    )
    nnx.update(model, bf16_state)

    tx = optax.inject_hyperparams(optax.adamw, hyperparam_dtype=jnp.float32)(
        learning_rate=1e-3
    )
    config = peft_trainer.TrainingConfig(eval_every_n_steps=100, max_steps=1)
    trainer = peft_trainer.PeftTrainer(model, tx, config)

    opt_state_dtypes = jax.tree_util.tree_leaves(
        jax.tree_util.tree_map(
            lambda v: v[...].dtype,
            nnx.state(trainer.optimizer, nnx.optimizer.OptState),
            is_leaf=lambda x: isinstance(x, nnx.Variable),
        )
    )
    float_dtypes = [
        d for d in opt_state_dtypes if jnp.issubdtype(d, jnp.floating)
    ]
    self.assertNotEmpty(float_dtypes)
    for d in float_dtypes:
      self.assertEqual(d, jnp.float32)


class TrainStepGlobalWeightedTest(parameterized.TestCase):
  """Pins that _train_step accumulates the GLOBAL weighted mean, not a mean-of-means.

  Emulates the _train_step recipe: grad(unreduced_sum_i) accumulated with the
  loss's real per-micro-batch denom -> get() = Sum grads / Sum denom. With
  uneven denoms this must equal the global weighting and differ from the old
  mean-of-means (grad_i/denom_i averaged equally).
  """

  @staticmethod
  def _unwrap(state):
    return jax.tree_util.tree_map(
        lambda v: v[...] if isinstance(v, nnx.Variable) else v,
        state,
        is_leaf=lambda x: isinstance(x, nnx.Variable),
    )

  @staticmethod
  def _loss_sum(model, x, y):
    return jnp.sum((model(x) - y) ** 2)

  def test_train_step_recipe_is_global_weighted(self):
    d1, d2 = 2, 6  # uneven micro-batch denominators
    rngs = nnx.Rngs(0)
    model = nnx.Linear(in_features=4, out_features=2, rngs=rngs)
    k = jax.random.split(jax.random.PRNGKey(0), 2)
    x = jax.random.normal(k[0], (d1 + d2, 4))
    y = jax.random.normal(k[1], (d1 + d2, 2))
    grad_fn = nnx.value_and_grad(self._loss_sum)
    _, g1 = grad_fn(model, x[:d1], y[:d1])
    _, g2 = grad_fn(model, x[d1:], y[d1:])
    g1, g2 = self._unwrap(g1), self._unwrap(g2)
    # New recipe: NO local 1/denom scale; add grad(sum) with the real denom.
    acc = peft_trainer.GradientAccumulator(model, nnx.Param)
    acc.add(g1, denom=jnp.asarray(float(d1)))
    acc.add(g2, denom=jnp.asarray(float(d2)))
    result = self._unwrap(acc.get())
    global_weighted = jax.tree_util.tree_map(
        lambda a, b: (a + b) / (d1 + d2), g1, g2
    )
    mean_of_means = jax.tree_util.tree_map(
        lambda a, b: (a / d1 + b / d2) / 2.0, g1, g2
    )
    # Now equals the global weighting ...
    jax.tree_util.tree_map(
        lambda a, e: np.testing.assert_allclose(a, e, rtol=1e-6, atol=1e-6),
        result,
        global_weighted,
    )
    # ... and is NOT the old mean-of-means (uneven denoms).
    max_diff = jax.tree_util.tree_reduce(
        jnp.maximum,
        jax.tree_util.tree_map(
            lambda a, b: jnp.max(jnp.abs(a - b)), result, mean_of_means
        ),
        initializer=jnp.asarray(0.0),
    )
    self.assertGreater(float(max_diff), 1e-3)


if __name__ == '__main__':
  absltest.main()
