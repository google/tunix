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

import functools
import os
from typing import Any, Tuple
from unittest import mock
from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
import jax.sharding as shd
import numpy as np
import optax
import orbax.checkpoint as ocp
from tunix.sft import checkpoint_manager
from tunix.sft import peft_trainer
from tunix.sft import profiler
from tunix.tests import test_common as tc


os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'


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


class PeftTrainerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.eval_ds = self.train_ds = dummy_datasets(batch_size=4)

  def test_basic_training(self):
    config = peft_trainer.TrainingConfig(eval_every_n_steps=2, max_steps=100)
    rngs = nnx.Rngs(0)
    model = tc.ToyTransformer(rngs=rngs)
    original_variables = jax.tree.map(jnp.copy, nnx.state(model, nnx.Param))

    trainer = peft_trainer.PeftTrainer(model, optax.sgd(1e-3), config)
    trainer = trainer.with_gen_model_input_fn(dummy_gen_model_input_fn)

    trainer.train(self.train_ds, self.eval_ds)
    variables = nnx.state(model, nnx.Param)

    jax.tree.map_with_path(tc.assert_not_equal, original_variables, variables)

    self.assertGreater(
        trainer._metrics_logger.get_metric('perplexity', 'train'), 0
    )
    self.assertGreater(
        trainer._metrics_logger.get_metric('perplexity', 'eval'), 0
    )
    self.assertGreater(trainer._train_steps, 0)
    self.assertLen(
        trainer._metrics_logger.get_metric_history('perplexity', 'train'),
        trainer._train_steps,
    )

    trainer.train(self.train_ds)  # No eval dataset.

  @mock.patch.object(profiler, 'Profiler')
  def test_basic_training_with_profiler(self, mock_profiler_init):
    self.train_ds = dummy_datasets(batch_size=4, repeat=4)
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
    model = tc.ToyTransformer(rngs=rngs)

    trainer = peft_trainer.PeftTrainer(model, optax.sgd(1e-3), config)
    trainer = trainer.with_gen_model_input_fn(dummy_gen_model_input_fn)
    trainer.train(self.train_ds)  # No eval dataset.

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
    mesh = shd.Mesh(
        devices=np.array(jax.devices()).reshape(2, 2), axis_names=('fsdp', 'tp')
    )
    rngs = nnx.Rngs(0)
    model, _ = create_sharded_model(tc.ToyTransformer, rngs, mesh)
    original_variables = jax.tree.map(jnp.copy, nnx.state(model, nnx.Param))

    config = peft_trainer.TrainingConfig(eval_every_n_steps=2, max_steps=100)
    trainer = peft_trainer.PeftTrainer(model, optax.sgd(1e-3), config)
    trainer = trainer.with_gen_model_input_fn(dummy_gen_model_input_fn)

    with mesh:
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
    unsharded_model = tc.ToyTransformer(rngs=rngs)
    trainer = peft_trainer.PeftTrainer(unsharded_model, optax.sgd(1e-3), config)
    trainer = trainer.with_gen_model_input_fn(dummy_gen_model_input_fn)
    trainer.train(self.train_ds, self.eval_ds)
    unsharded_variables = nnx.state(unsharded_model, nnx.Param)
    self.assertIsInstance(
        unsharded_variables.layers[0].w1.kernel.value.sharding,
        jax._src.lib.xla_client.SingleDeviceSharding,
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
    model = tc.ToyTransformer(rngs=rngs)
    original_variables = jax.tree.map(jnp.copy, nnx.state(model, nnx.Param))

    trainer = peft_trainer.PeftTrainer(model, optax.sgd(1e-3), config)
    trainer = trainer.with_gen_model_input_fn(
        dummy_gen_model_input_fn
    ).with_loss_fn(custom_loss_fn)
    trainer.train(self.train_ds, self.eval_ds)
    variables = nnx.state(model, nnx.Param)

    jax.tree.map_with_path(tc.assert_not_equal, original_variables, variables)

  def test_lora_training(self):
    config = peft_trainer.TrainingConfig(eval_every_n_steps=2, max_steps=100)
    rngs = nnx.Rngs(0)
    model = tc.get_lora_model(tc.ToyTransformer(rngs=rngs))

    original_params = jax.tree.map(
        jnp.copy, nnx.state(model, (nnx.filterlib.Not(nnx.LoRAParam)))
    )
    original_lora_params = jax.tree.map(
        jnp.copy, nnx.state(model, nnx.LoRAParam)
    )

    trainer = peft_trainer.PeftTrainer(model, optax.sgd(1e-3), config)
    trainer = trainer.with_gen_model_input_fn(dummy_gen_model_input_fn)

    trainer.train(self.train_ds, self.eval_ds)
    params = nnx.state(model, (nnx.filterlib.Not(nnx.LoRAParam)))
    lora_params = nnx.state(model, nnx.LoRAParam)

    jax.tree.map_with_path(tc.assert_equal, original_params, params)
    jax.tree.map_with_path(
        tc.assert_not_equal, original_lora_params, lora_params
    )

  def test_gradient_accumulation(self):
    def train(train_ds, gradient_accumulation_steps: int | None):
      config = peft_trainer.TrainingConfig(
          eval_every_n_steps=2,
          max_steps=100,
          gradient_accumulation_steps=gradient_accumulation_steps,
      )
      rngs = nnx.Rngs(0)
      model = tc.ToyTransformer(rngs=rngs)

      trainer = peft_trainer.PeftTrainer(model, optax.sgd(1e-3), config)
      trainer = trainer.with_gen_model_input_fn(dummy_gen_model_input_fn)

      trainer.train(train_ds, self.eval_ds)
      return nnx.state(model, nnx.Param)

    train_ds = dummy_datasets(batch_size=4, repeat=4)
    params = train(train_ds, gradient_accumulation_steps=None)
    params_with_grad_accumulation = train(
        dummy_datasets(batch_size=2, repeat=4),
        gradient_accumulation_steps=2,
    )
    jax.tree.map_with_path(
        functools.partial(tc.assert_close, atol=1e-7, rtol=1e-7),
        params,
        params_with_grad_accumulation,
    )

  @mock.patch.object(checkpoint_manager, 'CheckpointManager')
  def test_checkpointing(self, mock_checkpoint_manager_init):
    mock_checkpoint_manager = mock.MagicMock()
    mock_checkpoint_manager_init.return_value = mock_checkpoint_manager
    mock_checkpoint_manager.maybe_restore.return_value = 1
    mock_checkpoint_manager.save.return_value = True
    mock_checkpoint_manager.latest_step.return_value = 1
    checkpoint_options = ocp.CheckpointManagerOptions()
    config = peft_trainer.TrainingConfig(
        eval_every_n_steps=2,
        max_steps=100,
        checkpoint_root_directory='/tmp/checkpoint',
        checkpointing_options=checkpoint_options,
    )
    rngs = nnx.Rngs(0)
    model = tc.get_lora_model(tc.ToyTransformer(rngs=rngs))
    trainer = peft_trainer.PeftTrainer(model, optax.sgd(1e-3), config)
    trainer = trainer.with_gen_model_input_fn(dummy_gen_model_input_fn)

    trainer.train(self.train_ds, self.eval_ds)

    mock_checkpoint_manager_init.assert_called_once_with(
        root_directory='/tmp/checkpoint', options=checkpoint_options
    )
    # Assert that the checkpoint manager is called with the correct arguments
    # and does not have any unexpected calls.
    mock_checkpoint_manager.assert_has_calls(
        [
            mock.call.maybe_restore(mock.ANY, restore_only_lora_params=True),
            mock.call.save(2, mock.ANY, save_only_lora_params=True),
            mock.call.latest_step(),
            mock.call.save(2, mock.ANY, save_only_lora_params=True, force=True),
            mock.call.close(),
        ],
        any_order=True,
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
    model = tc.ToyTransformer(rngs=nnx.Rngs(0))

    trainer = CustomTrainer(model, optax.sgd(1e-3), config)
    trainer = trainer.with_gen_model_input_fn(
        dummy_gen_model_input_fn
    ).with_loss_fn(custom_loss_fn, has_aux=True)

    trainer.train(self.train_ds, self.eval_ds)
    self.assertEqual(train_invoke, {'foo': 2, 'bar': 4})
    self.assertEqual(eval_invoke, {'foo': 1, 'bar': 4})


if __name__ == '__main__':
  absltest.main()
