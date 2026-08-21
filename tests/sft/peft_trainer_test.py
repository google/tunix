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

  def test_p41_precomputed_transaction_has_one_microbatch(self):
    expected = peft_trainer._precomputed_expected_microbatches({  # pylint: disable=protected-access
        "CANON_P41_OPTIMIZER_BENCH": "1",
        "CANON_GSM8K_L3": "1",
        "CANON_GSM8K_UPDATE_CANARY": "1",
    })
    self.assertEqual(expected, 1)

  def test_p41_precomputed_transaction_rejects_missing_canary(self):
    with self.assertRaisesRegex(ValueError, "bounded GSM8K L3"):
      peft_trainer._precomputed_expected_microbatches({  # pylint: disable=protected-access
          "CANON_P41_OPTIMIZER_BENCH": "1",
      })

  def test_p41_single_precomputed_microbatch_allocates_and_commits(self):
    env = {
        "CANON_ALIGNMENT_GATE": "1",
        "CANON_ALIGNMENT_GATE_ONLY": "0",
        "CANON_ALIGNMENT_TRAIN": "0",
        "CANON_ALIGNMENT_UPDATE_CANARY": "1",
        "CANON_GSM8K_L3": "1",
        "CANON_GSM8K_UPDATE_CANARY": "1",
        "CANON_P28_G5C_ONLY": "0",
        "CANON_P28_G6_UPDATE": "1",
        "CANON_P28_SEGMENTED_TRAIN": "1",
        "CANON_P31_CONVERGENCE": "0",
        "CANON_P33_WORKLOAD_LAUNCH_ADMITTED": "0",
        "CANON_P41_OPTIMIZER_BENCH": "1",
    }
    config = peft_trainer.TrainingConfig(
        eval_every_n_steps=100,
        max_steps=1,
        gradient_accumulation_steps=1,
        checkpoint_root_directory=None,
    )
    with mock.patch.dict(os.environ, env, clear=False):
      model = tc.ToyTransformer(config=tc.ModelConfig(), rngs=nnx.Rngs(0))
      trainer = peft_trainer.PeftTrainer(model, optax.sgd(1e-3), config)
      self.assertGreater(len(jax.tree.leaves(trainer.grad_accumulator.grads)), 0)
      before = jax.tree.map(
          lambda value: np.asarray(value).copy(),
          nnx.state(trainer.model, nnx.Param),
      )
      gradient = jax.tree.map(
          lambda value: type(value)(jnp.ones_like(value[...], jnp.float32)),
          nnx.state(trainer.model, nnx.Param),
          is_leaf=lambda value: isinstance(value, nnx.VariableState),
      )
      norms = trainer.apply_precomputed_gradient_microbatches((gradient,))

    self.assertLen(norms, 1)
    self.assertGreater(float(norms[0]), 0.0)
    self.assertEqual(trainer.iter_steps, 1)
    self.assertEqual(trainer.train_steps, 1)
    for value in jax.tree.leaves(nnx.state(trainer.grad_accumulator)):
      np.testing.assert_array_equal(np.asarray(value), np.zeros_like(value))
    after = nnx.state(trainer.model, nnx.Param)
    self.assertTrue(any(
        not np.array_equal(old, np.asarray(new))
        for old, new in zip(
            jax.tree.leaves(before), jax.tree.leaves(after), strict=True
        )
    ))

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

  def test_compile_once_on_cond_path(self):
    """Depth>1 (accumulator + nnx.cond path) also traces exactly once.

    Guards the moment-dtype change (bf16 by default on the cond path) against
    re-tracing: the cond path must still compile just once.
    """
    class CountCompiledTimesTrainer(peft_trainer.PeftTrainer):

      def _train_step(
          self, model, optimizer, grad_accumulator, inputs, is_update_step
      ):
        global global_counter
        global_counter += 1
        return super()._train_step(
            model, optimizer, grad_accumulator, inputs, is_update_step
        )

    config = peft_trainer.TrainingConfig(
        eval_every_n_steps=2, max_steps=100, gradient_accumulation_steps=2
    )
    rngs = nnx.Rngs(0)
    model = tc.get_lora_model(
        tc.ToyTransformer(config=tc.ModelConfig(), rngs=rngs), mesh=self.mesh
    )
    trainer = CountCompiledTimesTrainer(model, optax.sgd(1e-3), config)
    trainer = trainer.with_gen_model_input_fn(dummy_gen_model_input_fn)
    global global_counter
    global_counter = 0
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

  def test_interval_only_policy_skips_forced_checkpoint_on_close(self):
    trainer = object.__new__(peft_trainer.PeftTrainer)
    trainer.checkpoint_manager = mock.MagicMock(save_on_close=False)

    trainer._save_last_checkpoint()

    trainer.checkpoint_manager.latest_step.assert_not_called()
    trainer.checkpoint_manager.save.assert_not_called()

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
    # Covers the manual gradient scaling in _train_step: grad(unreduced_sum) *
    # (1/d) must equal grad(sum/d). Uses a parameter-dependent loss because the
    # original test's constant loss has zero gradient and can't exercise it.
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

  def test_denominator_weighted_accumulation_matches_concatenated_batch(self):
    """Unequal effective rows divide once by their global denominator."""

    def loss_fn(
        model, input_tokens, input_mask, positions, attention_mask, images=None
    ):
      del attention_mask, images
      logits, _ = model(input_tokens, positions)
      row_mask = (input_mask.sum(axis=-1) > 0).astype(jnp.float32)
      unreduced = jnp.sum(logits * row_mask[:, None, None])
      return utils.LossOutput(
          primary_loss=utils.WeightedMetric(
              unreduced, row_mask.sum(), min_denom=1e-6
          ),
          aux_metrics={},
      )

    def make_example(offset, rows, effective_rows):
      tokens = jnp.arange(offset, offset + rows * 16).reshape(rows, 16)
      mask = jnp.zeros((rows, 16), dtype=jnp.int32)
      mask = mask.at[:effective_rows, 0].set(1)
      return peft_trainer.TrainingInput(
          input_tokens=tokens, input_mask=mask
      )

    first = make_example(0, 4, 1)
    second = make_example(64, 4, 3)
    combined = peft_trainer.TrainingInput(
        input_tokens=jnp.concatenate(
            [first.input_tokens, second.input_tokens], axis=0
        ),
        input_mask=jnp.concatenate(
            [first.input_mask, second.input_mask], axis=0
        ),
    )

    accumulated_model = tc.ToyTransformer(
        config=tc.ModelConfig(), rngs=nnx.Rngs(0)
    )
    accumulated_config = peft_trainer.TrainingConfig(
        eval_every_n_steps=100,
        max_steps=1,
        gradient_accumulation_steps=2,
        loss_denominator_weighted_accumulation=True,
    )
    accumulated = peft_trainer.PeftTrainer(
        accumulated_model, optax.sgd(1e-3), accumulated_config
    ).with_gen_model_input_fn(dummy_gen_model_input_fn).with_loss_fn(loss_fn)
    accumulated._train_step(
        accumulated.model,
        accumulated.optimizer,
        accumulated.grad_accumulator,
        first,
        jnp.asarray(False),
    )
    _, accumulated_aux, _ = accumulated._train_step(
        accumulated.model,
        accumulated.optimizer,
        accumulated.grad_accumulator,
        second,
        jnp.asarray(True),
    )

    full_model = tc.ToyTransformer(
        config=tc.ModelConfig(), rngs=nnx.Rngs(0)
    )
    full_config = peft_trainer.TrainingConfig(
        eval_every_n_steps=100,
        max_steps=1,
        loss_denominator_weighted_accumulation=True,
    )
    full = peft_trainer.PeftTrainer(
        full_model, optax.sgd(1e-3), full_config
    ).with_gen_model_input_fn(dummy_gen_model_input_fn).with_loss_fn(loss_fn)
    _, full_aux, _ = full._train_step(
        full.model,
        full.optimizer,
        full.grad_accumulator,
        combined,
        jnp.asarray(True),
    )

    chex.assert_trees_all_close(
        nnx.state(accumulated.model, nnx.Param),
        nnx.state(full.model, nnx.Param),
        atol=1e-6,
    )
    self.assertEqual(
        float(accumulated_aux['loss/accumulated_denominator']), 4.0
    )
    self.assertTrue(bool(accumulated_aux['loss/optimizer_committed']))
    self.assertTrue(bool(full_aux['loss/optimizer_committed']))

  def test_denominator_weighted_all_empty_skips_optimizer(self):
    """A zero global denominator does not execute an optimizer transaction."""

    def loss_fn(
        model, input_tokens, input_mask, positions, attention_mask, images=None
    ):
      del input_mask, attention_mask, images
      logits, _ = model(input_tokens, positions)
      return utils.LossOutput(
          primary_loss=utils.WeightedMetric(
              jnp.sum(logits * 0.0),
              jnp.asarray(0.0),
              min_denom=1e-6,
          ),
          aux_metrics={},
      )

    model = tc.ToyTransformer(config=tc.ModelConfig(), rngs=nnx.Rngs(0))
    config = peft_trainer.TrainingConfig(
        eval_every_n_steps=100,
        max_steps=1,
        loss_denominator_weighted_accumulation=True,
    )
    trainer = peft_trainer.PeftTrainer(
        model, optax.adam(1e-3), config
    ).with_gen_model_input_fn(dummy_gen_model_input_fn).with_loss_fn(loss_fn)
    model_before = jax.tree.map(
        lambda x: np.asarray(x).copy(), nnx.state(trainer.model, nnx.Param)
    )
    optimizer_before = jax.tree.map(
        lambda x: np.asarray(x).copy(), nnx.state(trainer.optimizer)
    )
    _, aux, _ = trainer._train_step(
        trainer.model,
        trainer.optimizer,
        trainer.grad_accumulator,
        self.train_ds[0],
        jnp.asarray(True),
    )
    chex.assert_trees_all_equal(
        model_before, nnx.state(trainer.model, nnx.Param)
    )
    chex.assert_trees_all_equal(optimizer_before, nnx.state(trainer.optimizer))
    self.assertFalse(bool(aux['loss/optimizer_committed']))

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

  def test_alignment_gate_only_returns_gradient_without_mutating_state(self):
    """The post-JIT host gate cannot race an optimizer/accumulator mutation."""

    def loss_fn(
        model, input_tokens, input_mask, positions, attention_mask, images=None
    ):
      del input_mask, attention_mask, images
      logits, _ = model(input_tokens, positions)
      return utils.LossOutput(
          primary_loss=utils.WeightedMetric(
              jnp.sum(logits), jnp.array(1.0, jnp.float32)
          ),
          aux_metrics={},
      )

    config = peft_trainer.TrainingConfig(eval_every_n_steps=100, max_steps=1)
    model = tc.ToyTransformer(config=tc.ModelConfig(), rngs=nnx.Rngs(0))
    trainer = peft_trainer.PeftTrainer(
        model, optax.sgd(1e-3), config
    ).with_gen_model_input_fn(dummy_gen_model_input_fn).with_loss_fn(loss_fn)

    def copy_state(value):
      return jax.tree.map(lambda x: np.asarray(x).copy(), nnx.state(value))

    model_before = copy_state(trainer.model)
    optimizer_before = copy_state(trainer.optimizer)
    accumulator_before = copy_state(trainer.grad_accumulator)
    with mock.patch.dict(
        os.environ,
        {
            'CANON_ALIGNMENT_GATE': '1',
            'CANON_ALIGNMENT_GATE_ONLY': '1',
        },
        clear=False,
    ):
      _, aux, grad_norm = trainer._train_step(
          trainer.model,
          trainer.optimizer,
          trainer.grad_accumulator,
          self.train_ds[0],
          jnp.array(True),
      )

    self.assertEqual(int(aux['canon/optimizer_skipped']), 1)
    self.assertGreater(float(grad_norm), 0.0)
    for before, after in zip(
        jax.tree_util.tree_leaves(model_before),
        jax.tree_util.tree_leaves(copy_state(trainer.model)),
    ):
      np.testing.assert_array_equal(before, after)
    for before, after in zip(
        jax.tree_util.tree_leaves(optimizer_before),
        jax.tree_util.tree_leaves(copy_state(trainer.optimizer)),
    ):
      np.testing.assert_array_equal(before, after)
    for before, after in zip(
        jax.tree_util.tree_leaves(accumulator_before),
        jax.tree_util.tree_leaves(copy_state(trainer.grad_accumulator)),
    ):
      np.testing.assert_array_equal(before, after)

  def test_alignment_gate_only_without_host_gate_is_rejected(self):
    config = peft_trainer.TrainingConfig(eval_every_n_steps=100, max_steps=1)
    model = tc.ToyTransformer(config=tc.ModelConfig(), rngs=nnx.Rngs(0))
    trainer = peft_trainer.PeftTrainer(
        model, optax.sgd(1e-3), config
    ).with_gen_model_input_fn(dummy_gen_model_input_fn)
    with mock.patch.dict(
        os.environ,
        {
            'CANON_ALIGNMENT_GATE': '0',
            'CANON_ALIGNMENT_GATE_ONLY': '1',
        },
        clear=False,
    ):
      with self.assertRaisesRegex(ValueError, 'requires CANON_ALIGNMENT_GATE=1'):
        trainer._train_step(
            trainer.model,
            trainer.optimizer,
            trainer.grad_accumulator,
            self.train_ds[0],
            jnp.array(True),
        )

  def test_alignment_update_canary_executes_optimizer(self):
    config = peft_trainer.TrainingConfig(eval_every_n_steps=100, max_steps=1)
    model = tc.ToyTransformer(config=tc.ModelConfig(), rngs=nnx.Rngs(0))
    trainer = peft_trainer.PeftTrainer(
        model, optax.sgd(1e-3), config
    ).with_gen_model_input_fn(dummy_gen_model_input_fn)

    before = jax.tree.map(lambda x: np.asarray(x).copy(), nnx.state(trainer.model))
    with mock.patch.dict(
        os.environ,
        {
            'CANON_ALIGNMENT_GATE': '1',
            'CANON_ALIGNMENT_GATE_ONLY': '0',
            'CANON_ALIGNMENT_UPDATE_CANARY': '1',
        },
        clear=False,
    ):
      _, aux, grad_norm = trainer._train_step(
          trainer.model,
          trainer.optimizer,
          trainer.grad_accumulator,
          self.train_ds[0],
          jnp.array(True),
      )

    self.assertEqual(int(aux['canon/optimizer_skipped']), 0)
    self.assertGreater(float(grad_norm), 0.0)
    after = jax.tree.map(lambda x: np.asarray(x).copy(), nnx.state(trainer.model))
    self.assertTrue(any(
        not np.array_equal(a, b)
        for a, b in zip(
            jax.tree_util.tree_leaves(before),
            jax.tree_util.tree_leaves(after),
        )
    ))

  def test_p28_g6_precomputed_four_microstep_update(self):
    config = peft_trainer.TrainingConfig(
        eval_every_n_steps=100,
        max_steps=1,
        gradient_accumulation_steps=4,
        checkpoint_root_directory=None,
    )
    model = tc.ToyTransformer(config=tc.ModelConfig(), rngs=nnx.Rngs(0))
    baseline_model = tc.ToyTransformer(
        config=tc.ModelConfig(), rngs=nnx.Rngs(0)
    )
    trainer = peft_trainer.PeftTrainer(model, optax.sgd(1e-3), config)
    baseline_trainer = peft_trainer.PeftTrainer(
        baseline_model, optax.sgd(1e-3), config
    )
    gradient_microbatches = tuple(
        jax.tree.map(
            lambda value, scale=scale: type(value)(
                jnp.full_like(value[...], scale)
            ),
            nnx.state(trainer.model, nnx.Param),
            is_leaf=lambda value: isinstance(value, nnx.VariableState),
        )
        for scale in (1.0, 2.0, 3.0, 4.0)
    )
    env = {
        "CANON_ALIGNMENT_GATE": "1",
        "CANON_ALIGNMENT_UPDATE_CANARY": "1",
        "CANON_ALIGNMENT_GATE_ONLY": "0",
        "CANON_ALIGNMENT_TRAIN": "0",
        "CANON_P28_SEGMENTED_TRAIN": "1",
        "CANON_P28_G5C_ONLY": "0",
        "CANON_P28_G6_UPDATE": "1",
    }
    with mock.patch.dict(os.environ, env, clear=False):
      norms = trainer.apply_precomputed_gradient_microbatches(
          gradient_microbatches
      )
      baseline_gradient_microbatches = tuple(
          jax.tree.map(
              lambda value: type(value)(jnp.full_like(value[...], 2.5)),
              nnx.state(baseline_model, nnx.Param),
              is_leaf=lambda value: isinstance(value, nnx.VariableState),
          )
          for _ in range(4)
      )
      baseline_trainer.apply_precomputed_gradient_microbatches(
          baseline_gradient_microbatches
      )

    self.assertLen(norms, 4)
    self.assertEqual(trainer.iter_steps, 4)
    self.assertEqual(trainer.train_steps, 1)
    for actual, expected in zip(
        jax.tree.leaves(nnx.state(trainer.model, nnx.Param)),
        jax.tree.leaves(nnx.state(baseline_model, nnx.Param)),
        strict=True,
    ):
      np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    for value in jax.tree.leaves(nnx.state(trainer.grad_accumulator)):
      np.testing.assert_array_equal(np.asarray(value), np.zeros_like(value))

    with mock.patch.dict(
        os.environ, {**env, "CANON_P28_G5C_ONLY": "1"}, clear=False
    ):
      with self.assertRaisesRegex(ValueError, "exclusive P28 G6"):
        trainer.apply_precomputed_gradient_microbatches(
            gradient_microbatches
        )

  def test_p58_precomputed_all_filtered_discard_resets_without_commit(self):
    config = peft_trainer.TrainingConfig(
        eval_every_n_steps=100,
        max_steps=1,
        gradient_accumulation_steps=4,
        checkpoint_root_directory=None,
    )
    model = tc.ToyTransformer(config=tc.ModelConfig(), rngs=nnx.Rngs(0))
    trainer = peft_trainer.PeftTrainer(model, optax.sgd(1e-3), config)
    gradients = tuple(
        jax.tree.map(
            lambda value: type(value)(jnp.zeros_like(value[...])),
            nnx.state(trainer.model, nnx.Param),
            is_leaf=lambda value: isinstance(value, nnx.VariableState),
        )
        for _ in range(4)
    )
    before = jax.tree.map(
        lambda value: np.asarray(value).copy(),
        nnx.state(trainer.model, nnx.Param),
    )
    env = {
        "CANON_ALIGNMENT_GATE": "1",
        "CANON_ALIGNMENT_UPDATE_CANARY": "1",
        "CANON_ALIGNMENT_GATE_ONLY": "0",
        "CANON_ALIGNMENT_TRAIN": "0",
        "CANON_P28_SEGMENTED_TRAIN": "1",
        "CANON_P28_G5C_ONLY": "0",
        "CANON_P28_G6_UPDATE": "1",
        "CANON_P58_DEEPSWE_TIM": "1",
    }
    with mock.patch.dict(os.environ, env, clear=False):
      for index, gradient in enumerate(gradients):
        trainer.accumulate_precomputed_gradient_microbatch(
            gradient, microbatch_index=index
        )
      denominator = trainer.discard_precomputed_gradients()

    self.assertEqual(float(denominator), 4.0)
    self.assertEqual(trainer.train_steps, 0)
    self.assertEqual(trainer._p28_precomputed_microstep, 0)
    for actual, expected in zip(
        jax.tree.leaves(nnx.state(trainer.model, nnx.Param)),
        jax.tree.leaves(before),
        strict=True,
    ):
      np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    for value in jax.tree.leaves(nnx.state(trainer.grad_accumulator)):
      np.testing.assert_array_equal(np.asarray(value), np.zeros_like(value))

  def test_p28_g6_checkpointing_is_isolated_to_signed_p45(self):
    checkpoint_directory = (
        "gs://yuxzhang-tunix-models/canon-zero-tim/checkpoints/"
        "frozenlake/fl-test/actor"
    )
    p45_contract = getattr(
        peft_trainer, "_P45_PRECOMPUTED_CHECKPOINT_CONTRACT"
    )
    env = {
        "CANON_ALIGNMENT_GATE": "1",
        "CANON_ALIGNMENT_GATE_ONLY": "0",
        "CANON_ALIGNMENT_TRAIN": "1",
        "CANON_P28_SEGMENTED_TRAIN": "1",
        "CANON_P28_G5C_ONLY": "0",
        "CANON_P28_G6_UPDATE": "1",
        "CANON_P33_WORKLOAD_LAUNCH_ADMITTED": "1",
        "CANON_LOCAL_TRAJECTORIES": "32",
        "CANON_P32_WORKLOAD": "frozenlake-dp8-tp8",
        "CANON_P33_RUN_STAGE": "full",
        "CANON_P33_NO_COMMIT": "0",
        "CANON_OPT_STATE_RESIDENT": "1",
        "CANON_P30_OPT_STATE_OFFLOAD": "0",
        "CANON_FROZENLAKE_CKPT_MODE": "new",
        "CANON_FROZENLAKE_CKPT_ROOT": (
            "gs://yuxzhang-tunix-models/canon-zero-tim/checkpoints/frozenlake"
        ),
        "CANON_FROZENLAKE_CKPT_TAG": "fl-test",
        "CANON_FROZENLAKE_CKPT_INTERVAL": "10",
        "CANON_FROZENLAKE_CKPT_MAX_TO_KEEP": "1",
        "ENABLE_PATHWAYS_PERSISTENCE": "1",
    }

    def make_trainer(contract):
      trainer = object.__new__(peft_trainer.PeftTrainer)
      trainer.config = peft_trainer.TrainingConfig(
          eval_every_n_steps=10,
          gradient_accumulation_steps=32,
          checkpoint_root_directory=checkpoint_directory,
          precomputed_gradient_checkpointing_contract=contract,
      )
      return trainer

    with mock.patch.dict(os.environ, env, clear=True):
      with self.assertRaisesRegex(ValueError, "checkpointing disabled"):
        make_trainer(None)._validate_precomputed_gradient_contract()
      make_trainer(p45_contract)._validate_precomputed_gradient_contract()

    drifted = {**env, "CANON_FROZENLAKE_CKPT_INTERVAL": "11"}
    with mock.patch.dict(os.environ, drifted, clear=True):
      with self.assertRaisesRegex(ValueError, "checkpointing disabled"):
        make_trainer(p45_contract)._validate_precomputed_gradient_contract()

  def test_p31_precomputed_sixteen_microstep_update(self):
    config = peft_trainer.TrainingConfig(
        eval_every_n_steps=100,
        max_steps=1,
        gradient_accumulation_steps=16,
        checkpoint_root_directory=None,
    )
    model = tc.ToyTransformer(config=tc.ModelConfig(), rngs=nnx.Rngs(0))
    baseline_model = tc.ToyTransformer(
        config=tc.ModelConfig(), rngs=nnx.Rngs(0)
    )
    trainer = peft_trainer.PeftTrainer(model, optax.sgd(1e-3), config)
    baseline = peft_trainer.PeftTrainer(
        baseline_model, optax.sgd(1e-3), config
    )
    gradients = tuple(
        jax.tree.map(
            lambda value, scale=scale: type(value)(
                jnp.full_like(value[...], scale)
            ),
            nnx.state(trainer.model, nnx.Param),
            is_leaf=lambda value: isinstance(value, nnx.VariableState),
        )
        for scale in range(1, 17)
    )
    baseline_gradients = tuple(
        jax.tree.map(
            lambda value: type(value)(jnp.full_like(value[...], 8.5)),
            nnx.state(baseline.model, nnx.Param),
            is_leaf=lambda value: isinstance(value, nnx.VariableState),
        )
        for _ in range(16)
    )
    env = {
        "CANON_ALIGNMENT_GATE": "1",
        "CANON_ALIGNMENT_GATE_ONLY": "0",
        "CANON_ALIGNMENT_UPDATE_CANARY": "0",
        "CANON_ALIGNMENT_TRAIN": "1",
        "CANON_P28_SEGMENTED_TRAIN": "1",
        "CANON_P28_G5C_ONLY": "0",
        "CANON_P28_G6_UPDATE": "1",
        "CANON_P31_CONVERGENCE": "1",
    }
    with mock.patch.dict(os.environ, env, clear=False):
      norms = trainer.apply_precomputed_gradient_microbatches(gradients)
      baseline.apply_precomputed_gradient_microbatches(baseline_gradients)

    self.assertLen(norms, 16)
    self.assertEqual(trainer.iter_steps, 16)
    self.assertEqual(trainer.train_steps, 1)
    for actual, expected in zip(
        jax.tree.leaves(nnx.state(trainer.model, nnx.Param)),
        jax.tree.leaves(nnx.state(baseline.model, nnx.Param)),
        strict=True,
    ):
      np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    for value in jax.tree.leaves(nnx.state(trainer.grad_accumulator)):
      np.testing.assert_array_equal(np.asarray(value), np.zeros_like(value))

  def test_p33_scaled_sixteen_group_update_matches_materialized_scale(self):
    def make_trainer():
      config = peft_trainer.TrainingConfig(
          eval_every_n_steps=100,
          max_steps=1,
          gradient_accumulation_steps=16,
          checkpoint_root_directory=None,
      )
      model = tc.ToyTransformer(config=tc.ModelConfig(), rngs=nnx.Rngs(0))
      return peft_trainer.PeftTrainer(model, optax.sgd(1e-3), config)

    scaled = make_trainer()
    materialized = make_trainer()
    env = {
        "CANON_ALIGNMENT_GATE": "1",
        "CANON_ALIGNMENT_GATE_ONLY": "0",
        "CANON_ALIGNMENT_UPDATE_CANARY": "0",
        "CANON_ALIGNMENT_TRAIN": "1",
        "CANON_P28_SEGMENTED_TRAIN": "1",
        "CANON_P28_G5C_ONLY": "0",
        "CANON_P28_G6_UPDATE": "1",
        "CANON_P31_CONVERGENCE": "0",
        "CANON_P33_WORKLOAD_LAUNCH_ADMITTED": "1",
    }
    multiplier = jnp.asarray(0.25, jnp.float32)
    with mock.patch.dict(os.environ, env, clear=False):
      for index in range(16):
        gradient = jax.tree.map(
            lambda value: type(value)(
                jnp.full_like(value[...], float(index + 1))
            ),
            nnx.state(scaled.model, nnx.Param),
            is_leaf=lambda value: isinstance(value, nnx.VariableState),
        )
        expected = jax.tree.map(
            lambda value: value * multiplier.astype(value.dtype), gradient
        )
        scaled_norm = scaled.accumulate_precomputed_scaled_gradient_microbatch(
            gradient, multiplier, microbatch_index=index
        )
        expected_norm = materialized.accumulate_precomputed_gradient_microbatch(
            expected, microbatch_index=index
        )
        np.testing.assert_array_equal(
            np.asarray(scaled_norm), np.asarray(expected_norm)
        )
      scaled_commit = scaled.commit_precomputed_gradients()
      expected_commit = materialized.commit_precomputed_gradients()
    np.testing.assert_array_equal(
        np.asarray(scaled_commit), np.asarray(expected_commit)
    )
    for actual, expected in zip(
        jax.tree.leaves(nnx.state(scaled.model, nnx.Param)),
        jax.tree.leaves(nnx.state(materialized.model, nnx.Param)),
        strict=True,
    ):
      np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    self.assertIsNone(scaled._jitted_precomputed_gradient_scaled_step_fn)
    self.assertIsNotNone(scaled._jitted_precomputed_gradient_scaled_step_impl)

  @parameterized.named_parameters(
      ("zero_lr", 0.0, False),
      ("constant_lr", 1.0e-3, True),
  )
  def test_p33_commit_evidence_tracks_effective_parameter_delta(
      self, learning_rate, expect_parameter_change
  ):
    schedule = optax.constant_schedule(learning_rate)
    config = peft_trainer.TrainingConfig(
        eval_every_n_steps=100,
        max_steps=1,
        gradient_accumulation_steps=16,
        checkpoint_root_directory=None,
    )
    model = tc.ToyTransformer(config=tc.ModelConfig(), rngs=nnx.Rngs(0))
    trainer = peft_trainer.PeftTrainer(
        model, optax.adamw(schedule), config
    )
    trainer.register_learning_rate_schedule(schedule)
    env = {
        "CANON_ALIGNMENT_GATE": "1",
        "CANON_ALIGNMENT_GATE_ONLY": "0",
        "CANON_ALIGNMENT_UPDATE_CANARY": "0",
        "CANON_ALIGNMENT_TRAIN": "1",
        "CANON_P28_SEGMENTED_TRAIN": "1",
        "CANON_P28_G5C_ONLY": "0",
        "CANON_P28_G6_UPDATE": "1",
        "CANON_P31_CONVERGENCE": "0",
        "CANON_P33_WORKLOAD_LAUNCH_ADMITTED": "1",
    }
    with mock.patch.dict(os.environ, env, clear=False):
      for index in range(16):
        gradient = jax.tree.map(
            lambda value: type(value)(jnp.ones_like(value[...])),
            nnx.state(trainer.model, nnx.Param),
            is_leaf=lambda value: isinstance(value, nnx.VariableState),
        )
        trainer.accumulate_precomputed_gradient_microbatch(
            gradient, microbatch_index=index
        )
      trainer.commit_precomputed_gradients()
    evidence = trainer.consume_precomputed_commit_evidence()
    self.assertEqual(evidence["effective_learning_rate"], learning_rate)
    self.assertGreater(evidence["gradient_nonzero_elements"], 0)
    self.assertTrue(evidence["gradient_finite"])
    self.assertTrue(evidence["parameter_delta_finite"])
    if expect_parameter_change:
      self.assertGreater(evidence["parameter_changed_elements"], 0)
      self.assertGreater(evidence["parameter_delta_max_abs"], 0.0)
    else:
      self.assertEqual(evidence["parameter_changed_elements"], 0)
      self.assertEqual(evidence["parameter_delta_max_abs"], 0.0)

  def test_p30_optimizer_offload_matches_two_device_commits(self):
    def make_trainer(*, optimizer_offload):
      config = peft_trainer.TrainingConfig(
          eval_every_n_steps=100,
          max_steps=2,
          gradient_accumulation_steps=4,
          checkpoint_root_directory=None,
          optimizer_state_dtype=jnp.float32,
          optimizer_offload=optimizer_offload,
      )
      model = tc.ToyTransformer(config=tc.ModelConfig(), rngs=nnx.Rngs(0))
      return peft_trainer.PeftTrainer(
          model, optax.adamw(1e-3), config
      )

    device_trainer = make_trainer(optimizer_offload=False)
    offload_trainer = make_trainer(optimizer_offload=True)
    self.assertEqual(
        offload_trainer.optimizer_state_memory_kinds(), ("pinned_host",)
    )
    self.assertEqual(device_trainer.optimizer_state_memory_kinds(), ("device",))

    env = {
        "CANON_ALIGNMENT_GATE": "1",
        "CANON_ALIGNMENT_UPDATE_CANARY": "1",
        "CANON_ALIGNMENT_GATE_ONLY": "0",
        "CANON_ALIGNMENT_TRAIN": "0",
        "CANON_P28_SEGMENTED_TRAIN": "1",
        "CANON_P28_G5C_ONLY": "0",
        "CANON_P28_G6_UPDATE": "1",
    }
    device_step_impl_ids = []
    device_commit_impl_ids = []
    offload_step_impl_ids = []
    offload_commit_impl_ids = []
    for commit, scales in enumerate(((1.0, 2.0, 3.0, 4.0),
                                     (-2.0, -1.0, 1.0, 3.0))):
      gradient_microbatches = tuple(
          jax.tree.map(
              lambda value, scale=scale: type(value)(
                  jnp.full_like(value[...], scale)
              ),
              nnx.state(device_trainer.model, nnx.Param),
              is_leaf=lambda value: isinstance(value, nnx.VariableState),
          )
          for scale in scales
      )
      with mock.patch.dict(os.environ, env, clear=False):
        device_norms = device_trainer.apply_precomputed_gradient_microbatches(
            gradient_microbatches
        )
      with mock.patch.dict(
          os.environ, {**env, "CANON_P30_POST_COMMIT_GC": "1"}, clear=False
      ):
        offload_norms = offload_trainer.apply_precomputed_gradient_microbatches(
            gradient_microbatches
        )

      self.assertEqual(device_trainer.train_steps, commit + 1)
      self.assertEqual(offload_trainer.train_steps, commit + 1)
      self.assertIsNone(device_trainer._jitted_precomputed_gradient_step_fn)
      self.assertIsNone(device_trainer._jitted_precomputed_gradient_commit_fn)
      self.assertIsNone(offload_trainer._jitted_precomputed_gradient_step_fn)
      self.assertIsNone(offload_trainer._jitted_precomputed_gradient_commit_fn)
      self.assertIsNotNone(
          device_trainer._jitted_precomputed_gradient_step_impl
      )
      self.assertIsNotNone(
          device_trainer._jitted_precomputed_gradient_commit_impl
      )
      self.assertIsNotNone(
          offload_trainer._jitted_precomputed_gradient_step_impl
      )
      self.assertIsNotNone(
          offload_trainer._jitted_precomputed_gradient_commit_impl
      )
      device_step_impl_ids.append(
          id(device_trainer._jitted_precomputed_gradient_step_impl)
      )
      device_commit_impl_ids.append(
          id(device_trainer._jitted_precomputed_gradient_commit_impl)
      )
      offload_step_impl_ids.append(
          id(offload_trainer._jitted_precomputed_gradient_step_impl)
      )
      offload_commit_impl_ids.append(
          id(offload_trainer._jitted_precomputed_gradient_commit_impl)
      )
      self.assertEqual(
          offload_trainer.optimizer_state_memory_kinds(), ("pinned_host",)
      )
      device_evidence = device_trainer.consume_precomputed_commit_evidence()
      offload_evidence = offload_trainer.consume_precomputed_commit_evidence()
      device_timing = device_evidence["optimizer_timing"]
      offload_timing = offload_evidence["optimizer_timing"]
      self.assertEqual(
          device_timing["optimizer_logical_bytes"],
          offload_timing["optimizer_logical_bytes"],
      )
      self.assertEqual(device_timing["optimizer_h2d_seconds"], 0.0)
      self.assertEqual(device_timing["optimizer_d2h_seconds"], 0.0)
      self.assertGreaterEqual(offload_timing["optimizer_h2d_seconds"], 0.0)
      self.assertGreaterEqual(offload_timing["optimizer_d2h_seconds"], 0.0)
      self.assertGreater(device_timing["adam_commit_seconds"], 0.0)
      self.assertGreater(offload_timing["adam_commit_seconds"], 0.0)
      for actual, expected in zip(device_norms, offload_norms, strict=True):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
      for device_value, offload_value in zip(
          jax.tree.leaves(nnx.state(device_trainer.model, nnx.Param)),
          jax.tree.leaves(nnx.state(offload_trainer.model, nnx.Param)),
          strict=True,
      ):
        np.testing.assert_array_equal(
            np.asarray(device_value), np.asarray(offload_value)
        )
      for device_value, offload_value in zip(
          jax.tree.leaves(
              nnx.state(device_trainer.optimizer, nnx.optimizer.OptState)
          ),
          jax.tree.leaves(
              nnx.state(offload_trainer.optimizer, nnx.optimizer.OptState)
          ),
          strict=True,
      ):
        self.assertEqual(device_value.shape, offload_value.shape)
        self.assertEqual(device_value.dtype, offload_value.dtype)
        np.testing.assert_array_equal(
            np.asarray(device_value), np.asarray(offload_value)
        )
    self.assertLen(set(device_step_impl_ids), 1)
    self.assertLen(set(device_commit_impl_ids), 1)
    self.assertLen(set(offload_step_impl_ids), 1)
    self.assertLen(set(offload_commit_impl_ids), 1)

  def test_p30_post_commit_gc_runs_after_cached_bindings_clear(self):
    config = peft_trainer.TrainingConfig(
        eval_every_n_steps=100,
        max_steps=1,
        gradient_accumulation_steps=4,
        checkpoint_root_directory=None,
        optimizer_state_dtype=jnp.float32,
    )
    model = tc.ToyTransformer(config=tc.ModelConfig(), rngs=nnx.Rngs(0))
    trainer = peft_trainer.PeftTrainer(model, optax.adamw(1e-3), config)
    gradients = tuple(
        jax.tree.map(
            lambda value: type(value)(jnp.ones_like(value[...])),
            nnx.state(trainer.model, nnx.Param),
            is_leaf=lambda value: isinstance(value, nnx.VariableState),
        )
        for _ in range(4)
    )
    env = {
        "CANON_ALIGNMENT_GATE": "1",
        "CANON_ALIGNMENT_UPDATE_CANARY": "1",
        "CANON_ALIGNMENT_GATE_ONLY": "0",
        "CANON_ALIGNMENT_TRAIN": "0",
        "CANON_P28_SEGMENTED_TRAIN": "1",
        "CANON_P28_G5C_ONLY": "0",
        "CANON_P28_G6_UPDATE": "1",
        "CANON_P30_POST_COMMIT_GC": "1",
    }

    def collect_after_clear():
      self.assertIsNone(trainer._jitted_precomputed_gradient_step_fn)
      self.assertIsNone(trainer._jitted_precomputed_gradient_pair_step_fn)
      self.assertIsNone(trainer._jitted_precomputed_gradient_commit_fn)
      return 7

    with mock.patch.dict(os.environ, env, clear=False):
      with mock.patch.object(
          peft_trainer.gc, "collect", side_effect=collect_after_clear
      ) as collect_mock:
        trainer.apply_precomputed_gradient_microbatches(gradients)
    collect_mock.assert_called_once_with()

  def test_p30_model_donation_matches_two_undonated_commits(self):
    def make_trainer():
      config = peft_trainer.TrainingConfig(
          eval_every_n_steps=100,
          max_steps=2,
          gradient_accumulation_steps=4,
          checkpoint_root_directory=None,
          optimizer_state_dtype=jnp.float32,
      )
      model = tc.ToyTransformer(config=tc.ModelConfig(), rngs=nnx.Rngs(0))
      return peft_trainer.PeftTrainer(model, optax.adamw(1e-3), config)

    undonated = make_trainer()
    donated = make_trainer()
    env = {
        "CANON_ALIGNMENT_GATE": "1",
        "CANON_ALIGNMENT_UPDATE_CANARY": "1",
        "CANON_ALIGNMENT_GATE_ONLY": "0",
        "CANON_ALIGNMENT_TRAIN": "0",
        "CANON_P28_SEGMENTED_TRAIN": "1",
        "CANON_P28_G5C_ONLY": "0",
        "CANON_P28_G6_UPDATE": "1",
    }
    donated_impl_ids = []
    for scales in ((1.0, 2.0, 3.0, 4.0), (-2.0, -1.0, 1.0, 3.0)):
      gradients = tuple(
          jax.tree.map(
              lambda value, scale=scale: type(value)(
                  jnp.full_like(value[...], scale)
              ),
              nnx.state(undonated.model, nnx.Param),
              is_leaf=lambda value: isinstance(value, nnx.VariableState),
          )
          for scale in scales
      )
      with mock.patch.dict(os.environ, env, clear=False):
        undonated_norms = undonated.apply_precomputed_gradient_microbatches(
            gradients
        )
      with mock.patch.dict(
          os.environ, {**env, "CANON_P30_DONATE_MODEL": "1"}, clear=False
      ):
        donated_norms = donated.apply_precomputed_gradient_microbatches(
            gradients
        )
      donated_impl_ids.append(
          id(donated._jitted_precomputed_gradient_commit_impl)
      )
      for actual, expected in zip(
          donated_norms, undonated_norms, strict=True
      ):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
      for actual_tree, expected_tree in (
          (
              nnx.state(donated.model, nnx.Param),
              nnx.state(undonated.model, nnx.Param),
          ),
          (
              nnx.state(donated.optimizer, nnx.optimizer.OptState),
              nnx.state(undonated.optimizer, nnx.optimizer.OptState),
          ),
          (
              nnx.state(donated.grad_accumulator),
              nnx.state(undonated.grad_accumulator),
          ),
      ):
        for actual, expected in zip(
            jax.tree.leaves(actual_tree),
            jax.tree.leaves(expected_tree),
            strict=True,
        ):
          np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    self.assertLen(set(donated_impl_ids), 1)

  def test_p30_accumulator_reshard_matches_two_unrepaired_commits(self):
    def make_trainer():
      model, _ = create_sharded_model(
          tc.ToyTransformer, nnx.Rngs(0), self.mesh
      )
      config = peft_trainer.TrainingConfig(
          eval_every_n_steps=100,
          max_steps=2,
          gradient_accumulation_steps=4,
          checkpoint_root_directory=None,
          optimizer_state_dtype=jnp.float32,
      )
      return peft_trainer.PeftTrainer(
          model, optax.adamw(1e-3), config
      )

    with compat.set_mesh(self.mesh):
      baseline = make_trainer()
      repaired = make_trainer()
      env = {
          "CANON_ALIGNMENT_GATE": "1",
          "CANON_ALIGNMENT_UPDATE_CANARY": "1",
          "CANON_ALIGNMENT_GATE_ONLY": "0",
          "CANON_ALIGNMENT_TRAIN": "0",
          "CANON_P28_SEGMENTED_TRAIN": "1",
          "CANON_P28_G5C_ONLY": "0",
          "CANON_P28_G6_UPDATE": "1",
      }
      for scales in ((1.0, 2.0, 3.0, 4.0), (-2.0, -1.0, 1.0, 3.0)):
        gradients = tuple(
            jax.tree.map(
                lambda value, scale=scale: type(value)(
                    jnp.full_like(value[...], scale)
                ),
                nnx.state(baseline.model, nnx.Param),
                is_leaf=lambda value: isinstance(value, nnx.VariableState),
            )
            for scale in scales
        )
        with mock.patch.dict(os.environ, env, clear=False):
          baseline_norms = baseline.apply_precomputed_gradient_microbatches(
              gradients
          )
        with mock.patch.dict(
            os.environ,
            {**env, "CANON_P30_RESHARD_ACCUMULATOR": "1"},
            clear=False,
        ):
          repaired_norms = repaired.apply_precomputed_gradient_microbatches(
              gradients
          )
        for actual, expected in zip(
            repaired_norms, baseline_norms, strict=True
        ):
          np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
        for actual_tree, expected_tree in (
            (
                nnx.state(repaired.model, nnx.Param),
                nnx.state(baseline.model, nnx.Param),
            ),
            (
                nnx.state(repaired.optimizer, nnx.optimizer.OptState),
                nnx.state(baseline.optimizer, nnx.optimizer.OptState),
            ),
            (
                nnx.state(repaired.grad_accumulator),
                nnx.state(baseline.grad_accumulator),
            ),
        ):
          for actual, expected in zip(
              jax.tree.leaves(actual_tree),
              jax.tree.leaves(expected_tree),
              strict=True,
          ):
            np.testing.assert_array_equal(
                np.asarray(actual), np.asarray(expected)
            )

      accumulator = repaired.grad_accumulator.grads
      pspecs = nnx.get_partition_spec(accumulator)
      for value, pspec in zip(
          jax.tree.leaves(accumulator),
          jax.tree.leaves(pspecs),
          strict=True,
      ):
        if isinstance(value, jax.Array):
          target = peft_trainer.sharding_utils.get_sharding(
              value,
              self.mesh,
              pspec if pspec is not None else shd.PartitionSpec(),
          )
          self.assertEqual(value.sharding, target)

  def test_p30_optimizer_offload_transfer_failure_is_loud(self):
    config = peft_trainer.TrainingConfig(
        eval_every_n_steps=100,
        max_steps=1,
        optimizer_offload=True,
    )
    model = tc.ToyTransformer(config=tc.ModelConfig(), rngs=nnx.Rngs(0))
    with mock.patch.object(
        peft_trainer,
        "_put_state_on_memory_kind",
        side_effect=RuntimeError("pinned host unavailable"),
    ):
      with self.assertRaisesRegex(RuntimeError, "pinned host unavailable"):
        peft_trainer.PeftTrainer(model, optax.adamw(1e-3), config)

  def test_p30_fused_pair_accumulation_matches_materialized_update(self):
    def make_trainer():
      config = peft_trainer.TrainingConfig(
          eval_every_n_steps=100,
          max_steps=2,
          gradient_accumulation_steps=4,
          checkpoint_root_directory=None,
          optimizer_state_dtype=jnp.float32,
      )
      model = tc.ToyTransformer(config=tc.ModelConfig(), rngs=nnx.Rngs(0))
      return peft_trainer.PeftTrainer(model, optax.adamw(1e-3), config)

    materialized = make_trainer()
    fused = make_trainer()
    env = {
        "CANON_ALIGNMENT_GATE": "1",
        "CANON_ALIGNMENT_UPDATE_CANARY": "1",
        "CANON_ALIGNMENT_GATE_ONLY": "0",
        "CANON_ALIGNMENT_TRAIN": "0",
        "CANON_P28_SEGMENTED_TRAIN": "1",
        "CANON_P28_G5C_ONLY": "0",
        "CANON_P28_G6_UPDATE": "1",
        "CANON_P30_FUSED_PAIR_ACCUMULATION": "1",
    }
    fused_impl_ids = []
    for commit in range(2):
      with mock.patch.dict(os.environ, env, clear=False):
        for index in range(4):
          left_scale = float(commit * 8 + index * 2 + 1)
          right_scale = float(commit * 8 + index * 2 + 2)
          left = jax.tree.map(
              lambda value: type(value)(
                  jnp.full_like(value[...], left_scale)
              ),
              nnx.state(materialized.model, nnx.Param),
              is_leaf=lambda value: isinstance(value, nnx.VariableState),
          )
          right = jax.tree.map(
              lambda value: type(value)(
                  jnp.full_like(value[...], right_scale)
              ),
              nnx.state(materialized.model, nnx.Param),
              is_leaf=lambda value: isinstance(value, nnx.VariableState),
          )
          multiplier = jnp.asarray(0.5, jnp.float32)
          legacy_gradient = jax.tree.map(
              lambda a, b: (a + b) * multiplier.astype(a.dtype),
              left,
              right,
          )
          legacy_norm = materialized.accumulate_precomputed_gradient_microbatch(
              legacy_gradient, microbatch_index=index
          )
          fused_norm = fused.accumulate_precomputed_gradient_pair_microbatch(
              left, right, multiplier, microbatch_index=index
          )
          np.testing.assert_array_equal(
              np.asarray(fused_norm), np.asarray(legacy_norm)
          )
        legacy_commit_norm = materialized.commit_precomputed_gradients()
        fused_commit_norm = fused.commit_precomputed_gradients()
      np.testing.assert_array_equal(
          np.asarray(fused_commit_norm), np.asarray(legacy_commit_norm)
      )
      self.assertIsNone(fused._jitted_precomputed_gradient_pair_step_fn)
      self.assertIsNotNone(fused._jitted_precomputed_gradient_pair_step_impl)
      fused_impl_ids.append(
          id(fused._jitted_precomputed_gradient_pair_step_impl)
      )
      for actual, expected in zip(
          jax.tree.leaves(nnx.state(fused.model, nnx.Param)),
          jax.tree.leaves(nnx.state(materialized.model, nnx.Param)),
          strict=True,
      ):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
      for actual, expected in zip(
          jax.tree.leaves(nnx.state(fused.optimizer, nnx.optimizer.OptState)),
          jax.tree.leaves(
              nnx.state(materialized.optimizer, nnx.optimizer.OptState)
          ),
          strict=True,
      ):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    self.assertLen(set(fused_impl_ids), 1)

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

  def test_setitem_vs_set_value_write_equivalence(self):
    """set_value(x) stores the same value/dtype/type as `v[...] = x`."""
    v_setitem = nnx.Variable(jnp.arange(6, dtype=jnp.float32).reshape(2, 3))
    v_setvalue = nnx.Variable(jnp.arange(6, dtype=jnp.float32).reshape(2, 3))
    new = jnp.full((2, 3), 7.0, dtype=jnp.float32)
    v_setitem[...] = new
    v_setvalue.set_value(new)
    np.testing.assert_array_equal(v_setitem[...], v_setvalue[...])
    self.assertEqual(v_setitem[...].dtype, v_setvalue[...].dtype)
    self.assertIs(type(v_setitem), type(v_setvalue))

  def test_depth1_single_add_denom_one_is_identity(self):
    """At depth 1, add(g, denom=1) -> get() returns g exactly (fast-path premise)."""
    model, acc = self._make_accumulator()
    grads = self._ones_like_params(model, scale=2.5)
    acc.add(grads, denom=jnp.asarray(1.0, dtype=jnp.float32))
    out = _unwrap(acc.get())
    jax.tree_util.tree_map(
        lambda g, o: np.testing.assert_allclose(o, g, rtol=1e-7, atol=1e-7),
        _unwrap(grads),
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
    # get() emits the configured accumulator dtype (here matched to the param
    # dtype); the trainer casts back to the param dtype before the update.
    acc = peft_trainer.GradientAccumulator(
        model, nnx.Param, accumulator_dtype=dtype
    )

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

  def test_weighted_get_is_sum_of_grads_over_sum_of_denoms(self):
    """get() == (Σ grads) / (Σ denom): the weighted mean for variable packs."""
    isvar = lambda x: isinstance(x, nnx.Variable)
    rngs = nnx.Rngs(0)
    model = nnx.Linear(8, 8, rngs=rngs, param_dtype=jnp.float32)

    def const_grads(c):
      return jax.tree_util.tree_map(
          lambda v: type(v)(jnp.full_like(v[...], c)),
          nnx.state(model, nnx.Param),
          is_leaf=isvar,
      )

    acc = peft_trainer.GradientAccumulator(model, nnx.Param)  # fp32 default
    acc.add(const_grads(1.0), denom=jnp.asarray(3.0, jnp.float32))
    acc.add(const_grads(2.0), denom=jnp.asarray(1.0, jnp.float32))
    out = acc.get()  # (1 + 2) / (3 + 1) = 0.75

    jax.tree_util.tree_map(
        lambda v: np.testing.assert_allclose(
            np.asarray(v[...]), 0.75, rtol=1e-6
        ),
        out,
        is_leaf=isvar,
    )

  def test_fp32_accumulation_closer_to_golden_than_bf16(self):
    """fp32 accumulator is far closer to the full-batch golden than bf16.

    Locks in the P2.0 finding: summing bf16 grads over many microbatches loses
    small contributions (swamping), while fp32 accumulation stays near-exact.
    Uses an fp32 golden (no x64): the fp32 accumulator should match it, the
    bf16 accumulator should not.
    """
    isvar = lambda x: isinstance(x, nnx.Variable)
    rngs = nnx.Rngs(0)
    model = nnx.Linear(64, 64, rngs=rngs, param_dtype=jnp.bfloat16)

    def micro(seed):
      return jax.tree_util.tree_map(
          lambda v: type(v)(
              jax.random.normal(jax.random.PRNGKey(seed), v.shape, jnp.bfloat16)
          ),
          nnx.state(model, nnx.Param),
          is_leaf=isvar,
      )

    grads = [micro(s) for s in range(32)]

    def accumulate(dtype):
      acc = peft_trainer.GradientAccumulator(
          model, nnx.Param, accumulator_dtype=dtype
      )
      for g in grads:
        acc.add(g, denom=jnp.asarray(1.0, jnp.float32))
      return acc.get()

    golden = jax.tree_util.tree_map(  # fp32 mean of the bf16 grads
        lambda *gs: type(gs[0])(
            sum(g[...].astype(jnp.float32) for g in gs) / len(gs)
        ),
        *grads,
        is_leaf=isvar,
    )

    def rel_err(out):
      gl = jax.tree_util.tree_leaves(golden, is_leaf=isvar)
      ol = jax.tree_util.tree_leaves(out, is_leaf=isvar)
      num = sum(
          float(jnp.sum((o[...].astype(jnp.float32) - g[...]) ** 2))
          for o, g in zip(ol, gl)
      )
      den = sum(float(jnp.sum(g[...] ** 2)) for g in gl)
      return (num**0.5) / (den**0.5)

    err_bf16 = rel_err(accumulate(jnp.bfloat16))
    err_fp32 = rel_err(accumulate(jnp.float32))
    self.assertLess(err_fp32 * 50, err_bf16)  # fp32 at least 50x closer

  def test_cond_path_keeps_bf16_moments_after_update(self):
    """A depth>1 adam step keeps moments bf16 through the fp32 accumulator.

    The accumulator sums in float32, but `apply_updates` casts back to the param
    dtype before `optimizer.update`; otherwise the bf16 moments would promote to
    float32 and the two nnx.cond branches would mismatch (raising at trace time).
    Runs the production `_train_step` for one skip + one update micro-step.
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
    config = peft_trainer.TrainingConfig(
        eval_every_n_steps=100, max_steps=2, gradient_accumulation_steps=2
    )
    trainer = peft_trainer.PeftTrainer(
        model, optax.adamw(1e-3), config
    ).with_gen_model_input_fn(dummy_gen_model_input_fn)
    # Accumulator grads live in float32 (default), but moments stay bf16.
    acc_float_dtypes = {
        str(x.dtype)
        for x in jax.tree_util.tree_leaves(trainer.grad_accumulator.grads)
        if hasattr(x, 'dtype') and jnp.issubdtype(x.dtype, jnp.floating)
    }
    self.assertEqual(acc_float_dtypes, {'float32'})

    ds = dummy_datasets(batch_size=4)
    for flag in (jnp.array(False), jnp.array(True)):  # accumulate, then apply
      trainer._train_step(
          trainer.model,
          trainer.optimizer,
          trainer.grad_accumulator,
          ds[0],
          flag,
      )

    moment_dtypes = {
        str(x.dtype)
        for x in jax.tree_util.tree_leaves(
            nnx.state(trainer.optimizer, nnx.optimizer.OptState)
        )
        if hasattr(x, 'dtype') and jnp.issubdtype(x.dtype, jnp.floating)
    }
    self.assertEqual(moment_dtypes, {'bfloat16'})

  def test_peft_trainer_keeps_bf16_moments_on_cond_path(self):
    """Default None keeps bf16 moments on the cond path, even with an fp32

    inject_hyperparams learning rate. Depth>1 (and packing) take the `nnx.cond`
    path; bf16 moments trace fine there, so nothing is promoted to float32. Set
    `optimizer_state_dtype=jnp.float32` to force fp32 moments.
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
    config = peft_trainer.TrainingConfig(
        eval_every_n_steps=100, max_steps=1, gradient_accumulation_steps=2
    )  # depth>1 -> cond path; moments stay bf16 (param dtype)
    trainer = peft_trainer.PeftTrainer(model, tx, config)

    opt_state_dtypes = jax.tree_util.tree_leaves(
        jax.tree_util.tree_map(
            lambda v: v[...].dtype,
            nnx.state(trainer.optimizer, nnx.optimizer.OptState),
            is_leaf=lambda x: isinstance(x, nnx.Variable),
        )
    )
    float_dtypes = {
        str(d) for d in opt_state_dtypes if jnp.issubdtype(d, jnp.floating)
    }
    # Moments (mu/nu, the bulk of opt-state) stay bf16 — not promoted to fp32.
    # The only fp32 float leaf is the injected learning rate (hyperparam_dtype).
    self.assertIn('bfloat16', float_dtypes)
    self.assertEqual(float_dtypes - {'bfloat16'}, {'float32'})


class Depth1FastPathTest(parameterized.TestCase):
  """Depth-1 fast path: numeric equivalence, accumulator untouched, no

  `cond` in the depth-1 jaxpr, packing keeps the cond path.
  """

  def _make_trainer(self, accum_steps=None, max_seq_token=None):
    config = peft_trainer.TrainingConfig(
        eval_every_n_steps=1,
        max_steps=4,
        gradient_accumulation_steps=accum_steps,
        max_seq_token_per_tpu=max_seq_token,
    )
    rngs = nnx.Rngs(0)
    model = tc.ToyTransformer(config=tc.ModelConfig(), rngs=rngs)
    trainer = peft_trainer.PeftTrainer(model, optax.sgd(1e-3), config)
    return trainer.with_gen_model_input_fn(dummy_gen_model_input_fn)

  def _train_step_primitives(self, trainer):
    """Top-level jaxpr primitive names of one traced `_train_step` call."""
    x = dummy_datasets(batch_size=4)[0]
    graphdef, state = nnx.split(
        (trainer.model, trainer.optimizer, trainer.grad_accumulator)
    )

    def pure_step(state, tokens, mask, flag):
      model, optimizer, accumulator = nnx.merge(graphdef, state)
      out = trainer._train_step(
          model,
          optimizer,
          accumulator,
          peft_trainer.TrainingInput(input_tokens=tokens, input_mask=mask),
          flag,
      )
      _, new_state = nnx.split((model, optimizer, accumulator))
      return out, new_state

    jaxpr = jax.make_jaxpr(pure_step)(
        state, x.input_tokens, x.input_mask, jnp.array(True)
    )
    return {eqn.primitive.name for eqn in jaxpr.eqns}

  def test_depth1_fast_path_matches_accumulator_path(self):
    """Direct update from grads matches add(1) -> get() -> update -> reset."""
    rngs_a, rngs_b = nnx.Rngs(0), nnx.Rngs(0)
    model_a = nnx.Linear(in_features=4, out_features=2, rngs=rngs_a)
    model_b = nnx.Linear(in_features=4, out_features=2, rngs=rngs_b)
    opt_a = nnx.Optimizer(model_a, optax.adamw(1e-3), wrt=nnx.Param)
    opt_b = nnx.Optimizer(model_b, optax.adamw(1e-3), wrt=nnx.Param)
    grads = jax.tree_util.tree_map(
        lambda x: 0.5 * jnp.ones_like(x), nnx.state(model_a, nnx.Param)
    )

    # Old accumulator path (what depth 1 used to run).
    acc = peft_trainer.GradientAccumulator(model_a, nnx.Param)
    acc.add(grads, denom=jnp.asarray(1.0, dtype=jnp.float32))
    acc_grads = acc.get()
    norm_a = optax.global_norm(
        jax.tree_util.tree_map(lambda x: x.astype(jnp.float32), acc_grads)
    )
    opt_a.update(model_a, acc_grads)
    acc.reset()

    # New fast path (direct update).
    norm_b = optax.global_norm(
        jax.tree_util.tree_map(lambda x: x.astype(jnp.float32), grads)
    )
    opt_b.update(model_b, grads)

    np.testing.assert_allclose(norm_a, norm_b, rtol=1e-7)
    jax.tree_util.tree_map(
        lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6),
        _unwrap(nnx.state(model_a, nnx.Param)),
        _unwrap(nnx.state(model_b, nnx.Param)),
    )

  def test_depth1_accumulator_untouched(self):
    """Depth-1 training must not write the accumulator (keeps shardings stable)."""
    trainer = self._make_trainer(accum_steps=None)
    trainer.train(dummy_datasets(batch_size=4))
    jax.tree_util.tree_map(
        lambda v: np.testing.assert_array_equal(v, jnp.zeros_like(v)),
        _unwrap(trainer.grad_accumulator.grads),
    )
    np.testing.assert_array_equal(trainer.grad_accumulator.denom[...], 0.0)

  def test_depth1_jaxpr_has_no_cond(self):
    """Sentinel: the depth-1 step jaxpr must contain no `cond` primitive."""
    self.assertNotIn('cond', self._train_step_primitives(self._make_trainer()))

  def test_depth_gt1_jaxpr_has_cond(self):
    """Depth>1 keeps the cond path (update cadence needs it)."""
    self.assertIn(
        'cond', self._train_step_primitives(self._make_trainer(accum_steps=2))
    )

  def test_packing_config_keeps_cond_path(self):
    """Packing keeps the cond path at depth 1 (data-driven update cadence)."""
    trainer = self._make_trainer(accum_steps=None, max_seq_token=64)
    self.assertIn('cond', self._train_step_primitives(trainer))

  def test_packing_config_respects_skip_step(self):
    """With packing config, is_update_step=False must not update weights."""
    trainer = self._make_trainer(accum_steps=None, max_seq_token=64)
    before = jax.tree.map(
        jnp.copy, _unwrap(nnx.state(trainer.model, nnx.Param))
    )
    _, _, grad_norm = trainer._train_step(
        trainer.model,
        trainer.optimizer,
        trainer.grad_accumulator,
        dummy_datasets(batch_size=4)[0],
        jnp.array(False),
    )
    np.testing.assert_array_equal(grad_norm, 0.0)
    jax.tree_util.tree_map(
        np.testing.assert_array_equal,
        before,
        _unwrap(nnx.state(trainer.model, nnx.Param)),
    )

  def test_data_driven_false_flag_raises_on_fast_path_config(self):
    """A data-driven is_update_step=False must fail loudly at depth 1."""

    class _FlaggedInput:

      def __init__(self, base):
        self.input_tokens = base.input_tokens
        self.input_mask = base.input_mask
        self.is_update_step = False

    trainer = self._make_trainer(accum_steps=None)
    ds = [_FlaggedInput(dummy_datasets(batch_size=4)[0])]
    with self.assertRaisesRegex(ValueError, 'is_update_step=False'):
      trainer.train(ds)

  def test_depth2_cadence(self):
    """Depth 2: skip step accumulates only; update step applies and resets."""
    trainer = self._make_trainer(accum_steps=2)
    ds = dummy_datasets(batch_size=4)
    before = jax.tree.map(
        jnp.copy, _unwrap(nnx.state(trainer.model, nnx.Param))
    )

    # Micro-step 1: accumulate only.
    trainer._train_step(
        trainer.model,
        trainer.optimizer,
        trainer.grad_accumulator,
        ds[0],
        jnp.array(False),
    )
    jax.tree_util.tree_map(
        np.testing.assert_array_equal,
        before,
        _unwrap(nnx.state(trainer.model, nnx.Param)),
    )
    np.testing.assert_array_equal(trainer.grad_accumulator.denom[...], 1.0)

    # Micro-step 2: update and reset.
    trainer._train_step(
        trainer.model,
        trainer.optimizer,
        trainer.grad_accumulator,
        ds[1],
        jnp.array(True),
    )
    with self.assertRaises(AssertionError):
      jax.tree_util.tree_map(
          np.testing.assert_array_equal,
          before,
          _unwrap(nnx.state(trainer.model, nnx.Param)),
      )
    jax.tree_util.tree_map(
        lambda v: np.testing.assert_array_equal(v, jnp.zeros_like(v)),
        _unwrap(trainer.grad_accumulator.grads),
    )
    np.testing.assert_array_equal(trainer.grad_accumulator.denom[...], 0.0)


class OptimizerMemoryTest(parameterized.TestCase):
  """Optimizer-state HBM knobs: moment dtype + depth-1 accumulator skip."""

  def _bf16_model(self):
    model = tc.ToyTransformer(config=tc.ModelConfig(), rngs=nnx.Rngs(0))
    state = nnx.state(model, nnx.Param)
    state = jax.tree.map(
        lambda v: v.astype(jnp.bfloat16)
        if jnp.issubdtype(v.dtype, jnp.floating)
        else v,
        state,
    )
    nnx.update(model, state)
    return model

  def _moment_float_dtypes(self, trainer):
    st = nnx.state(trainer.optimizer, nnx.optimizer.OptState)
    return {
        str(x.dtype)
        for x in jax.tree_util.tree_leaves(st)
        if hasattr(x, 'dtype') and jnp.issubdtype(x.dtype, jnp.floating)
    }

  @parameterized.named_parameters(
      ('float32', jnp.float32, 'float32'),
      ('bfloat16', jnp.bfloat16, 'bfloat16'),
  )
  def test_optimizer_state_dtype_casts_moments(self, dtype, expected):
    """`optimizer_state_dtype` casts the Adam moment trees to that dtype."""
    model = self._bf16_model()
    config = peft_trainer.TrainingConfig(
        eval_every_n_steps=100, max_steps=1, optimizer_state_dtype=dtype
    )
    trainer = peft_trainer.PeftTrainer(model, optax.adamw(1e-3), config)
    self.assertEqual(self._moment_float_dtypes(trainer), {expected})

  def test_optimizer_state_dtype_none_keeps_param_dtype_on_fast_path(self):
    """Default None + depth-1 keeps moments at the param dtype (no forced fp32)."""
    model = self._bf16_model()  # bf16 params
    config = peft_trainer.TrainingConfig(eval_every_n_steps=100, max_steps=1)
    trainer = peft_trainer.PeftTrainer(model, optax.adamw(1e-3), config)
    # bf16 params -> bf16 moments (matching optax), NOT promoted to fp32.
    self.assertEqual(self._moment_float_dtypes(trainer), {'bfloat16'})

  def test_optimizer_state_dtype_none_keeps_param_dtype_on_cond_path(self):
    """Default None + depth>1 keeps moments at the param dtype (no forced fp32)."""
    model = self._bf16_model()  # bf16 params
    config = peft_trainer.TrainingConfig(
        eval_every_n_steps=100, max_steps=1, gradient_accumulation_steps=2
    )
    trainer = peft_trainer.PeftTrainer(model, optax.adamw(1e-3), config)
    # bf16 params -> bf16 moments on the cond path too (set fp32 to override).
    self.assertEqual(self._moment_float_dtypes(trainer), {'bfloat16'})

  def test_accumulator_grads_skipped_at_depth1(self):
    """At depth-1 (non-packing) the accumulator grad tree is not allocated."""
    model = tc.ToyTransformer(config=tc.ModelConfig(), rngs=nnx.Rngs(0))
    config = peft_trainer.TrainingConfig(
        eval_every_n_steps=100, max_steps=1
    )  # gradient_accumulation_steps=None -> depth 1
    trainer = peft_trainer.PeftTrainer(model, optax.sgd(1e-3), config)
    self.assertEmpty(
        jax.tree_util.tree_leaves(trainer.grad_accumulator.grads)
    )

  @parameterized.named_parameters(
      ('depth2', 2, None),
      ('packing', None, 16),
  )
  def test_accumulator_grads_allocated_when_used(self, steps, max_seq_token):
    """Depth>1 or packing keeps the accumulator grad tree allocated."""
    model = tc.ToyTransformer(config=tc.ModelConfig(), rngs=nnx.Rngs(0))
    config = peft_trainer.TrainingConfig(
        eval_every_n_steps=100,
        max_steps=1,
        gradient_accumulation_steps=steps,
        max_seq_token_per_tpu=max_seq_token,
    )
    trainer = peft_trainer.PeftTrainer(model, optax.sgd(1e-3), config)
    self.assertNotEmpty(
        jax.tree_util.tree_leaves(trainer.grad_accumulator.grads)
    )


if __name__ == '__main__':
  absltest.main()
