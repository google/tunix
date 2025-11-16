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

"""Disttillation trainer unittest."""

import os
from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
import jax.sharding as shd
import numpy as np
import optax
from tunix.distillation import distillation_trainer
from tunix.distillation import strategies
from tunix.tests import test_common as tc


os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'
_VOCAB_SIZE = 256


def create_sharded_model(model_ctor, rngs, mesh):
  @nnx.jit(static_argnums=(0,))
  def _create_sharded_model(model_ctor, rngs):
    model = model_ctor(rngs, vocab_size=_VOCAB_SIZE)
    state = nnx.state(model)
    pspecs = nnx.get_partition_spec(state)
    sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
    nnx.update(model, sharded_state)
    return model, state

  with mesh:
    model, state = _create_sharded_model(model_ctor, rngs)
  state_sharding = nnx.get_named_sharding(state, mesh)
  return model, state_sharding


def dummy_gen_model_input_fn(x: distillation_trainer.TrainingInput):
  return {
      'input_tokens': x.input_tokens,
      'input_mask': x.input_mask,
      'positions': jnp.arange(x.input_tokens.shape[1]),
      'attention_mask': jnp.ones_like(x.input_tokens),
  }


def get_labels_fn(
    input_tokens: jax.Array,
    input_mask: jax.Array,
    **kwargs,
):
  del kwargs
  target_tokens = input_tokens[:, 1:]
  target_mask = input_mask[:, 1:]
  # Convert the target labels to one-hot encoded vectors.
  labels = jax.nn.one_hot(target_tokens, _VOCAB_SIZE)
  # Don't update on unwanted tokens.
  return labels * target_mask.astype(labels.dtype)[..., None]


def get_model_outputs_fn(
    model: nnx.Module,
    input_tokens: jax.Array,
    input_mask: jax.Array,
    positions: jax.Array,
    attention_mask: jax.Array,
):
  del input_mask
  logits, _ = model(
      input_tokens,
      positions,
      None,
      attention_mask,
  )
  # Exclude the last step as it does not appear in the targets.
  return logits[:, :-1, :]


def get_toy_logit_strategy():
  return strategies.LogitStrategy(
      student_forward_fn=get_model_outputs_fn,
      teacher_forward_fn=get_model_outputs_fn,
      labels_fn=get_labels_fn,
  )


def get_toy_attention_transfer_strategy():
  return strategies.AttentionTransferStrategy(
      student_forward_fn=get_model_outputs_fn,
      teacher_forward_fn=get_model_outputs_fn,
      labels_fn=get_labels_fn,
      attention_layer=nnx.MultiHeadAttention,
  )


def dummy_datasets(batch_size: int, repeat: int = 1):
  # (num_batch, batch_size, seq_len)
  dummy_input = np.arange(128).reshape((-1, batch_size, 16))
  return [
      distillation_trainer.TrainingInput(
          input_tokens=x, input_mask=jnp.ones(x.shape, dtype=jnp.int32)
      )
      for x in dummy_input
  ] * repeat


class DistillationTrainerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.eval_ds = self.train_ds = dummy_datasets(batch_size=4)

  def test_with_loss_fn_raises_exception(self):
    student_rngs = nnx.Rngs(0)
    teacher_rngs = nnx.Rngs(1)
    student_model = tc.ToyTransformer(rngs=student_rngs, vocab_size=_VOCAB_SIZE)
    teacher_model = tc.ToyTransformer(rngs=teacher_rngs, vocab_size=_VOCAB_SIZE)
    strategy = get_toy_logit_strategy()
    config = distillation_trainer.TrainingConfig(
        eval_every_n_steps=2, max_steps=100
    )

    with self.assertRaises(NotImplementedError):
      distillation_trainer.DistillationTrainer(
          student_model, teacher_model, strategy, optax.sgd(1e-3), config
      ).with_loss_fn(lambda a, b: 0.0)

  def test_basic_training(self):
    student_rngs = nnx.Rngs(0)
    teacher_rngs = nnx.Rngs(1)
    student_model = tc.ToyTransformer(rngs=student_rngs, vocab_size=_VOCAB_SIZE)
    teacher_model = tc.ToyTransformer(rngs=teacher_rngs, vocab_size=_VOCAB_SIZE)
    original_variables = jax.tree.map(
        jnp.copy, nnx.state(student_model, nnx.Param)
    )
    strategy = get_toy_logit_strategy()
    config = distillation_trainer.TrainingConfig(
        eval_every_n_steps=2, max_steps=100
    )
    trainer = distillation_trainer.DistillationTrainer(
        student_model, teacher_model, strategy, optax.sgd(1e-3), config
    ).with_gen_model_input_fn(dummy_gen_model_input_fn)

    trainer.train(self.train_ds, self.eval_ds)
    variables = nnx.state(student_model, nnx.Param)

    jax.tree.map_with_path(tc.assert_not_equal, original_variables, variables)

  def test_complex_strategy_training(self):
    student_rngs = nnx.Rngs(0)
    teacher_rngs = nnx.Rngs(1)
    student_model = tc.ToyTransformer(rngs=student_rngs, vocab_size=_VOCAB_SIZE)
    teacher_model = tc.ToyTransformer(
        rngs=teacher_rngs, vocab_size=_VOCAB_SIZE, num_layers=6
    )
    original_variables = jax.tree.map(
        jnp.copy, nnx.state(student_model, nnx.Param)
    )
    strategy = get_toy_attention_transfer_strategy()
    config = distillation_trainer.TrainingConfig(
        eval_every_n_steps=2, max_steps=100
    )
    trainer = distillation_trainer.DistillationTrainer(
        student_model, teacher_model, strategy, optax.sgd(1e-3), config
    ).with_gen_model_input_fn(dummy_gen_model_input_fn)

    trainer.train(self.train_ds, self.eval_ds)
    variables = nnx.state(student_model, nnx.Param)

    jax.tree.map_with_path(tc.assert_not_equal, original_variables, variables)

  def test_distributed_training(self):
    total_devices = jax.device_count()
    mesh = shd.Mesh(
        devices=np.array(jax.devices()).reshape(2, total_devices // 2),
        axis_names=('fsdp', 'tp'),
    )
    student_rngs = nnx.Rngs(0)
    teacher_rngs = nnx.Rngs(1)
    student_model, _ = create_sharded_model(
        tc.ToyTransformer, student_rngs, mesh
    )
    teacher_model, _ = create_sharded_model(
        tc.ToyTransformer, teacher_rngs, mesh
    )
    original_variables = jax.tree.map(
        jnp.copy, nnx.state(student_model, nnx.Param)
    )
    strategy = get_toy_attention_transfer_strategy()
    config = distillation_trainer.TrainingConfig(
        eval_every_n_steps=2, max_steps=100
    )
    trainer = distillation_trainer.DistillationTrainer(
        student_model, teacher_model, strategy, optax.sgd(1e-3), config
    ).with_gen_model_input_fn(dummy_gen_model_input_fn)

    with mesh:
      trainer.train(self.train_ds, self.eval_ds)
    variables = nnx.state(student_model, nnx.Param)

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
    student_rngs = nnx.Rngs(0)
    teacher_rngs = nnx.Rngs(1)
    unsharded_student_model = tc.ToyTransformer(
        rngs=student_rngs, vocab_size=_VOCAB_SIZE
    )
    unsharded_teacher_model = tc.ToyTransformer(
        rngs=teacher_rngs, vocab_size=_VOCAB_SIZE
    )
    trainer = distillation_trainer.DistillationTrainer(
        unsharded_student_model,
        unsharded_teacher_model,
        strategy,
        optax.sgd(1e-3),
        config,
    ).with_gen_model_input_fn(dummy_gen_model_input_fn)
    trainer.train(self.train_ds, self.eval_ds)
    unsharded_variables = nnx.state(unsharded_student_model, nnx.Param)

    self.assertIsInstance(
        unsharded_variables.layers[0].w1.kernel.value.sharding,
        jax.sharding.SingleDeviceSharding,
    )
    jax.tree.map_with_path(tc.assert_close, variables, unsharded_variables)


if __name__ == '__main__':
  absltest.main()
