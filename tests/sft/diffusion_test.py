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

"""Tests for diffusion supervised fine-tuning loss."""

from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tunix.diffusion import types as diffusion_types
from tunix.sft import diffusion
from tunix.sft import peft_trainer
from tunix.sft import utils as sft_utils


class _LogitModel(nnx.Module):

  def __init__(self, logits):
    self.logits = nnx.Param(logits)


def _score_fn(
    model: nnx.Module,
    model_inputs: diffusion_types.ModelInputs,
) -> jax.Array:
  return model.logits[model_inputs["example_ids"]]


def _batch(
    *,
    example_ids,
    target_ids,
    loss_weights,
) -> diffusion_types.DiffusionTokenBatch:
  return diffusion_types.DiffusionTokenBatch.create(
      model_inputs={"example_ids": jnp.asarray(example_ids, dtype=jnp.int32)},
      target_ids=jnp.asarray(target_ids, dtype=jnp.int32),
      loss_weights=jnp.asarray(loss_weights),
  )


def _loss_value(
    model: nnx.Module,
    batch: diffusion_types.DiffusionTokenBatch,
) -> jax.Array:
  return diffusion.diffusion_loss_fn(
      model, batch, _score_fn
  ).primary_loss.compute()


class DiffusionLossTest(absltest.TestCase):

  def test_configure_diffusion_sft_wires_trainer(self):
    model = _LogitModel(
        jnp.array([[[1.0, -1.0], [-0.5, 0.5]]], dtype=jnp.float32)
    )
    trainer = peft_trainer.PeftTrainer(
        model,
        optax.sgd(0.1),
        peft_trainer.TrainingConfig(eval_every_n_steps=100, max_steps=1),
    )
    raw_batch = {
        "example_ids": [0],
        "target_ids": [[1, 0]],
        "loss_weights": [[1.0, 1.0]],
    }
    adapter_inputs = []

    def batch_adapter(value):
      adapter_inputs.append(value)
      return _batch(
          example_ids=value["example_ids"],
          target_ids=value["target_ids"],
          loss_weights=value["loss_weights"],
      )

    configured_trainer = diffusion.configure_diffusion_sft(
        trainer, batch_adapter, _score_fn
    )
    original_logits = jnp.copy(model.logits[...])
    loss, aux, grad_norm = configured_trainer.create_train_step_fn()(
        model,
        configured_trainer.optimizer,
        configured_trainer.grad_accumulator,
        raw_batch,
        jnp.asarray(True),
    )

    self.assertIs(configured_trainer, trainer)
    self.assertLen(adapter_inputs, 1)
    self.assertIs(adapter_inputs[0], raw_batch)
    self.assertTrue(bool(jnp.isfinite(loss)))
    self.assertIsInstance(aux, sft_utils.LossOutput)
    self.assertEmpty(aux.aux_metrics)
    self.assertGreater(float(grad_norm), 0.0)
    self.assertFalse(bool(jnp.array_equal(model.logits[...], original_logits)))

  def test_uses_target_aligned_logits_without_shifting(self):
    logits = jnp.array(
        [[[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]]],
        dtype=jnp.float32,
    )
    batch = _batch(
        example_ids=[0],
        target_ids=[[0, 1, 2]],
        loss_weights=[[1.0, 0.5, 0.0]],
    )

    output = diffusion.diffusion_loss_fn(_LogitModel(logits), batch, _score_fn)
    expected_token_losses = optax.softmax_cross_entropy_with_integer_labels(
        logits, batch.target_ids
    )
    expected_sum = jnp.sum(
        expected_token_losses * batch.loss_weights, dtype=jnp.float32
    )

    self.assertIsInstance(output, sft_utils.LossOutput)
    np.testing.assert_allclose(
        output.primary_loss.unreduced_sum, expected_sum, rtol=1e-6
    )
    self.assertEqual(float(output.primary_loss.denominator), 1.5)
    self.assertEmpty(output.aux_metrics)

  def test_cross_entropy_and_metrics_are_float32(self):
    logits = jnp.array([[[2.0, -1.0], [-2.0, 3.0]]], dtype=jnp.bfloat16)
    batch = _batch(
        example_ids=[0],
        target_ids=[[0, 1]],
        loss_weights=jnp.array([[1, 2]], dtype=jnp.int32),
    )

    metric = diffusion.diffusion_loss_fn(
        _LogitModel(logits), batch, _score_fn
    ).primary_loss

    self.assertEqual(metric.unreduced_sum.dtype, jnp.float32)
    self.assertEqual(metric.denominator.dtype, jnp.float32)
    self.assertEqual(metric.compute().dtype, jnp.float32)

  def test_zero_total_weight_has_zero_loss_and_gradient(self):
    model = _LogitModel(
        jnp.array([[[jnp.inf, -jnp.inf], [jnp.nan, 0.5]]], dtype=jnp.float32)
    )
    batch = _batch(
        example_ids=[0],
        target_ids=[[0, 1]],
        loss_weights=[[0.0, 0.0]],
    )

    @nnx.jit
    def loss_and_grad(model, batch):
      return nnx.value_and_grad(_loss_value)(model, batch)

    metric = diffusion.diffusion_loss_fn(model, batch, _score_fn).primary_loss
    loss, grads = loss_and_grad(model, batch)

    self.assertEqual(float(metric.unreduced_sum), 0.0)
    self.assertEqual(float(metric.denominator), 0.0)
    self.assertEqual(float(loss), 0.0)
    for leaf in jax.tree.leaves(grads):
      self.assertTrue(bool(jnp.all(jnp.isfinite(leaf))))
      np.testing.assert_array_equal(leaf, jnp.zeros_like(leaf))

  def test_jitted_gradients_are_finite_and_nonzero(self):
    model = _LogitModel(
        jnp.array([[[1.0, -1.0], [-0.5, 0.5]]], dtype=jnp.float32)
    )
    batch = _batch(
        example_ids=[0],
        target_ids=[[1, 0]],
        loss_weights=[[1.0, 1.0]],
    )

    @nnx.jit
    def loss_and_grad(model, batch):
      return nnx.value_and_grad(_loss_value)(model, batch)

    loss, grads = loss_and_grad(model, batch)
    gradient_leaves = jax.tree.leaves(grads)

    self.assertTrue(bool(jnp.isfinite(loss)))
    self.assertTrue(
        any(bool(jnp.any(jnp.abs(leaf) > 0)) for leaf in gradient_leaves)
    )
    for leaf in gradient_leaves:
      self.assertTrue(bool(jnp.all(jnp.isfinite(leaf))))

  def test_equal_weight_microbatches_match_full_batch(self):
    logits = jnp.array(
        [
            [[2.0, -1.0], [0.5, -0.5]],
            [[-1.0, 2.0], [-0.5, 0.5]],
            [[0.0, 1.0], [1.5, -0.5]],
            [[1.0, 0.0], [-1.5, 0.5]],
        ],
        dtype=jnp.float32,
    )
    targets = jnp.array([[0, 1], [1, 0], [0, 0], [1, 1]])
    weights = jnp.ones((4, 2), dtype=jnp.float32)
    full_batch = _batch(
        example_ids=jnp.arange(4),
        target_ids=targets,
        loss_weights=weights,
    )
    microbatches = [
        _batch(
            example_ids=jnp.arange(start, start + 2),
            target_ids=targets[start : start + 2],
            loss_weights=weights[start : start + 2],
        )
        for start in (0, 2)
    ]
    model = _LogitModel(logits)

    full_output = diffusion.diffusion_loss_fn(model, full_batch, _score_fn)
    micro_outputs = [
        diffusion.diffusion_loss_fn(model, batch, _score_fn)
        for batch in microbatches
    ]
    _, full_grads = nnx.value_and_grad(_loss_value)(model, full_batch)
    micro_grads = [
        nnx.value_and_grad(_loss_value)(model, batch)[1]
        for batch in microbatches
    ]
    mean_micro_grads = jax.tree.map(
        lambda first, second: (first + second) / 2,
        micro_grads[0],
        micro_grads[1],
    )

    np.testing.assert_allclose(
        full_output.primary_loss.unreduced_sum,
        sum(output.primary_loss.unreduced_sum for output in micro_outputs),
        rtol=1e-6,
    )
    self.assertEqual(
        float(full_output.primary_loss.denominator),
        sum(float(output.primary_loss.denominator) for output in micro_outputs),
    )
    jax.tree.map(
        lambda actual, expected: np.testing.assert_allclose(
            actual, expected, rtol=1e-6, atol=1e-6
        ),
        mean_micro_grads,
        full_grads,
    )

  def test_rejects_misaligned_logits(self):
    batch = _batch(
        example_ids=[0],
        target_ids=[[0, 1]],
        loss_weights=[[1.0, 1.0]],
    )

    def misaligned_score_fn(model, model_inputs):
      del model, model_inputs
      return jnp.ones((1, 1, 2), dtype=jnp.float32)

    with self.assertRaisesRegex(ValueError, "align with target_ids"):
      diffusion.diffusion_loss_fn(
          _LogitModel(jnp.zeros((1, 2, 2))), batch, misaligned_score_fn
      )


if __name__ == "__main__":
  absltest.main()
