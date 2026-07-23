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

"""Tests for target-aligned diffusion on-policy distillation."""

from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tunix.diffusion import types as diffusion_types
from tunix.distillation import diffusion as diffusion_distillation
from tunix.distillation import diffusion_opd
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
    student_logits,
    teacher_logits,
    target_ids,
    loss_weights,
) -> tuple[_LogitModel, diffusion_distillation.DiffusionDistillationBatch]:
  """Builds a test model and its aligned distillation batch."""

  model = _LogitModel(jnp.asarray(student_logits))
  student_batch = diffusion_types.DiffusionTokenBatch.create(
      model_inputs={
          "example_ids": jnp.arange(len(target_ids), dtype=jnp.int32)
      },
      target_ids=jnp.asarray(target_ids, dtype=jnp.int32),
      loss_weights=jnp.asarray(loss_weights),
  )
  return model, diffusion_distillation.DiffusionDistillationBatch.create(
      student_batch=student_batch,
      teacher_logits=jnp.asarray(teacher_logits),
  )


def _loss_value(
    model: nnx.Module,
    batch: diffusion_distillation.DiffusionDistillationBatch,
    **kwargs,
) -> jax.Array:
  return diffusion_opd.diffusion_opd_loss_fn(
      model, batch, _score_fn, **kwargs
  ).primary_loss.compute()


class DiffusionOpdLossTest(absltest.TestCase):

  def test_rejects_student_teacher_vocabulary_mismatch(self):
    model, batch = _batch(
        student_logits=jnp.ones((1, 2, 3), dtype=jnp.float32),
        teacher_logits=jnp.ones((1, 2, 2), dtype=jnp.float32),
        target_ids=[[0, 1]],
        loss_weights=[[1.0, 1.0]],
    )

    with self.assertRaisesRegex(ValueError, "identical shapes"):
      diffusion_opd.diffusion_opd_loss_fn(model, batch, _score_fn)

  def test_uses_forward_kl_and_target_aligned_hard_loss(self):
    student_logits = jnp.array([[[2.0, -1.0], [-0.5, 0.5]]], dtype=jnp.float32)
    teacher_logits = jnp.array(
        [[[-1.0, 2.0], [0.75, -0.75]]], dtype=jnp.float32
    )
    model, batch = _batch(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        target_ids=[[1, 0]],
        loss_weights=[[1.0, 0.5]],
    )
    temperature = 2.0

    output = diffusion_opd.diffusion_opd_loss_fn(
        model,
        batch,
        _score_fn,
        temperature=temperature,
        soft_loss_weight=0.75,
        hard_loss_weight=0.25,
    )
    teacher_probs = jax.nn.softmax(teacher_logits / temperature, axis=-1)
    expected_soft = optax.kl_divergence(
        jax.nn.log_softmax(student_logits / temperature, axis=-1),
        teacher_probs,
    ) * (temperature**2)
    expected_hard = optax.softmax_cross_entropy_with_integer_labels(
        student_logits, batch.student_batch.target_ids
    )
    weights = batch.student_batch.loss_weights
    soft_sum = jnp.sum(expected_soft * weights)
    hard_sum = jnp.sum(expected_hard * weights)

    self.assertIsInstance(output, sft_utils.LossOutput)
    np.testing.assert_allclose(
        output.primary_loss.unreduced_sum,
        0.75 * soft_sum + 0.25 * hard_sum,
        rtol=1e-6,
    )
    self.assertEqual(float(output.primary_loss.denominator), 1.5)
    np.testing.assert_allclose(
        output.aux_metrics["distill/soft_loss"].unreduced_sum,
        soft_sum,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        output.aux_metrics["distill/hard_loss"].unreduced_sum,
        hard_sum,
        rtol=1e-6,
    )

  def test_teacher_logits_are_stop_gradient(self):
    model, batch = _batch(
        student_logits=jnp.array([[[2.0, -1.0]]], dtype=jnp.float32),
        teacher_logits=jnp.array([[[-1.0, 2.0]]], dtype=jnp.float32),
        target_ids=[[1]],
        loss_weights=[[1.0]],
    )

    def loss_from_teacher(teacher_logits):
      return _loss_value(
          model,
          batch.replace(teacher_logits=teacher_logits),  # pylint: disable=no-member
      )

    teacher_grads = jax.grad(loss_from_teacher)(batch.teacher_logits)
    _, student_grads = nnx.value_and_grad(_loss_value)(model, batch)

    np.testing.assert_array_equal(teacher_grads, jnp.zeros_like(teacher_grads))
    self.assertTrue(
        any(
            bool(jnp.any(jnp.abs(leaf) > 0))
            for leaf in jax.tree.leaves(student_grads)
        )
    )

  def test_fractional_weights_apply_to_loss_and_metrics(self):
    student_logits = jnp.array(
        [[[1.0, 0.0], [0.0, 1.0], [2.0, -2.0]]], dtype=jnp.float32
    )
    teacher_logits = jnp.array(
        [[[0.0, 1.0], [0.0, 1.0], [-2.0, 2.0]]], dtype=jnp.float32
    )
    model, batch = _batch(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        target_ids=[[1, 1, 0]],
        loss_weights=[[2.0, 0.25, 0.0]],
    )

    output = diffusion_opd.diffusion_opd_loss_fn(model, batch, _score_fn)
    per_token = optax.kl_divergence(
        jax.nn.log_softmax(student_logits, axis=-1),
        jax.nn.softmax(teacher_logits, axis=-1),
    )
    expected_sum = jnp.sum(per_token * batch.student_batch.loss_weights)

    np.testing.assert_allclose(
        output.primary_loss.unreduced_sum, expected_sum, rtol=1e-6
    )
    self.assertEqual(float(output.primary_loss.denominator), 2.25)
    self.assertEqual(
        float(output.aux_metrics["distill/soft_loss"].denominator), 2.25
    )

  def test_zero_total_weight_has_zero_loss_and_gradients(self):
    model, batch = _batch(
        student_logits=jnp.array(
            [[[jnp.inf, -jnp.inf], [jnp.nan, 0.0]]], dtype=jnp.float32
        ),
        teacher_logits=jnp.array(
            [[[jnp.nan, 0.0], [jnp.inf, -jnp.inf]]], dtype=jnp.float32
        ),
        target_ids=[[0, 1]],
        loss_weights=[[0.0, 0.0]],
    )

    @nnx.jit
    def loss_and_grad(model, batch):
      return nnx.value_and_grad(_loss_value)(model, batch)

    output = diffusion_opd.diffusion_opd_loss_fn(
        model,
        batch,
        _score_fn,
        soft_loss_weight=0.5,
        hard_loss_weight=0.5,
    )
    loss, grads = loss_and_grad(model, batch)

    self.assertEqual(float(output.primary_loss.unreduced_sum), 0.0)
    self.assertEqual(float(output.primary_loss.denominator), 0.0)
    self.assertEqual(float(loss), 0.0)
    for leaf in jax.tree.leaves(grads):
      self.assertTrue(bool(jnp.all(jnp.isfinite(leaf))))
      np.testing.assert_array_equal(leaf, jnp.zeros_like(leaf))

  def test_inactive_invalid_logits_do_not_affect_active_loss_or_gradients(self):
    model, batch = _batch(
        student_logits=jnp.array(
            [[[2.0, -1.0], [jnp.nan, jnp.inf]]], dtype=jnp.float32
        ),
        teacher_logits=jnp.array(
            [[[-1.0, 2.0], [jnp.inf, jnp.nan]]], dtype=jnp.float32
        ),
        target_ids=[[1, -1]],
        loss_weights=[[1.0, 0.0]],
    )

    loss, grads = nnx.value_and_grad(_loss_value)(model, batch)

    self.assertTrue(bool(jnp.isfinite(loss)))
    for leaf in jax.tree.leaves(grads):
      self.assertTrue(bool(jnp.all(jnp.isfinite(leaf))))

  def test_rejects_invalid_active_teacher_logits(self):
    model, batch = _batch(
        student_logits=jnp.array([[[1.0, -1.0]]], dtype=jnp.float32),
        teacher_logits=jnp.array([[[jnp.nan, jnp.inf]]], dtype=jnp.float32),
        target_ids=[[0]],
        loss_weights=[[1.0]],
    )

    with self.assertRaisesRegex(ValueError, "active teacher_logits"):
      diffusion_opd.diffusion_opd_loss_fn(model, batch, _score_fn)

  def test_rejects_active_hard_targets_outside_vocabulary(self):
    with self.assertRaisesRegex(ValueError, "smaller than vocabulary size"):
      _batch(
          student_logits=jnp.zeros((1, 1, 2), dtype=jnp.float32),
          teacher_logits=jnp.zeros((1, 1, 2), dtype=jnp.float32),
          target_ids=[[2]],
          loss_weights=[[1.0]],
      )

  def test_disabled_soft_loss_ignores_invalid_teacher_values(self):
    model, batch = _batch(
        student_logits=jnp.array([[[1.0, -1.0]]], dtype=jnp.float32),
        teacher_logits=jnp.array([[[jnp.nan, jnp.inf]]], dtype=jnp.float32),
        target_ids=[[0]],
        loss_weights=[[1.0]],
    )

    output = diffusion_opd.diffusion_opd_loss_fn(
        model,
        batch,
        _score_fn,
        soft_loss_weight=0.0,
        hard_loss_weight=1.0,
    )
    expected = optax.softmax_cross_entropy_with_integer_labels(
        model.logits[...], batch.student_batch.target_ids
    )

    self.assertTrue(bool(jnp.isfinite(output.primary_loss.compute())))
    np.testing.assert_allclose(
        output.primary_loss.unreduced_sum, jnp.sum(expected), rtol=1e-6
    )
    self.assertEqual(
        float(output.aux_metrics["distill/soft_loss"].unreduced_sum), 0.0
    )

  def test_rejects_invalid_loss_configuration(self):
    model, batch = _batch(
        student_logits=jnp.zeros((1, 1, 2), dtype=jnp.float32),
        teacher_logits=jnp.zeros((1, 1, 2), dtype=jnp.float32),
        target_ids=[[0]],
        loss_weights=[[1.0]],
    )

    with self.assertRaisesRegex(ValueError, "temperature must be positive"):
      diffusion_opd.diffusion_opd_loss_fn(
          model, batch, _score_fn, temperature=0.0
      )
    with self.assertRaisesRegex(ValueError, "must be nonnegative"):
      diffusion_opd.diffusion_opd_loss_fn(
          model, batch, _score_fn, soft_loss_weight=-1.0
      )
    with self.assertRaisesRegex(ValueError, "at least one"):
      diffusion_opd.diffusion_opd_loss_fn(
          model,
          batch,
          _score_fn,
          soft_loss_weight=0.0,
          hard_loss_weight=0.0,
      )
    for name, overrides in (
        ("temperature", {"temperature": jnp.nan}),
        ("temperature", {"temperature": jnp.inf}),
        ("soft_loss_weight", {"soft_loss_weight": jnp.nan}),
        ("soft_loss_weight", {"soft_loss_weight": jnp.inf}),
        ("hard_loss_weight", {"hard_loss_weight": jnp.nan}),
        ("hard_loss_weight", {"hard_loss_weight": jnp.inf}),
    ):
      with self.subTest(name=name, overrides=overrides):
        with self.assertRaisesRegex(ValueError, "finite real scalar"):
          diffusion_opd.diffusion_opd_loss_fn(
              model,
              batch,
              _score_fn,
              **overrides,
          )

  def test_configure_prepared_diffusion_opd_wires_public_trainer(self):
    model = _LogitModel(
        jnp.array([[[1.0, -1.0], [-0.5, 0.5]]], dtype=jnp.float32)
    )
    trainer = peft_trainer.PeftTrainer(
        model,
        optax.sgd(0.1),
        peft_trainer.TrainingConfig(eval_every_n_steps=100, max_steps=1),
    )
    raw_batch = {
        "target_ids": [[1, 0]],
        "loss_weights": [[1.0, 0.5]],
        "teacher_logits": [[[-1.0, 1.0], [1.0, -1.0]]],
    }
    adapter_inputs = []

    def batch_adapter(value):
      adapter_inputs.append(value)
      student_batch = diffusion_types.DiffusionTokenBatch.create(
          model_inputs={"example_ids": jnp.array([0], dtype=jnp.int32)},
          target_ids=jnp.asarray(value["target_ids"], dtype=jnp.int32),
          loss_weights=jnp.asarray(value["loss_weights"]),
      )
      return diffusion_distillation.DiffusionDistillationBatch.create(
          student_batch=student_batch,
          teacher_logits=jnp.asarray(value["teacher_logits"]),
      )

    configured_trainer = diffusion_opd.configure_prepared_diffusion_opd(
        trainer,
        batch_adapter,
        _score_fn,
        temperature=2.0,
        soft_loss_weight=0.8,
        hard_loss_weight=0.2,
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
    self.assertCountEqual(
        aux.aux_metrics, ["distill/soft_loss", "distill/hard_loss"]
    )
    self.assertGreater(float(grad_norm), 0.0)
    self.assertFalse(bool(jnp.array_equal(model.logits[...], original_logits)))


if __name__ == "__main__":
  absltest.main()
