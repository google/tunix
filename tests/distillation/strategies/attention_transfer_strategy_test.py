# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for attention transfer strategy."""

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
import numpy.testing as npt
from tunix.distillation.strategies import attention_transfer_strategy

jax.config.update("jax_threefry_partitionable", False)

AttentionTransferStrategy = (
    attention_transfer_strategy.AttentionTransferStrategy
)


class DummyModel(nnx.Module):

  def __call__(self, x):
    return x


class AttentionTransferStrategyTest(parameterized.TestCase):

  def _get_dummy_logits(self, key, batch_size, num_classes):
    return jax.random.normal(key, (batch_size, num_classes))

  def _get_dummy_labels(self, key, batch_size, num_classes):
    labels_int = jax.random.randint(key, (batch_size,), 0, num_classes)
    return jax.nn.one_hot(labels_int, num_classes)

  def dummy_fn(self):
    pass

  def _get_dummy_attentions(
      self, key, num_layers=3, batch_size=2, num_heads=4, seq_len=8
  ):
    attns = []
    for _ in range(num_layers):
      key, subkey = jax.random.split(key)
      shape = (batch_size, num_heads, seq_len, seq_len)
      attns.append(jax.random.uniform(subkey, shape))
    return jnp.stack(attns, axis=0)

  @parameterized.named_parameters(
      ("alpha_half", 0.5),
      ("alpha_zero", 0.0),
      ("alpha_one", 1.0),
  )
  def test_init_valid(self, alpha):
    strategy = AttentionTransferStrategy(
        student_forward_fn=self.dummy_fn,
        teacher_forward_fn=self.dummy_fn,
        labels_fn=self.dummy_fn,
        alpha=alpha,
    )
    self.assertIsInstance(strategy, AttentionTransferStrategy)
    self.assertEqual(strategy.alpha, alpha)

  @parameterized.named_parameters(
      ("alpha_neg", -0.1),
      ("alpha_over", 1.1),
  )
  def test_init_invalid(self, alpha):
    with self.assertRaises(ValueError):
      AttentionTransferStrategy(
          student_forward_fn=self.dummy_fn,
          teacher_forward_fn=self.dummy_fn,
          labels_fn=self.dummy_fn,
          alpha=alpha,
      )

  @parameterized.named_parameters(
      ("alpha_one", 1.0, 0.162183),
      ("alpha_half", 0.5, 1.07354),
      ("alpha_zero", 0.0, 1.984897),
  )
  def test_compute_loss_default_cosine_distance(self, alpha, expected_loss):
    key = jax.random.key(2)
    output_key, s_attn_key, t_attn_key, label_key = jax.random.split(key, 4)
    num_layers, batch_size, num_heads, seq_len, num_classes = 3, 2, 4, 8, 5

    student_logits = self._get_dummy_logits(output_key, batch_size, num_classes)
    student_attns = self._get_dummy_attentions(
        s_attn_key, num_layers, batch_size, num_heads, seq_len
    )
    teacher_attns = self._get_dummy_attentions(
        t_attn_key, num_layers * 2, batch_size, num_heads * 2, seq_len
    )
    labels = self._get_dummy_labels(label_key, batch_size, num_classes)
    student_outputs = {"logits": student_logits, "attentions": student_attns}
    strategy = AttentionTransferStrategy(
        student_forward_fn=self.dummy_fn,
        teacher_forward_fn=self.dummy_fn,
        labels_fn=self.dummy_fn,
        alpha=alpha,
    )

    computed_loss = strategy.compute_loss(
        student_output=student_outputs,
        teacher_output=teacher_attns,
        labels=labels,
    )
    npt.assert_allclose(computed_loss, expected_loss, rtol=1e-6, atol=1e-6)

  def test_compute_loss_empty_attentions(self):
    key = jax.random.key(3)
    l_key, label_key = jax.random.split(key, 2)
    batch_size, num_classes = 2, 5
    student_logits = self._get_dummy_logits(l_key, batch_size, num_classes)
    labels = self._get_dummy_labels(label_key, batch_size, num_classes)
    student_outputs = {"logits": student_logits, "attentions": jnp.array([])}
    alpha = 0.6
    strategy = AttentionTransferStrategy(
        student_forward_fn=self.dummy_fn,
        teacher_forward_fn=self.dummy_fn,
        labels_fn=self.dummy_fn,
        alpha=alpha,
    )
    expected_combined_loss = jnp.array(1.444023)

    computed_loss = strategy.compute_loss(
        student_output=student_outputs,
        teacher_output=jnp.array([]),
        labels=labels,
    )
    npt.assert_allclose(computed_loss, expected_combined_loss, rtol=1e-6)

  @parameterized.named_parameters(
      ("alpha_one", 1.0),
      ("alpha_half", 0.5),
      ("alpha_zero", 0.0),
  )
  def test_compute_eval_loss(self, alpha):
    s_key, l_key = jax.random.split(jax.random.key(0), 2)
    batch_size, num_classes = 4, 10
    student_logits = self._get_dummy_logits(s_key, batch_size, num_classes)
    student_output = {"logits": student_logits, "attentions": jnp.array([])}
    labels = self._get_dummy_labels(l_key, batch_size, num_classes)
    strategy = AttentionTransferStrategy(
        student_forward_fn=self.dummy_fn,
        teacher_forward_fn=self.dummy_fn,
        labels_fn=self.dummy_fn,
        alpha=alpha,
    )
    expected_loss = 2.49732

    computed_loss = strategy.compute_eval_loss(
        student_output=student_output,
        labels=labels,
    )

    npt.assert_allclose(computed_loss, expected_loss, rtol=1e-6)

  def test_get_train_loss(self):
    strategy = AttentionTransferStrategy(
        student_forward_fn=lambda _, student_output, **kwargs: student_output,
        teacher_forward_fn=lambda _, teacher_output, **kwargs: teacher_output,
        labels_fn=lambda labels, **kwargs: labels,
    )
    inputs = {
        "student_output": {
            "logits": jnp.array([1.0, 2.0, 3.0]),
            "attentions": jnp.array([4.0, 5.0, 6.0]),
        },
        "teacher_output": jnp.array([7.0, 8.0, 9.0]),
        "labels": jnp.array([10.0, 11.0, 12.0]),
    }
    expected_loss = strategy.compute_loss(
        student_output=inputs["student_output"],
        teacher_output=inputs["teacher_output"],
        labels=inputs["labels"],
    )

    teacher_output = strategy.get_teacher_outputs(
        teacher_model=DummyModel(), inputs=inputs
    )
    computed_loss = strategy.get_train_loss(
        student_model=DummyModel(), teacher_output=teacher_output, inputs=inputs
    )

    npt.assert_allclose(teacher_output, inputs["teacher_output"], rtol=1e-6)
    npt.assert_allclose(computed_loss, expected_loss, rtol=1e-6)

  def test_get_eval_loss(self):
    strategy = AttentionTransferStrategy(
        student_forward_fn=lambda _, student_output, **kwargs: student_output,
        teacher_forward_fn=lambda _, teacher_output, **kwargs: teacher_output,
        labels_fn=lambda labels, **kwargs: labels,
    )
    inputs = {
        "student_output": {
            "logits": jnp.array([1.0, 2.0, 3.0]),
            "attentions": jnp.array([4.0, 5.0, 6.0]),
        },
        "teacher_output": jnp.array([]),
        "labels": jnp.array([10.0, 11.0, 12.0]),
    }
    expected_loss = strategy.compute_eval_loss(
        student_output=inputs["student_output"],
        labels=inputs["labels"],
    )

    computed_loss = strategy.get_eval_loss(
        student_model=DummyModel(), inputs=inputs
    )

    npt.assert_allclose(computed_loss, expected_loss, rtol=1e-6)


if __name__ == "__main__":
  absltest.main()
