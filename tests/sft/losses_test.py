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

"""Tests for SFT loss functions."""

import os

from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tunix.sft import losses
from tunix.sft import peft_trainer
from tunix.tests import test_common as tc

os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'


def _dummy_gen_model_input_fn(x: peft_trainer.TrainingInput):
  return {
      'input_tokens': x.input_tokens,
      'input_mask': x.input_mask,
      'positions': jnp.arange(x.input_tokens.shape[1]),
      'attention_mask': jnp.ones_like(x.input_tokens),
  }


def _dummy_datasets(batch_size: int, repeat: int = 1):
  dummy_input = np.arange(128).reshape((-1, batch_size, 16))
  return [
      peft_trainer.TrainingInput(
          input_tokens=x, input_mask=jnp.ones(x.shape, dtype=jnp.int32)
      )
      for x in dummy_input
  ] * repeat


def _make_model_and_inputs(seed=0):
  model = tc.ToyTransformer(config=tc.ModelConfig(), rngs=nnx.Rngs(seed))
  rng = jax.random.PRNGKey(42)
  input_tokens = jax.random.randint(rng, (2, 16), 0, 256)
  input_mask = jnp.ones((2, 16), dtype=jnp.int32)
  positions = jnp.arange(16)
  attention_mask = jnp.ones((2, 16), dtype=jnp.int32)
  return model, input_tokens, input_mask, positions, attention_mask


def _train_with_loss_fn(loss_fn):
  config = peft_trainer.TrainingConfig(eval_every_n_steps=2, max_steps=100)
  model = tc.ToyTransformer(config=tc.ModelConfig(), rngs=nnx.Rngs(0))
  trainer = peft_trainer.PeftTrainer(model, optax.sgd(1e-3), config)
  trainer = trainer.with_gen_model_input_fn(
      _dummy_gen_model_input_fn
  ).with_loss_fn(loss_fn)
  trainer.train(_dummy_datasets(batch_size=4))
  return model, trainer


class CrossEntropyLossFnTest(absltest.TestCase):
  """Tests for cross_entropy_loss_fn (moved from peft_trainer)."""

  def test_returns_scalar(self):
    model, *args = _make_model_and_inputs()
    loss = losses.cross_entropy_loss_fn(model, *args)
    self.assertEqual(loss.shape, ())
    self.assertGreater(float(loss), 0.0)

  def test_training_updates_params(self):
    model, trainer = _train_with_loss_fn(losses.cross_entropy_loss_fn)
    original = jax.tree.map(
        jnp.copy,
        nnx.state(
            tc.ToyTransformer(config=tc.ModelConfig(), rngs=nnx.Rngs(0)),
            nnx.Param,
        ),
    )
    updated = nnx.state(model, nnx.Param)
    jax.tree.map_with_path(tc.assert_not_equal, original, updated)


class DFTRescaleTest(absltest.TestCase):
  """Tests for the dft_rescale utility function."""

  def test_non_negative(self):
    xent = jnp.array([0.1, 0.5, 1.0, 2.0, 5.0])
    rescaled = losses.dft_rescale(xent)
    np.testing.assert_array_less(-1e-7, np.asarray(rescaled))

  def test_less_than_or_equal_input(self):
    """p(y_t) <= 1, so dft_rescale(xent) <= xent."""
    xent = jnp.array([0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
    rescaled = losses.dft_rescale(xent)
    np.testing.assert_array_less(
        np.asarray(rescaled), np.asarray(xent) + 1e-6
    )

  def test_zero_input(self):
    xent = jnp.array([0.0, 0.0])
    rescaled = losses.dft_rescale(xent)
    np.testing.assert_allclose(np.asarray(rescaled), 0.0, atol=1e-7)

  def test_uniform_distribution_ratio(self):
    """For uniform predictions p=1/V, ratio should be exactly 1/V."""
    vocab_size = 8
    uniform_xent = jnp.array([jnp.log(vocab_size)])
    rescaled = losses.dft_rescale(uniform_xent)
    expected_ratio = 1.0 / vocab_size
    actual_ratio = float(rescaled[0]) / float(uniform_xent[0])
    self.assertAlmostEqual(actual_ratio, expected_ratio, places=5)

  def test_stop_gradient_on_probability(self):
    """Gradients must differ with and without stop_gradient on p(y_t)."""

    def with_sg(xent):
      return jnp.sum(losses.dft_rescale(xent))

    def without_sg(xent):
      return jnp.sum(jnp.exp(-xent) * xent)

    xent = jnp.array([0.5, 1.0, 2.0])
    grad_with = jax.grad(with_sg)(xent)
    grad_without = jax.grad(without_sg)(xent)

    self.assertFalse(jnp.allclose(grad_with, grad_without, atol=1e-6))

  def test_gradient_differs_from_identity(self):
    """DFT gradients should differ from unrescaled gradients."""

    def standard(xent):
      return jnp.sum(xent)

    def dft(xent):
      return jnp.sum(losses.dft_rescale(xent))

    xent = jnp.array([0.5, 1.0, 2.0])
    self.assertFalse(
        jnp.allclose(jax.grad(standard)(xent), jax.grad(dft)(xent), atol=1e-6)
    )

  def test_batched_input(self):
    xent = jax.random.uniform(jax.random.PRNGKey(0), (2, 4)) + 0.1
    rescaled = losses.dft_rescale(xent)
    self.assertEqual(rescaled.shape, xent.shape)
    np.testing.assert_array_less(np.asarray(rescaled), np.asarray(xent) + 1e-6)


class DFTLossFnTest(absltest.TestCase):
  """Tests for dft_loss_fn integrated with PeftTrainer."""

  def test_returns_scalar(self):
    model, *args = _make_model_and_inputs()
    loss = losses.dft_loss_fn(model, *args)
    self.assertEqual(loss.shape, ())
    self.assertGreater(float(loss), 0.0)

  def test_less_than_or_equal_cross_entropy(self):
    model, *args = _make_model_and_inputs()
    ce = losses.cross_entropy_loss_fn(model, *args)
    dft = losses.dft_loss_fn(model, *args)
    self.assertLessEqual(float(dft), float(ce) + 1e-6)

  def test_differs_from_cross_entropy(self):
    model, *args = _make_model_and_inputs()
    ce = losses.cross_entropy_loss_fn(model, *args)
    dft = losses.dft_loss_fn(model, *args)
    self.assertNotAlmostEqual(float(dft), float(ce), places=3)

  def test_training_updates_params(self):
    model, trainer = _train_with_loss_fn(losses.dft_loss_fn)
    original = jax.tree.map(
        jnp.copy,
        nnx.state(
            tc.ToyTransformer(config=tc.ModelConfig(), rngs=nnx.Rngs(0)),
            nnx.Param,
        ),
    )
    updated = nnx.state(model, nnx.Param)
    jax.tree.map_with_path(tc.assert_not_equal, original, updated)
    self.assertGreater(trainer._train_steps, 0)

  def test_training_differs_from_cross_entropy(self):
    """Training with DFT should produce different params than CE."""
    _, ce_trainer = _train_with_loss_fn(losses.cross_entropy_loss_fn)
    _, dft_trainer = _train_with_loss_fn(losses.dft_loss_fn)

    ce_loss = ce_trainer.metrics_logger.get_metric('', 'loss', 'train')
    dft_loss = dft_trainer.metrics_logger.get_metric('', 'loss', 'train')
    self.assertNotAlmostEqual(float(ce_loss), float(dft_loss), places=3)


if __name__ == '__main__':
  absltest.main()
