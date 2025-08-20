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

import logging
from unittest import mock

from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
import optax
from tunix.sft import system_metrics_calculator

_PARAMS = 1_000_000_000
_GLOBAL_BATCH_SIZE = 32
_STEP_TIME = 0.5


def _fake_train_step(model, optimizer, inputs):
  """A representative train step function for testing."""
  grad_fn = nnx.value_and_grad(lambda m, x: jnp.sum(m(x)))
  loss, grads = grad_fn(model, **inputs)
  optimizer.update(grads)
  return loss, {}


class SystemMetricsCalculatorTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.model = nnx.Linear(10, 2, rngs=nnx.Rngs(0))
    self.optimizer = nnx.ModelAndOptimizer(self.model, optax.sgd(0.1))
    self.train_example = {'x': jnp.ones((1, 10))}
    self.jitted_train_step = nnx.jit(_fake_train_step)

  def test_measure_tflops_per_step_success(self):
    """Tests that measure_tflops_per_step returns a valid float."""
    tflops_per_step = system_metrics_calculator.measure_tflops_per_step(
        train_step_fn=self.jitted_train_step,
        model=self.model,
        optimizer=self.optimizer,
        train_example=self.train_example,
    )
    self.assertIsInstance(tflops_per_step, float)
    # The exact value depends on the JAX implementation and the model.
    self.assertBetween(tflops_per_step, 0.0, 0.0000001)

  def test_measure_tflops_per_step_not_jitted(self):
    """Tests that measure_tflops_per_step returns None for a non-jitted function."""
    with self.assertLogs(level=logging.WARNING) as cm:
      tflops_per_step = system_metrics_calculator.measure_tflops_per_step(
          train_step_fn=_fake_train_step,  # Passing the non-jitted version
          model=self.model,
          optimizer=self.optimizer,
          train_example=self.train_example,
      )
      self.assertIsNone(tflops_per_step)
      self.assertIn('must be a JIT-compiled function', cm.output[0])

  @mock.patch.object(
      jax.stages.Lowered, 'compile', side_effect=ValueError('Test Error')
  )
  def test_measure_tflops_per_step_jax_error(self, _):
    """Tests that measure_tflops_per_step handles internal JAX errors."""
    with self.assertLogs(level=logging.WARNING) as cm:
      tflops_per_step = system_metrics_calculator.measure_tflops_per_step(
          train_step_fn=self.jitted_train_step,
          model=self.model,
          optimizer=self.optimizer,
          train_example=self.train_example,
      )
      self.assertIsNone(tflops_per_step)
      self.assertIn('Could not measure TFLOPs', cm.output[0])
      self.assertIn('Test Error', cm.output[0])

  def test_approximate_tflops_per_second(self):
    """Tests TFLOPS approximation calculation."""
    expected_tflops = 6 * _GLOBAL_BATCH_SIZE * _PARAMS / _STEP_TIME / 1e12

    result = system_metrics_calculator.approximate_tflops_per_second(
        total_model_params=_PARAMS,
        global_batch_size=_GLOBAL_BATCH_SIZE,
        step_time_delta=_STEP_TIME,
    )

    self.assertAlmostEqual(result, expected_tflops, places=6)

  def test_approximate_tflops_per_second_invalid_step_time_delta(self):
    """Tests TFLOPS approximation returns 0.0 when step_time_delta is zero."""
    with self.assertLogs(level=logging.WARNING) as cm:
      result = system_metrics_calculator.approximate_tflops_per_second(
          total_model_params=_PARAMS,
          global_batch_size=_GLOBAL_BATCH_SIZE,
          step_time_delta=0.0,
      )
      self.assertLen(cm.output, 1)
      self.assertIn(
          'TFLOPS cannot be approximated',
          cm.output[0],
      )

    self.assertEqual(result, 0.0)

  def test_approximate_tflops_per_second_invalid_total_model_params(self):
    """Tests TFLOPS approximation returns 0.0 when total_model_params is zero."""
    with self.assertLogs(level=logging.WARNING) as cm:
      result = system_metrics_calculator.approximate_tflops_per_second(
          total_model_params=0,
          global_batch_size=_GLOBAL_BATCH_SIZE,
          step_time_delta=_STEP_TIME,
      )
      self.assertLen(cm.output, 1)
      self.assertIn(
          'TFLOPS cannot be approximated',
          cm.output[0],
      )

    self.assertEqual(result, 0.0)


if __name__ == '__main__':
  absltest.main()
