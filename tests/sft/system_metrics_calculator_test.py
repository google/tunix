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
from tunix.sft import system_metrics_calculator

SystemMetricsCalculator = system_metrics_calculator.SystemMetricsCalculator

_PARAMS = 1_000_000_000
_BATCH_SIZE = 32
_STEP_TIME = 0.5


class SystemMetricsCalculatorTest(absltest.TestCase):

  @mock.patch('jax.device_count', return_value=1)
  def test_tflops_per_second_single_device(self, mock_device_count):
    """Tests tflops_per_second calculation with a single JAX device."""
    expected_tflops = 6 * _BATCH_SIZE * _PARAMS / _STEP_TIME / 1e12

    result = SystemMetricsCalculator.tflops_per_second(
        total_model_params=_PARAMS,
        per_device_mini_batch_size=_BATCH_SIZE,
        step_time_delta=_STEP_TIME,
    )

    self.assertAlmostEqual(result, expected_tflops, places=6)
    mock_device_count.assert_called_once()

  @mock.patch('jax.device_count', return_value=8)
  def test_tflops_per_second_multi_device(self, mock_device_count):
    """Tests tflops_per_second calculation with multiple JAX devices."""
    expected_tflops = 6 * 8 * _BATCH_SIZE * _PARAMS / _STEP_TIME / 1e12

    result = SystemMetricsCalculator.tflops_per_second(
        total_model_params=_PARAMS,
        per_device_mini_batch_size=_BATCH_SIZE,
        step_time_delta=_STEP_TIME,
    )

    self.assertAlmostEqual(result, expected_tflops, places=6)
    mock_device_count.assert_called_once()

  @mock.patch('jax.device_count', return_value=1)
  def test_tflops_per_second_invalid_step_time_delta(self, mock_device_count):
    """Tests tflops_per_second returns 0.0 when step_time_delta is zero."""
    with self.assertLogs(level=logging.WARNING) as cm:
      result = SystemMetricsCalculator.tflops_per_second(
          total_model_params=_PARAMS,
          per_device_mini_batch_size=_BATCH_SIZE,
          step_time_delta=0.0,
      )
      self.assertLen(cm.output, 1)
      self.assertIn(
          'Step duration is zero or negative (0.0000 s), TFLOPs/sec cannot be'
          ' calculated and will be returned as 0.0.',
          cm.output[0],
      )

    self.assertEqual(result, 0.0)
    mock_device_count.assert_not_called()

  @mock.patch('jax.device_count', return_value=1)
  def test_tflops_per_second_invalud_total_model_params(
      self, mock_device_count
  ):
    """Tests tflops_per_second returns 0.0 when total_model_params is zero."""
    with self.assertLogs(level=logging.WARNING) as cm:
      result = SystemMetricsCalculator.tflops_per_second(
          total_model_params=0,
          per_device_mini_batch_size=_BATCH_SIZE,
          step_time_delta=_STEP_TIME,
      )
      self.assertLen(cm.output, 1)
      self.assertIn(
          'total_model_params is zero or negative (0), TFLOPs/sec cannot be'
          ' calculated and will be returned as 0.0.',
          cm.output[0],
      )

    self.assertEqual(result, 0.0)
    mock_device_count.assert_not_called()


if __name__ == '__main__':
  absltest.main()
