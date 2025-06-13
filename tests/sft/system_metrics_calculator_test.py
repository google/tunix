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

from absl.testing import absltest
from tunix.sft import system_metrics_calculator

_PARAMS = 1_000_000_000
_GLOBAL_BATCH_SIZE = 32
_STEP_TIME = 0.5


class SystemMetricsCalculatorTest(absltest.TestCase):

  def test_tflops(self):
    """Tests tflops calculation."""
    expected_tflops = 6 * _GLOBAL_BATCH_SIZE * _PARAMS / _STEP_TIME / 1e12

    result = system_metrics_calculator.tflops(
        total_model_params=_PARAMS,
        global_batch_size=_GLOBAL_BATCH_SIZE,
        step_time_delta=_STEP_TIME,
    )

    self.assertAlmostEqual(result, expected_tflops, places=6)

  def test_tflops_invalid_step_time_delta(self):
    """Tests tflops returns 0.0 when step_time_delta is zero."""
    with self.assertLogs(level=logging.WARNING) as cm:
      result = system_metrics_calculator.tflops(
          total_model_params=_PARAMS,
          global_batch_size=_GLOBAL_BATCH_SIZE,
          step_time_delta=0.0,
      )
      self.assertLen(cm.output, 1)
      self.assertIn(
          'Step duration is zero or negative (0.0000 s), TFLOPS cannot be'
          ' calculated and will be returned as 0.0.',
          cm.output[0],
      )

    self.assertEqual(result, 0.0)

  def test_tflops_invalid_total_model_params(self):
    """Tests tflops returns 0.0 when total_model_params is zero."""
    with self.assertLogs(level=logging.WARNING) as cm:
      result = system_metrics_calculator.tflops(
          total_model_params=0,
          global_batch_size=_GLOBAL_BATCH_SIZE,
          step_time_delta=_STEP_TIME,
      )
      self.assertLen(cm.output, 1)
      self.assertIn(
          'total_model_params is zero or negative (0), TFLOPS cannot be'
          ' calculated and will be returned as 0.0.',
          cm.output[0],
      )

    self.assertEqual(result, 0.0)


if __name__ == '__main__':
  absltest.main()
