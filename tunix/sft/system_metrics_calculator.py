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

"""System metrics calculator for Tunix."""

from absl import logging
import jax


class SystemMetricsCalculator:
  """Utility class for calculating system-related performance metrics."""

  @staticmethod
  def tflops_per_second(
      total_model_params: int,
      per_device_mini_batch_size: int,
      step_time_delta: float,
  ) -> float:
    """Calculates Model TeraFLOPs/second throughput for a single mini-batch step.

    This estimation uses the heuristic of 6 FLOPs per parameter for
    the combined forward and backward pass.

    Args:
      total_model_params: The total number of trainable parameters in the model.
      per_device_mini_batch_size: The batch size processed by each device for a
        single mini-batch.
      step_time_delta: The time taken for one mini-batch training step (forward
        + backward + partial optimizer update).

    Returns:
      The estimated TFLOPs/second throughput achieved during processing.
    """
    if total_model_params <= 0:
      logging.warning(
          "total_model_params is zero or negative (%d), TFLOPs/sec cannot be"
          " calculated and will be returned as 0.0.",
          total_model_params,
      )
      return 0.0
    if step_time_delta <= 0:
      logging.warning(
          "Step duration is zero or negative (%.4f s), TFLOPs/sec cannot be"
          " calculated and will be returned as 0.0.",
          step_time_delta,
      )
      return 0.0

    # Total batch size in a single mini-batch step across all devices.
    current_mini_batch_global_size = (
        per_device_mini_batch_size * jax.device_count()
    )

    # Estimated FLOPs for the work done in one mini-batch (forward + backward).
    # Heuristic: 6 * params for forward + backward pass.
    flops_per_mini_batch_step = (
        6 * current_mini_batch_global_size * total_model_params
    )

    flops_per_second = flops_per_mini_batch_step / step_time_delta
    tflops_per_second = flops_per_second / 1e12

    return tflops_per_second
