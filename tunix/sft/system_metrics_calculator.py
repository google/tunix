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


def tflops(
    total_model_params: int,
    global_batch_size: int,
    step_time_delta: float,
) -> float:
  """Calculates Model TFLOPS throughput for a single mini-batch step.

  TFLOPS, or TeraFLOPS, stands for Tera Floating Point Operations Per Second.

  This estimation uses the heuristic of 6 FLOPs per parameter for the combined
  forward and backward pass of a dense model. It assumes the `global_batch_size`
  represents the total number of examples processed across all devices in a
  single step.

  Args:
    total_model_params: The total number of trainable parameters in the model.
    global_batch_size: The total number of examples processed in one mini-batch
      step across all participating devices (e.g., sum of per-device batch sizes
      in data parallelism).
    step_time_delta: The time taken for one mini-batch training step (forward +
      backward + partial optimizer update).

  Returns:
    The estimated TFLOPS throughput achieved during processing.
  """
  if total_model_params <= 0:
    logging.warning(
        "total_model_params is zero or negative (%d), TFLOPS cannot be"
        " calculated and will be returned as 0.0.",
        total_model_params,
    )
    return 0.0
  if step_time_delta <= 0:
    logging.warning(
        "Step duration is zero or negative (%.4f s), TFLOPS cannot be"
        " calculated and will be returned as 0.0.",
        step_time_delta,
    )
    return 0.0

  # Estimated FLOPs for the work done in one mini-batch (forward + backward).
  # Heuristic: 6 * params for forward + backward pass.
  flops_per_mini_batch_step = 6 * global_batch_size * total_model_params

  flops_per_second = flops_per_mini_batch_step / step_time_delta
  calculated_tflops = flops_per_second / 1e12

  return calculated_tflops
