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

from typing import Any, Callable

from absl import logging
from flax import nnx


def measure_tflops_per_step(
    train_step_fn: Callable[..., Any],
    model: nnx.Module,
    optimizer: nnx.ModelAndOptimizer,
    train_example: Any,
) -> float | None:
  """Performs a one-time static measurement of TFLOPs using JAX's cost analysis."""
  if not hasattr(train_step_fn, "lower"):
    logging.warning(
        "Cannot measure TFLOPs. The provided 'train_step_fn' must be a"
        " JIT-compiled function."
    )
    return None

  try:
    compiled = train_step_fn.lower(model, optimizer, train_example).compile()
    cost = compiled.cost_analysis()
    flops = cost.get("flops")
    if flops is None:
      logging.warning("JAX cost_analysis did not return a 'flops' value.")
      return None
    # Convert FLOPs to TFLOPs
    return float(flops) / 1e12
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.error("Could not measure TFLOPs due to an error: %s", e)
    return None


def approximate_tflops_per_second(
    total_model_params: int,
    global_batch_size: int,
    step_time_delta: float,
) -> float:
  """Approximates the TFLOPS (per second) rate using a 6*params heuristic."""
  if total_model_params <= 0:
    logging.warning(
        "total_model_params is zero or negative (%d), TFLOPS cannot be"
        " approximated and will be returned as 0.0.",
        total_model_params,
    )
    return 0.0
  if step_time_delta <= 0:
    logging.warning(
        "Step duration is zero or negative (%.4f s), TFLOPS cannot be"
        " approximated and will be returned as 0.0.",
        step_time_delta,
    )
    return 0.0

  # Heuristic: 6 * params for forward + backward pass.
  flops_per_step = 6 * global_batch_size * total_model_params
  flops_per_second = flops_per_step / step_time_delta
  # Convert FLOPS to TFLOPS
  return flops_per_second / 1e12
