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

"""Utilities for mutating feature shapes via pooling for distillation."""

import enum

from flax import nnx
import jax.numpy as jnp


class PaddingMode(enum.Enum):
  """Feature shape padding modes."""

  VALID = "VALID"
  SAME = "SAME"


def avg_pool_array_to_target_shape(
    input_array: jnp.ndarray,
    target_shape: tuple[int, ...],
    padding_mode: PaddingMode = PaddingMode.VALID,
    count_include_pad_for_same_padding: bool = False,
) -> jnp.ndarray:
  """Reduces a JAX numpy array to a target shape using average pooling.

  This function reshapes the input to treat all its dimensions as spatial
  for the pooling operation, then calculates window sizes and strides
  to achieve the specified target shape.

  Args:
      input_array: The input JAX numpy array.
      target_shape: A tuple representing the desired output shape. Each
        dimension must be less than or equal to the corresponding input
        dimension and be at least 1.
      padding_mode: If 'VALID' padding, kernel and stride are calculated to
        precisely achieve the target dimensions. If 'SAME' padding, strides are
        calculated based on ceil(input_dim / target_dim), and kernel size is
        typically set equal to stride.
      count_include_pad_for_same_padding: Boolean, relevant only if using 'SAME'
        padding. Determines if padded elements are included in the average
        calculation's denominator.

  Returns:
      A new JAX numpy array with the target shape.

  Raises:
      ValueError: If input/target shape ranks don't match, if target
                  dimensions are invalid, or if the pooling operation
                  does not result in the exact target shape.
  """
  if input_array.shape == target_shape:
    return input_array

  input_rank = input_array.ndim
  if input_rank != len(target_shape):
    raise ValueError(
        f"Input array rank ({input_rank}) must match target_shape rank"
        f" ({len(target_shape)})."
    )

  for i in range(input_rank):
    if not (1 <= target_shape[i] <= input_array.shape[i]):
      raise ValueError(
          f"Target dimension target_shape[{i}]={target_shape[i]} is invalid. "
          f"Must be 1 <= target_dim <= input_dim ({input_array.shape[i]})."
      )

  # Reshape input to (1, D1, D2, ..., Dn, 1) to treat all original dims as
  # spatial B=1, S=(D1,...,Dn), C=1 for the avg_pool function
  reshaped_input = input_array.reshape(1, *input_array.shape, 1)

  calculated_windows = []
  calculated_strides = []

  for i in range(input_rank):
    input_shape_i = input_array.shape[i]
    output_shape_i = target_shape[i]

    if output_shape_i == input_shape_i:
      window_shape_i = 1
      stride_shape_i = 1
    else:
      if padding_mode == PaddingMode.VALID:
        stride_shape_i = input_shape_i // output_shape_i
        window_shape_i = input_shape_i - (output_shape_i - 1) * stride_shape_i
      else:  # Use 'SAME' padding
        # For 'SAME' padding, output_size O = ceil(D / S).
        # stride_shape_i = ceil(input_shape_i / output_shape_i).
        stride_shape_i = (input_shape_i + output_shape_i - 1) // output_shape_i
        window_shape_i = stride_shape_i

    calculated_windows.append(window_shape_i)
    calculated_strides.append(stride_shape_i)

  try:
    pooled_output = nnx.avg_pool(
        reshaped_input,
        window_shape=tuple(calculated_windows),
        strides=tuple(calculated_strides),
        padding=padding_mode.value,
        count_include_pad=count_include_pad_for_same_padding,
    )
  except Exception as e:
    raise RuntimeError(
        "Underlying avg_pool function failed with parameters:"
        f" window={calculated_windows}, strides={calculated_strides},"
        f" padding='{padding_mode}'."
    ) from e

  final_output = jnp.squeeze(pooled_output, axis=(0, -1))

  if final_output.shape != target_shape:
    raise ValueError(
        f"Squeezed output shape {final_output.shape} does not match"
        f" target_shape {target_shape}. An unexpected error occurred in"
        " reshaping or squeezing."
    )

  return final_output
