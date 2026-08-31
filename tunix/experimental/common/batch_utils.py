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

"""Utilities for processing arrays."""

from typing import Any, Callable

import numpy as np


def apply_chunked(
    fn: Callable[..., Any],
    chunk_size: int | None,
    *input_arrays: np.ndarray,
) -> Any:
  """Splits batched inputs into chunks and applies `fn` to each chunk.

  Splitting occurs along the batch dimension (axis 0). Each chunk is processed
  independently, and the results are concatenated to match the output of a
  single pass.

  Args:
    fn: Function to apply to each chunk. Takes the sliced arrays as arguments.
    chunk_size: Optional maximum batch size for processing to reduce peak
      memory.
    *input_arrays: The arrays to chunk. Must have matching sizes in dimension 0.

  Returns:
    The concatenated result of applying `fn` over all chunks.
  """
  if not input_arrays:
    raise ValueError("At least one array must be provided.")

  for a in input_arrays:
    if getattr(a, "ndim", 0) < 1:
      raise ValueError("Arrays must have at least 1 dimension.")

  batch_size = input_arrays[0].shape[0]
  if any(a.shape[0] != batch_size for a in input_arrays):
    raise ValueError("Mismatched batch sizes.")

  if chunk_size is not None and chunk_size <= 0:
    raise ValueError("chunk_size must be positive.")

  if not chunk_size or chunk_size >= batch_size:
    return np.asarray(fn(*input_arrays))

  chunks = []
  for start in range(0, batch_size, chunk_size):
    end = start + chunk_size
    chunk_arrays = [a[start:end] for a in input_arrays]
    chunks.append(np.asarray(fn(*chunk_arrays)))
  return np.concatenate(chunks, axis=0)
