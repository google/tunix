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

"""Framework-neutral test helpers for Tunix."""

from typing import Any

import numpy as np


class CustomTensor:
  """Minimal non-NumPy tensor exposing only the `TensorLike` contract."""

  def __init__(self, value: np.ndarray):
    self._value = value

  @property
  def shape(self) -> tuple[int, ...]:
    return self._value.shape

  @property
  def dtype(self) -> Any:
    return self._value.dtype

  def __array__(self) -> np.ndarray[Any, np.dtype[Any]]:
    return self._value
