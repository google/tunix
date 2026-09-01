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

"""Logging utilities for experimental RL training pipelines."""

from collections.abc import Sequence
from typing import Any


def summarize_list(
    input_list: Sequence[Any],
    max_length: int = 4,
) -> str:
  """Returns a compressed string representation of a sequence of items."""
  valid_items = [str(x) for x in input_list if x is not None and str(x) != ""]
  if not valid_items:
    return "[]"

  if max_length < 2:
    raise ValueError(f"max_length must be at least 2, got {max_length}")

  if len(valid_items) <= max_length:
    return f"[{', '.join(valid_items)}]"

  head = tail = max_length // 2
  if max_length % 2 != 0:
    head += 1
  return (
      f"[{', '.join(valid_items[:head])}, ...,"
      f" {', '.join(valid_items[-tail:])}]"
  )


# Alias for backwards compatibility
summarize_ids = summarize_list
