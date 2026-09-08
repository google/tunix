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

"""Lazy runtime interpretation of the core canonical training selectors.

Read only the requested option, on every call. This module does not select a
workload, admit a topology, cache an environment, or change legacy defaults.
Callers supply their existing ValueError subclass without a dependency on the
adapter or JAX here. Workload admission and option combinations remain at the
existing execution boundaries.
"""

import os


def keep_tape_mode(*, error_type: type[ValueError] = ValueError) -> str:
  """Returns '', 'batch', or 'stream'; empty and explicit 0 remain off."""
  value = os.environ.get("CANON_P32_KEEP_TAPE", "")
  if value in ("", "0"):
    return ""
  if value == "1":
    return "batch"
  if value == "stream":
    return "stream"
  raise error_type(
      f"CANON_P32_KEEP_TAPE must be unset, 0, 1, or stream, got {value!r}"
  )


def reduce_once_enabled(*, error_type: type[ValueError] = ValueError) -> bool:
  """Returns whether reduction follows the staged cross-group accumulation."""
  value = os.environ.get("CANON_DP_REDUCE_ONCE", "")
  if value in ("", "0"):
    return False
  if value == "1":
    return True
  raise error_type(
      f"CANON_DP_REDUCE_ONCE must be unset, 0, or 1, got {value!r}"
  )


def rank_parallel_backward(*, error_type: type[ValueError] = ValueError) -> bool:
  """Returns the exact default-off rank-parallel backward selection."""
  value = os.environ.get("CANON_P59_RANK_PARALLEL_BACKWARD", "")
  if value not in ("", "0", "1"):
    raise error_type(
        "CANON_P59_RANK_PARALLEL_BACKWARD must be unset/0/1, "
        f"got {value!r}"
    )
  return value == "1"
