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

"""Pure fail-closed validators for the Phase4 first optimizer admission."""

from __future__ import annotations

import math
from typing import Any, Mapping


STABLE_NORM_MAX = 1.0e6


def validate_precommit(
    record: Mapping[str, Any],
    *,
    workload: str,
    dp: int,
    tp: int,
    microsteps: int,
) -> tuple[str, ...]:
  """Returns contract violations for one complete pre-AdamW accumulator."""
  expected = {
      "schema": "canon-v1-first-update-precommit-v1",
      "update": 0,
      "workload": workload,
      "dp": dp,
      "tp": tp,
      "microsteps": microsteps,
      "accumulator_denominator": float(microsteps),
      "stable_norm_max": STABLE_NORM_MAX,
      "all_finite": True,
      "any_nonzero": True,
  }
  reasons = [
      f"{name}={record.get(name)!r} expected={value!r}"
      for name, value in expected.items()
      if record.get(name) != value
  ]
  stable_norm = record.get("stable_norm")
  if not (
      isinstance(stable_norm, (int, float))
      and not isinstance(stable_norm, bool)
      and math.isfinite(float(stable_norm))
      and 0.0 < float(stable_norm) <= STABLE_NORM_MAX
  ):
    reasons.append(
        f"stable_norm={stable_norm!r} expected=(0,{STABLE_NORM_MAX}]"
    )
  return tuple(reasons)

def validate_commit(
    record: Mapping[str, Any],
    *,
    workload: str,
    dp: int,
    tp: int,
) -> tuple[str, ...]:
  """Returns contract violations before outer weight sync/checkpoint."""
  expected = {
      "schema": "canon-v1-first-update-commit-v1",
      "update": 0,
      "workload": workload,
      "dp": dp,
      "tp": tp,
      "train_steps_before": 0,
      "train_steps_after": 1,
      "optimizer_transaction_valid": True,
      "gradient_finite": True,
      "parameter_delta_finite": True,
      "outer_weight_sync_pending": True,
  }
  reasons = [
      f"{name}={record.get(name)!r} expected={value!r}"
      for name, value in expected.items()
      if record.get(name) != value
  ]
  changed = record.get("parameter_changed_elements")
  learning_rate = record.get("effective_learning_rate")
  changed_valid = (
      isinstance(changed, int)
      and not isinstance(changed, bool)
      and changed >= 0
  )
  learning_rate_valid = (
      isinstance(learning_rate, (int, float))
      and not isinstance(learning_rate, bool)
      and math.isfinite(float(learning_rate))
      and float(learning_rate) >= 0.0
  )
  if not changed_valid:
    reasons.append(f"parameter_changed_elements={changed!r}")
  if not learning_rate_valid:
    reasons.append(f"effective_learning_rate={learning_rate!r}")
  elif float(learning_rate) > 0.0 and (not changed_valid or changed == 0):
    reasons.append("positive_learning_rate_without_parameter_change")
  return tuple(reasons)
