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

"""Small, default-off helpers for bounded semantic Perfetto captures."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import types
from typing import Any


def single_step_export_fn(
    export_fn: Callable[[Mapping[str, Any]], dict[str, Any]],
    *,
    target_step: int,
) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
  """Returns an exporter that writes only one committed training step.

  The agentic learner exports after zero-origin training steps 0, 1, ... .
  Tunix timelines retain all committed history, so merely skipping exporter
  calls would still write every earlier step at the target call.  This wrapper
  passes immutable, duck-typed timeline snapshots containing only the newest
  committed step.  Collection remains enabled for semantic attribution, while
  serialization and disk I/O occur once.
  """
  if target_step < 0:
    raise ValueError(f"target_step must be non-negative, got {target_step}")
  export_calls = 0

  def export_one_step(timelines: Mapping[str, Any]) -> dict[str, Any]:
    nonlocal export_calls
    current_step = export_calls
    export_calls += 1
    if current_step != target_step:
      return {}

    snapshots = {}
    for timeline_id, timeline in timelines.items():
      committed = tuple(timeline.committed_steps)
      if not committed:
        continue
      snapshots[timeline_id] = types.SimpleNamespace(
          id=timeline.id,
          born=timeline.born,
          committed_steps=[committed[-1]],
      )
    if not snapshots:
      raise RuntimeError(
          "single-step Perfetto target has no committed timeline spans"
      )
    result = export_fn(snapshots)
    print(
        f"[V1.PERFETTO] captured training_step={target_step} "
        f"timelines={len(snapshots)}",
        flush=True,
    )
    return result

  return export_one_step
