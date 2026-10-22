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

"""Core data types and context utilities for Distributed RL Lineage Tracking.

Defines:
- LineageEvent: Individual telemetry event for a component/operation.
- LineageContext: Provenance tree node carrying tracking IDs and DAG causal edges.
"""

from collections.abc import Sequence
import dataclasses
import time
from typing import Any


@dataclasses.dataclass(kw_only=True)
class LineageEvent:
  """Individual event recording an operation on a tracked lineage entity.

  Attributes:
    component: System component that generated the event (e.g.,
      "engine.dispatch", "worker.rollout", "orchestrator.assembler",
      "worker.trainer").
    operation: Operation name (e.g., "rollout", "generate", "pack",
      "fwd_bwd", "train_step").
    timestamp_s: Unix timestamp in seconds when the event was recorded.
    attributes: Arbitrary key-value metadata associated with the event (e.g.,
      worker_id, latency_ms, loss, metrics).
  """

  component: str
  operation: str
  timestamp_s: float = dataclasses.field(default_factory=time.time)
  attributes: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(kw_only=True)
class LineageContext:
  """Provenance tracking context for artifacts in the Distributed RL workflow.

  Attributes:
    tracking_id: Unique semantic tracking identifier for this entity (e.g.,
      "traj_prompt_1_0", "batch_0", "policy_v2").
    parent_tracking_ids: List of upstream tracking IDs that contributed to this
      entity (directed causal edges in DAG).
    events: Chronological sequence of lifecycle events for this entity.
  """

  tracking_id: str
  parent_tracking_ids: list[str] = dataclasses.field(default_factory=list)
  events: list[LineageEvent] = dataclasses.field(default_factory=list)

  def add_event(
      self,
      component: str,
      operation: str,
      attributes: dict[str, Any] | None = None,
  ) -> LineageEvent:
    """Appends a new LineageEvent to this context.

    Args:
      component: Component name emitting the event.
      operation: Operation name.
      attributes: Optional key-value metadata.

    Returns:
      The created and appended LineageEvent.
    """
    event = LineageEvent(
        component=component,
        operation=operation,
        attributes=dict(attributes) if attributes else {},
    )
    self.events.append(event)
    return event

  @classmethod
  def merge(
      cls,
      batch_id: str,
      contexts: Sequence["LineageContext | None"],
      component: str,
      operation: str,
      attributes: dict[str, Any] | None = None,
  ) -> "LineageContext":
    """Merges multiple upstream lineage contexts into a single child batch context.

    Used by Batch Assemblers to merge N trajectories into a single parent batch
    lineage.

    Args:
      batch_id: Tracking ID for the merged child entity.
      contexts: List of upstream LineageContext objects. None entries are
        ignored.
      component: Component name performing the merge.
      operation: Operation name.
      attributes: Optional metadata attributes for the merge event.

    Returns:
      A new LineageContext with parent_tracking_ids set to the upstream IDs
      and an initial merge event recorded.
    """
    parent_ids = list(
        dict.fromkeys(
            ctx.tracking_id
            for ctx in contexts
            if ctx is not None and ctx.tracking_id
        )
    )
    new_ctx = cls(tracking_id=batch_id, parent_tracking_ids=parent_ids)
    new_ctx.add_event(component, operation, attributes)
    return new_ctx
