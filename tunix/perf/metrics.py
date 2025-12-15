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

"""APIs of performance metrics for RL workflows.

Config API:

    from tunix import PerfMetricsConfig
    from tunix import PerfMetricsExport

    # 1. Create a PerfMetricsConfig object.

    perf_config = PerfMetricsConfig()

    # 2. Create a metrics export function.

    # Let PerfMetricsExport create a metrics export function based on the mesh
    # topology in cluster config. See PerfMetricsExport for more details.
    perf_config.custom_export_fn = (
      PerfMetricsExport.from_cluster_config(cluster_config)
    )

    # Write your own custom export function.
    def my_custom_export_fn(query: PerfMetricsQuery) -> MetricsT:
      ...
    perf_config.custom_export_fn = my_custom_export_fn

    # 3. Pass the PerfMetricsConfig when creating rl cluster.

    rl_cluster.RLCluster(..., perf_config=perf_config)
"""

from __future__ import annotations

import dataclasses
from typing import Any
from typing import Callable, Dict, Tuple

from jax import typing
from tunix.perf import span


ArrayLike = typing.ArrayLike
Timeline = Any  # tunix.perf.span.Timeline
Span = span.Span
SpanGroup = span.SpanGroup

MetricsT = Dict[
    str, Tuple[ArrayLike | str, Callable[[ArrayLike], ArrayLike] | None]
]  # Metrics to be buffered: name -> (values, optional agg_fn)


@dataclasses.dataclass(slots=True)
class MetricsBuffer:
  global_steps: int
  # Metrics to be buffered: name -> (list of (values), optional agg_fn)
  metrics: dict[
      str, tuple[list[ArrayLike | str], Callable[[ArrayLike], ArrayLike] | None]
  ] = dataclasses.field(default_factory=dict)
  mode: str = "train"


class PerfMetricsConfig:
  # (query, epoch) -> metrics
  custom_export_fn: Callable[[PerfSpanQuery], MetricsT] | None = None


class PerfSpanQuery:
  """Query API for PerfMetrics.

  Format:
    query().<timeline_selector>.<group_selector>.get()

  Timeline selector (required):
    .main()
    .timeline(id)

  Group selector (optional):
    .group(name)
    .group(a).group(b).group(c) # nested groups

  Examples:

    # Get the last SpanGroup with name "global_step" in the main thread
    # timeline.
    query.main().group("global_step").get()

    # Get the last SpanGroup with name "global_step", then get the last
    # SpanGroup with name "mini_batch" in the tpu0 timeline.
    query.timeline("tpu0").group("global_step").group("mini_batch").get()
  """

  def __init__(self, timelines: dict[str, Timeline], main_thread_id: str):
    self._timelines: dict[str, Timeline] = timelines
    self._main_thread_id = main_thread_id

    self._select_timeline: str | None = None

    # (name, type, arg)
    # type: 0 - first, 1 - last, 2 - nth, 3 - all
    self._select_groups: list[tuple[str, int, int]] = []

  def __call__(self) -> PerfSpanQuery:
    query = PerfSpanQuery(self._timelines, self._main_thread_id)
    query._select_timeline = self._select_timeline
    query._select_groups = self._select_groups.copy()
    return query

  def timeline(self, id: str) -> PerfSpanQuery:
    self._select_timeline = id
    return self

  def main(self) -> PerfSpanQuery:
    self._select_timeline = self._main_thread_id
    return self

  def first_group(self, name: str) -> PerfSpanQuery:
    self._select_groups.append((name, 0, 0))
    return self

  def last_group(self, name: str) -> PerfSpanQuery:
    self._select_groups.append((name, 1, 0))
    return self

  def nth_group(self, name: str, index: int) -> PerfSpanQuery:
    self._select_groups.append((name, 2, index))
    return self

  def all_groups(self, name: str) -> PerfSpanQuery:
    self._select_groups.append((name, 3, 0))
    return self

  def get(self) -> list[SpanGroup]:
    """Returns SpanGroups gathered by the given selectors."""

    if self._select_timeline not in self._timelines:
      raise ValueError(f"timeline '{self._select_timeline}' not found.")

    curr_groups: list[SpanGroup] = [self._timelines[self._select_timeline].root]
    next_groups: list[SpanGroup] = []
    for name, type, index in self._select_groups:
      assert 0 <= type <= 3, f"invalid query type: {type}"
      match type:
        case 0:  # first
          next_groups = span.span_group_batch_query_first(curr_groups, name)
        case 1:  # last
          next_groups = span.span_group_batch_query_last(curr_groups, name)
        case 2:  # nth
          next_groups = span.span_group_batch_query_nth(
              curr_groups, name, index
          )
        case 3:  # all
          next_groups = span.span_group_batch_query_all(curr_groups, name)
      curr_groups = next_groups
      next_groups = []
    return curr_groups
