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
      PerfMetricsExport.create_metrics_export_fn(cluster_config)
    )

    # Write your own custom export function.
    def my_custom_export_fn(
        query: PerfMetricsQuery, context: PerfMetricsContext
    ) -> MetricsT:
      ...
    perf_config.custom_export_fn = my_custom_export_fn

    # 3. Pass the PerfMetricsConfig when creating rl cluster.

    rl_cluster.RLCluster(..., perf_config=perf_config)
"""

from __future__ import annotations

import dataclasses
import itertools
from typing import Any
from typing import Callable, Dict, Tuple

import jax
from jax import typing
import numpy as np
from graphviz import Digraph
from typing import Any


ArrayLike = typing.ArrayLike
BaseTimeline = Any  # tunix.perf.trace.BaseTimeline

MetricsT = Dict[
    str, Tuple[ArrayLike | str, Callable[[jax.Array], jax.Array] | None]
]  # Metrics to be buffered: name -> (values, optional agg_fn)


@dataclasses.dataclass(slots=True)
class MetricsBuffer:
  global_steps: int
  # Metrics to be buffered: name -> (list of (values), optional agg_fn)
  metrics: dict[
      str, tuple[list[ArrayLike | str], Callable[[ArrayLike], ArrayLike] | None]
  ] = dataclasses.field(default_factory=dict)
  mode: str = "train"


@dataclasses.dataclass
class PerfMetricsContext:
  epoch: int | None = None
  global_steps: int | None = None

class PerfMetricsConfig:
  # (query, epoch) -> metrics
  custom_export_fn: (
      Callable[[PerfMetricsQuery, PerfMetricsContext], MetricsT] | None
  ) = None


class PerfMetricsApi:

  def __init__(self, timelines: dict[str, BaseTimeline], main_thread_id: str):
    self._timelines: dict[str, BaseTimeline] = timelines
    self._main_thread_id = main_thread_id

    def dump(self) -> None:
        for timeline in self._timelines.values():
            print(timeline)

  def query(self) -> PerfMetricsQuery:
    return PerfMetricsQuery(self._timelines, self._main_thread_id)


#
# Query API
#
#   query.timeline("tpu0").epoch(*).idle().sum()
#   query.timeline("tpu0").epoch(*).idle().mean()
#   query.timeline("tpu0").epoch(*).idle().tolist()
#   query.timeline("tpu0").epoch(*).idle().epoch_sum()
#   query.timeline("tpu0").epoch(*).idle().epoch_mean()
#   query.timeline("tpu0").epoch(*).idle().epoch_tolist()
#
#   query.timeline("tpu2").epoch(*).busy(*).sum()
#   query.timeline("tpu2").epoch(*).busy(*).mean()
#   query.timeline("tpu2").epoch(*).busy(*).tolist()
#   query.timeline("tpu2").epoch(*).busy(*).epoch_sum()
#   query.timeline("tpu2").epoch(*).busy(*).epoch_mean()
#   query.timeline("tpu2").epoch(*).busy(*).epoch_tolist()
#
class PerfMetricsQuery:

  Format:
    query.<timeline_selector>.<epoch_selector>.<busy_or_idle>.<aggregation>()

  Timeline selector (required):
    .main()
    .timeline(id)

  Epoch selector (optional):
    .epoch(epoch_ids)

  Busy or idle (required):
    .busy(labels)  # labels: list of interval labels to select
    .idle()

  Aggregation (required):
    .sum()  # sum over all selected intervals
    .mean()  # mean over all selected intervals
    .tolist()  # list of all selected intervals
    .epoch_sum()  # sum over all selected intervals, grouped by epochs
    .epoch_mean()  # mean over all selected intervals, grouped by epochs
    .epoch_tolist()  # list of all selected intervals, grouped by epochs

  Examples:

    # Sum of idle time for main thread over all epochs.
    query.main().idle().sum()

    # Sum of busy time for main thread over epoch 1.
    query.main().epoch([1]).busy().sum()

    # Sum of idle time for tpu0 timeline over all epochs.
    query.timeline("tpu0").idle().tolist()

    # Mean of task1 time for tpu2 timeline over all epochs, grouped by epochs.
    query.timeline("tpu2").busy(["task1"]).epoch_mean()
  """

  def __init__(self, timelines: dict[str, BaseTimeline], main_thread_id: str):
    self._timelines: dict[str, BaseTimeline] = timelines
    self._main_thread_id = main_thread_id

    def timeline(self, id: str) -> PerfMetricsQuery:
        self._select_timeline = id
        return self

    def epoch(self, epochs: list[int]) -> PerfMetricsQuery:
        self._select_epochs = epochs
        return self

    def idle(self) -> PerfMetricsQuery:
        self._select_busy = False
        return self

  def main(self) -> PerfMetricsQuery:
    self._select_timeline = self._main_thread_id
    return self

  def epoch(self, epochs: list[int]) -> PerfMetricsQuery:
    self._select_epochs = epochs
    return self

    def _gather(self) -> list[list[float]]:
        if self._select_timeline is None:
            raise ValueError("No timeline selected.")
        if self._select_timeline not in self._timelines:
            raise ValueError(f"Timeline '{self._select_timeline}' not found.")
        timeline = self._timelines[self._select_timeline]

  def busy(self, labels: list[str] | None = None) -> PerfMetricsQuery:
    self._select_labels = labels
    self._select_busy = True
    return self

  def _gather_intervals(self) -> list[list[tuple[float, float]]]:
    """Gathers the intervals from the selected timeline, grouped by epochs."""
    if self._select_timeline is None:
      raise ValueError("No timeline selected.")
    if self._select_timeline not in self._timelines:
      raise ValueError(f"Timeline '{self._select_timeline}' not found.")
    timeline = self._timelines[self._select_timeline]

    if self._select_epochs is not None:
      for select_epoch in self._select_epochs:
        if select_epoch >= len(timeline.epochs):
          raise ValueError(
              f"Epoch {select_epoch} exceeds max epoch"
              f" {len(timeline.epochs) - 1}."
          )
      epochs = [timeline.epochs[i] for i in self._select_epochs]
    else:
      epochs = timeline.epochs

    if self._select_busy is None:
      raise ValueError("None of idle() or busy() is called.")

    intervals_by_epochs: list[list[tuple[float, float]]] = []

    epoch_begin = 0
    interval_begin = timeline.born
    for epoch_end in epochs:
      intervals = []
      for i in range(epoch_begin, epoch_end):
        if self._select_busy:
          if (
              self._select_labels is None
              or timeline.labels[i] in self._select_labels
          ):
            intervals.append(
                (timeline.intervals[i][0], timeline.intervals[i][1])
            )
        else:
          # Idle intervals are the gaps between busy intervals.
          intervals.append((interval_begin, timeline.intervals[i][0]))
          interval_begin = timeline.intervals[i][1]
      intervals_by_epochs.append(intervals)
      epoch_begin = epoch_end

    return intervals_by_epochs

  def _gather_deltas(self):
    """Gathers the deltas from the selected timeline, grouped by epochs."""
    intervals_by_epochs: list[list[tuple[float, float]]] = (
        self._gather_intervals()
    )

    deltas_by_epochs: list[list[float]] = []
    for intervals in intervals_by_epochs:
      deltas = [interval[1] - interval[0] for interval in intervals]
      deltas_by_epochs.append(deltas)

    return deltas_by_epochs

  def intervals(self) -> list[tuple[float, float]]:
    return list(itertools.chain(*self._gather_intervals()))

  def sum(self) -> float:
    return np.sum(self.tolist()).item()

        epoch_begin = 0
        delta_begin = timeline.born
        for epoch_end in epochs:
            deltas = []
            for i in range(epoch_begin, epoch_end):
                if self._select_busy:
                    if self._select_labels is None or timeline.labels[i] in self._select_labels:
                        deltas.append(timeline.intervals[i][1] - timeline.intervals[i][0])
                else:
                    deltas.append(timeline.intervals[i][0] - delta_begin)
                    delta_begin = timeline.intervals[i][1]
            deltas_by_epochs.append(deltas)
            epoch_begin = epoch_end

  def tolist(self) -> list[float]:
    return list(itertools.chain(*self._gather_deltas()))

  def epoch_intervals(self) -> list[list[tuple[float, float]]]:
    return self._gather_intervals()

  def epoch_sum(self) -> list[float]:
    deltas_by_epochs = self._gather_deltas()
    return [np.sum(deltas).item() for deltas in deltas_by_epochs]

  def epoch_mean(self) -> list[float]:
    deltas_by_epochs = self._gather_deltas()
    return [np.mean(deltas).item() for deltas in deltas_by_epochs]

  def epoch_tolist(self) -> list[list[float]]:
    return self._gather_deltas()
