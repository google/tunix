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

"""API of Metrics for RL workflows."""

from __future__ import annotations

import itertools
import numpy as np
from graphviz import Digraph
from typing import Any


BaseTimeline = Any

class PerfMetricsConfig:
    pass

class PerfMetricsApi:

    def __init__(self, timelines: dict[str, BaseTimeline]):
        self._timelines: dict[str, BaseTimeline] = timelines

    def dump(self) -> None:
        for timeline in self._timelines.values():
            print(timeline)

    def draw_dot(self) -> Digraph:
        dot = Digraph(name="Timelines")
        dot.node_attr.update(shape="box", style="filled")

        for timeline in self._timelines.values():
            dot.node(f"{timeline.id}-h", f"{timeline.id}")
            dot.edge(f"{timeline.id}-h", f"{timeline.id}-e0")
            with dot.subgraph(name=f"{timeline.id}") as t:
                epoch_begin = 0
                epoch_ends = timeline.epochs + [len(timeline.labels)]
                for epoch in range(len(epoch_ends)):
                    epoch_end = epoch_ends[epoch]
                    
                    if epoch_begin == epoch_end:
                        break

                    # epoch head
                    t.node(f"{timeline.id}-e{epoch}", f"epoch [{epoch}]", fillcolor="grey")

                    # epoch body
                    for i in range(epoch_begin, epoch_end):
                        t.node(f"{timeline.id}-{i}", f"{timeline.labels[i]}\n{(timeline.intervals[i][1] - timeline.intervals[i][0])*1000.0:.2f} ms")

                    # previous epoch body -> epoch head
                    if epoch > 0:
                        t.edge(f"{timeline.id}-{epoch_begin-1}", f"{timeline.id}-e{epoch}")

                    # epoch head -> epoch body
                    if epoch_begin < len(timeline.intervals):
                        if epoch_begin > 0:
                            epoch_idle = timeline.intervals[epoch_begin][0] - timeline.intervals[epoch_begin-1][1]
                        else:
                            epoch_idle = timeline.intervals[epoch_begin][0] - timeline.born
                        t.edge(f"{timeline.id}-e{epoch}", f"{timeline.id}-{epoch_begin}", f"{epoch_idle*1000.0:.2f} ms")

                    # epoch body -> epoch body
                    for i in range(epoch_begin, epoch_end-1):
                        t.edge(f"{timeline.id}-{i}", f"{timeline.id}-{i+1}", f"{(timeline.intervals[i+1][0] - timeline.intervals[i][1])*1000.0:.2f} ms")

                    epoch_begin = epoch_end
        return dot

    def query(self) -> PerfMetricsQuery:
        return PerfMetricsQuery(self._timelines)


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

    def __init__(self, timelines: dict[str, BaseTimeline]):
        self._timelines: dict[str, BaseTimeline] = timelines

        self._select_timeline: str | None = None
        self._select_epochs: list[int] | None = None
        self._select_busy: bool | None = None
        self._select_labels: list[str] | None = None

    def timeline_ids(self) -> list[str]:
        return list(self._timelines.keys())

    def timeline(self, id: str) -> PerfMetricsQuery:
        self._select_timeline = id
        return self

    def epoch(self, epochs: list[int]) -> PerfMetricsQuery:
        self._select_epochs = epochs
        return self

    def idle(self) -> PerfMetricsQuery:
        self._select_busy = False
        return self

    def busy(self, labels: list[str] | None = None) -> PerfMetricsQuery:
        self._select_labels = labels
        self._select_busy = True
        return self

    def _gather(self) -> list[list[float]]:
        if self._select_timeline is None:
            raise ValueError("No timeline selected.")
        if self._select_timeline not in self._timelines:
            raise ValueError(f"Timeline '{self._select_timeline}' not found.")
        timeline = self._timelines[self._select_timeline]

        if self._select_epochs is not None:
            for select_epoch in self._select_epochs:
                if select_epoch >= len(timeline.epochs):
                    raise ValueError(f"Epoch {select_epoch} exceeds max epoch {len(timeline.epochs) - 1}.")
            epochs = self._select_epochs
        else:
            epochs = timeline.epochs

        if self._select_busy is None:
            raise ValueError("None of idle() or busy() is called.")

        deltas_by_epochs: list[list[float]] = []

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

        return deltas_by_epochs

    def sum(self) -> float:
        return np.sum(self.tolist()).item()

    def mean(self) -> float:
        return np.mean(self.tolist()).item()

    def tolist(self) -> list[float]:
        return list(itertools.chain(*self._gather()))

    def epoch_sum(self) -> list[float]:
        deltas_by_epochs = self._gather()
        return [np.sum(deltas).item() for deltas in deltas_by_epochs]

    def epoch_mean(self) -> list[float]:
        deltas_by_epochs = self._gather()
        return [np.mean(deltas).item() for deltas in deltas_by_epochs]

    def epoch_tolist(self) -> list[list[float]]:
        return self._gather()
