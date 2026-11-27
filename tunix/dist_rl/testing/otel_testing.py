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

"""Test helpers for reading back OpenTelemetry metrics from dist_rl workers."""

import collections
import itertools
from typing import Any

from opentelemetry.sdk.metrics import export as sdk_metrics_export


def collect_metrics(
    reader: sdk_metrics_export.InMemoryMetricReader,
) -> collections.defaultdict[str, list[tuple[Any, int | None, dict[str, Any]]]]:
  """Reads every data point currently held by `reader`.

  Args:
    reader: The in-memory reader attached to the test `MeterProvider`.

  Returns:
    `{metric_name: [(value_or_sum, count_or_none, attributes), ...]}`, where
    histograms report `(sum, count, attributes)` and counters or gauges report
    `(value, None, attributes)`. Empty when nothing has been recorded.
  """
  collected = collections.defaultdict(list)
  data = reader.get_metrics_data()
  resource_metrics = data.resource_metrics if data is not None else ()
  scope_metrics = itertools.chain.from_iterable(
      rm.scope_metrics for rm in resource_metrics
  )
  metrics = itertools.chain.from_iterable(sm.metrics for sm in scope_metrics)
  for metric in metrics:
    for dp in metric.data.data_points:
      attributes = dict(dp.attributes or {})
      if isinstance(dp, sdk_metrics_export.NumberDataPoint):
        collected[metric.name].append((dp.value, None, attributes))
      else:
        collected[metric.name].append((dp.sum, dp.count, attributes))
  return collected
