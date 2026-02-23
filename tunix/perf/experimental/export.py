# Copyright 2026 Google LLC
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

"""Metrics export functions."""

from absl import logging
from tunix.perf import metrics

MetricsT = metrics.MetricsT
PerfSpanQuery = metrics.PerfSpanQuery


class DebugMetricsExport:
  """Debug metrics export functions."""

  @staticmethod
  def log_export_fn(query: PerfSpanQuery) -> MetricsT:
    """Logs the export function."""
    logging.info("\n=== Exporting Timelines ===\n %s", query)
    return {}
