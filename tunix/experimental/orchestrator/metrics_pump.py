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

"""Metrics pump: pulls worker metric buffers, dedups, and stamps the step.

Workers buffer metrics on-device and hand them to the orchestrator on a pull, one
sealed buffer at a time. Each buffer carries a monotonic seal id; because a pull
may be retried (at-least-once), the pump drops any seal id it has already seen
per worker (idempotency) and stamps each newly-accepted buffer with the current
orchestrator step. This replaces per-worker step arithmetic with a single
step-stamping point.
"""

import collections
import dataclasses
from typing import Any

from tunix.experimental.metrics import metrics


@dataclasses.dataclass(kw_only=True)
class MetricRecord:
  """One accepted metric buffer, stamped with the orchestrator step.

  Attributes:
    step: Orchestrator step this buffer was pumped at.
    worker_id: The worker the buffer came from.
    seal_id: The buffer's seal id (as reported by the worker).
    scalars: Scalar metric values.
    mode: "train" or "eval".
  """

  step: int
  worker_id: str
  seal_id: Any
  scalars: dict[str, Any]
  mode: str


def _seal_key(seal_id: Any) -> Any:
  if isinstance(seal_id, (int, str)):
    return seal_id
  return int(seal_id)  # numpy/jax scalar -> hashable python int


class MetricsPump:
  """Drain-once metrics collector with per-worker seal-id dedup."""

  def __init__(self):
    self._seen: dict[str, set[Any]] = collections.defaultdict(set)
    self._records: list[MetricRecord] = []

  def pull(
      self, worker_id: str, buffer: metrics.MetricsBuffer, *, step: int
  ) -> bool:
    """Accepts a metric buffer unless its seal id was already seen.

    Args:
      worker_id: The source worker.
      buffer: The sealed metric buffer from `get_metrics()`.
      step: The orchestrator step to stamp an accepted buffer with.

    Returns:
      True if the buffer was newly accepted; False if it was a re-delivery.
    """
    key = _seal_key(buffer.id)
    if key in self._seen[worker_id]:
      return False
    self._seen[worker_id].add(key)
    self._records.append(
        MetricRecord(
            step=step,
            worker_id=worker_id,
            seal_id=buffer.id,
            scalars=dict(buffer.scalar_metrics),
            mode=buffer.mode,
        )
    )
    return True

  def records(self) -> list[MetricRecord]:
    """All accepted records, in pull order."""
    return list(self._records)

  def records_for(self, worker_id: str) -> list[MetricRecord]:
    return [r for r in self._records if r.worker_id == worker_id]
