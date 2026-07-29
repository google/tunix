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

"""A unified fake worker for orchestrator tests."""

import collections
from typing import Any

from tunix.experimental.common import datatypes
from tunix.experimental.worker import abstract_worker


class FakeWorker(abstract_worker.Worker):
  """A unified fake worker for orchestrator tests.

  This mock worker allows tests to track the lifecycle states and call counts
  of its methods. It optionally supports appending lifecycle events to a shared
  log to verify execution ordering across multiple workers, and can simulate
  failures during shutdown.
  """

  def __init__(
      self,
      worker_id: str,
      roles: set[str] | frozenset[str],
      resources: dict[str, Any] | None = None,
      log: list[str] | None = None,
      fail_stop: bool = False,
  ):
    """Initializes the fake worker.

    Args:
      worker_id: The unique identifier for this worker.
      roles: The roles this worker satisfies (e.g., {"trainer", "rollout"}).
      resources: Optional dict of resources.
      log: An optional shared list. If provided, the worker will append
        lifecycle transitions (e.g., "{worker_id}:start") to this list, which
        is useful for asserting the global ordering of calls across multiple
        workers.
      fail_stop: If True, the worker will raise a RuntimeError when stop() is
        called to simulate a failure during shutdown.
    """
    self._info = datatypes.WorkerInfo(
        worker_id=worker_id, roles=frozenset(roles), resources=resources or {}
    )
    self.state = "PENDING"
    # Tracks how many times each method was called
    self.call_counts = collections.Counter()
    self._log = log
    self._fail_stop = fail_stop

  def info(self) -> datatypes.WorkerInfo:
    return self._info

  def heartbeat(self) -> datatypes.HealthReport:
    return datatypes.HealthReport(state=self.state)

  def initialize(self) -> datatypes.Response:
    self.call_counts["initialize"] += 1
    self.state = "INITIALIZED"
    if self._log is not None:
      self._log.append(f"{self._info.worker_id}:initialize")
    return datatypes.Response()

  def compile(self, dummy_data: Any = None) -> datatypes.Response:
    self.call_counts["compile"] += 1
    self.state = "COMPILED"
    if self._log is not None:
      self._log.append(f"{self._info.worker_id}:compile")
    return datatypes.Response()

  def start(self) -> datatypes.Response:
    self.call_counts["start"] += 1
    self.state = "READY"
    if self._log is not None:
      self._log.append(f"{self._info.worker_id}:start")
    return datatypes.Response()

  def stop(self) -> datatypes.Response:
    self.call_counts["stop"] += 1
    self.state = "STOPPED"
    if self._log is not None:
      self._log.append(f"{self._info.worker_id}:stop")
    if self._fail_stop:
      raise RuntimeError(f"{self._info.worker_id} refused to stop")
    return datatypes.Response()
