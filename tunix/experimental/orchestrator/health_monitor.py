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

"""Health monitor for the control plane.

Concurrently polls each worker's `heartbeat()` across the WorkerRegistry.
"""

import concurrent.futures
import logging
import threading

from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import worker_registry


class HealthMonitor:
  """Polls registered workers' heartbeat RPCs using a thread pool."""

  def __init__(
      self,
      registry: worker_registry.WorkerRegistry,
      *,
      max_workers: int = 32,
      executor: concurrent.futures.ThreadPoolExecutor | None = None,
  ):
    self._registry = registry
    self._max_workers = max_workers
    if executor is not None:
      self._executor = executor
      self._owns_executor = False
    else:
      self._executor = concurrent.futures.ThreadPoolExecutor(
          max_workers=self._max_workers
      )
      self._owns_executor = True

  def close(self) -> None:
    """Shuts down the thread pool executor if owned by this monitor."""
    if self._owns_executor:
      self._executor.shutdown(wait=True)

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.close()
    return False

  def poll(self) -> dict[str, datatypes.HealthReport]:
    """Polls every worker once.

    Returns:
      A mapping of worker_id -> the HealthReport captured this poll.
    """
    reports: dict[str, datatypes.HealthReport] = {}
    worker_ids = self._registry.worker_ids()
    abort_event = threading.Event()

    def _poll_worker(wid: str) -> tuple[str, datatypes.HealthReport | None]:
      if abort_event.is_set():
        return wid, None
      try:
        handle = self._registry.handle(wid)
      except KeyError:
        logging.warning(
            "Worker %r unregistered concurrently, skipping poll.", wid
        )
        return wid, None
      try:
        return wid, handle.submit("heartbeat")
      except BaseException:
        abort_event.set()
        raise

    futures = [
        self._executor.submit(_poll_worker, wid) for wid in worker_ids
    ]
    try:
      for future in concurrent.futures.as_completed(futures):
        wid, report = future.result()
        if report is None:
          continue
        reports[wid] = report
    finally:
      abort_event.set()
      for future in futures:
        future.cancel()

    return reports
