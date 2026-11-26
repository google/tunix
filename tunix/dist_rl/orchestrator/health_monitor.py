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

from collections.abc import Collection
import concurrent.futures
import threading

from absl import logging
from tunix.dist_rl.common import datatypes
from tunix.dist_rl.orchestrator import worker_registry
from tunix.dist_rl.worker import remote_execution


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

  def _is_current_handle(
      self, wid: str, handle: remote_execution.ActorHandle
  ) -> bool:
    """Returns True if `handle` is still the active registered handle for `wid`."""
    try:
      return self._registry.handle(wid) is handle
    except KeyError:
      return False

  def _poll_worker(
      self,
      info: datatypes.WorkerInfo,
      handle: remote_execution.ActorHandle,
      abort_event: threading.Event,
  ) -> tuple[str, datatypes.HealthReport | None]:
    """Polls `handle` if still current, skipping concurrent displacements."""
    wid = info.worker_id
    if abort_event.is_set():
      return wid, None
    if not self._is_current_handle(wid, handle):
      logging.warning(
          "Worker %r unregistered or re-registered concurrently, skipping"
          " poll.",
          wid,
      )
      return wid, None
    try:
      return wid, handle.submit("heartbeat")
    except Exception:  # pylint: disable=broad-exception-caught
      if not self._is_current_handle(wid, handle):
        logging.warning(
            "Worker %r unregistered or re-registered during heartbeat, skipping"
            " poll.",
            wid,
        )
        return wid, None
      abort_event.set()
      raise
    except BaseException:
      abort_event.set()
      raise

  def poll(
      self,
      *,
      statuses: Collection[worker_registry.AdmissionStatus] | None = None,
  ) -> dict[str, datatypes.HealthReport]:
    """Polls matching workers once.

    Args:
      statuses: Admission statuses to include, or `None` to poll every
        registered worker regardless of admission status.

    Returns:
      A mapping of worker_id -> the HealthReport captured this poll.
    """
    reports: dict[str, datatypes.HealthReport] = {}
    targets = self._registry.targets(statuses=statuses)
    abort_event = threading.Event()
    futures = [
        self._executor.submit(self._poll_worker, info, handle, abort_event)
        for info, handle in targets
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
