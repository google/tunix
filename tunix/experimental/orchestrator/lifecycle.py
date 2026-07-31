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

"""Lifecycle driver for bringing a fleet of workers up and down.

Sequences the `Worker` lifecycle (`initialize` -> `compile` -> `start`) across
an
entire registry, phase by phase, so every worker finishes one phase before the
next begins (e.g. all compiles happen before any worker starts serving).
Shutdown is best-effort: every worker gets a `stop()` even if an earlier one
raised, and the collected failures are reported together.
"""

import concurrent.futures
import logging
from typing import Any

from tunix.experimental.orchestrator import worker_registry


class LifecycleError(RuntimeError):
  """Aggregated failures from a lifecycle phase across multiple workers."""

  def __init__(self, phase: str, failures: list[tuple[str, BaseException]]):
    self.phase = phase
    self.failures = failures
    detail = "; ".join(f"{wid}: {err!r}" for wid, err in failures)
    super().__init__(f"{phase} failed for {len(failures)} worker(s): {detail}")


class LifecycleDriver:
  """Drives a WorkerRegistry through the worker lifecycle phases."""

  def __init__(
      self, registry: worker_registry.WorkerRegistry, max_workers: int = 32
  ):
    self._registry = registry
    self._max_workers = max_workers

  def bring_up(self, dummy_data: Any) -> None:
    """Runs initialize -> compile -> start across all workers, phase by phase.

    Each phase runs to completion for every worker before the next phase begins.
    A phase aborts on the first failure (fail-fast), since a half-initialized
    fleet should not proceed to compile or serve.
    Args:
      dummy_data: Dummy data each worker uses to synthesize warmup dummies.
    """
    workers = self._registry.workers()
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=self._max_workers
    ) as pool:
      # TODO(noghabi): Refactor to allow successful workers to proceed to
      # compile/start and potentially retry failed ones, rather than fail-fast
      # on the first error.
      list(pool.map(lambda w: w.initialize(), workers))
      list(pool.map(lambda w: w.compile(dummy_data), workers))
      list(pool.map(lambda w: w.start(), workers))

  def shutdown(self) -> None:
    """Stops every worker best-effort, then raises if any stop() failed."""
    failures: list[tuple[str, BaseException]] = []
    worker_ids = self._registry.worker_ids()

    def _stop_worker(wid: str) -> None:
      try:
        worker = self._registry.get(wid)
      except KeyError:
        logging.warning(
            "Worker %r unregistered concurrently, nothing to stop.", wid
        )
        return  # Worker unregistered concurrently, nothing to stop.
      worker.stop()

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=self._max_workers
    ) as pool:
      futures = {pool.submit(_stop_worker, wid): wid for wid in worker_ids}
      for future in concurrent.futures.as_completed(futures):
        wid = futures[future]
        try:
          future.result()
        except Exception as err:  # pylint: disable=broad-except
          failures.append((wid, err))
    if failures:
      raise LifecycleError("shutdown", failures)
