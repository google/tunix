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

  def bring_up(
      self,
      dummy_data: Any,
      *,
      require_all: bool = True,
      max_attempts: int = 1,
  ) -> list[str]:
    """Runs initialize -> compile -> start across all workers, phase by phase.

    Each phase completes for every worker before the next begins, and a worker
    that fails a phase is dropped from the following ones rather than taking
    the fleet with it. Which workers failed, and on which phase, is reported
    together: bringing a fleet up one failure per restart is its own kind of
    outage.

    Args:
      dummy_data: Dummy data each worker uses to synthesize warmup dummies.
      require_all: Raise unless every worker came up. Turning this off is for
        fleets that can run degraded -- the caller then has to check what came
        back.
      max_attempts: Attempts per worker per phase. A worker whose startup is
        merely slow or racing a dependency often succeeds on a second try.

    Returns:
      The ids of workers that completed every phase.

    Raises:
      LifecycleError: If any worker failed and `require_all` is set.
      ValueError: If `max_attempts` is not positive.
    """
    if max_attempts < 1:
      raise ValueError(f"max_attempts must be >= 1, got {max_attempts}.")

    survivors = {w.info().worker_id: w for w in self._registry.workers()}
    failures: list[tuple[str, BaseException]] = []

    phases = (
        ("initialize", lambda w: w.initialize()),
        ("compile", lambda w: w.compile(dummy_data)),
        ("start", lambda w: w.start()),
    )
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=self._max_workers
    ) as pool:
      for phase, action in phases:
        if not survivors:
          break
        failed = self._run_phase(
            pool, phase, action, survivors, max_attempts, failures
        )
        for worker_id in failed:
          del survivors[worker_id]

    if failures and require_all:
      raise LifecycleError("bring_up", failures)
    if failures:
      logging.error(
          "Fleet came up degraded: %d of %d workers failed to start.",
          len(failures),
          len(failures) + len(survivors),
      )
    return sorted(survivors)

  def _run_phase(
      self,
      pool: concurrent.futures.ThreadPoolExecutor,
      phase: str,
      action: Any,
      workers: dict[str, Any],
      max_attempts: int,
      failures: list[tuple[str, BaseException]],
  ) -> list[str]:
    """Runs one phase across `workers`; returns the ids that failed it."""

    def _attempt(worker_id: str) -> tuple[str, BaseException | None]:
      last_error: BaseException | None = None
      for attempt in range(max_attempts):
        try:
          action(workers[worker_id])
          return worker_id, None
        except Exception as err:  # pylint: disable=broad-except
          last_error = err
          logging.warning(
              "Worker %r failed %s (attempt %d of %d): %r",
              worker_id,
              phase,
              attempt + 1,
              max_attempts,
              err,
          )
      return worker_id, last_error

    failed: list[str] = []
    futures = [pool.submit(_attempt, wid) for wid in list(workers)]
    for future in concurrent.futures.as_completed(futures):
      worker_id, error = future.result()
      if error is not None:
        failures.append((worker_id, error))
        failed.append(worker_id)
    return failed

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
