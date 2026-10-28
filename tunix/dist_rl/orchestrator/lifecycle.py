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

Sequences the worker lifecycle (`initialize` -> `compile` -> `start`) across an
entire registry, phase by phase, so every worker finishes one phase before the
next begins (e.g. all compiles happen before any worker starts serving).
Shutdown is best-effort: every worker gets a `stop()` even if an earlier one
raised, and the collected failures are reported together.
"""

from collections.abc import Callable, Sequence
import concurrent.futures
import logging
from typing import Any

from tunix.dist_rl.common import datatypes
from tunix.dist_rl.orchestrator import worker_registry
from tunix.dist_rl.worker import remote_execution

_INIT_METADATA_KEYS = (
    "trajectory_store_config",
    "drain_timeout_s",
    "partial_rollout",
)


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
      self,
      registry: worker_registry.WorkerRegistry,
      max_workers: int = 32,
      verify_fn: Callable[[Sequence[datatypes.WorkerInfo]], None] | None = None,
  ):
    self._registry = registry
    self._max_workers = max_workers
    self._verify_fn = verify_fn

  def bring_up(
      self,
      dummy_data: Any = None,
  ) -> None:
    """Runs initialize -> verify -> compile -> start across workers, phase by phase.

    Each phase runs across all target workers before the next phase begins. If
    any worker raises an exception during a phase, the exception is re-raised
    after all workers have attempted the phase, preventing subsequent phases
    from running.

    Args:
      dummy_data: Dummy data each worker uses to synthesize warmup dummies.

    Raises:
      RuntimeError: If a target worker was unregistered or re-registered with a
        different handle during bring-up.
    """
    resolved_targets = self._registry.targets()

    def _init_worker(
        target: tuple[datatypes.WorkerInfo, remote_execution.ActorHandle],
    ) -> None:
      info, handle = target
      resp = handle.submit("initialize")
      metadata = {
          k: resp.metadata[k] for k in _INIT_METADATA_KEYS if k in resp.metadata
      }
      if not self._registry.update_resources(
          info.worker_id, metadata, expected_handle=handle
      ):
        raise RuntimeError(
            f"worker {info.worker_id!r} was unregistered or re-registered"
            " with a different handle during bring-up"
        )

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=self._max_workers
    ) as pool:
      # TODO(noghabi): Refactor to allow successful workers to proceed to
      # compile/start and potentially retry failed ones, rather than fail-fast
      # on the first error.
      list(pool.map(_init_worker, resolved_targets))
      if self._verify_fn is not None:
        self._verify_fn([info for info, _ in resolved_targets])
      list(
          pool.map(
              lambda target: target[1].submit("compile", dummy_data),
              resolved_targets,
          )
      )
      list(
          pool.map(
              lambda target: target[1].submit("start"),
              resolved_targets,
          )
      )

  def shutdown(self) -> None:
    """Stops every worker best-effort, then raises if any stop() failed."""
    failures: list[tuple[str, BaseException]] = []
    worker_ids = self._registry.worker_ids()

    def _stop_worker(wid: str) -> None:
      try:
        handle = self._registry.handle(wid)
      except KeyError:
        logging.warning(
            "Worker %r unregistered concurrently, nothing to stop.", wid
        )
        return  # Worker unregistered concurrently, nothing to stop.
      handle.submit("stop")

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
