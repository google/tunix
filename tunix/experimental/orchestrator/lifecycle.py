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

Sequences the `Worker` lifecycle (`initialize` -> `compile` -> `start`) across an
entire registry, phase by phase, so every worker finishes one phase before the
next begins (e.g. all compiles happen before any worker starts serving).
Shutdown is best-effort: every worker gets a `stop()` even if an earlier one
raised, and the collected failures are reported together.
"""

from tunix.experimental.common import datatypes
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

  def __init__(self, registry: worker_registry.WorkerRegistry):
    self._registry = registry

  def bring_up(self, shape_config: datatypes.ShapeConfig) -> None:
    """Runs initialize -> compile -> start across all workers, phase by phase.

    Each phase runs to completion for every worker before the next phase begins.
    A phase aborts on the first failure (fail-fast), since a half-initialized
    fleet should not proceed to compile or serve.
    Args:
      shape_config: Shape hints each worker uses to synthesize warmup dummies.
    """
    for worker in self._registry.workers():
      worker.initialize()
    for worker in self._registry.workers():
      worker.compile(shape_config)
    for worker in self._registry.workers():
      worker.start()

  def shutdown(self) -> None:
    """Stops every worker best-effort, then raises if any stop() failed."""
    failures: list[tuple[str, BaseException]] = []
    for worker_id in self._registry.worker_ids():
      try:
        self._registry.get(worker_id).stop()
      except Exception as err:  # pylint: disable=broad-except
        failures.append((worker_id, err))
    if failures:
      raise LifecycleError("shutdown", failures)
