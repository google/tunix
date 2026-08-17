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

"""Traffic controller for rollout worker admission and state management."""

import asyncio
import threading
from typing import Any

from tunix.experimental.common import datatypes

WorkerState = datatypes.WorkerState


class AdmissionClosedError(RuntimeError):
  """A rollout request arrived while admission is closed for weight sync."""


class TrafficController:
  """Encapsulates worker lifecycle state and task admission control.

  Provides a single source of truth for the rollout worker's state and active
  async tasks, avoiding distributed locks across the worker and manager.
  """

  def __init__(self):
    self._lock = threading.RLock()
    self._state = WorkerState.READY
    self._admission_open = asyncio.Event()
    self._admission_open.set()
    self._active_tasks: set[asyncio.Task[Any]] = set()

  @property
  def state(self) -> WorkerState:
    """Returns the current worker state."""
    with self._lock:
      return self._state

  async def wait_for_admission(self) -> None:
    """Blocks until admission is open."""
    await self._admission_open.wait()

  def is_admission_open(self) -> bool:
    """Returns True if admission is currently open."""
    with self._lock:
      return self._admission_open.is_set()

  def try_admit(self, task: asyncio.Task[Any]) -> bool:
    """Attempts to admit a task if the gate is open.

    If admission is open and the state is READY, the task is tracked.
    If not, the task is immediately cancelled.

    Args:
      task: An asyncio.Task to admit.

    Returns:
      True if admitted, False if rejected.
    """
    with self._lock:
      if not self._admission_open.is_set() or self._state != WorkerState.READY:
        task.cancel()
        return False
      self.track(task)
      return True

  def track(self, task: asyncio.Task[Any]) -> None:
    """Tracks an in-flight task; it removes itself on completion."""
    with self._lock:
      self._active_tasks.add(task)
    task.add_done_callback(self._on_task_done)

  def _on_task_done(self, task: asyncio.Task[Any]) -> None:
    with self._lock:
      self._active_tasks.discard(task)

  def get_active_tasks(self) -> list[asyncio.Task[Any]]:
    """Returns a snapshot of currently running tasks."""
    with self._lock:
      return list(self._active_tasks)

  async def drain(self, timeout_s: float) -> None:
    """Waits for tracked tasks to finish, up to the timeout."""
    with self._lock:
      tasks = list(self._active_tasks)
    if tasks:
      await asyncio.wait(tasks, timeout=timeout_s)

  def transition_to_syncing(self) -> None:
    """Transitions state to SYNCING and closes admission.

    Raises:
      RuntimeError: If transitioning from an invalid state.
    """
    with self._lock:
      if self._state not in (WorkerState.READY, WorkerState.SYNCING):
        raise RuntimeError(
            f"Cannot transition to SYNCING from {self._state.value}"
        )
      self._state = WorkerState.SYNCING
      self._admission_open.clear()

  def reopen(self) -> bool:
    """Reopens admission and sets state to READY.

    Returns:
      True if successfully reopened, False if the worker was STOPPED.
    """
    with self._lock:
      if self._state == WorkerState.STOPPED:
        return False
      self._state = WorkerState.READY
      self._admission_open.set()
      return True

  def stop_and_cancel_all(self) -> list[asyncio.Task[Any]]:
    """Permanently sets state to STOPPED, closes admission, and cancels tasks.

    Returns:
      A list of the cancelled tasks.
    """
    with self._lock:
      self._state = WorkerState.STOPPED
      self._admission_open.clear()
      tasks = list(self._active_tasks)

    # Cancel outside the lock to avoid reentrancy if callbacks are invoked immediately
    for task in tasks:
      task.cancel()
    return tasks
