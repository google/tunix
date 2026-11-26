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

"""Measures how long a device sat idle before each computation sent to it."""

from collections.abc import Callable
from concurrent import futures
import functools
import queue
import threading
import time
from typing import Any, TypeVar

_T = TypeVar("_T")
# A call to run on the observer thread, and the future awaiting its result.
_Task = tuple[Callable[[], Any], futures.Future[Any] | None]


class DeviceIdleTracker:
  """Records the idle time of a device before each computation sent to it.

  Dispatch returns before the device finishes, so the caller cannot tell when
  the device became free. A daemon observer thread waits for each computation
  in dispatch order and records how long the device had been free when the next
  computation arrived. Work that occupies the device without being recorded is
  reported with `mark_busy_until_now`.

  `_free_at`, `_idle_times` and `_error` are only touched on the observer
  thread.
  """

  def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
    self._clock = clock
    self._free_at: float = clock()
    self._idle_times: list[float] = []
    self._error: Exception | None = None
    self._lock = threading.Lock()
    self._closed: bool = False
    self._tasks: queue.SimpleQueue[_Task | None] = queue.SimpleQueue()
    self._thread: threading.Thread | None = None

  def now(self) -> float:
    """Returns the current reading of the tracker's clock."""
    return self._clock()

  def reset(self) -> None:
    """Drops recorded idle times and errors; the device is busy until now."""
    self._call(functools.partial(self._reset, self._clock()))

  def mark_busy_until_now(self) -> None:
    """Notes that unrecorded work occupied the device until now."""
    self._post(functools.partial(self._advance_free_at, self._clock()))

  def record(
      self,
      t_enter: float,
      wait_until_done: Callable[[], object],
  ) -> None:
    """Records a computation that arrived at `t_enter` and was just dispatched.

    Args:
      t_enter: `now()` reading from when the computation arrived.
      wait_until_done: Blocks until the device has finished the computation,
        e.g. `jax.Array.block_until_ready`.
    """
    self._post(
        functools.partial(
            self._observe, t_enter, self._clock(), wait_until_done
        )
    )

  def flush(self) -> tuple[float, ...]:
    """Returns and clears idle times in seconds in dispatch order.

    Waits for every recorded computation to finish first.

    Raises:
      Exception: The first error raised while waiting for a computation since
        the last `flush()` or `reset()`.
    """
    return self._call(self._flush)

  def close(self, *, wait: bool) -> None:
    """Stops the observer thread, joining it only when `wait=True`.

    With `wait=False` a computation that never finishes cannot block the
    caller: queued work is dropped and pending calls fail.
    """
    with self._lock:
      thread = self._thread
      if not self._closed:
        self._closed = True
        if thread is not None:
          if not wait:
            self._fail_queued_tasks()
          self._tasks.put(None)
    if wait and thread is not None:
      thread.join()

  def _post(
      self, fn: Callable[[], _T], future: futures.Future[_T] | None = None
  ) -> bool:
    """Queues `fn` for the observer thread; returns False once closed."""
    with self._lock:
      if self._closed:
        return False
      if self._thread is None:
        self._thread = threading.Thread(
            target=self._run, name="device_idle_tracker", daemon=True
        )
        self._thread.start()
      self._tasks.put((fn, future))
      return True

  def _call(self, fn: Callable[[], _T]) -> _T:
    """Runs `fn` on the observer thread after all queued work."""
    future: futures.Future[_T] = futures.Future()
    if not self._post(fn, future):
      raise RuntimeError("DeviceIdleTracker is closed.")
    return future.result()

  def _fail_queued_tasks(self) -> None:
    while True:
      try:
        task = self._tasks.get_nowait()
      except queue.Empty:
        return
      if task is not None and task[1] is not None:
        task[1].set_exception(
            RuntimeError("DeviceIdleTracker closed before the call completed.")
        )

  def _run(self) -> None:
    while True:
      task = self._tasks.get()
      if task is None:
        return
      fn, future = task
      if future is None:
        fn()
      else:
        try:
          future.set_result(fn())
        except Exception as exc:  # pylint: disable=broad-exception-caught
          future.set_exception(exc)
      # Do not keep a finished computation alive while idle.
      del task, fn, future

  def _reset(self, t: float) -> None:
    self._free_at = max(self._free_at, t)
    self._idle_times.clear()
    self._error = None

  def _advance_free_at(self, t: float) -> None:
    self._free_at = max(self._free_at, t)

  def _observe(
      self,
      t_enter: float,
      t_dispatch: float,
      wait_until_done: Callable[[], object],
  ) -> None:
    if self._error is not None:
      return
    self._idle_times.append(max(0.0, t_enter - self._free_at))
    try:
      wait_until_done()
    except Exception as exc:  # pylint: disable=broad-exception-caught
      self._error = exc
    self._free_at = max(self._free_at, t_dispatch, self._clock())

  def _flush(self) -> tuple[float, ...]:
    error, self._error = self._error, None
    idle_times = tuple(self._idle_times)
    self._idle_times.clear()
    if error is not None:
      raise error
    return idle_times
