# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Multiplexed out-of-order task execution session (`TaskSession`, `DefaultRouter`)."""

import asyncio
import hashlib
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from tunix.experimental.worker import remote_execution

ActorHandle = remote_execution.ActorHandle
ExecutionResponse = remote_execution.ExecutionResponse
LONG_POLL_TIMEOUT_S = remote_execution.LONG_POLL_TIMEOUT_S

RouteFn = Callable[[Sequence[ActorHandle], Optional[Any]], ActorHandle]


def stable_route_hash(route_key: Any) -> int:
  """Maps a sticky routing key to a process-stable non-negative integer.

  Args:
    route_key: The sticky routing key. Non-negative integers are returned
      as-is; any other value is hashed via its `str()` representation.

  Returns:
    A non-negative integer suitable for `% num_actors` bucketing.

  Raises:
    ValueError: If `route_key` is a negative integer.
  """
  if isinstance(route_key, int):
    if route_key < 0:
      raise ValueError(
          "An integer route_key is an explicit shard index and must be"
          f" non-negative, got {route_key}."
      )
    return route_key
  digest = hashlib.blake2b(str(route_key).encode("utf-8"), digest_size=8)
  return int.from_bytes(digest.digest(), "big")


class DefaultRouter:
  """Routes by process-stable `route_key` hash when given, else round-robin."""

  def __init__(self) -> None:
    self._idx = 0

  def __call__(
      self,
      actors: Sequence[ActorHandle],
      route_key: Optional[Any] = None,
  ) -> ActorHandle:
    """Selects a target `ActorHandle` from `actors`.

    Args:
      actors: Available worker `ActorHandle` sequence.
      route_key: Optional sticky routing key.

    Returns:
      The selected target `ActorHandle`.

    Raises:
      RuntimeError: If `actors` is empty.
    """
    if not actors:
      raise RuntimeError("TaskSession has no registered ActorHandles.")
    if route_key is not None:
      # Not builtin `hash()`: it salts `str` per process, so placement would
      # not survive a restart.
      # TODO(tunix-dev): `% len(actors)` remaps every key when pool
      # membership changes, not just the keys on the affected actor. Switch to
      # rendezvous (HRW) hashing before adding actor eviction.
      return actors[stable_route_hash(route_key) % len(actors)]
    actor = actors[self._idx % len(actors)]
    self._idx += 1
    return actor


class TaskSession:
  """Multiplexed out-of-order task execution session over ActorHandles."""

  def __init__(
      self,
      actors: Union[
          Sequence[ActorHandle],
          Callable[[], Sequence[ActorHandle]],
      ],
      *,
      route_fn: Optional[RouteFn] = None,
  ):
    """Initializes the task execution session.

    Args:
      actors: A sequence of `ActorHandle` instances or a zero-arg callable
        returning the current sequence of `ActorHandle`s.
      route_fn: Optional routing callable `(actors, route_key) -> ActorHandle`.
        Defaults to `DefaultRouter()`.

    Raises:
      TypeError: If any item in a static `actors` sequence is not an
        `ActorHandle`.
    """
    if callable(actors):
      self._actors_fn: Callable[[], Sequence[ActorHandle]] = actors
    else:
      for actor in actors:
        if not isinstance(actor, ActorHandle):
          raise TypeError(f"Expected ActorHandle, got {type(actor)}")
      static_actors = tuple(actors)
      self._actors_fn = lambda: static_actors

    self._route_fn: RouteFn = (
        route_fn if route_fn is not None else DefaultRouter()
    )
    self._response_queue: asyncio.Queue[Any] = asyncio.Queue()
    self._active_workers: set[ActorHandle] = set()
    self._dispatched_tasks: Dict[ActorHandle, set[str]] = {}
    self._poll_tasks: set[asyncio.Task[Any]] = set()
    self._in_flight = 0
    self._closed = False
    self._sentinel = object()

  async def __aenter__(self) -> "TaskSession":
    return self

  async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
    del exc_type, exc_val, exc_tb
    await self.close()

  def _notify_if_zero_flight(self) -> None:
    if self._in_flight == 0:
      self._response_queue.put_nowait(self._sentinel)

  def route(self, route_key: Optional[Any] = None) -> ActorHandle:
    """Selects a target `ActorHandle` from current live actors via `route_fn`."""
    return self._route_fn(self._actors_fn(), route_key)

  async def submit(
      self,
      request_id: str,
      method_name: Optional[str] = None,
      *args: Any,
      route_key: Optional[Any] = None,
      **kwargs: Any,
  ) -> str:
    """Dispatches a task to a routed worker and tracks its completion.

    Args:
      request_id: Unique identifier for the in-flight request.
      method_name: Remote method name to invoke on the selected worker.
      *args: Positional arguments forwarded to `dispatch_task`.
      route_key: Optional sticky routing key consumed by `route_fn`.
      **kwargs: Keyword arguments forwarded to `dispatch_task`.

    Returns:
      The submitted `request_id`.

    Raises:
      RuntimeError: If the session is closed or no `ActorHandle`s are
        available.
    """
    if self._closed:
      raise RuntimeError("TaskSession is closed.")
    actor = self.route(route_key)

    # Increment in_flight and register request_id BEFORE awaiting dispatch_task
    # so a fast worker completing before dispatch_task's coroutine resumes
    # always finds request_id in _dispatched_tasks[actor].
    self._in_flight += 1
    dispatched_set = self._dispatched_tasks.setdefault(actor, set())
    dispatched_set.add(request_id)
    self._ensure_worker_polling(actor)

    try:
      await actor.dispatch_task(
          request_id, method_name, *args, **kwargs
      )
      # Re-ensure worker polling is active in case the previous polling loop
      # exited or died while dispatch_task was awaiting.
      self._ensure_worker_polling(actor)
      return request_id
    except BaseException:
      # Roll back _in_flight on any exception or cancellation if
      # _poll_worker_loop hasn't already failed and cleared it.
      if request_id in dispatched_set:
        dispatched_set.remove(request_id)
        self._in_flight = max(0, self._in_flight - 1)
        self._notify_if_zero_flight()
      raise

  def _ensure_worker_polling(self, actor: ActorHandle) -> None:
    if actor in self._active_workers:
      return
    self._active_workers.add(actor)
    task = asyncio.create_task(self._poll_worker_loop(actor))
    self._poll_tasks.add(task)
    task.add_done_callback(self._poll_tasks.discard)

  async def _poll_worker_loop(self, actor: ActorHandle) -> None:
    """Long-polls a single active worker until its in-flight set drains."""
    try:
      while not self._closed:
        dispatched_set = self._dispatched_tasks.setdefault(actor, set())
        if not dispatched_set:
          break
        try:
          response = await actor.poll_responses(timeout_s=LONG_POLL_TIMEOUT_S)
          if isinstance(response, ExecutionResponse):
            if (
                not response.request_id
                or response.request_id not in dispatched_set
            ):
              raise RuntimeError(
                  "Received ExecutionResponse with unknown or missing"
                  f" request_id={response.request_id!r}; in-flight"
                  f" request_ids={sorted(dispatched_set)!r}."
              )
            dispatched_set.remove(response.request_id)
            self._in_flight = max(0, self._in_flight - 1)
            try:
              res = response.unwrap()
              self._response_queue.put_nowait((res, None))
            except Exception as exc:  # pylint: disable=broad-exception-caught
              self._response_queue.put_nowait((None, exc))
            self._notify_if_zero_flight()
        except asyncio.CancelledError:
          break
        except Exception as exc:  # pylint: disable=broad-exception-caught
          # Transport or polling failure on this worker; fail all dispatched
          # tasks on this worker.
          failed_count = len(dispatched_set)
          dispatched_set.clear()
          if failed_count > 0:
            for _ in range(failed_count):
              self._response_queue.put_nowait((None, exc))
            self._in_flight = max(0, self._in_flight - failed_count)
            self._notify_if_zero_flight()
          break
    finally:
      self._active_workers.discard(actor)

  async def poll_completed(
      self, timeout_s: float = LONG_POLL_TIMEOUT_S
  ) -> List[Tuple[Any, Optional[Exception]]]:
    """Waits up to `timeout_s` for the first completion, then drains ready items."""
    if self._closed:
      return []

    batch: List[Tuple[Any, Optional[Exception]]] = []
    # When idle (_in_flight == 0), yield briefly (<= 50ms) so concurrent
    # dispatch coroutines can run without stalling for the full 50s timeout.
    wait_s = (
        min(timeout_s, 0.05)
        if (self._in_flight == 0 and self._response_queue.empty())
        else timeout_s
    )
    try:
      # Block until the first real (result, exc) completion arrives.
      while not batch:
        item = await asyncio.wait_for(
            self._response_queue.get(), timeout=wait_s
        )
        if item is self._sentinel:
          # Sentinel marks _in_flight reaching 0 or session close; return early
          # if drained, otherwise skip stale sentinels when new tasks are in
          # flight or queued behind it.
          if self._in_flight == 0 and self._response_queue.empty():
            return []
          continue
        batch.append(item)
    except asyncio.TimeoutError:
      return []

    # Greedily drain any additional completions already queued by background
    # worker loops.
    while not self._response_queue.empty():
      item = self._response_queue.get_nowait()
      if item is not self._sentinel:
        batch.append(item)
    return batch

  async def as_completed(
      self,
  ) -> AsyncIterator[Tuple[Any, Optional[Exception]]]:
    """Yields `(result, exception)` tuples as tasks complete out-of-order."""
    while not self._closed:
      item = await self._response_queue.get()
      if item is self._sentinel:
        if self._in_flight == 0 and self._response_queue.empty():
          break
        continue
      result, exc = item
      yield result, exc

  async def close(self) -> None:
    """Closes the session and cancels all background polling tasks."""
    self._closed = True
    self._response_queue.put_nowait(self._sentinel)
    for t in list(self._poll_tasks):
      if not t.done():
        t.cancel()
    if self._poll_tasks:
      await asyncio.gather(*self._poll_tasks, return_exceptions=True)
