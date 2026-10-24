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
import dataclasses
import hashlib
import pickle
import random
import time
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

from absl import logging
from tunix.dist_rl.worker import remote_execution

ActorHandle = remote_execution.ActorHandle
ExecutionResponse = remote_execution.ExecutionResponse
LONG_POLL_TIMEOUT_S = remote_execution.LONG_POLL_TIMEOUT_S
TransientTransportError = remote_execution.TransientTransportError

DEFAULT_POLL_RETRY_BUDGET_S: float = 300.0
DEFAULT_POLL_RETRY_INITIAL_BACKOFF_S: float = 0.5
DEFAULT_POLL_RETRY_MAX_BACKOFF_S: float = 10.0

_monotonic: Callable[[], float] = time.monotonic

RouteFn = Callable[[Sequence[ActorHandle], Optional[Any]], ActorHandle]
OnWorkerUnreachableFn = Callable[[ActorHandle, Exception], None]


@dataclasses.dataclass(frozen=True)
class TaskSessionFaultToleranceConfig:
  """Fault-tolerance configuration for `TaskSession`.

  Attributes:
    max_task_retries: Maximum number of times an in-flight task is re-dispatched
      onto surviving healthy workers when its assigned worker fails with an
      unrecoverable transport error. Defaults to `0` (disabled). Requires
      `on_worker_unreachable` to be set when `> 0`.
    on_worker_unreachable: Optional synchronous upward notification callback
      `(actor, exc) -> None` invoked on the event loop when `TaskSession`
      detects an unrecoverable worker-level transport failure. Must not block,
      must be idempotent (it may be invoked more than once for the same worker
      when concurrent dispatches or pollers fail), and must remove `actor` from
      the live membership source backing `actors`. Defaults to `None`.
  """

  # TODO(b/577214994): Keep `max_task_retries=0` and
  # `on_worker_unreachable=None` as the production default until control-plane
  # dynamic worker eviction support is wired up.
  max_task_retries: int = 0
  on_worker_unreachable: Optional[OnWorkerUnreachableFn] = None

  def __post_init__(self) -> None:
    if self.max_task_retries < 0:
      raise ValueError(
          f"max_task_retries must be >= 0, got {self.max_task_retries}."
      )
    if self.max_task_retries > 0 and self.on_worker_unreachable is None:
      raise ValueError(
          "max_task_retries > 0 requires on_worker_unreachable to evict "
          "failed workers from the active actors view before retrying."
      )


@dataclasses.dataclass(slots=True)
class _InFlightTask:
  """State of an in-flight task retained until completion for fault retry."""

  request_id: str
  method_name: Optional[str]
  args: Tuple[Any, ...]
  kwargs: Dict[str, Any]
  route_key: Optional[Any] = None
  retries: int = 0
  dispatching: bool = False


class WorkerUnreachableError(ConnectionError):
  """Raised when a worker remains unreachable past the poll retry budget."""


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
      poll_retry_budget_s: float = DEFAULT_POLL_RETRY_BUDGET_S,
      poll_retry_initial_backoff_s: float = DEFAULT_POLL_RETRY_INITIAL_BACKOFF_S,
      poll_retry_max_backoff_s: float = DEFAULT_POLL_RETRY_MAX_BACKOFF_S,
      fault_tolerance_config: Optional[TaskSessionFaultToleranceConfig] = None,
  ):
    """Initializes the task execution session.

    Args:
      actors: A sequence of `ActorHandle` instances or a zero-arg callable
        returning the current sequence of `ActorHandle`s.
      route_fn: Optional routing callable `(actors, route_key) -> ActorHandle`.
        Defaults to `DefaultRouter()`.
      poll_retry_budget_s: Maximum wall-clock duration in seconds to retry
        consecutive `TransientTransportError` failures when polling a worker
        before failing all of its in-flight tasks with `WorkerUnreachableError`.
        Set to `0.0` to fail immediately on the first transient poll error.
      poll_retry_initial_backoff_s: Initial exponential backoff delay in seconds
        after the first transient poll failure.
      poll_retry_max_backoff_s: Maximum backoff delay cap in seconds between
        transient poll retries.
      fault_tolerance_config: Optional `TaskSessionFaultToleranceConfig`
        controlling task retry budget (`max_task_retries`) and upward worker
        eviction notification (`on_worker_unreachable`). Defaults to
        `TaskSessionFaultToleranceConfig()` (`max_task_retries=0`,
        `on_worker_unreachable=None`).

    Raises:
      TypeError: If any item in a static `actors` sequence is not an
        `ActorHandle`.
      ValueError: If any poll retry or fault-tolerance parameter is invalid.
    """
    if poll_retry_budget_s < 0.0:
      raise ValueError(
          f"poll_retry_budget_s must be >= 0.0, got {poll_retry_budget_s}."
      )
    if poll_retry_initial_backoff_s <= 0.0:
      raise ValueError(
          "poll_retry_initial_backoff_s must be > 0.0, got"
          f" {poll_retry_initial_backoff_s}."
      )
    if poll_retry_max_backoff_s < poll_retry_initial_backoff_s:
      raise ValueError(
          "poll_retry_max_backoff_s"
          f" ({poll_retry_max_backoff_s}) must be >="
          f" poll_retry_initial_backoff_s ({poll_retry_initial_backoff_s})."
      )
    ft_cfg = (
        fault_tolerance_config
        if fault_tolerance_config is not None
        else TaskSessionFaultToleranceConfig()
    )
    if (
        ft_cfg.max_task_retries > 0 or ft_cfg.on_worker_unreachable is not None
    ) and not callable(actors):
      raise ValueError(
          "TaskSession fault tolerance requires a dynamic `actors` callable "
          "so `on_worker_unreachable` can remove evicted workers from routing."
      )
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
    self._poll_retry_budget_s = float(poll_retry_budget_s)
    self._poll_retry_initial_backoff_s = float(poll_retry_initial_backoff_s)
    self._poll_retry_max_backoff_s = float(poll_retry_max_backoff_s)
    self._max_task_retries = int(ft_cfg.max_task_retries)
    self._on_worker_unreachable = ft_cfg.on_worker_unreachable
    self._fault_tolerance_enabled = (
        self._max_task_retries > 0 or self._on_worker_unreachable is not None
    )
    self._response_queue: asyncio.Queue[Any] = asyncio.Queue()
    self._dispatched_tasks: Dict[ActorHandle, Dict[str, _InFlightTask]] = {}
    self._poll_tasks: Dict[ActorHandle, asyncio.Task[Any]] = {}
    self._retry_tasks: set[asyncio.Task[Any]] = set()
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
    # TODO(tunix-dev): Incorporate per-worker in-flight load
    # (`len(self._dispatched_tasks.get(actor, ()))`) into routing so retried
    # and newly submitted tasks avoid queueing behind heavily loaded workers.
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
    task = _InFlightTask(
        request_id=request_id,
        method_name=method_name,
        args=args,
        kwargs=dict(kwargs) if self._max_task_retries > 0 else kwargs,
        route_key=route_key,
    )
    # Increment in_flight BEFORE awaiting dispatch_task so a fast worker
    # completing before dispatch_task's coroutine resumes always finds
    # request_id in _dispatched_tasks[actor].
    self._in_flight += 1
    try:
      await self._dispatch_with_retries(task)
    except BaseException:
      # `task.dispatching` stays True only if another path (the poller or
      # `close()`) already accounted for this task's `_in_flight` slot.
      if not task.dispatching:
        self._in_flight = max(0, self._in_flight - 1)
        self._notify_if_zero_flight()
      raise
    return request_id

  def _fail_task_to_queue(self, exc: Exception) -> None:
    """Delivers a terminal `(None, exc)` outcome and releases its slot."""
    self._response_queue.put_nowait((None, exc))
    self._in_flight = max(0, self._in_flight - 1)
    self._notify_if_zero_flight()

  def _can_retry(self, task: _InFlightTask) -> bool:
    return (
        not self._closed
        and task.retries < self._max_task_retries
        and bool(self._actors_fn())
    )

  async def _dispatch_with_retries(self, task: _InFlightTask) -> None:
    """Routes and dispatches `task`, retrying on surviving workers on failure.

    Returns normally once `task` is acknowledged by a worker, or if another
    path (the poller) already settled it. Raises only while this coroutine still
    owns `task` (after removing it from the ledger), or on cancellation.

    Args:
      task: The in-flight task to route and dispatch.
    """
    while True:
      actor = self.route(task.route_key)
      dispatched_map = self._dispatched_tasks.setdefault(actor, {})
      task.dispatching = True
      dispatched_map[task.request_id] = task
      self._ensure_worker_polling(actor)
      try:
        await actor.dispatch_task(
            task.request_id, task.method_name, *task.args, **task.kwargs
        )
        # `_ensure_worker_polling` registered a poller for `actor` above; it
        # can only have disappeared if `_handle_worker_failure(actor, ...)` ran
        # while `dispatch_task` was awaiting.
        if (
            task.request_id in dispatched_map
            and not self._closed
            and actor not in self._poll_tasks
        ):
          if self._fault_tolerance_enabled:
            # `actor` was evicted; never respawn a poller on it.
            raise WorkerUnreachableError(
                f"Worker {actor!r} failed while dispatching"
                f" request_id={task.request_id!r}."
            )
          # Without fault tolerance `actor` stays a member (existing
          # behavior): poll it again so this task completes or fails there.
          self._ensure_worker_polling(actor)
      except BaseException as exc:  # pylint: disable=broad-exception-caught
        current = asyncio.current_task()
        if (
            isinstance(exc, asyncio.CancelledError)
            and self._fault_tolerance_enabled
            and not self._closed
            and current is not None
            and current.cancelling() == 0
        ):
          # Not a cancellation of this coroutine: the transport cancelled the
          # in-flight call, e.g. because the worker's channel was closed after
          # the worker was evicted by a concurrent failure. Treat it as a
          # worker-level transport fault so the task is re-dispatched instead
          # of surfacing a spurious cancellation to the caller.
          cancel_err = WorkerUnreachableError(
              f"Worker {actor!r} dispatch_task raised unexpected"
              f" CancelledError for request_id={task.request_id!r}."
          )
          cancel_err.__cause__ = exc
          exc = cancel_err
        if dispatched_map.pop(task.request_id, None) is None:
          # The poller already delivered this task's result and released its
          # `_in_flight` slot (or `close()` cleared it).
          if isinstance(exc, Exception):
            task.dispatching = False
            return
          raise
        task.dispatching = False
        if (
            isinstance(exc, Exception)
            and not isinstance(exc, (TypeError, ValueError, pickle.PickleError))
            and not self._closed
            and self._fault_tolerance_enabled
        ):
          self._handle_worker_failure(actor, exc)
          if self._can_retry(task):
            task.retries += 1
            continue
        raise exc
      else:
        task.dispatching = False
        if self._max_task_retries == 0:
          task.args = ()
          task.kwargs = {}
        return

  def _handle_worker_failure(self, actor: ActorHandle, exc: Exception) -> None:
    """Evicts a failed worker and schedules retry or terminal failure for orphans."""
    poll_task = self._poll_tasks.pop(actor, None)
    if (
        poll_task is not None
        and poll_task is not asyncio.current_task()
        and not poll_task.done()
    ):
      poll_task.cancel()
    if self._on_worker_unreachable is not None:
      try:
        self._on_worker_unreachable(actor, exc)
      except Exception:  # pylint: disable=broad-exception-caught
        logging.exception(
            "Error in on_worker_unreachable callback for worker %r", actor
        )
    dispatched_map = self._dispatched_tasks.get(actor)
    if not dispatched_map:
      return
    # Tasks still awaiting `dispatch_task` are owned by their own
    # `_dispatch_with_retries` coroutine, which retries or fails them itself.
    orphaned_tasks = [
        task for task in dispatched_map.values() if not task.dispatching
    ]
    for task in orphaned_tasks:
      dispatched_map.pop(task.request_id, None)
      if self._can_retry(task):
        task.retries += 1
        retry_task = asyncio.create_task(self._retry_or_fail_task(task, exc))
        self._retry_tasks.add(retry_task)
        retry_task.add_done_callback(self._retry_tasks.discard)
      else:
        self._fail_task_to_queue(exc)

  async def _retry_or_fail_task(
      self, task: _InFlightTask, worker_exc: Exception
  ) -> None:
    """Re-dispatches an orphaned `task` onto a surviving worker or fails it."""
    try:
      await self._dispatch_with_retries(task)
    except Exception as retry_exc:  # pylint: disable=broad-exception-caught
      if retry_exc.__cause__ is None and retry_exc is not worker_exc:
        # Keep the original worker failure visible, e.g. when no surviving
        # worker remains to route the retry to.
        retry_exc.__cause__ = worker_exc
      self._fail_task_to_queue(retry_exc)

  def _ensure_worker_polling(self, actor: ActorHandle) -> None:
    if actor in self._poll_tasks:
      return
    if not self._dispatched_tasks.get(actor):
      return
    self._poll_tasks[actor] = asyncio.create_task(self._poll_worker_loop(actor))

  async def _poll_worker_loop(self, actor: ActorHandle) -> None:
    """Long-polls a single active worker until its in-flight set drains."""
    # TODO: b/577214994 - Add ACK-based response redelivery so a response whose
    # final frame was handed to the server transport just before a connection
    # drop is retained until acknowledged by the client. Currently no outer
    # orchestrator timeout covers this window, so a lost response stalls
    # polling_stage until ACK redelivery lands.
    # TODO: b/577214994 - Detect in-place worker restarts. If a worker process
    # restarts at the same address within `poll_retry_budget_s` (e.g., a k8s
    # `restartPolicy: OnFailure` container restart), the reconnected poll
    # returns `None` from the new process's empty response queue and resets
    # the outage budget below, so tasks dispatched to the old process are
    # never completed or failed. Planned fix: have the server attach a
    # per-process incarnation token to `PollResponses` and `DispatchTask`
    # responses, and raise a non-transient `WorkerRestartedError` from
    # `GrpcRemoteActorHandle` when it changes.
    outage_start: Optional[float] = None
    retry_attempts = 0
    try:
      while not self._closed:
        dispatched_map = self._dispatched_tasks.setdefault(actor, {})
        if not dispatched_map:
          break
        backoff_s: Optional[float] = None
        try:
          response = await actor.poll_responses(timeout_s=LONG_POLL_TIMEOUT_S)
          outage_start = None
          retry_attempts = 0
          if isinstance(response, ExecutionResponse):
            if (
                not response.request_id
                or response.request_id not in dispatched_map
            ):
              raise RuntimeError(
                  "Received ExecutionResponse with unknown or missing"
                  f" request_id={response.request_id!r}; in-flight"
                  f" request_ids={sorted(dispatched_map)!r}."
              )
            dispatched_map.pop(response.request_id)
            self._in_flight = max(0, self._in_flight - 1)
            try:
              res = response.unwrap()
              self._response_queue.put_nowait((res, None))
            except Exception as exc:  # pylint: disable=broad-exception-caught
              self._response_queue.put_nowait((None, exc))
            self._notify_if_zero_flight()
        except asyncio.CancelledError as exc:
          if self._closed or not dispatched_map:
            break
          curr_task = asyncio.current_task()
          if curr_task is not None and curr_task.cancelling() > 0:
            break
          cancel_err = RuntimeError(
              f"Worker {actor!r} poll_responses raised unexpected"
              " CancelledError while session is active."
          )
          cancel_err.__cause__ = exc
          self._handle_worker_failure(actor, cancel_err)
          break
        except TransientTransportError as exc:
          if self._closed or not dispatched_map:
            break
          if actor not in self._actors_fn():
            evicted_err = WorkerUnreachableError(
                f"Worker {actor!r} was removed from the session's active"
                f" actors during a transient poll error: {exc}"
            )
            evicted_err.__cause__ = exc
            self._handle_worker_failure(actor, evicted_err)
            break
          now = _monotonic()
          if outage_start is None:
            outage_start = now
          elapsed_s = now - outage_start
          retry_attempts += 1
          if (
              self._poll_retry_budget_s <= 0.0
              or elapsed_s >= self._poll_retry_budget_s
          ):
            unreachable_err = WorkerUnreachableError(
                f"Worker {actor!r} remained unreachable for {elapsed_s:.1f}s"
                f" (budget {self._poll_retry_budget_s:.1f}s) across"
                f" {retry_attempts} poll retry attempt(s): {exc}"
            )
            unreachable_err.__cause__ = exc
            self._handle_worker_failure(actor, unreachable_err)
            break
          remaining_budget_s = self._poll_retry_budget_s - elapsed_s
          raw_backoff_s = min(
              self._poll_retry_max_backoff_s,
              self._poll_retry_initial_backoff_s
              * (2 ** min(retry_attempts - 1, 60)),
          )
          jittered_backoff_s = raw_backoff_s * random.uniform(0.8, 1.0)
          backoff_s = min(jittered_backoff_s, remaining_budget_s)
          if retry_attempts == 1:
            logging.warning(
                "Transient transport error polling worker %r (attempt %d,"
                " retrying in %.2fs, budget=%.1fs): %s",
                actor,
                retry_attempts,
                backoff_s,
                self._poll_retry_budget_s,
                exc,
            )
          else:
            logging.debug(
                "Transient transport error polling worker %r (attempt %d,"
                " elapsed=%.1fs/%.1fs, retrying in %.2fs): %s",
                actor,
                retry_attempts,
                elapsed_s,
                self._poll_retry_budget_s,
                backoff_s,
                exc,
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
          if self._closed or not dispatched_map:
            break
          # Non-transient transport or polling failure on this worker; evict
          # and retry or fail all dispatched tasks on this worker immediately.
          self._handle_worker_failure(actor, exc)
          break
        if backoff_s is not None:
          await asyncio.sleep(backoff_s)
    finally:
      # No respawn here: a normal exit only happens once the in-flight map is
      # empty (with no `await` in between), and every other exit is either a
      # worker failure handled by `_handle_worker_failure` or session close.
      if self._poll_tasks.get(actor) is asyncio.current_task():
        self._poll_tasks.pop(actor, None)

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
    """Closes the session and cancels all background polling and retry tasks."""
    self._closed = True
    self._response_queue.put_nowait(self._sentinel)
    tasks = list(self._poll_tasks.values()) + list(self._retry_tasks)
    for t in tasks:
      if not t.done():
        t.cancel()
    if tasks:
      await asyncio.gather(*tasks, return_exceptions=True)
