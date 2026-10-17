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

"""Multi-node actor routing pool and out-of-order execution session (`RoutingActorPool`, `PoolExecutionSession`)."""

import asyncio
import contextlib
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


class RoutingActorPool:
  """Actor pool with smart task routing (`route_key affinity, round-robin fallback`).

  Args:
    actors: Initial sequence of worker actor handles or string URI targets.
    actors_fn: Optional zero-arg callable returning the live sequence of
      `ActorHandle`s (mutually exclusive with `actors`).
    router: Optional custom routing callable `(actors, method_name, args,
      kwargs) -> ActorHandle` or router module/object providing per-method
      handlers matching `method_name` with signature `(actors, args, kwargs) ->
      ActorHandle`.
  """

  def __init__(
      self,
      actors: Optional[Sequence[Union[str, ActorHandle]]] = None,
      *,
      actors_fn: Optional[Callable[[], Sequence[ActorHandle]]] = None,
      router: Optional[Union[Callable[..., ActorHandle], Any]] = None,
  ):
    if actors is not None and actors_fn is not None:
      raise ValueError("Cannot specify both `actors` and `actors_fn`.")
    self._actors_fn = actors_fn
    self._actors: List[ActorHandle] = []
    for a in actors or []:
      self.add_actor(a)
    self._idx = 0
    self.router = router

  @property
  def actors(self) -> Sequence[ActorHandle]:
    if self._actors_fn is not None:
      return self._actors_fn()
    return self._actors

  def add_actor(self, actor: Union[str, ActorHandle]) -> None:
    if self._actors_fn is not None:
      raise RuntimeError(
          "Cannot call add_actor() on a RoutingActorPool backed by actors_fn."
      )
    if isinstance(actor, str):
      self._actors.append(ActorHandle.from_address(actor))
    elif isinstance(actor, ActorHandle):
      self._actors.append(actor)
    else:
      raise TypeError(f"Expected str or ActorHandle, got {type(actor)}")

  def _get_next_actor(
      self,
      method_name: Optional[str] = None,
      args: Sequence[Any] = (),
      kwargs: Optional[Dict[str, Any]] = None,
  ) -> ActorHandle:
    """Selects target actor via custom router, route_key affinity, or round-robin.

    Args:
      method_name: Target remote method being invoked.
      args: Positional arguments passed to the method call.
      kwargs: Keyword arguments passed to the method call. If this dictionary
        contains `route_key`, process-stable hash routing
        (`stable_route_hash(route_key) % N`) is used for sticky endpoint
        affinity (popped prior to remote dispatch).

    Returns:
      The selected `ActorHandle` target worker.

    Raises:
      RuntimeError: If the pool contains no registered ActorHandles.
    """
    actors = self.actors
    if not actors:
      raise RuntimeError(
          "RoutingActorPool contains no registered ActorHandles."
      )

    kwargs = kwargs or {}
    if self.router is not None:
      if (
          method_name
          and hasattr(self.router, method_name)
          and callable(getattr(self.router, method_name))
      ):
        return getattr(self.router, method_name)(actors, args, kwargs)
      elif callable(self.router):
        return self.router(actors, method_name, args, kwargs)
      else:
        raise TypeError(
            f"Router object {type(self.router)} must provide a method matching "
            f"'{method_name}' or be callable."
        )

    # Check for sticky routing key (e.g. route_key for KV-cache locality)
    route_key = kwargs.get("route_key")

    if route_key is not None:
      # Not builtin `hash()`: it salts `str` per process, so placement would
      # not survive a restart.
      # TODO(tunix-dev): `% len(actors)` remaps every key when pool
      # membership changes, not just the keys on the affected actor. Switch to
      # rendezvous (HRW) hashing before adding actor eviction.
      return actors[stable_route_hash(route_key) % len(actors)]

    # Default fallback: round-robin load balancing across all endpoints
    actor = actors[self._idx % len(actors)]
    self._idx += 1
    return actor

  def submit(self, method_name: Optional[str] = None, *args, **kwargs) -> Any:
    actor = self._get_next_actor(method_name, args, kwargs)
    kwargs.pop("route_key", None)
    return actor.submit(method_name, *args, **kwargs)

  async def asubmit(
      self, method_name: Optional[str] = None, *args, **kwargs
  ) -> Any:
    actor = self._get_next_actor(method_name, args, kwargs)
    kwargs.pop("route_key", None)
    return await actor.asubmit(method_name, *args, **kwargs)

  async def as_completed_stream(
      self,
      tasks: Sequence[
          Tuple[str, str, Sequence[Any], Dict[str, Any]]
      ],
  ) -> AsyncIterator[Any]:
    """Dispatches a static batch of tasks across pool workers and yields results as they complete.

    Intended Use Case:
      Simple, one-shot static batch processing where the full list of tasks is
      known upfront (similar to `asyncio.as_completed`). This convenience
      wrapper provides fail-fast behavior: if any task raises an exception, the
      exception is immediately re-raised in the caller's stream and remaining
      tasks are cancelled.

      For dynamic task enqueuing (e.g. submitting new tasks as existing ones
      finish) or fault-isolated streaming (where individual task errors do not
      terminate the stream), use `execution_session()` instead.

    Args:
      tasks: Sequence of task specifications formatted as 4-tuples:
        `(request_id, method_name, args, kwargs)`, where:
          - request_id: Unique request identifier string.
          - method_name: Target remote method name to execute on the worker.
          - args: Positional arguments sequence passed to the remote method.
          - kwargs: Keyword arguments dictionary passed to the remote method.
    """
    if not self.actors:
      raise RuntimeError(
          "RoutingActorPool contains no registered ActorHandles."
      )
    if not tasks:
      return
    async with self.execution_session(tasks) as session:
      async for result, exc in session.as_completed():
        if exc is not None:
          raise exc
        yield result

  @contextlib.asynccontextmanager
  async def execution_session(
      self,
      initial_tasks: Optional[
          Sequence[Tuple[str, str, Sequence[Any], Dict[str, Any]]]
      ] = None,
  ) -> AsyncIterator["PoolExecutionSession"]:
    """Creates a dynamic, fault-isolated execution session over the worker pool.

    Intended Use Case:
      Long-running worker pipelines, dynamic task enqueuing, and fault-tolerant
      batch processing. Within the `async with` session block, callers can:
        1. Dynamically enqueue new tasks at any time via `await
        session.submit()`.
        2. Consume completions out-of-order via `session.as_completed()`, which
           yields `(result, exception)` tuples without terminating the stream
           when an individual task fails.
        3. Rely on automatic background worker polling and clean task
        cancellation
           upon session exit.

    Args:
      initial_tasks: Optional sequence of initial task specifications to
        dispatch upon entering the session, formatted as 4-tuples
        `(request_id, method_name, args, kwargs)`, where:
          - request_id: Unique request identifier string.
          - method_name: Target remote method name to execute on the worker.
          - args: Positional arguments sequence passed to the remote method.
          - kwargs: Keyword arguments dictionary passed to the remote method.
    """
    session = PoolExecutionSession(self)
    try:
      if initial_tasks:
        for request_id, method_name, args, kwargs in initial_tasks:
          await session.submit(request_id, method_name, *args, **kwargs)
      yield session
    finally:
      await session.close()


class PoolExecutionSession:
  """Dynamic, fault-isolated execution session for a RoutingActorPool.

  Intended Use Case:
    Managed by `RoutingActorPool.execution_session()`. Provides an interactive
    handle to submit tasks (`submit`) and consume out-of-order completions and
    exceptions (`as_completed`) without terminating the stream on individual
    task
    failures.
  """

  def __init__(
      self,
      pool: Optional[RoutingActorPool] = None,
      *,
      actors_fn: Optional[Callable[[], Sequence[ActorHandle]]] = None,
  ):
    if (pool is None) == (actors_fn is None):
      raise ValueError("Must specify exactly one of `pool` or `actors_fn`.")
    self._pool = (
        pool if pool is not None else RoutingActorPool(actors_fn=actors_fn)
    )
    self._response_queue: asyncio.Queue[Any] = asyncio.Queue()
    self._active_workers: set[ActorHandle] = set()
    self._dispatched_tasks: Dict[ActorHandle, set[str]] = {}
    self._poll_tasks: set[asyncio.Task[Any]] = set()
    self._in_flight = 0
    self._closed = False
    self._sentinel = object()

  def _notify_if_zero_flight(self) -> None:
    if self._in_flight == 0:
      self._response_queue.put_nowait(self._sentinel)

  async def asubmit(
      self, method_name: Optional[str] = None, *args, **kwargs
  ) -> Any:
    """Selects a worker from the underlying pool and awaits a one-shot RPC."""
    return await self._pool.asubmit(method_name, *args, **kwargs)

  async def submit(
      self,
      request_id: str,
      method_name: Optional[str] = None,
      *args,
      **kwargs,
  ) -> str:
    """Dispatches a task to a worker in the pool and tracks its completion."""
    if self._closed:
      raise RuntimeError("PoolExecutionSession is closed.")
    actor = self._pool._get_next_actor(method_name, args, kwargs)
    kwargs.pop("route_key", None)  # remove route_key from worker method args

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
          # Transport or polling failure on this worker; fail all dispatched tasks on this worker.
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
    # When idle (_in_flight == 0), yield briefly (<= 50ms) so concurrent dispatch
    # coroutines can run without stalling for the full 50s long-poll timeout.
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
          # Sentinel marks _in_flight reaching 0 or session close; return early if drained,
          # otherwise skip stale sentinels when new tasks are in flight or queued behind it.
          if self._in_flight == 0 and self._response_queue.empty():
            return []
          continue
        batch.append(item)
    except asyncio.TimeoutError:
      return []

    # Greedily drain any additional completions already queued by background worker loops.
    while not self._response_queue.empty():
      item = self._response_queue.get_nowait()
      if item is not self._sentinel:
        batch.append(item)
    return batch

  async def as_completed(
      self,
  ) -> AsyncIterator[Tuple[Any, Optional[Exception]]]:
    """Yields (result, exception) tuples as tasks complete out-of-order."""
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
