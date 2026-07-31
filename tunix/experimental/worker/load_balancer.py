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

"""Capacity-aware load balancing across a pool of workers.

`RoutingActorPool` routes round-robin, which is fine when every call is a
blocking request/response but not when work is dispatched fire-and-forget: a
round-robin pool will keep piling requests onto a worker that is already
saturated, because nothing tracks what is still running there.

`CapacityRouter` gives each worker a fixed number of concurrent slots. It
picks the worker with the fewest outstanding requests, refuses to exceed
`max_concurrency` on any one of them, and hands a slot back when a request
finishes -- so a worker that completes a trajectory is immediately the most
attractive target for the next one.

Ownership contract: routing a request takes a slot, and the caller that
dispatched it must return the slot exactly once via `release()`, whether the
request succeeded, failed, or was never dispatched at all.
"""

from __future__ import annotations

import asyncio
import dataclasses
from typing import Any, AsyncIterator, Dict, Iterable, Optional, Sequence, Tuple

from tunix.experimental.worker import remote_execution


class NoCapacityError(RuntimeError):
  """Raised when every worker in the pool is already at `max_concurrency`."""


class CapacityRouter:
  """Least-outstanding router with a hard per-worker concurrency cap.

  Usable directly as the `router` of a
  `remote_execution.RoutingActorPool`, in which case routing a call also
  reserves the slot:

    router = CapacityRouter(max_concurrency=4)
    pool = remote_execution.RoutingActorPool(handles, router=router)

  Attributes:
    max_concurrency: Slots per worker.
  """

  def __init__(self, max_concurrency: int = 1):
    if max_concurrency < 1:
      raise ValueError(
          f"max_concurrency must be >= 1, got {max_concurrency}."
      )
    self.max_concurrency = max_concurrency
    self._outstanding: Dict[remote_execution.ActorHandle, int] = {}
    self._slot_freed = asyncio.Event()

  def outstanding(self, actor: remote_execution.ActorHandle) -> int:
    """Number of requests currently in flight on `actor`."""
    return self._outstanding.get(actor, 0)

  def snapshot(self) -> Dict[remote_execution.ActorHandle, int]:
    """Per-worker outstanding counts, for logging and metrics."""
    return dict(self._outstanding)

  def total_outstanding(self) -> int:
    return sum(self._outstanding.values())

  def has_capacity(
      self, actors: Sequence[remote_execution.ActorHandle]
  ) -> bool:
    """True if any worker in `actors` has a free slot."""
    return any(self.outstanding(a) < self.max_concurrency for a in actors)

  def select(
      self, actors: Sequence[remote_execution.ActorHandle]
  ) -> Optional[remote_execution.ActorHandle]:
    """Returns the least-loaded worker with a free slot, or None if saturated.

    Ties are broken by pool order, so an idle pool fills evenly: with N workers
    and `max_concurrency` slots each, the first N * max_concurrency requests
    are spread `max_concurrency` per worker before anything has to wait.
    """
    best: Optional[remote_execution.ActorHandle] = None
    best_load = self.max_concurrency
    for actor in actors:
      load = self.outstanding(actor)
      if load < best_load:
        best, best_load = actor, load
        if load == 0:
          break
    return best

  def acquire(self, actor: remote_execution.ActorHandle) -> None:
    """Reserves a slot on `actor`, over-subscribing it if it is already full."""
    self._outstanding[actor] = self.outstanding(actor) + 1

  def release(self, actor: remote_execution.ActorHandle) -> None:
    """Returns a slot to `actor` and wakes anyone waiting for capacity."""
    remaining = self.outstanding(actor) - 1
    if remaining <= 0:
      self._outstanding.pop(actor, None)
    else:
      self._outstanding[actor] = remaining
    self._slot_freed.set()

  async def wait_for_capacity(
      self, actors: Sequence[remote_execution.ActorHandle]
  ) -> None:
    """Blocks until some worker in `actors` has a free slot.

    Only safe to await from a single dispatch loop: two waiters can both wake
    on the same freed slot and only one of them will win the following
    `select()`.
    """
    if not actors:
      raise RuntimeError("Cannot wait for capacity on an empty worker pool.")
    while not self.has_capacity(actors):
      self._slot_freed.clear()
      if self.has_capacity(actors):
        return
      await self._slot_freed.wait()

  def __call__(
      self,
      actors: Sequence[remote_execution.ActorHandle],
      method_name: Optional[str] = None,
      args: Sequence[Any] = (),
      kwargs: Optional[Dict[str, Any]] = None,
  ) -> remote_execution.ActorHandle:
    """Router entry point for `RoutingActorPool`; reserves the chosen slot.

    Raises:
      NoCapacityError: If every worker is already at `max_concurrency`. Callers
        that would rather wait than fail should await `wait_for_capacity()`
        before submitting.
    """
    del method_name, args, kwargs  # Routing is load-based only.
    actor = self.select(actors)
    if actor is None:
      raise NoCapacityError(
          f"All {len(actors)} workers are at max_concurrency="
          f"{self.max_concurrency} ({self.total_outstanding()} requests in"
          " flight)."
      )
    self.acquire(actor)
    return actor


@dataclasses.dataclass(frozen=True)
class Task:
  """One unit of work to run on some worker in the pool.

  Attributes:
    request_id: Caller-chosen id, echoed back with the result so a batch can be
      reassembled in its original order.
    method_name: Method to invoke on the worker's bound instance.
    args: Positional arguments for the method.
    kwargs: Keyword arguments for the method.
  """

  request_id: str
  method_name: str
  args: Tuple[Any, ...] = ()
  kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)


class BalancedDispatcher:
  """Runs a stream of tasks across a worker pool, `max_concurrency` at a time.

  Work is submitted fire-and-forget (`dispatch_task`) and collected by the
  session's background long-poll loops, so a slow worker never blocks the
  others. The dispatcher keeps every worker filled to `max_concurrency` and
  releases a slot the moment a task settles, which is immediately reused by
  the next queued task -- so a worker that finishes a trajectory gets the next
  one while its peers keep running.

  Results are yielded as they arrive, which is out of submission order.
  """

  def __init__(
      self,
      actors: Sequence[remote_execution.ActorHandle],
      *,
      max_concurrency: int = 1,
      router: Optional[CapacityRouter] = None,
  ):
    if not actors:
      raise ValueError("BalancedDispatcher requires at least one worker.")
    self.router = router or CapacityRouter(max_concurrency=max_concurrency)
    self._pool = remote_execution.RoutingActorPool(
        list(actors), router=self.router
    )
    self._actors = list(actors)

  @property
  def actors(self) -> Sequence[remote_execution.ActorHandle]:
    return tuple(self._actors)

  @property
  def max_in_flight(self) -> int:
    """Ceiling on concurrently running tasks across the whole pool."""
    return len(self._actors) * self.router.max_concurrency

  async def run(
      self, tasks: Iterable[Task]
  ) -> AsyncIterator[Tuple[str, Any, Optional[Exception]]]:
    """Dispatches `tasks` across the pool, yielding results as they settle.

    A task that fails is reported as its own `(request_id, None, exception)`
    tuple rather than aborting the batch, so one bad prompt does not lose the
    rest of the work already running on other workers.

    Args:
      tasks: Work to run. Consumed lazily, so this may be a generator that
        produces more work while the batch is in flight.

    Yields:
      `(request_id, result, exception)` in completion order.
    """
    pending = iter(tasks)
    async with self._pool.execution_session(
        on_task_settled=self._on_settled
    ) as session:
      # Prime every worker up to its concurrency limit before consuming any
      # results, so all workers start together rather than serially.
      accepted_any = False
      while self.router.has_capacity(self._actors):
        submitted = await self._submit_next(session, pending)
        if submitted is None:
          break
        request_id, dispatch_error = submitted
        if dispatch_error is None:
          accepted_any = True
        else:
          yield request_id, None, dispatch_error

      if not accepted_any:
        # Nothing is in flight, so no completion will ever arrive and the
        # session publishes no end-of-stream marker to wake the consumer.
        # Awaiting here would hang on an empty or entirely-failed batch.
        return

      async for request_id, result, exc in session.as_completed_with_ids():
        yield request_id, result, exc
        # The finished worker's slot was released as the task settled, so the
        # next task lands on it. Keep going past tasks that fail to dispatch,
        # otherwise one bad request would leave the slot idle.
        while True:
          submitted = await self._submit_next(session, pending)
          if submitted is None:
            break
          next_id, dispatch_error = submitted
          if dispatch_error is None:
            break
          yield next_id, None, dispatch_error

  async def _submit_next(
      self,
      session: remote_execution.PoolExecutionSession,
      pending: Any,
  ) -> Optional[Tuple[str, Optional[Exception]]]:
    """Submits the next pending task.

    Returns:
      `(request_id, dispatch_error)` for the task that was attempted, or None
      when `pending` is exhausted. A dispatch error means the worker never
      accepted the task, so it will never appear in the completion stream.
    """
    task = next(pending, None)
    if task is None:
      return None
    await self.router.wait_for_capacity(self._actors)
    try:
      await session.submit(
          task.request_id, task.method_name, *task.args, **task.kwargs
      )
      return task.request_id, None
    except Exception as exc:  # pylint: disable=broad-exception-caught
      return task.request_id, exc

  def _on_settled(
      self, actor: remote_execution.ActorHandle, request_id: str
  ) -> None:
    del request_id
    self.router.release(actor)
