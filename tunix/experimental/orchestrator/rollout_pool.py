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

"""A rollout worker backed by a pool of remote rollout workers.

`RemoteRolloutWorker` sends a whole batch to one worker with a blocking
`submit`, so the caller is stuck behind the slowest prompt in the batch and
extra workers sit idle. `PooledRolloutWorker` implements the same
`RolloutWorker` contract but spreads the batch over a pool:

  * each request is dispatched fire-and-forget with `dispatch_task`, so no
    generation blocks the dispatch of the next one;
  * completions come back through the session's long-poll loops, one loop per
    worker, and are surfaced as soon as they land;
  * `CapacityRouter` keeps each worker filled to `max_concurrency`
    trajectories and hands its freed slot to the next queued request as soon
    as one finishes.

Requests complete out of order, so responses are correlated by request id and
reassembled into the caller's order before being returned. Callers that would
rather consume trajectories as they finish can pass `on_complete`.
"""

from __future__ import annotations

import asyncio
import collections
import traceback as traceback_lib
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from absl import logging

from tunix.experimental.common import datatypes
from tunix.experimental.worker import load_balancer
from tunix.experimental.worker import remote_execution
from tunix.experimental.worker import rollout_worker


# Mirrors agent_types.TrajectoryStatus.FAILED without importing the agentic
# stack into the transport layer.
_FAILED_STATUS = "FAILED"

RequestOrBatch = Union[
    datatypes.RolloutRequest, Sequence[datatypes.RolloutRequest]
]


class DuplicateRequestIdError(ValueError):
  """Two rollout requests in one batch carried the same id."""


def _as_actor_handle(worker: Any) -> remote_execution.ActorHandle:
  """Returns an actor handle addressing `worker`, whatever transport it uses."""
  if isinstance(worker, remote_execution.ActorHandle):
    return worker
  actor = getattr(worker, "actor", None)
  if isinstance(actor, remote_execution.ActorHandle):
    return actor
  if not callable(getattr(worker, "generate", None)):
    raise TypeError(
        f"{type(worker).__name__} is not usable as a rollout worker: it is"
        " neither an ActorHandle nor an object exposing generate()."
    )
  return remote_execution.InProcessActorHandle(
      remote_execution.InProcessRemoteExecutionServer(instance=worker)
  )


class PooledRolloutWorker(rollout_worker.RolloutWorker):
  """Fans rollout requests across a pool of rollout workers.

  A worker may have only one consumer polling it at a time. Its completed
  responses sit in a single queue that any poller drains, and the wire has no
  way to ask for only one's own: a second consumer therefore takes responses
  belonging to the first, which are then discarded as unrecognized. The first
  waits forever for work that was already answered, and the capacity it
  reserved is never returned.

  Concurrent `generate` calls are serialized here to keep that from happening.
  It costs pipelining -- a second batch cannot overlap the first -- and it only
  holds within one pool object; two pools sharing a worker still collide.
  Per-consumer response queues on the worker are the real fix.

  Attributes:
    worker_id: Identifier for this pool, as seen by the control plane.
  """

  def __init__(
      self,
      actors: Sequence[remote_execution.ActorHandle],
      *,
      max_concurrency: int = 1,
      worker_id: str = "rollout_pool",
      method_name: str = "generate",
      batch_timeout_s: Optional[float] = None,
  ):
    """Initializes the pool.

    Args:
      actors: Handles to the rollout workers to spread work across.
      max_concurrency: Trajectories each worker may generate at once. The pool
        runs at most `len(actors) * max_concurrency` generations concurrently.
      worker_id: Identifier for this pool.
      method_name: Method to invoke on each remote rollout worker.
      batch_timeout_s: How long to wait for a batch before giving up on
        whatever has not answered. Nothing else bounds this: the dispatcher
        waits indefinitely, so a single lost trajectory would stall the step
        that needs it. Stragglers come back as failed responses. None waits
        forever, which is only safe when the workers themselves time out.
    """
    super().__init__(worker_id=worker_id)
    self._dispatcher = load_balancer.BalancedDispatcher(
        actors, max_concurrency=max_concurrency
    )
    self._method_name = method_name
    self._batch_timeout_s = batch_timeout_s
    # Enforces one consumer per worker; see the class docstring.
    self._one_consumer = asyncio.Lock()

  @classmethod
  def from_workers(
      cls, workers: Sequence[Any], **kwargs
  ) -> "PooledRolloutWorker":
    """Builds a pool from rollout workers of any transport.

    Accepts actor handles, the RPC handles the fleet holds (anything exposing
    an `actor`), and plain in-process worker objects, which are bound to an
    in-process server so the pool can drive them the same way.

    Args:
      workers: The rollout workers to pool. Must not be empty.
      **kwargs: Forwarded to the constructor.
    """
    return cls([_as_actor_handle(worker) for worker in workers], **kwargs)

  @property
  def dispatcher(self) -> load_balancer.BalancedDispatcher:
    return self._dispatcher

  @property
  def max_in_flight(self) -> int:
    """Trajectories the pool can have running at once."""
    return self._dispatcher.max_in_flight

  async def generate(
      self,
      requests: RequestOrBatch,
      on_complete: Optional[
          Callable[[datatypes.RolloutResponse], None]
      ] = None,
  ) -> Union[datatypes.RolloutResponse, Sequence[datatypes.RolloutResponse]]:
    """Generates rollouts for `requests`, spread across the pool.

    Args:
      requests: A single request or a batch of them.
      on_complete: Optional callback invoked with each response the moment it
        arrives, i.e. out of order, before the full batch is done.

    Returns:
      A single response for a single request, otherwise responses in the same
      order as `requests`. A request that fails comes back as a response with
      `error` set and a non-success `status`, never as a missing entry.

      Calls are serialized: a second call waits for the first to finish, so
      that no worker is ever polled by two consumers at once.
    """
    is_single = isinstance(requests, datatypes.RolloutRequest)
    batch: List[datatypes.RolloutRequest] = (
        [requests] if is_single else list(requests)
    )
    if not batch:
      return []

    async with self._one_consumer:
      return await self._generate_batch(batch, is_single, on_complete)

  async def _generate_batch(
      self,
      batch: List[datatypes.RolloutRequest],
      is_single: bool,
      on_complete: Optional[Callable[[datatypes.RolloutResponse], None]],
  ) -> Union[datatypes.RolloutResponse, Sequence[datatypes.RolloutResponse]]:
    """Runs one batch through the dispatcher, as the sole consumer.

    Raises:
      DuplicateRequestIdError: If two requests share a request id.
    """
    order = [self._task_id(req, i) for i, req in enumerate(batch)]
    repeated = sorted(
        {request_id for request_id, count in collections.Counter(order).items()
         if count > 1}
    )
    if repeated:
      # Responses are matched by id. Two requests under one id means one gets
      # dispatched twice and the other never, and a single response resolves
      # both -- so the batch would come back plausible and wrong.
      raise DuplicateRequestIdError(
          f"Rollout requests must have unique ids within a batch; {repeated}"
          " appear more than once."
      )
    by_id: Dict[str, datatypes.RolloutRequest] = dict(zip(order, batch))
    tasks = (
        load_balancer.Task(
            request_id=request_id,
            method_name=self._method_name,
            args=(by_id[request_id],),
        )
        for request_id in order
    )

    responses: Dict[str, datatypes.RolloutResponse] = {}

    async def _collect() -> None:
      async for request_id, result, exc in self._dispatcher.run(tasks):
        response = self._as_response(request_id, by_id[request_id], result, exc)
        responses[request_id] = response
        if on_complete is not None:
          on_complete(response)

    try:
      await asyncio.wait_for(_collect(), timeout=self._batch_timeout_s)
    except asyncio.TimeoutError:
      logging.warning(
          "Rollout batch timed out after %ss with %d of %d answered; the rest"
          " are reported as failures.",
          self._batch_timeout_s,
          len(responses),
          len(order),
      )

    ordered = [
        responses.get(request_id) or self._missing_response(request_id)
        for request_id in order
    ]
    return ordered[0] if is_single else ordered

  def _task_id(self, request: datatypes.RolloutRequest, index: int) -> str:
    """Returns a per-batch unique id used to correlate the response back."""
    if request.request_id:
      return request.request_id
    # prompt_id repeats across a GRPO group, so it alone cannot correlate.
    return f"{request.prompt_id}:{index}"

  def _as_response(
      self,
      request_id: str,
      request: datatypes.RolloutRequest,
      result: Any,
      exc: Optional[Exception],
  ) -> datatypes.RolloutResponse:
    """Normalizes whatever a worker returned into one stamped RolloutResponse."""
    if exc is not None:
      return self._error_response(request_id, exc)

    # A worker whose generate() accepts a batch may answer with a one-element
    # sequence even though we sent it a single request.
    if isinstance(result, Sequence) and not isinstance(
        result, (str, bytes, datatypes.RolloutResponse)
    ):
      result = result[0] if len(result) == 1 else None

    if not isinstance(result, datatypes.RolloutResponse):
      return self._error_response(
          request_id,
          TypeError(
              f"Rollout worker returned {type(result).__name__} for prompt"
              f" {request.prompt_id!r}, expected a RolloutResponse."
          ),
      )

    if not result.request_id:
      result.request_id = request_id
    return result

  def _error_response(
      self, request_id: str, exc: BaseException
  ) -> datatypes.RolloutResponse:
    return datatypes.RolloutResponse(
        request_id=request_id,
        status=_FAILED_STATUS,
        error=datatypes.ErrorInfo(
            error_type=type(exc).__name__,
            message=str(exc),
            traceback="".join(
                traceback_lib.format_exception(
                    type(exc), exc, exc.__traceback__
                )
            ),
        ),
    )

  def _missing_response(self, request_id: str) -> datatypes.RolloutResponse:
    """Fallback so the returned batch always lines up with the requests."""
    return self._error_response(
        request_id,
        RuntimeError(
            f"Rollout request {request_id!r} never produced a response."
        ),
    )
