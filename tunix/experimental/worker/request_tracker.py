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

"""Caller-side tracking that matches responses back to outgoing requests.

Under the fire-and-forget pattern a caller does not block on the RPC: it
dispatches, gets an immediate ack, and later polls for the result. That only
works if every outgoing request is remembered and every arriving response can be
attributed to one, because responses come back in completion order, not dispatch
order.

`RequestTracker` is that book: it records each dispatched request by
`request_id` and resolves responses against it, with the properties the pattern
needs:

  * out-of-order delivery -- responses are keyed, never positional;
  * first-wins de-duplication -- a redelivered response is ignored, so a retry
    or an at-least-once transport cannot double-count;
  * orphan tolerance -- a response for an unknown id is reported to the caller
    rather than raising or being silently dropped;
  * liveness -- requests that never came back can be listed by age, which is
    what a timeout/retry policy is built on.

`TrackedActorHandle` wraps an `ActorHandle` so the dispatch/ack/poll cycle keeps
the book automatically.
"""

import dataclasses
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from absl import logging
from tunix.experimental.worker import remote_execution


@dataclasses.dataclass
class PendingRequest:
  """One outgoing request and, once it arrives, its response.

  Attributes:
    request_id: Correlation id shared by the request and its response.
    method_name: Remote method invoked, for diagnostics.
    dispatched_at: Monotonic timestamp taken when the request was registered.
    response: The matched response, or None while still in flight.
    completed_at: Monotonic timestamp when the response was matched.
  """

  request_id: str
  method_name: str = ""
  dispatched_at: float = 0.0
  response: Optional[remote_execution.ExecutionResponse] = None
  completed_at: Optional[float] = None

  @property
  def is_complete(self) -> bool:
    return self.response is not None

  @property
  def latency_s(self) -> Optional[float]:
    if self.completed_at is None:
      return None
    return self.completed_at - self.dispatched_at


class RequestTracker:
  """Tracks outgoing requests and matches responses to them by request id."""

  def __init__(self, *, time_fn: Callable[[], float] = time.monotonic):
    self._time_fn = time_fn
    self._records: Dict[str, PendingRequest] = {}
    self._orphans: int = 0
    self._duplicates: int = 0
    self._lock = threading.Lock()

  def register(
      self, request_id: str, method_name: str = ""
  ) -> PendingRequest:
    """Records a dispatched request.

    Re-registering an id already in flight returns the existing record rather
    than resetting it, so a retry of the same id cannot lose the original
    dispatch time.
    """
    with self._lock:
      existing = self._records.get(request_id)
      if existing is not None:
        return existing
      record = PendingRequest(
          request_id=request_id,
          method_name=method_name,
          dispatched_at=self._time_fn(),
      )
      self._records[request_id] = record
      return record

  def resolve(
      self, response: remote_execution.ExecutionResponse
  ) -> Optional[PendingRequest]:
    """Attaches a response to its request.

    Args:
      response: A response carrying the `request_id` it is answering.

    Returns:
      The completed record, or None if the response is an orphan (unknown id)
      or a duplicate of one already matched.
    """
    request_id = getattr(response, "request_id", "") or ""
    with self._lock:
      record = self._records.get(request_id)
      if record is None:
        self._orphans += 1
        logging.warning(
            "Dropping response for unknown request_id %r; it was never"
            " registered as outgoing (or has already been taken).",
            request_id,
        )
        return None
      if record.is_complete:
        # First response wins: an at-least-once transport may redeliver.
        self._duplicates += 1
        return None
      record.response = response
      record.completed_at = self._time_fn()
      return record

  def take(
      self, request_id: str
  ) -> Optional[remote_execution.ExecutionResponse]:
    """Removes and returns a completed response, or None if not ready."""
    with self._lock:
      record = self._records.get(request_id)
      if record is None or not record.is_complete:
        return None
      del self._records[request_id]
      return record.response

  def is_pending(self, request_id: str) -> bool:
    """True when the request is being tracked and has no response yet."""
    with self._lock:
      record = self._records.get(request_id)
      return record is not None and not record.is_complete

  def pending_ids(self) -> List[str]:
    """Ids of requests still awaiting a response, oldest first."""
    with self._lock:
      pending = [r for r in self._records.values() if not r.is_complete]
    pending.sort(key=lambda r: r.dispatched_at)
    return [r.request_id for r in pending]

  def overdue(self, timeout_s: float) -> List[PendingRequest]:
    """Requests still in flight after `timeout_s`, oldest first.

    This is the input to a timeout/retry policy; the tracker itself never
    expires anything, so the caller decides whether to retry or fail.
    """
    now = self._time_fn()
    with self._lock:
      stale = [
          r
          for r in self._records.values()
          if not r.is_complete and (now - r.dispatched_at) >= timeout_s
      ]
    stale.sort(key=lambda r: r.dispatched_at)
    return stale

  def forget(self, request_id: str) -> bool:
    """Stops tracking a request (e.g. it was cancelled). True if it was known."""
    with self._lock:
      return self._records.pop(request_id, None) is not None

  @property
  def in_flight(self) -> int:
    with self._lock:
      return sum(1 for r in self._records.values() if not r.is_complete)

  @property
  def orphan_count(self) -> int:
    """Responses that arrived for an id that was never registered."""
    return self._orphans

  @property
  def duplicate_count(self) -> int:
    """Responses discarded because that request was already answered."""
    return self._duplicates


class TrackedActorHandle:
  """An `ActorHandle` whose fire-and-forget traffic is tracked and correlated.

  Wraps dispatch/poll so the caller works in terms of request ids:

      request_id = await tracked.dispatch("generate", req)   # returns immediately
      ...
      response = await tracked.await_response(request_id)    # correlated result
  """

  def __init__(
      self,
      handle: remote_execution.ActorHandle,
      tracker: Optional[RequestTracker] = None,
  ):
    self._handle = handle
    self._tracker = tracker or RequestTracker()

  @property
  def tracker(self) -> RequestTracker:
    return self._tracker

  @property
  def handle(self) -> remote_execution.ActorHandle:
    return self._handle

  async def dispatch(
      self, method_name: Optional[str] = None, *args, **kwargs
  ) -> str:
    """Fire-and-forget: dispatches, records the request, returns the ack id."""
    request_id = await self._handle.dispatch_task(method_name, *args, **kwargs)
    self._tracker.register(request_id, method_name or "__call__")
    return request_id

  async def await_response(
      self, request_id: str, timeout_s: float = 10.0
  ) -> Optional[remote_execution.ExecutionResponse]:
    """Polls until this request's response arrives, or the timeout elapses.

    Responses for other requests seen along the way are recorded against their
    own ids, so nothing is lost by waiting on this one.
    """
    already = self._tracker.take(request_id)
    if already is not None:
      return already

    response = await self._handle.poll_responses(
        timeout_s=timeout_s, request_id=request_id
    )
    if response is None:
      return None
    self._tracker.resolve(response)
    return self._tracker.take(request_id) or response

  async def drain(self, timeout_s: float = 0.0) -> int:
    """Collects whatever responses are ready, matching each to its request.

    Returns:
      How many responses were matched to a tracked request.
    """
    matched = 0
    while True:
      response = await self._handle.poll_responses(timeout_s=timeout_s)
      if response is None:
        return matched
      if self._tracker.resolve(response) is not None:
        matched += 1

  def take(
      self, request_id: str
  ) -> Optional[remote_execution.ExecutionResponse]:
    """Removes and returns an already-collected response, if ready."""
    return self._tracker.take(request_id)
