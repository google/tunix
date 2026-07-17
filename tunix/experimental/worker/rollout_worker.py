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

"""Top-level RolloutWorker abstractions."""

from typing import Sequence

from tunix.experimental.common import datatypes
from tunix.experimental.worker import abstract_worker


class TicketNotFound(Exception):
  """Raised when a rollout ticket is unknown (never issued, or already expired).

  Not retryable: the caller should re-dispatch the work rather than re-poll.
  """


class RolloutWorker(abstract_worker.Worker):
  """Worker wrapper for rollout collection.

  Encapsulates RolloutManager and executes concurrent episode loops. Completed
  trajectories are delivered through a ticketed, cursor-read channel
  (generate -> get_completed -> ack) rather than a return value or a callback,
  so delivery is idempotent and a dropped/retried poll never loses results.
  """

  def __init__(self, worker_id: str, **kwargs):
    del kwargs
    self.worker_id = worker_id

  def get_worker_id(self) -> str:
    """Returns the unique worker ID."""
    return self.worker_id

  def initialize(self) -> None:
    pass

  def compile(self, shape_config: datatypes.ShapeConfig) -> None:
    pass

  def start(self) -> None:
    pass

  def stop(self) -> None:
    pass

  def health(self) -> datatypes.HealthReport:
    """Returns a liveness/status snapshot."""
    raise NotImplementedError()

  def info(self) -> datatypes.WorkerInfo:
    """Returns the worker's static description."""
    raise NotImplementedError()

  async def generate(
      self, requests: Sequence[datatypes.TrajectoryRequest]
  ) -> str:
    """Enqueues a batch of requests and returns a ticket to read results from.

    Enqueue-only: this returns immediately with a ticket. Completed results are
    read out-of-order via `get_completed` and acknowledged via `ack`.

    Args:
      requests: The requests to generate trajectories for.

    Returns:
      A ticket string identifying this batch's completion stream.
    """
    raise NotImplementedError()

  async def get_completed(
      self,
      ticket: str,
      after_seq: int,
      max_items: int = 16,
      wait_s: float = 0.0,
  ) -> datatypes.CompletedPage:
    """Cursor-read of completed results for a ticket (non-destructive).

    Returns results whose sequence number exceeds `after_seq`; the worker
    retains results until they are `ack`ed, so a retried read with the same
    `after_seq` re-reads the same page (idempotent). `wait_s > 0` long-polls up
    to that many seconds; an empty page with `done=False` is valid.

    Args:
      ticket: The ticket returned by `generate`.
      after_seq: Read results whose sequence number exceeds this cursor.
      max_items: Maximum number of results to return in this page.
      wait_s: Optional bounded long-poll timeout.

    Returns:
      A CompletedPage of results plus the new cursor and done flag.
    """
    raise NotImplementedError()

  async def ack(self, ticket: str, upto_seq: int) -> None:
    """Acknowledges results up to `upto_seq`, letting the worker GC them."""
    raise NotImplementedError()

  async def cancel(
      self, ticket: str, request_ids: Sequence[str] | None = None
  ) -> None:
    """Cancels a ticket's episodes (or a subset), emitting CANCELLED results.

    Args:
      ticket: The ticket to cancel.
      request_ids: Specific requests to cancel; None cancels the whole ticket.
    """
    raise NotImplementedError()

  def prepare_weight_sync(self, meta: datatypes.WeightSyncMetadata) -> None:
    """Fences the worker for an upcoming weight sync.

    Stops admitting new episodes and awaits in-flight episode drain so weights
    can be installed without corrupting trajectories in flight.

    Args:
      meta: Metadata locating the weights to be installed.
    """
    raise NotImplementedError()

  def sync_weights(self, meta: datatypes.WeightSyncMetadata) -> int:
    """Fetches and installs new weights, returning the installed version.

    Args:
      meta: Metadata locating the weights to sync.

    Returns:
      The installed weight (policy) version.
    """
    raise NotImplementedError()
