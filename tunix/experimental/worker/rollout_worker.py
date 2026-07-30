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

from typing import Any, AsyncIterator, Callable, Sequence

from tunix.experimental.common import datatypes
from tunix.experimental.worker import abstract_worker

WorkerState = datatypes.WorkerState


class RolloutWorker(abstract_worker.Worker):
  """Worker wrapper for rollout collection.

  Encapsulates RolloutManager and executes concurrent episode loops.
  """

  def __init__(self, worker_id: str, **kwargs):
    del kwargs
    self.worker_id = worker_id

  def get_worker_id(self) -> str:
    """Returns the unique worker ID."""
    return self.worker_id

  def info(self) -> datatypes.WorkerInfo:
    return datatypes.WorkerInfo(
        worker_id=self.worker_id, roles=frozenset({"rollout"})
    )

  def initialize(self) -> datatypes.Response:
    self.state = WorkerState.INITIALIZING
    try:
      return datatypes.Response()
    finally:
      self.state = WorkerState.READY

  def compile(self, dummy_data: Any) -> datatypes.Response:
    self.state = WorkerState.COMPILING
    try:
      return datatypes.Response()
    finally:
      self.state = WorkerState.READY

  def start(self) -> datatypes.Response:
    return datatypes.Response()

  def stop(self) -> datatypes.Response:
    self.state = WorkerState.STOPPED
    return datatypes.Response()

  def pause(self) -> datatypes.Response:
    raise NotImplementedError()

  def resume(self) -> datatypes.Response:
    raise NotImplementedError()

  def heartbeat(self) -> datatypes.HealthReport:
    return datatypes.HealthReport(state=self.state)

  async def generate(
      self,
      requests: datatypes.RolloutRequest | Sequence[datatypes.RolloutRequest],
      on_complete: Callable[[datatypes.RolloutResponse], None] | None = None,
  ) -> datatypes.RolloutResponse | Sequence[datatypes.RolloutResponse]:
    """Coroutine method for single or batched generate requests.

    Args:
      requests: A single RolloutRequest or a sequence of them to process.
      on_complete: An optional callback invoked immediately as each individual
        RolloutResponse is successfully generated. This allows the caller to
        stream results asynchronously without waiting for the entire batch to
        finish.

    Returns:
      A single RolloutResponse (if a single request was provided) or a sequence
      of
      completed RolloutResponses corresponding to the batch of requests.
    """
    raise NotImplementedError()

  async def pop_next_completed(self) -> datatypes.RolloutResponse:
    """Pull-based stream: yields whichever trajectory finishes first out-of-order.

    This provides an alternative to the `on_complete` callback for consumers
    who prefer to actively await the next available trajectory from the worker.

    Returns:
      The next completed RolloutResponse.
    """
    raise NotImplementedError()

  def as_completed_stream(self) -> AsyncIterator[datatypes.RolloutResponse]:
    """Async stream yielding completed trajectories or errors strictly out-of-order.

    Yields:
      Completed RolloutResponse objects as they finish generation.
    """
    # Convert `datatypes.Trajectory` to a `RolloutResponse` using
    # `datatypes.RolloutResponse.from_trajectory()` before yielding
    raise NotImplementedError()

  def prepare_weight_sync(self, metadata: Any) -> datatypes.Response:
    """Prepares the worker for an upcoming weight synchronization step.

    This is used to fence off state or pause ongoing execution to ensure
    safe memory updates without race conditions.

    Args:
      metadata: Any metadata required to prepare the sync (e.g. sync IDs).
    """
    self.state = WorkerState.SYNCING
    del metadata
    try:
      raise NotImplementedError()
    finally:
      self.state = WorkerState.READY

  def sync_weights(self, metadata: Any) -> int:
    """Synchronizes the worker's internal model weights.

    Args:
      metadata: Metadata locating the weights to sync (e.g. from Raiden).

    Returns:
      The version identifier (policy version) of the newly synced weights.
    """
    self.state = WorkerState.SYNCING
    del metadata
    try:
      raise NotImplementedError()
    finally:
      self.state = WorkerState.READY
