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

  def initialize(self) -> None:
    pass

  def compile(self, dummy_data: Any) -> None:
    pass

  def start(self) -> None:
    pass

  def stop(self) -> None:
    pass

  def pause(self) -> None:
    raise NotImplementedError()

  def resume(self) -> None:
    raise NotImplementedError()

  async def generate(
      self,
      requests: (
          datatypes.TrajectoryRequest | Sequence[datatypes.TrajectoryRequest]
      ),
      on_complete: (
          Callable[[datatypes.TrajectoryResult], None] | None
      ) = None,
  ) -> datatypes.TrajectoryResult | Sequence[datatypes.TrajectoryResult]:
    """Coroutine method for single or batched generate requests.

    Args:
      requests: A single TrajectoryRequest or a sequence of them to process.
      on_complete: An optional callback invoked immediately as each individual
        TrajectoryResult is produced. This allows the caller to stream results
        asynchronously without waiting for the entire batch to finish.

    Returns:
      A single TrajectoryResult (if a single request was provided) or a sequence
      of TrajectoryResults corresponding to the batch of requests.
    """
    raise NotImplementedError()

  async def pop_next_completed(self) -> datatypes.TrajectoryResult:
    """Pull-based stream: yields whichever result finishes first out-of-order.

    This provides an alternative to the `on_complete` callback for consumers
    who prefer to actively await the next available result from the worker.

    Returns:
      The next completed TrajectoryResult.
    """
    raise NotImplementedError()

  def as_completed_stream(self) -> AsyncIterator[datatypes.TrajectoryResult]:
    """Async stream yielding completed results (or errors) strictly out-of-order.

    Yields:
      TrajectoryResult objects as episodes finish generation.
    """
    raise NotImplementedError()

  def prepare_weight_sync(self, metadata: Any) -> None:
    """Prepares the worker for an upcoming weight synchronization step.

    This is used to fence off state or pause ongoing execution to ensure
    safe memory updates without race conditions.

    Args:
      metadata: Any metadata required to prepare the sync (e.g. sync IDs).
    """
    raise NotImplementedError()

  def sync_weights(self, metadata: Any) -> int:
    """Synchronizes the worker's internal model weights.

    Args:
      metadata: Metadata locating the weights to sync (e.g. from Raiden).

    Returns:
      The version identifier (policy version) of the newly synced weights.
    """
    raise NotImplementedError()
