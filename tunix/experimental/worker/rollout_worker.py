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
      requests: Any | Sequence[Any],
      on_complete: Callable[[Any], None] | None = None,
  ) -> Any | Sequence[Any]:
    """Coroutine method for single or batched generate requests."""
    raise NotImplementedError()

  async def pop_next_completed(self) -> Any:
    """Pull-based stream: yields whichever trajectory finishes first out-of-order."""
    raise NotImplementedError()

  def as_completed_stream(self) -> AsyncIterator[Any]:
    """Async stream yielding completed trajectories or errors strictly out-of-order."""
    raise NotImplementedError()

  def prepare_weight_sync(self, metadata: Any) -> None:
    """Synchronous method for prepare_weight_sync."""
    raise NotImplementedError()

  def sync_weights(self, metadata: Any) -> int:
    """Synchronous method for sync_weights."""
    raise NotImplementedError()
