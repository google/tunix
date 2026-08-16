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

"""Runs weight sync rounds from synchronous code on a resident event loop."""

import asyncio
import threading
from typing import Any, Callable, Coroutine


class WeightSyncDriver:
  """Bridges a synchronous training loop to the async weight sync coordinator.

  gRPC channels are bound to the loop that created them, so the driver owns
  one loop for its whole lifetime and builds every component on it.
  """

  def __init__(
      self,
      components_factory: Callable[[int], Coroutine[Any, Any, Any]],
      *,
      initial_uuid: int,
      initial_policy_version: int,
      round_timeout_s: float = 3600.0,
  ):
    self._round_timeout_s = round_timeout_s
    self._policy_version = initial_policy_version
    self._loop = asyncio.new_event_loop()
    self._thread = threading.Thread(
        target=self._loop.run_forever, name="weight-sync-driver", daemon=True
    )
    self._thread.start()
    self._components = self._run(components_factory(initial_uuid))

  @property
  def policy_version(self) -> int:
    return self._policy_version

  def _run(self, coro: Coroutine[Any, Any, Any]) -> Any:
    """Runs a coroutine on the driver loop and blocks for its result."""
    future = asyncio.run_coroutine_threadsafe(coro, self._loop)
    return future.result(self._round_timeout_s)

  def sync_weights(self, **kwargs) -> Any:
    """Runs one round and advances the policy version on commit."""
    version = self._policy_version + 1
    result = self._run(
        self._components.coordinator.sync(policy_version=version, **kwargs)
    )
    self._policy_version = version
    return result

  def close(self) -> None:
    """Closes the components on the driver loop, then retires the loop."""
    close = getattr(self._components, "close", None)
    if close is not None:
      self._run(close())
    self._loop.call_soon_threadsafe(self._loop.stop)
    self._thread.join()
    self._loop.close()
