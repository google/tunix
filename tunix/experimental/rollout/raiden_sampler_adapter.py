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

"""Vanilla sampler adapter with raiden weight sync as the destination."""

from __future__ import annotations

import os
from typing import Any, List

from absl import logging

from tunix.experimental.rollout import vanilla_sampler_adapter
from tunix.experimental.worker import raiden_synchronizer


class RaidenSamplerAdapter(vanilla_sampler_adapter.VanillaSamplerAdapter):
  """Serves with the in-process sampler; syncs weights over raiden.

  The destination side of a weight sync round: bind_weight_sync binds the
  sampler's transformer state to the raiden transport, the transfer lands
  in host staging, and weight_sync installs it on device.

  Caveat: the transport binds the live transformer state directly, so
  weight_sync writes into the serving copy rather than a shadow buffer.
  Serving is protected by the manager's closed-admission window, but an
  abort after a partial weight_sync cannot restore the previous weights.
  """

  def __init__(self, *args, worker_index: int = 0, **kwargs):
    super().__init__(*args, **kwargs)
    self._synchronizers: List[Any] = [
        raiden_synchronizer.RaidenSynchronizer(
            "rollout", worker_index=worker_index, auto_h2d=True
        )
    ]
    self._version = 0

  def _bound_synchronizers(self) -> List[Any]:
    if self.sampler is None:
      raise RuntimeError("initialize the sampler before weight sync")
    for sync in self._synchronizers:
      # The state arrays never change, so one bind covers every round.
      if not sync.bound:
        sync.bind(self.sampler.transformer_state)
    return self._synchronizers

  async def bind_weight_sync(self, sync_request: Any = None, **kwargs) -> Any:
    del sync_request, kwargs
    self._bound_synchronizers()
    return None

  async def get_weight_sync_metadata(self, **kwargs) -> Any:
    del kwargs
    return [s.work_unit_metadata() for s in self._bound_synchronizers()]

  async def pre_weight_sync(self, sync_request: Any = None, **kwargs) -> Any:
    del sync_request, kwargs
    self._bound_synchronizers()
    return True

  async def weight_sync(self, sync_request: Any = None, **kwargs) -> Any:
    del kwargs
    for sync in self._synchronizers:
      if not sync.bound:
        raise RuntimeError("bind_weight_sync must run before weight_sync")
      # auto_h2d installs chunks as they arrive; this call is the round's
      # awaited install, so completion is guaranteed before checksums/post.
      sync.h2d()
      if os.environ.get("VERIFY_WEIGHTS", "").lower() == "true":
        logging.info("destination checksums: %s", sync.checksums())
    version = getattr(sync_request, "policy_version", 0)
    self._version = version if version else self._version + 1
    return self._version

  async def post_weight_sync(self, sync_request: Any = None, **kwargs) -> Any:
    del sync_request, kwargs
    if os.environ.get("VERIFY_WEIGHTS", "").lower() == "true":
      for sync in self._synchronizers:
        logging.info("raiden metrics: %s", sync.metrics())
    return True