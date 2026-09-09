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

"""Raiden weight sync delegate for destination-side rollout workers."""

from __future__ import annotations

import asyncio
import os
from typing import Any, List, Mapping, Optional

from absl import logging
from flax import nnx
import jax
from tunix.experimental.weight_sync import raiden_synchronizer
from tunix.experimental.weight_sync import weight_sync_coordinator


class RaidenWeightSyncDelegate:
  """Manages weight synchronization over Raiden for sampler adapters.

  The destination side of a weight sync round: bind_weight_sync binds the
  sampler's transformer state to the raiden transport, the transfer lands
  in host staging, and weight_sync installs it on device.

  Caveat: the transport binds the live transformer state directly, so
  weight_sync writes into the serving copy rather than a shadow buffer.
  Serving is protected by the manager's closed-admission window, but an
  abort after a partial weight_sync cannot restore the previous weights.
  """

  def __init__(
      self,
      *args,
      job_name: Optional[str] = None,
      server_id: Optional[str] = None,
      worker_index: int = 0,
      sampler: Optional[Any] = None,
      **kwargs,
  ):
    del args, kwargs
    # Raiden partitions the weights across every unit sharing a job_name, so
    # replicas that all call themselves "rollout" get a slice each instead of a
    # copy each. server_id is already unique per replica and shared across the
    # hosts within one, which is exactly the grouping job_name needs.
    self.server_id = server_id
    self.job_name = job_name or server_id or "rollout"
    self._sampler = sampler
    self._synchronizers: List[Any] = [
        raiden_synchronizer.RaidenSynchronizer(
            self.job_name,
            worker_index=worker_index,
            auto_h2d=True,
        )
    ]
    self._version = 0
    self._sync_lock = asyncio.Lock()
    self._tracker = weight_sync_coordinator.WorkerRoundTracker()

  def is_bounded(
      self,
  ) -> bool:
    """Returns whether all managed synchronizers are bound."""
    return all(s.bound for s in self._synchronizers)

  async def bind_weight_sync(
      self, sync_request: Any = None, state: Any = None, sampler: Any = None, **kwargs
  ) -> Any:
    """Binds destination-side transport resources for weight sync."""
    del sync_request, kwargs
    if sampler is not None:
      self._sampler = sampler

    for sync in self._synchronizers:
      # The state arrays never change, so one bind covers every round.
      if not sync.bound:
        sync.bind(state)

    return True

  async def get_weight_sync_metadata(self, **kwargs) -> Any:
    """Retrieves destination worker metadata for the sync coordinator."""
    del kwargs
    return [s.work_unit_metadata() for s in self._synchronizers]

  def _has_round(self, sync_request: Any) -> bool:
    extra = getattr(sync_request, "extra_config", None) or {}
    return extra.get("req_id") is not None

  async def pre_weight_sync(self, sync_request: Any = None, **kwargs) -> Any:
    """Pre-sync phase hook executed before weight transfer begins."""
    del kwargs
    async with self._sync_lock:
      if self._has_round(sync_request):
        if not self._tracker.admit(sync_request, "prepared"):
          return True
        self._tracker.complete(sync_request, "prepared")

      sampler = getattr(self, "_sampler", None)
      if sampler is not None:
        llm = getattr(sampler, "llm", None)
        driver = getattr(sampler, "_driver", None)
        if llm is not None:
          llm.reset_prefix_cache()
          llm.collective_rpc("delete_kv_cache")
        elif driver is not None:
          driver.llm_engine.reset_prefix_cache()
          driver.llm_engine.collective_rpc("delete_kv_cache")
        jax.effects_barrier()

      return True

  async def weight_sync(self, sync_request: Any = None, **kwargs) -> Any:
    """Executes weight installation on device from host staging buffer."""
    del kwargs
    async with self._sync_lock:
      if self._has_round(sync_request):
        if not self._tracker.admit(sync_request, "h2d_done"):
          return self._version

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

      if self._has_round(sync_request):
        self._tracker.complete(sync_request, "h2d_done")

      return self._version

  async def post_weight_sync(self, sync_request: Any = None, **kwargs) -> Any:
    """Post-sync phase hook executed after weight installation completes."""
    del kwargs
    async with self._sync_lock:
      if self._has_round(sync_request):
        if not self._tracker.admit(sync_request, "committed"):
          return True

      if os.environ.get("VERIFY_WEIGHTS", "").lower() == "true":
        for sync in self._synchronizers:
          logging.info("raiden metrics: %s", sync.metrics())

      sampler = getattr(self, "_sampler", None)
      if sampler is not None:
        model_runner = getattr(sampler, "_model_runner", None)
        if model_runner is not None and hasattr(model_runner, "state_leaves"):
          state = getattr(model_runner, "state", None)
          if state is not None:
            model_runner.state_leaves = tuple(
                jax.tree_util.tree_leaves(state)
            )

        llm = getattr(sampler, "llm", None)
        driver = getattr(sampler, "_driver", None)
        if llm is not None:
          llm.collective_rpc("reinitialize_kv_cache")
        elif driver is not None:
          driver.llm_engine.collective_rpc("reinitialize_kv_cache")

      if self._has_round(sync_request):
        self._tracker.complete(sync_request, "committed")

      return True

  async def abort_weight_sync(self, sync_request: Any = None, **kwargs) -> Any:
    """Safely handles abort of weight sync round."""
    del kwargs
    async with self._sync_lock:
      if self._has_round(sync_request):
        if not self._tracker.admit(sync_request, "aborted"):
          return False
        self._tracker.complete(sync_request, "aborted")
      return True

  def get_weight_sync_status(self) -> Mapping[str, Any]:
    """Reports worker-side round status for coordinator recovery checks."""
    return self._tracker.report()
