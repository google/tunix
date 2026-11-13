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

"""Destination-side weight sync delegate for basic weight sync."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
import logging
import threading
import time
from typing import Any, Protocol, runtime_checkable

import numpy as np
from tunix.dist_rl.common import datatypes
from tunix.dist_rl.weight_sync import basic_handler
from tunix.dist_rl.weight_sync import weight_sync
from tunix.dist_rl.weight_sync import weight_sync_coordinator


@runtime_checkable
class HfWeightUpdatableSampler(Protocol):
  """Framework-neutral protocol for samplers accepting canonical HF weights."""

  def expected_hf_variables(
      self,
  ) -> tuple[weight_sync.TensorMetadata, ...]:
    ...

  def update_params_from_hf(
      self,
      hf_weights: Mapping[str, basic_handler.RawTensorPayload | np.ndarray],
      *,
      preserve_active_kv_cache: bool | None = None,
  ) -> None:
    ...


def _extract_round_id(
    sync_request: datatypes.WeightSyncRequest | None,
) -> tuple[str, int]:
  """Extracts and strictly validates req_id and generation from sync_request."""
  if sync_request is None:
    raise ValueError("WeightSyncRequest cannot be None")
  if not isinstance(sync_request, datatypes.WeightSyncRequest):
    raise TypeError(f"Expected WeightSyncRequest, got {type(sync_request)}")
  extra = sync_request.extra_config
  if not extra or "req_id" not in extra or not extra["req_id"]:
    raise ValueError(
        f"WeightSyncRequest.extra_config missing required 'req_id': {extra}"
    )
  if "uuid" not in extra or extra["uuid"] is None:
    raise ValueError(
        f"WeightSyncRequest.extra_config missing required 'uuid': {extra}"
    )
  generation = int(extra["uuid"])
  if generation <= 0:
    raise ValueError(
        f"WeightSyncRequest uuid must be positive, got {generation}"
    )
  return str(extra["req_id"]), generation


class BasicDestinationDelegate:
  """Manages host-staged basic weight synchronization for rollout samplers.

  Incoming weights from trainer rank 0 are streamed over TCP into host RAM
  buffers managed by `BasicReceiverServer`. Model weights on TPU/GPU are
  updated only during the `weight_sync()` phase. Built for simplicity and
  generality across framework pairs, not peak performance.
  """

  def __init__(
      self,
      worker_index: int = 0,
      server_id: str = "rollout",
      partial_rollout: bool | None = None,
      bind_port: int | None = None,
      advertised_host: str | None = None,
  ):
    self._server_id = server_id
    self._worker_index = worker_index
    self.partial_rollout: bool | None = (
        None if partial_rollout is None else bool(partial_rollout)
    )
    self._bind_port = bind_port
    self._advertised_host = advertised_host
    self._variables: tuple[weight_sync.TensorMetadata, ...] = ()

    self._sampler: HfWeightUpdatableSampler | None = None
    self._version = 0
    self._tracker = weight_sync_coordinator.WorkerRoundTracker()
    self._phase_lock = threading.Lock()
    self._state_lock = threading.Lock()
    self._receiver: basic_handler.BasicReceiverServer | None = None
    self._uncommitted_h2d = False
    self._serving_corrupted = False
    self._bound = False
    self._closed = False

  @property
  def serving_corrupted(self) -> bool:
    with self._state_lock:
      return self._serving_corrupted

  @property
  def has_uncommitted_mutation(self) -> bool:
    """True when an in-place H2D weight write has not yet been committed."""
    with self._state_lock:
      return self._uncommitted_h2d

  def is_bounded(self) -> bool:
    """Returns whether destination receiver is bound and listening."""
    with self._state_lock:
      return self._bound

  def _require_open_locked(self) -> None:
    if self._closed:
      raise RuntimeError(
          f"BasicDestinationDelegate [{self._server_id}] is closed."
      )

  def _bind_weight_sync_sync(
      self,
      sampler: HfWeightUpdatableSampler | None = None,
  ) -> bool:
    """Synchronously binds receiver and queries expected HF metadata."""
    with self._phase_lock:
      with self._state_lock:
        self._require_open_locked()
        if self._bound:
          return True

      if sampler is None or not isinstance(sampler, HfWeightUpdatableSampler):
        raise TypeError(
            f"Sampler {type(sampler)} does not implement"
            " HfWeightUpdatableSampler"
        )

      variables = tuple(sampler.expected_hf_variables())
      receiver = basic_handler.BasicReceiverServer(
          bind_port=self._bind_port,
          advertised_host=self._advertised_host,
      )

      with self._state_lock:
        self._sampler = sampler
        self._variables = variables
        self._receiver = receiver
        self._bound = True
      return True

  async def bind_weight_sync(
      self,
      sync_request: datatypes.WeightSyncRequest | None = None,
      state: Any = None,
      sampler: HfWeightUpdatableSampler | None = None,
      **kwargs: Any,
  ) -> bool:
    """Binds destination-side transport receiver for weight sync."""
    del sync_request, state, kwargs
    return await asyncio.to_thread(
        self._bind_weight_sync_sync,
        sampler=sampler,
    )

  def _get_weight_sync_metadata_sync(
      self,
  ) -> list[weight_sync.WorkUnitMetadata]:
    """Synchronously reads destination unit metadata."""
    with self._state_lock:
      self._require_open_locked()
      bound = self._bound
      receiver = self._receiver
      variables = self._variables
    if not bound or receiver is None:
      raise RuntimeError(
          f"BasicDestinationDelegate [{self._server_id}] is not bound."
      )
    return [
        weight_sync.WorkUnitMetadata(
            unit=weight_sync.WorkUnitId(
                job_name=self._server_id,
                data_replica_idx=self._worker_index,
            ),
            control_plane_rpc_address=receiver.address,
            variables=variables,
            transport_mode="basic",
            use_ffi=False,
            artifact_uri=receiver.auth_uri,
        )
    ]

  async def get_weight_sync_metadata(
      self, **kwargs: Any
  ) -> list[weight_sync.WorkUnitMetadata]:
    """Retrieves destination worker metadata for the sync coordinator."""
    del kwargs
    return await asyncio.to_thread(self._get_weight_sync_metadata_sync)

  def _pre_weight_sync_sync(
      self,
      sync_request: datatypes.WeightSyncRequest | None = None,
  ) -> bool:
    """Synchronously admits and prepares a weight sync round."""
    with self._phase_lock:
      _extract_round_id(sync_request)
      with self._state_lock:
        self._require_open_locked()
        if self._serving_corrupted:
          raise RuntimeError(
              "Serving weights are corrupted from a previous failed H2D; cannot"
              " admit new round."
          )
        if not self._tracker.admit(sync_request, "prepared"):
          return True
        self._tracker.complete(sync_request, "prepared")
      return True

  async def pre_weight_sync(
      self,
      sync_request: datatypes.WeightSyncRequest | None = None,
      **kwargs: Any,
  ) -> bool:
    """Pre-sync phase hook executed before weight transfer begins."""
    del kwargs
    return await asyncio.to_thread(self._pre_weight_sync_sync, sync_request)

  def _weight_sync_sync(
      self,
      sync_request: datatypes.WeightSyncRequest | None = None,
      *,
      partial_rollout: bool | None = None,
  ) -> int:
    """Synchronously pops staged weights and installs them on device."""
    with self._phase_lock:
      req_id, generation = _extract_round_id(sync_request)
      assert sync_request is not None
      with self._state_lock:
        self._require_open_locked()
        if self._serving_corrupted:
          raise RuntimeError(
              "Serving weights are corrupted from a previous failed H2D; cannot"
              " admit weight_sync."
          )
        if not self._tracker.admit(sync_request, "h2d_done"):
          return self._version
        receiver = self._receiver
        sampler = self._sampler

      if receiver is None:
        raise RuntimeError(
            "BasicDestinationDelegate receiver is not initialized."
        )

      round_weights = receiver.pop_round(req_id, generation)
      num_tensors = len(round_weights)

      if sampler is None:
        raise RuntimeError("Sampler is not available for weight sync")

      if not isinstance(sampler, HfWeightUpdatableSampler):
        raise TypeError(
            f"Sampler {type(sampler)} does not implement"
            " HfWeightUpdatableSampler (missing update_params_from_hf)"
        )

      preserve_kv = (
          self.partial_rollout
          if partial_rollout is None
          else bool(partial_rollout)
      )

      with self._state_lock:
        self._uncommitted_h2d = True

      t_install_start = time.perf_counter()
      try:
        sampler.update_params_from_hf(
            round_weights, preserve_active_kv_cache=preserve_kv
        )
      except Exception as exc:
        with self._state_lock:
          self._serving_corrupted = True
        logging.error(
            "BasicDestinationDelegate: failed during H2D"
            " update_params_from_hf: %s",
            exc,
        )
        raise
      finally:
        del round_weights

      with self._state_lock:
        self._version = int(sync_request.policy_version)
        installed_version = self._version
        self._tracker.complete(sync_request, "h2d_done")

      install_elapsed = time.perf_counter() - t_install_start
      logging.info(
          "[BASIC] installed %d tensors on device (policy_version=%d) in"
          " %.3fs",
          num_tensors,
          installed_version,
          install_elapsed,
      )
      return installed_version

  async def weight_sync(
      self,
      sync_request: datatypes.WeightSyncRequest | None = None,
      *,
      partial_rollout: bool | None = None,
      **kwargs: Any,
  ) -> int:
    """Executes weight installation on device from host staging buffer."""
    del kwargs
    return await asyncio.to_thread(
        self._weight_sync_sync,
        sync_request,
        partial_rollout=partial_rollout,
    )

  def _post_weight_sync_sync(
      self,
      sync_request: datatypes.WeightSyncRequest | None = None,
  ) -> bool:
    """Synchronously commits a completed weight sync round."""
    with self._phase_lock:
      with self._state_lock:
        self._require_open_locked()
        if self._serving_corrupted:
          raise RuntimeError(
              "Serving weights are corrupted from a previous failed H2D; cannot"
              " commit round."
          )
        _extract_round_id(sync_request)
        if not self._tracker.admit(sync_request, "committed"):
          return True
        self._uncommitted_h2d = False
        self._tracker.complete(sync_request, "committed")
      return True

  async def post_weight_sync(
      self,
      sync_request: datatypes.WeightSyncRequest | None = None,
      **kwargs: Any,
  ) -> bool:
    """Post-sync phase hook executed after weight installation completes."""
    del kwargs
    return await asyncio.to_thread(self._post_weight_sync_sync, sync_request)

  def _abort_weight_sync_sync(
      self,
      sync_request: datatypes.WeightSyncRequest | None = None,
  ) -> bool:
    """Synchronously aborts a round or latches corruption after H2D."""
    with self._phase_lock:
      with self._state_lock:
        self._require_open_locked()
        if self._serving_corrupted:
          raise RuntimeError(
              "Cannot abort weight sync after partial H2D failure; serving"
              " weights are corrupted."
          )
      req_id, generation = _extract_round_id(sync_request)
      with self._state_lock:
        if not self._tracker.admit(sync_request, "aborted"):
          return False
        uncommitted_h2d = self._uncommitted_h2d
        if uncommitted_h2d:
          self._serving_corrupted = True
        receiver = self._receiver

      if receiver is not None:
        receiver.discard_round(req_id, generation)
      if uncommitted_h2d:
        raise RuntimeError(
            "Cannot roll back uncommitted in-place H2D weight update when"
            f" aborting round ({req_id!r}, {generation}); serving weights are"
            " un-rolled-back."
        )
      with self._state_lock:
        self._tracker.complete(sync_request, "aborted")
      return True

  async def abort_weight_sync(
      self,
      sync_request: datatypes.WeightSyncRequest | None = None,
      **kwargs: Any,
  ) -> bool:
    """Safely handles abort of weight sync round."""
    del kwargs
    return await asyncio.to_thread(self._abort_weight_sync_sync, sync_request)

  def get_weight_sync_status(self) -> Mapping[str, Any]:
    """Reports worker-side round status for coordinator recovery checks."""
    with self._state_lock:
      return dict(self._tracker.report())

  def close(self) -> None:
    """Shuts down receiver server and marks delegate permanently closed."""
    with self._phase_lock:
      with self._state_lock:
        if self._closed:
          return
        self._closed = True
        receiver = self._receiver
        self._receiver = None
        self._bound = False
      if receiver is not None:
        receiver.close()
