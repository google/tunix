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

"""Mixin providing destination-side Raiden weight sync for rollout samplers."""

from __future__ import annotations

from typing import Any

from flax import nnx
import jax
from tunix.common import configs
from tunix.dist_rl.common import datatypes
from tunix.dist_rl.rollout import sampler as base_sampler_lib


class RaidenDestinationWeightSyncMixin:
  """Mixin providing destination-side weight sync hooks for samplers."""

  server_id: str
  config: configs.RolloutConfig
  enable_raiden: bool
  raiden_sync_delegate: Any

  @property
  def transformer_state(self) -> Any:
    """Returns the sampler's transformer state."""
    raise NotImplementedError(
        f"{self.__class__.__name__} [{self.server_id}] does not expose"
        " transformer_state."
    )

  def update_params(
      self,
      updated_weights: Any,
      filter_types: tuple[Any, ...] | None = None,
  ) -> None:
    """Updates model parameters in the sampler."""
    del updated_weights, filter_types
    raise NotImplementedError(
        f"{self.__class__.__name__} [{self.server_id}] does not implement"
        " update_params."
    )

  def delete_cache(self) -> None:
    """Deletes the KV cache before weight sync."""
    raise NotImplementedError(
        f"{self.__class__.__name__} [{self.server_id}] does not implement"
        " delete_cache."
    )

  def reset_prefix_cache(self) -> None:
    """Resets the prefix cache before weight sync."""
    raise NotImplementedError(
        f"{self.__class__.__name__} [{self.server_id}] does not implement"
        " reset_prefix_cache."
    )

  def reinitialize_cache(self) -> None:
    """Reallocates the KV cache after weight sync."""
    raise NotImplementedError(
        f"{self.__class__.__name__} [{self.server_id}] does not implement"
        " reinitialize_cache."
    )

  def refresh_state_leaves(self) -> None:
    """Refreshes cached state leaves after weight sync."""
    raise NotImplementedError(
        f"{self.__class__.__name__} [{self.server_id}] does not implement"
        " refresh_state_leaves."
    )

  def _check_weight_sync_boundness(self) -> None:
    """Verifies that the Raiden delegate has been bound before executing sync phases."""
    if not self.enable_raiden:
      return

    if not self.raiden_sync_delegate.is_bounded():
      raise RuntimeError(
          f"{self.__class__.__name__} [{self.server_id}] weight sync delegate"
          " is not bounded."
      )

  async def get_weight_sync_metadata(self, **kwargs) -> Any:
    """Returns sharding specs and layout metadata across devices for weights."""
    self._check_weight_sync_boundness()

    if self.enable_raiden:
      return await self.raiden_sync_delegate.get_weight_sync_metadata(**kwargs)

    raise NotImplementedError(
        f"{self.__class__.__name__} [{self.server_id}] does not support"
        " get_weight_sync_metadata when Raiden is disabled."
    )

  async def bind_weight_sync(
      self,
      sync_request: base_sampler_lib.WeightSyncRequest | Any = None,
      **kwargs,
  ) -> Any:
    """Binds destination-side transport resources for weight transfer."""
    if self.enable_raiden:
      state = self.transformer_state
      if self.raiden_sync_delegate.is_bounded():
        return True

      return await self.raiden_sync_delegate.bind_weight_sync(
          sync_request=sync_request,
          state=state,
          sampler=self,
          **kwargs,
      )
    return None

  def get_target_state(self) -> Any:
    """Returns target state shape/dtype pytree for weight conversion."""
    state = self.transformer_state
    return jax.tree.map(
        lambda x: nnx.Param(jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype)),
        state,
        is_leaf=lambda x: isinstance(x, nnx.Variable),
    )

  # TODO(tunix-dev): Make `pause`/`resume` abstract (or type the host sampler
  # with a Protocol) so a sampler that forgets to override them fails loudly
  # instead of silently skipping the pause around in-place weight writes.
  async def pause(self, **kwargs) -> str | None | Any:
    """Pauses inference processing on this worker slice."""
    del kwargs
    return True

  async def resume(self, **kwargs) -> str | None | Any:
    """Resumes inference processing on this worker slice."""
    del kwargs
    return True

  async def pre_weight_sync(
      self,
      sync_request: (
          base_sampler_lib.WeightSyncRequest
          | datatypes.WeightSyncMetadata
          | None
      ) = None,
      **kwargs,
  ) -> str | None | Any:
    """Prepares staging handshake prior to policy weight update."""
    self._check_weight_sync_boundness()
    # TODO(tunix-dev): Hold a per-sampler lock across pause -> delegate here
    # and delegate -> resume in post/abort. The coordinator requires
    # destinations to serialize whole phase bodies, because a timed-out phase
    # (e.g. a long pause drain) can still be running when the abort arrives.
    partial_rollout = self.config.partial_rollout
    await self.pause(
        mode="keep" if partial_rollout else "wait",
        clear_cache=not partial_rollout,
    )

    if self.enable_raiden:
      return await self.raiden_sync_delegate.pre_weight_sync(
          sync_request=sync_request, **kwargs
      )
    return True

  async def weight_sync(
      self,
      sync_request: base_sampler_lib.WeightSyncRequest | Any = None,
      **kwargs,
  ) -> int | None:
    """Updates model weights in-place from the specified controller or request."""
    self._check_weight_sync_boundness()

    if self.enable_raiden:
      return await self.raiden_sync_delegate.weight_sync(
          sync_request=sync_request, **kwargs
      )
    else:
      if sync_request is None:
        raise ValueError(
            f"{self.__class__.__name__} Fallback mode [{self.server_id}]"
            " weight_sync: sync_request is None."
        )
      weights = sync_request.weights
      if weights is None:
        raise ValueError(
            f"{self.__class__.__name__} [{self.server_id}] weight_sync:"
            " weights not found in sync_request."
        )
      self.update_params(weights)
      return None

  async def post_weight_sync(
      self,
      sync_request: base_sampler_lib.WeightSyncRequest | Any = None,
      **kwargs,
  ) -> str | None | Any:
    """Finalizes and switches active policy weights after transfer completion."""
    try:
      if self.enable_raiden:
        return await self.raiden_sync_delegate.post_weight_sync(
            sync_request=sync_request, **kwargs
        )
      return True
    finally:
      await self.resume()

  async def abort_weight_sync(
      self,
      sync_request: base_sampler_lib.WeightSyncRequest | Any = None,
      **kwargs,
  ) -> str | None | Any:
    """Safely aborts weight sync round."""
    try:
      if self.enable_raiden and self.raiden_sync_delegate is not None:
        return await self.raiden_sync_delegate.abort_weight_sync(
            sync_request=sync_request, **kwargs
        )
      return True
    finally:
      await self.resume()

  async def get_weight_sync_status(self) -> Any:
    """Reports worker-side round status for coordinator recovery checks."""
    if self.enable_raiden and self.raiden_sync_delegate is not None:
      return self.raiden_sync_delegate.get_weight_sync_status()
    return {}
