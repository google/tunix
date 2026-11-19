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

import logging
from typing import Any

from flax import nnx
import jax
from tunix.dist_rl.rollout import sampler as base_sampler_lib
from tunix.dist_rl.weight_sync import weight_sync


class RaidenDestinationWeightSyncMixin:
  """Mixin providing destination-side weight sync hooks for samplers."""

  server_id: str
  enable_raiden: bool
  raiden_sync_delegate: Any

  def _get_underlying_sampler(self) -> Any:
    return (
        getattr(self, "sampler", None)
        or getattr(self, "vllm_sampler", None)
        or (self if hasattr(self, "transformer_state") else None)
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
      sampler = self._get_underlying_sampler()
      if sampler is None or not hasattr(sampler, "transformer_state"):
        raise RuntimeError(
            f"{self.__class__.__name__} [{self.server_id}] sampler does not"
            " expose transformer_state for Raiden weight sync."
        )

      if self.raiden_sync_delegate.is_bounded():
        return True

      state = sampler.transformer_state
      return await self.raiden_sync_delegate.bind_weight_sync(
          sync_request=sync_request, state=state, sampler=sampler, **kwargs
      )
    return None

  def get_target_state(self) -> Any:
    """Returns target state shape/dtype pytree for weight conversion."""
    sampler = self._get_underlying_sampler()
    if sampler is None:
      raise RuntimeError(
          f"{self.__class__.__name__} [{self.server_id}] sampler is not"
          " initialized."
      )
    if sampler is not self and hasattr(sampler, "get_target_state"):
      return sampler.get_target_state()
    if hasattr(sampler, "transformer_state"):
      state = sampler.transformer_state
      return jax.tree.map(
          lambda x: nnx.Param(
              jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype)
          ),
          state,
          is_leaf=lambda x: isinstance(x, nnx.Variable),
      )
    raise AttributeError(
        f"{self.__class__.__name__} [{self.server_id}] cannot extract"
        " target_state."
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
      sync_request: base_sampler_lib.WeightSyncRequest | Any = None,
      **kwargs,
  ) -> str | None | Any:
    """Prepares staging handshake prior to policy weight update."""
    self._check_weight_sync_boundness()
    # TODO(tunix-dev): Hold a per-sampler lock across pause -> delegate here
    # and delegate -> resume in post/abort. The coordinator requires
    # destinations to serialize whole phase bodies, because a timed-out phase
    # (e.g. a long pause drain) can still be running when the abort arrives.
    partial_rollout = bool(getattr(self, "partial_rollout", False))
    use_partial = bool(kwargs.get("partial_rollout", partial_rollout))
    await self.pause(
        mode="keep" if use_partial else "wait",
        clear_cache=not use_partial,
    )
    if partial_rollout:
      kwargs.setdefault("partial_rollout", True)

    if self.enable_raiden:
      return await self.raiden_sync_delegate.pre_weight_sync(
          sync_request=sync_request, **kwargs
      )
    return True

  async def weight_sync(
      self,
      sync_request: base_sampler_lib.WeightSyncRequest | Any = None,
      **kwargs,
  ) -> str | None | Any:
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
      sampler = self._get_underlying_sampler()
      if sampler and hasattr(sampler, "update_params"):
        weights = sync_request.weights
        if weights is None:
          raise ValueError(
              f"{self.__class__.__name__} [{self.server_id}] weight_sync:"
              " weights not found in sync_request."
          )
        sampler.update_params(weights)
      else:
        raise RuntimeError(
            f"{self.__class__.__name__} [{self.server_id}] does not support"
            " Raiden weight sync, while the fallback path missing required"
            " components."
        )
      return True

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
