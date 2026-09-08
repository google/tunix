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

"""Mixin class providing destination-side Raiden weight sync for rollout adapters."""

from __future__ import annotations

import logging
from typing import Any

from flax import nnx
import jax
from tunix.experimental.rollout import sampler as base_sampler_lib
from tunix.experimental.weight_sync import weight_sync


class RaidenDestinationWeightSyncMixin:
  """Mixin providing destination-side weight sync hooks for sampler adapters."""

  def _get_underlying_sampler(self) -> Any:
    return getattr(self, "sampler", None) or getattr(self, "vllm_sampler", None)

  def _check_weight_sync_boundness(self) -> None:
    """Verifies that the Raiden delegate has been bound before executing sync phases."""
    if not getattr(self, "enable_raiden", False):
      return

    if not self.raiden_sync_delegate.is_bounded():
      raise RuntimeError(
          f"{self.__class__.__name__} [{self.server_id}] weight sync delegate"
          " is not bounded."
      )

  async def get_weight_sync_metadata(self, **kwargs) -> Any:
    """Returns sharding specs and layout metadata across devices for weights."""
    self._check_weight_sync_boundness()

    if getattr(self, "enable_raiden", False):
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
    if getattr(self, "enable_raiden", False):
      sampler = self._get_underlying_sampler()
      if sampler is None or not hasattr(sampler, "transformer_state"):
        raise RuntimeError(
            f"{self.__class__.__name__} [{self.server_id}] sampler does not expose"
            " transformer_state for Raiden weight sync."
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
          f"{self.__class__.__name__} [{self.server_id}] sampler is not initialized."
      )
    if hasattr(sampler, "get_target_state"):
      return sampler.get_target_state()
    if hasattr(sampler, "transformer_state"):
      state = sampler.transformer_state
      return jax.tree.map(
          lambda x: nnx.Param(jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype)),
          state,
          is_leaf=lambda x: isinstance(x, nnx.Variable),
      )
    raise AttributeError(
        f"{self.__class__.__name__} [{self.server_id}] cannot extract target_state."
    )

  async def pre_weight_sync(
      self,
      sync_request: base_sampler_lib.WeightSyncRequest | Any = None,
      **kwargs,
  ) -> str | None | Any:
    """Prepares staging handshake prior to policy weight update."""
    self._check_weight_sync_boundness()

    if getattr(self, "enable_raiden", False):
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

    if getattr(self, "enable_raiden", False):
      return await self.raiden_sync_delegate.weight_sync(
          sync_request=sync_request, **kwargs
      )
    else:
      if sync_request is None:
        raise ValueError(
            f"{self.__class__.__name__} Fallback mode [{self.server_id}] weight_sync:"
            " sync_request is None."
        )
      sampler = self._get_underlying_sampler()
      if sampler and hasattr(sampler, "update_params"):
        weights = getattr(sync_request, "weights", None)
        if weights is None:
          raise ValueError(
              f"{self.__class__.__name__} [{self.server_id}] weight_sync: weights not found"
              " in sync_request."
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
    if getattr(self, "enable_raiden", False):
      return await self.raiden_sync_delegate.post_weight_sync(
          sync_request=sync_request, **kwargs
      )
    return True

  async def abort_weight_sync(
      self,
      sync_request: base_sampler_lib.WeightSyncRequest | Any = None,
      **kwargs,
  ) -> str | None | Any:
    """Safely aborts weight sync round."""
    if getattr(self, "enable_raiden", False) and getattr(self, "raiden_sync_delegate", None):
      if hasattr(self.raiden_sync_delegate, "abort_weight_sync"):
        return await self.raiden_sync_delegate.abort_weight_sync(
            sync_request=sync_request, **kwargs
        )
    return True
