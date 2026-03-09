# Copyright 2025 Google LLC
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

"""Rollout worker group."""

import dataclasses
import enum
from typing import Any, Dict, Optional, Tuple

import jax
import jaxtyping
from tunix.rl.rollout import base_rollout


class RolloutTrafficRoutingConfig(enum.Enum):
  """Rollout traffic routing algorithm."""

  pass


@dataclasses.dataclass(kw_only=True, frozen=True)
class RolloutWeightSyncConfig:
  """Configuration for the weight sync strategy."""

  pass


@dataclasses.dataclass(kw_only=True, frozen=True)
class RolloutEngineGroupConfig:
  """Configuration for the rollout engine group."""

  # number of total rollout engines
  num_engines: int

  # config template for all engines
  engine_config_template: base_rollout.RolloutConfig

  # specify traffic routing algorithm
  # by default it's round-robin (based on trajectory-id, not per-request)
  traffic_routing_config: RolloutTrafficRoutingConfig

  # specify weight sync algorithm, by default it's round-robin
  weight_sync_config: RolloutWeightSyncConfig


class RolloutEngineGroup:
  """RolloutEngineGroup manages multiple rollout engines, and route traffic to them based on routing config."""

  def generate(
      self, prompts, rollout_config: base_rollout.RolloutConfig, **kwargs
  ) -> base_rollout.RolloutOutput:
    raise NotImplementedError("Not implemented for RolloutEngineGroup.")

  def get_per_token_logps(
      self,
      prompt_tokens: jax.Array,
      completion_tokens: jax.Array,
      completion_mask: jax.Array | None = None,
      **kwargs
  ) -> jax.Array:
    raise NotImplementedError("Not implemented for RolloutEngineGroup.")

  def update_params(
      self,
      params: jaxtyping.PyTree,
      filter_types: Optional[Tuple[Any, ...]] = None,
  ):
    raise NotImplementedError("Not implemented for RolloutEngineGroup.")

  def pad_id(self) -> int:
    raise NotImplementedError("Not implemented for RolloutEngineGroup.")

  def eos_id(self) -> int:
    raise NotImplementedError("Not implemented for RolloutEngineGroup.")

  def model(self) -> Any:
    # if use RolloutEngineGroup, we disable CPU offloading, which
    # is the only caller of this function.
    return None
