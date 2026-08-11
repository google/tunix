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

"""The RL engine interface (Layer 1 Compute Routing Protocol) following Orchestrator V2."""

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable
from tunix.experimental.common import datatypes


@runtime_checkable
class AbstractRLEngine(Protocol):
  """Stateless compute primitives for distributed worker meshes."""

  async def dispatch_rollouts(
      self, prompts: Sequence[Any], **kwargs: Any
  ) -> list[str]:
    """Dispatches rollout requests across workers (constructing RolloutRequests internally)."""
    ...

  async def poll_rollouts(
      self, timeout_s: float = 0.1
  ) -> list[datatypes.TrajectoryItem]:
    """Retrieves completed rollout responses from workers via long-polling."""
    ...

  async def generate(
      self,
      prompts: Sequence[Any],
      generation_args: datatypes.GenerationArgs | None = None,
      metadata: Mapping[str, Any] | None = None,
      **kwargs: Any,
  ) -> list[datatypes.TrajectoryItem]:
    """Synchronous batched rollout generation over rollout workers."""
    ...

  async def score(
      self, role: datatypes.Role, items: Sequence[Any], **kwargs: Any
  ) -> list[float]:
    """Scores responses under a reward model."""
    ...

  async def per_token_logps(
      self, role: datatypes.Role, items: Sequence[Any], **kwargs: Any
  ) -> Any:
    """Computes per-token log probabilities under a reference/actor model."""
    ...

  async def train_step(
      self,
      payload: datatypes.RLTrainerPayload,
      role: datatypes.Role = datatypes.Role.ACTOR,
      accumulate_gradients: bool = False,
      apply_optimizer: bool = True,
      skip_jit: bool = False,
      **kwargs: Any,
  ) -> dict[str, Any]:
    """Executes forward/backward gradient update on trainer workers."""
    ...

  async def sync_weights(
      self,
      role: datatypes.Role = datatypes.Role.ACTOR,
      target_roles: Sequence[datatypes.Role] | None = None,
      **kwargs: Any,
  ) -> int:
    """Coordinates decentralized peer-to-peer weight sync across worker roles."""
    ...
