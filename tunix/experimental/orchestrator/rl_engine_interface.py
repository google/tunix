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
from tunix.experimental.metrics import metrics as exp_metrics
from tunix.experimental.worker import remote_execution


@runtime_checkable
class AbstractRLEngine(Protocol):
  """Stateless compute primitives for distributed worker meshes."""

  async def dispatch_rollout_requests(
      self,
      requests: Sequence[datatypes.RolloutRequest],
  ) -> list[str]:
    """Low-level primitive: Dispatches pre-formed RolloutRequests using prefix routing.

    Callers must ensure that all requests have unique, collision-free
    `request_id` and `prompt_id` attributes.

    Args:
      requests: Sequence of pre-formed `datatypes.RolloutRequest` DTOs.

    Returns:
      List of dispatched `request_id` strings.
    """
    ...

  async def dispatch_rollouts(
      self,
      prompts: Sequence[Any],
      *,
      group_size: int = 1,
      policy_version: int = 0,
      generation_args: datatypes.GenerationArgs | None = None,
      route_metadata: Mapping[str, Any] | None = None,
      **kwargs: Any,
  ) -> list[str]:
    """High-level convenience: Expands prompts by group_size and dispatches rollouts.

    Contract & Invariants:
      1. Every item in `prompts` MUST have a unique, collision-free `prompt_id`
         (provided as an attribute `p.prompt_id` or dict key `p["prompt_id"]`).
         The engine does not synthesize fallback IDs; missing IDs raise a
         `ValueError` immediately.
      2. The engine expands each prompt into `group_size` independent rollout
         requests with deterministic IDs `req_{prompt_id}_g{g_idx}_v{version}`
         and sets `group_index = g_idx` (0..G-1).

    Args:
      prompts: Sequence of prompt items (dicts, objects, or RolloutRequests).
        Every prompt item MUST provide a unique `prompt_id`.
      group_size: Number of rollout trajectories to generate per prompt (G).
      policy_version: Active policy version for generation.
      generation_args: Optional generation parameters (temperature, max steps).
      route_metadata: Optional routing metadata (e.g. `prefix_hash`).
      **kwargs: Optional additional metadata.

    Returns:
      List of dispatched `request_id` strings.

    Raises:
      ValueError: If any item in `prompts` lacks a `prompt_id`.
    """
    ...

  async def poll_rollouts(
      self, timeout_s: float = remote_execution.LONG_POLL_TIMEOUT_S
  ) -> list[datatypes.TrajectoryItem]:
    """Retrieves completed rollout responses from workers via long-polling."""
    ...

  async def generate(
      self,
      prompts: Sequence[Any],
      generation_args: datatypes.GenerationArgs | None = None,
      route_metadata: Mapping[str, Any] | None = None,
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
      self, role: datatypes.Role, items: Any, **kwargs: Any
  ) -> Any:
    """Computes per-token log probabilities for a padded batch/request."""
    ...

  async def train_step(
      self,
      payload: datatypes.RLTrainerPayload,
      role: datatypes.Role = datatypes.Role.ACTOR,
      accumulate_gradients: bool = False,
      apply_optimizer: bool = True,
      skip_jit: bool = False,
      **kwargs: Any,
  ) -> Any:
    """Executes forward/backward gradient update on trainer workers."""
    ...

  # TODO: b/552087289 - Generalize get_metrics to support querying metrics
  # across all worker roles (trainer, rollout, critique) or worker pools.
  async def get_metrics(
      self,
      role: datatypes.Role = datatypes.Role.ACTOR,
      **kwargs: Any,
  ) -> (
      exp_metrics.MetricsBuffer
      | Sequence[exp_metrics.MetricsBuffer]
      | dict[str, Any]
      | None
  ):
    """Retrieves step metrics from the worker for the specified role."""
    ...

  async def prepare_rollout_policy(
      self,
      role: datatypes.Role = datatypes.Role.ACTOR,
      sync_weights: bool = True,
      policy_version: int | None = None,
      **kwargs: Any,
  ) -> int | None:
    """Bootstraps rollout-facing policy state before step 0 dispatch begins."""
    ...

  async def sync_weights(
      self,
      role: datatypes.Role = datatypes.Role.ACTOR,
      target_roles: Sequence[datatypes.Role] | None = None,
      policy_version: int | None = None,
      **kwargs: Any,
  ) -> int:
    """Coordinates decentralized peer-to-peer weight sync across worker roles."""
    ...

  async def save_checkpoint(
      self,
      role: datatypes.Role = datatypes.Role.ACTOR,
      metadata: Any = None,
      **kwargs: Any,
  ) -> Any:
    """Requests the trainer worker for `role` to save a checkpoint."""
    ...
