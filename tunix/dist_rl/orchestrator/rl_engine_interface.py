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

import asyncio
from collections.abc import Mapping, Sequence
import dataclasses
from typing import Any, Protocol, runtime_checkable
import numpy as np
from tunix.dist_rl.common import datatypes
from tunix.dist_rl.orchestrator import batch_assembly
from tunix.dist_rl.worker import remote_execution
from tunix.rl import algorithm_config


@dataclasses.dataclass(frozen=True)
class RoutedExpertsFetchResult:
  """Materialized staged `routed_experts` arrays plus unrecoverable refs.

  Attributes:
    fetched: Mapping from resolved `RoutedExpertsRef` handles to materialized
      wire-encoded `routed_experts` arrays.
    failed_refs: Set of `RoutedExpertsRef` handles whose rollout worker was
      evicted, non-`ACTIVE`, restarted, or unreachable under fault tolerance.
  """

  fetched: dict[datatypes.RoutedExpertsRef, np.ndarray]
  failed_refs: frozenset[datatypes.RoutedExpertsRef]


@runtime_checkable
class AbstractRLEngine(Protocol):
  """Stateless compute primitives for distributed worker meshes."""

  @property
  def restored_checkpoint_metadata(self) -> dict[str, Any] | None:
    """Metadata mapping from the most recent checkpoint restoration, if any."""
    ...

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
      num_generations: int = 1,
      policy_version: int = 0,
      route_metadata: Mapping[str, Any] | None = None,
      stage_routed_experts: bool = False,
      max_response_length: int | None = None,
  ) -> list[str]:
    """High-level convenience: Expands prompts by num_generations and dispatches rollouts.

    Contract & Invariants:
      1. Every item in `prompts` MUST have a unique, collision-free `prompt_id`
         (provided as an attribute `p.prompt_id` or dict key `p["prompt_id"]`).
         The engine does not synthesize fallback IDs; missing IDs raise a
         `ValueError` immediately.
      2. The engine expands each prompt into `num_generations` independent
         rollout requests with deterministic IDs
         `req_{prompt_id}_g{g_idx}_v{version}` (appending `_e{epoch}` after
         `clear_all_staged_routed_experts()` resets the rollout epoch) and sets
         `group_index = g_idx` (0..G-1).

    Args:
      prompts: Sequence of prompt items (dicts, objects, or RolloutRequests).
        Every prompt item MUST provide a unique `prompt_id`.
      num_generations: Number of rollout trajectories to generate per prompt
        (G).
      policy_version: Active policy version for generation.
      route_metadata: Optional routing metadata merged into each request's
        metadata.
      stage_routed_experts: Whether rollout workers stage MoE `routed_experts`
        arrays locally and return lightweight `RoutedExpertsRef` handles.
      max_response_length: Default response token budget stamped onto requests
        that omit `max_response_length`, and upper bound when
        `stage_routed_experts=True`.

    Returns:
      List of dispatched `request_id` strings.

    Raises:
      ValueError: If any item in `prompts` lacks a `prompt_id`, or if
        `stage_routed_experts=True` and a request's `max_response_length` is not
        in `(0, max_response_length]`.
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
      route_metadata: Mapping[str, Any] | None = None,
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

  # TODO(tunix-dev): Fuse `apply_optimizer=True` into a single atomic
  # Trainer RPC and attach `(step_id, mb_index, num_microbatches)` accounting on
  # TrainerWorker so lost/retried microbatches across reconnects are detected
  # end-to-end.
  async def train_step(
      self,
      payload: datatypes.RLTrainerPayload,
      role: datatypes.Role = datatypes.Role.ACTOR,
      accumulate_gradients: bool = False,
      apply_optimizer: bool = True,
  ) -> Any:
    """Executes forward/backward gradient update on trainer workers.

    Ordering Contract:
      When multiple `train_step` coroutines are started concurrently for the
      same `role` (e.g. double-buffered microbatch pipelining in `train_stage`),
      implementations MUST ensure that when `apply_optimizer=True`, all prior
      in-flight `fwd_bwd` calls for `role` have completed before `update` is
      invoked, and callers MUST drain in-flight `train_step` calls at every
      `apply_optimizer=True` boundary before starting subsequent microbatches.

    Args:
      payload: Microbatch payload for the trainer worker.
      role: Target worker role to execute the training step on.
      accumulate_gradients: Whether to accumulate gradients across microbatches.
      apply_optimizer: Whether to apply the optimizer update after `fwd_bwd`.

    Returns:
      Engine-specific training step result or metrics.
    """
    ...

  # TODO: b/552087289 - Generalize get_metrics to support querying metrics
  # across all worker roles (trainer, rollout, critique) or worker pools.
  async def get_metrics(
      self,
      role: datatypes.Role = datatypes.Role.ACTOR,
      **kwargs: Any,
  ) -> (
      datatypes.MetricsBuffer
      | Sequence[datatypes.MetricsBuffer]
      | dict[str, Any]
      | None
  ):
    """Retrieves step metrics from the worker(s) for the specified role.

    For `Role.ROLLOUT`, returns `[]` under fault tolerance when zero rollout
    workers are currently `ACTIVE` mid-run, skips workers failing with transport
    or closed-handle faults, and propagates `RemoteCallError` application
    exceptions (failing fast on all errors when fault tolerance is disabled).

    Args:
      role: Worker role whose metrics should be queried.
      **kwargs: Additional engine- or worker-specific metric query options.

    Returns:
      Sequence of per-worker metric payloads for `Role.ROLLOUT`, or the single
      worker's metric payload for singleton roles.
    """
    ...

  def configure_worker(
      self,
      role: datatypes.Role = datatypes.Role.ACTOR,
      *,
      algo_config: algorithm_config.AlgorithmConfig,
      assembler: batch_assembly.BatchAssembler[Any],
      **kwargs: Any,
  ) -> None:
    """Configures worker(s) under the specified role with algorithm or runtime settings."""
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
      *,
      min_retained_policy_version: int | None = None,
      source_prepared: asyncio.Event | None = None,
      **kwargs: Any,
  ) -> int:
    """Coordinates decentralized peer-to-peer weight sync across worker roles.

    Args:
      role: Source trainer role publishing updated weights.
      target_roles: Optional destination worker roles to synchronize.
      policy_version: Optional explicit target policy version.
      min_retained_policy_version: Optional zombie-eviction floor forwarded to
        rollout workers during `post_weight_sync`.
      source_prepared: Set once the trainer has staged its weights and may
        train again while the rest of the sync runs. The sync fails if its
        transport would read the trainer's live weights after that.
      **kwargs: Additional engine-specific sync options.

    Returns:
      The committed `policy_version`.
    """
    ...

  def routed_experts_spec(
      self,
      *,
      encode_mode: datatypes.RoutedExpertsEncodeMode = "auto",
      num_experts: int | None = None,
  ) -> batch_assembly.RoutedExpertsSpec:
    """Resolves unified MoE routing spec across active rollout workers.

    Pure query without mutating engine state.

    Args:
      encode_mode: Requested MoE routing wire encoding mode.
      num_experts: Total routed experts per layer when `encode_mode="auto"`.

    Returns:
      Resolved `RoutedExpertsSpec` shared across all active rollout workers.

    Raises:
      ValueError: If no rollout workers are active, or any worker omits or
        disagrees on `routed_experts_shape`.
    """
    ...

  def configure_staging_liveness(
      self,
      *,
      worst_case_bytes: int,
      required_slots: int,
      overcommit_factor: float = 1.0,
  ) -> None:
    """Validates and installs cluster-wide Relay staging slot liveness bounds.

    Validates active rollout workers before updating internal state so failed
    checks leave previous bounds intact. Reset by
    `clear_all_staged_routed_experts()`.

    Args:
      worst_case_bytes: Worst-case staged bytes reserved per rollout trajectory.
      required_slots: Minimum cluster-wide worst-case staging slots required to
        guarantee deadlock-free cohort progress.
      overcommit_factor: Soft orchestrator-side slot multiplier (`>= 1.0`).
        Values `> 1.0` relax the strict deadlock-free guarantee under the
        assumption that average trajectories are shorter than
        `max_response_length`.

    Raises:
      ValueError: If `worst_case_bytes <= 0`, `required_slots < 0`,
        `overcommit_factor` is not a finite float `>= 1.0`, or any active
        rollout worker omits `staging_max_bytes`.
      RuntimeError: If any active rollout worker has 0 physical staging slots
        (`staging_max_bytes < worst_case_bytes`), or if cluster-wide scaled
        staging slots are below `required_slots`.
    """
    ...

  async def fetch_routed_experts(
      self,
      refs: Sequence[datatypes.RoutedExpertsRef],
  ) -> RoutedExpertsFetchResult:
    """Fetches worker-staged `routed_experts` arrays grouped by worker instance.

    Args:
      refs: Staged `RoutedExpertsRef` handles to materialize.

    Returns:
      `RoutedExpertsFetchResult` with materialized arrays in `fetched` and, when
      fault tolerance is enabled, any refs from evicted, non-`ACTIVE`,
      restarted, or unreachable rollout workers in `failed_refs`.

    Raises:
      RuntimeError: If fault tolerance is disabled and a referenced rollout
        worker was evicted, non-`ACTIVE`, or restarted, or if an application
        error occurs on `fetch_staged`.
      KeyError: If an active matching rollout worker omits a requested `ref_id`.
    """
    ...

  async def release_routed_experts(
      self,
      refs: Sequence[datatypes.RoutedExpertsRef],
  ) -> None:
    """Releases worker-staged `routed_experts` entries grouped by worker instance.

    Args:
      refs: Staged `RoutedExpertsRef` handles to free (evicted, non-`ACTIVE`, or
        instance-mismatched workers—and unreachable workers when fault tolerance
        is enabled—are logged and skipped).

    Raises:
      RuntimeError: If releasing more refs for a `(worker_id, instance_id)` than
        currently tracked as unreleased.
    """
    ...

  async def clear_all_staged_routed_experts(self) -> None:
    """Abandons in-flight rollouts and clears staged routing on workers."""
    ...

  def fault_tolerance_metrics(self) -> Mapping[str, float]:
    """Returns fault-tolerance metrics (`ft/*`) to log with each train step."""
    ...

  async def save_checkpoint(
      self,
      role: datatypes.Role = datatypes.Role.ACTOR,
      metadata: Any = None,
      *,
      step: int | None = None,
      force: bool = False,
  ) -> Any:
    """Requests the trainer worker for `role` to save a checkpoint."""
    ...

  async def resume_from_checkpoint(
      self,
      role: datatypes.Role = datatypes.Role.ACTOR,
  ) -> int:
    """Restores the trainer checkpoint for `role` and aligns engine policy state.

    Restores the trainer checkpoint for `role` and aligns the engine's internal
    policy version and restored checkpoint metadata with the restored step
    boundary. Callers synchronize rollout worker weights separately via
    `prepare_rollout_policy`.

    Args:
      role: Trainer role whose checkpoint should be restored.

    Returns:
      The restored step.
    """
    ...
