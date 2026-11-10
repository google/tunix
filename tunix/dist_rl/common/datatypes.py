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

"""Common data types and DTOs for the Tunix Orchestrator and Workers.

This module centralizes type aliases and dataclasses used for:
1) Routing data and commands between Orchestrator and workers.
2) Defining common data structures used by Orchestrator and workers.
"""

from collections.abc import Callable, Mapping
import dataclasses
import enum
import math
import time
from typing import Any, Dict, Protocol
import uuid
import numpy as np
from tunix.common import datatypes as common_datatypes
from tunix.rl.agentic.agents import agent_types

##### Worker-internal datatypes #####

# Worker-internal episode representation produced during rollout.
Trajectory = agent_types.Trajectory
Step = agent_types.Step
TrajectoryStatus = agent_types.TrajectoryStatus
TrajectoryItem = agent_types.TrajectoryItem
format_traj_id = agent_types.format_traj_id
Role = common_datatypes.Role
ROUTED_EXPERTS_ENCODE_MODES = common_datatypes.ROUTED_EXPERTS_ENCODE_MODES
RoutedExpertsEncodeMode = common_datatypes.RoutedExpertsEncodeMode
validate_routed_experts_encode_config = (
    common_datatypes.validate_routed_experts_encode_config
)

# Marks a router-replay slot the trainer must not replay, so the model falls
# back to its own gate there. Matches what MaxText's replay path expects.
UNSET_ROUTED_EXPERT = -1


# Re-export assistant_text and prompt_text from canonical agent_types module.
assistant_text = agent_types.assistant_text
prompt_text = agent_types.prompt_text


##### Common DTOs (Data Transfer Objects) #####


@dataclasses.dataclass(kw_only=True)
class ErrorInfo:
  """Structured description of a failed request, carried in-band on a result.

  Attributes:
    error_type: Short classifier for the failure (e.g. an exception class name).
    message: Human-readable failure description.
    retryable: Whether re-issuing the request could plausibly succeed.
    traceback: Optional captured traceback, for diagnostics.
  """

  error_type: str
  message: str
  retryable: bool = False
  traceback: str = ""


def _generate_request_id() -> str:
  return f"req_{uuid.uuid4().hex[:12]}"


@dataclasses.dataclass(kw_only=True)
class Request:
  """Standard base for generic RPC requests.

  Attributes:
    request_id: Unique identifier for this request, echoed back on the
      corresponding response so callers can correlate responses.
    metadata: Optional free-form data attached to the request.
  """

  request_id: str = dataclasses.field(default_factory=_generate_request_id)
  metadata: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(kw_only=True)
class Response:
  """Standard response for generic RPC requests.

  Attributes:
    request_id: Echoes the originating request_id for correlation.
    error: Structured failure details when the operation failed, else None.
    metadata: Optional free-form data attached to the response.
  """

  request_id: str = ""
  error: ErrorInfo | None = None
  metadata: dict[str, Any] = dataclasses.field(default_factory=dict)


class WorkerState(str, enum.Enum):
  """Worker lifecycle states.

  Attributes:
    PENDING: Worker is created but not yet initialized.
    INITIALIZING: Worker is currently allocating resources and running setup.
    COMPILING: Worker is compiling models or graphs for execution.
    READY: Worker is fully initialized and ready to accept requests.
    SYNCING: Worker is synchronizing model weights or policies.
    DRAINING: Worker is gracefully shutting down and finishing pending requests.
    STOPPED: Worker is stopped and no longer accepting requests.
    ERROR: Worker encountered an unrecoverable error.
  """

  PENDING = "PENDING"
  INITIALIZING = "INITIALIZING"
  COMPILING = "COMPILING"
  READY = "READY"
  SYNCING = "SYNCING"
  DRAINING = "DRAINING"
  STOPPED = "STOPPED"
  ERROR = "ERROR"

  def can_transition_to(self, new_state: "WorkerState") -> bool:
    """Checks if the transition to the new state is valid."""
    return new_state in _ALLOWED_TRANSITIONS.get(self, set())


_ALLOWED_TRANSITIONS: dict[WorkerState, set[WorkerState]] = {
    WorkerState.PENDING: {
        WorkerState.INITIALIZING,
        WorkerState.STOPPED,
        WorkerState.ERROR,
    },
    WorkerState.INITIALIZING: {
        WorkerState.READY,
        WorkerState.STOPPED,
        WorkerState.ERROR,
    },
    WorkerState.COMPILING: {
        WorkerState.READY,
        WorkerState.STOPPED,
        WorkerState.ERROR,
    },
    WorkerState.READY: {
        WorkerState.COMPILING,
        WorkerState.SYNCING,
        WorkerState.DRAINING,
        WorkerState.STOPPED,
        WorkerState.ERROR,
    },
    WorkerState.SYNCING: {
        WorkerState.READY,
        WorkerState.STOPPED,
        WorkerState.ERROR,
    },
    WorkerState.DRAINING: {WorkerState.STOPPED, WorkerState.ERROR},
    WorkerState.STOPPED: set(),
    WorkerState.ERROR: {WorkerState.STOPPED},
}


class AdmissionStatus(enum.Enum):
  """Control-plane admission status of a registered worker.

  Tracked by the orchestrator to decide which workers may receive data-plane
  traffic. Distinct from `WorkerState`, the lifecycle state a worker
  self-reports via `heartbeat()`: a worker can report `READY` while still
  `PENDING_WEIGHT_SYNC` here.

  Attributes:
    REGISTERED: Registered but not yet brought up by `LifecycleDriver`.
    PENDING_WEIGHT_SYNC: Brought up and awaiting its first committed weight
      sync; a weight sync destination but not routable.
    ACTIVE: Routable by the data plane.
  """

  REGISTERED = "registered"
  PENDING_WEIGHT_SYNC = "pending_weight_sync"
  ACTIVE = "active"


@dataclasses.dataclass(kw_only=True)
class HealthReport:
  """A snapshot of a worker's health and readiness state.

  Attributes:
    state: The current lifecycle state (e.g., WorkerState.READY).
    inflight: Number of active requests currently being processed.
    queue_depth: Number of pending requests queued by the worker.
    policy_version: The version of the weights currently loaded.
    last_error: A string summarizing the most recent error, if any.
    heartbeat_unix_s: The unix timestamp when this report was generated.
  """

  state: WorkerState
  inflight: int = 0
  queue_depth: int = 0
  policy_version: int = 0
  last_error: str | None = None
  heartbeat_unix_s: float = dataclasses.field(default_factory=time.time)


@dataclasses.dataclass(kw_only=True)
class WorkerInfo:
  """Static metadata describing a worker's identity and capabilities.

  Attributes:
    worker_id: The unique identifier for this worker.
    roles: The orchestrator roles this worker can serve (e.g., "trainer",
      "rollout").
    resources: Unstructured dictionary of hardware or configuration details
      (e.g., tokenizer_hash, fsdp_size) used during startup validation.
  """

  worker_id: str
  roles: frozenset[str] = frozenset()
  resources: dict[str, Any] = dataclasses.field(default_factory=dict)


# Default poll retry budgets used when `poll_retry_budget_s` is None.
# `_FT_OFF_DEFAULT_POLL_RETRY_BUDGET_S` must stay in sync with
# `task_session.DEFAULT_POLL_RETRY_BUDGET_S` (verified by
# `distributed_rl_engine_test`).
_FT_OFF_DEFAULT_POLL_RETRY_BUDGET_S = 300.0
_FT_ON_DEFAULT_POLL_RETRY_BUDGET_S = 60.0


@dataclasses.dataclass(frozen=True, kw_only=True)
class FaultToleranceConfig:
  """Configuration for worker fault tolerance.

  Attributes:
    enabled: Whether worker fault tolerance (automatic eviction and task retry)
      is enabled; defaults to `True`. Transient transport errors are retried in
      place on the same worker—up to 3 total attempts (1 initial send + 2
      serialized probes, ~1.5s default backoff, capped by
      `effective_poll_retry_budget_s`) on `dispatch_task`, and up to
      `effective_poll_retry_budget_s` (`60.0s` by default) on
      `poll_responses`—before evicting the worker and re-dispatching its
      in-flight tasks (`ft/rollout/task_retries_total` counts both cross-worker
      re-dispatches and in-place retry probes/resends). When `False`, fault
      tolerance is disabled (`effective_max_task_retries` is `0` and
      `min_active_rollout_ratio` is ignored). Known limitations when `True`:
      weight_sync: `TaskSession` detects failures only on workers holding
        in-flight tasks, so a rollout worker that dies with none (after its last
        delivery and before or during the next weight-sync round) is noticed
        only by the next weight-sync round, which fails and aborts the program,
        as with `enabled=False`. If the worker restarts and re-registers through
        discovery before that round starts, it re-enters `REGISTERED` ->
        `PENDING_WEIGHT_SYNC` and rejoins without failing the round (provided at
        least one rollout worker is `ACTIVE` or `PENDING_WEIGHT_SYNC` when the
        round starts). TODO(b/577214994): Add weight-sync fault tolerance after
        the weight-sync redesign.
      sampler_errors: Detects worker failures only via transport errors (e.g.
        process exit or unreachability). If the worker process stays up, a hung
        generation never returns (the first turn has no episode timeout) and
        sampler errors return `FAILED` trajectories without evicting the worker.
        TODO(b/577214994): Make fatal sampler errors exit the worker process so
        they surface as worker failures.
      live_evicted_workers: Rollout workers register with discovery only once at
        startup, so a live worker evicted after exhausting its retry budget does
        not rejoin (sustained pool shrink is bounded by
        `min_active_rollout_ratio`).
      poison_prompts: A prompt that deterministically crashes workers evicts up
        to `max_task_retries + 1` workers before surfacing the terminal error
        and aborting the program (as with `enabled=False`).
    max_task_retries: Maximum number of times an in-flight task is re-dispatched
      onto surviving workers after a worker failure when `enabled=True`. Because
      a task is re-dispatched only after its worker is evicted,
      `max_task_retries` also bounds the number of workers a single request can
      evict; defaults to `1`.
    poll_retry_budget_s: Optional override for the maximum wall-clock duration
      in seconds to retry transient `poll_responses` errors (and cap in-place
      `dispatch_task` retry backoff when `enabled=True`) on a rollout worker
      before failing its in-flight tasks (`0.0` fails immediately on the first
      transient error). When `None` (default), `effective_poll_retry_budget_s`
      resolves to `60.0` when `enabled=True` (gRPC keepalive ~30s plus margin;
      bounds step stall when a pod is rescheduled to a new IP) and `300.0` when
      `enabled=False` (matching `TaskSession`'s standalone default budget).
    min_active_rollout_ratio: Minimum ratio of currently `ACTIVE` rollout
      workers to the peak `ACTIVE` rollout worker count observed since startup,
      in `(0.0, 1.0]`, or `None` to disable the capacity-floor guardrail;
      defaults to `0.5`. When `enabled=True` and set, `FaultToleranceManager`
      raises `RolloutCapacityError` once `active / peak_active <
      min_active_rollout_ratio` (strictly below; e.g. `1 / 2` and `2 / 4` at
      `0.5` do not trigger) continuously for at least
      `min_active_rollout_grace_s` seconds after the first rollout worker has
      reached `ACTIVE`. Ignored when `enabled=False`.
    min_active_rollout_grace_s: Wall-clock grace period in seconds (`>= 0.0`,
      default `900.0`) that the active rollout ratio may remain strictly below
      `min_active_rollout_ratio` before `RolloutCapacityError` is raised,
      allowing crashed workers time to be restarted and rejoin. The grace timer
      starts on worker eviction, while the floor is checked at
      `DistributedRLEngine` construction, after each `poll_rollouts()`, and
      after `sync_weights()` promotion, so `RolloutCapacityError` surfaces on
      the first poll or weight-sync boundary once `min_active_rollout_grace_s`
      has elapsed.
  """

  enabled: bool = True
  max_task_retries: int = 1
  poll_retry_budget_s: float | None = None
  min_active_rollout_ratio: float | None = 0.5
  min_active_rollout_grace_s: float = 900.0

  def __post_init__(self) -> None:
    if self.max_task_retries < 0:
      raise ValueError(
          f"max_task_retries must be >= 0, got {self.max_task_retries}"
      )
    if self.poll_retry_budget_s is not None and self.poll_retry_budget_s < 0.0:
      raise ValueError(
          f"poll_retry_budget_s must be >= 0, got {self.poll_retry_budget_s}"
      )
    if self.min_active_rollout_ratio is not None and not (
        0.0 < self.min_active_rollout_ratio <= 1.0
    ):
      raise ValueError(
          "min_active_rollout_ratio must be in (0.0, 1.0], got"
          f" {self.min_active_rollout_ratio}"
      )
    if (
        math.isnan(self.min_active_rollout_grace_s)
        or self.min_active_rollout_grace_s < 0.0
    ):
      raise ValueError(
          "min_active_rollout_grace_s must be >= 0, got"
          f" {self.min_active_rollout_grace_s}"
      )

  @property
  def effective_max_task_retries(self) -> int:
    """Returns the active task retry budget (`0` when `enabled=False`)."""
    return self.max_task_retries if self.enabled else 0

  @property
  def effective_poll_retry_budget_s(self) -> float:
    """Returns the resolved poll retry budget in seconds."""
    if self.poll_retry_budget_s is not None:
      return self.poll_retry_budget_s
    return (
        _FT_ON_DEFAULT_POLL_RETRY_BUDGET_S
        if self.enabled
        else _FT_OFF_DEFAULT_POLL_RETRY_BUDGET_S
    )


##### Rollout DTOs #####


@dataclasses.dataclass(frozen=True, kw_only=True)
class GenerationArgs:
  """Typed per-turn generation arguments used by the orchestrator generate API.

  This holds per-turn sampling and generation configuration (as opposed to
  episode-level configuration like `max_turns` or `max_response_length` on
  `RolloutRequest`).

  Attributes:
    max_generation_steps: Maximum number of tokens to generate in a single turn.
    temperature: Sampling temperature for generation.
    top_p: Nucleus sampling probability threshold.
    top_k: Top-k sampling cutoff.
    seed: Random seed for reproducible generation.
    return_logprobs: Whether to return token log probabilities.
  """

  max_generation_steps: int | None = None
  temperature: float | None = None
  top_p: float | None = None
  top_k: int | None = None
  seed: int | None = None
  return_logprobs: bool | None = None
  return_routed_experts: bool | None = None

  def as_kwargs(self) -> dict[str, Any]:
    return {
        field.name: getattr(self, field.name)
        for field in dataclasses.fields(self)
        if getattr(self, field.name) is not None
    }


@dataclasses.dataclass(kw_only=True)
class RolloutRequest(Request):
  """Request to generate a rollout from a given prompt.

  Attributes:
    prompt: The prompt to generate from (e.g. formatted string, token array, or
      chat dictionary).
    prompt_id: Unique identifier for this prompt within a task or dataset.
    group_index: Optional index within a group for group-based algorithms (e.g.,
      GRPO). Defaults to 0 for ungrouped.
    generation_kwargs: Additional per-turn keyword arguments for generation
      (e.g. sampling parameters like max_generation_steps and temperature).
    max_turns: Maximum number of conversation turns for environment interaction.
    max_response_length: Optional cumulative response token budget across all
      turns of the rollout episode.
    target_policy_version: Policy model version identifier to use for rollout
      generation.
  """

  prompt: Any = ""
  prompt_id: str = ""
  group_index: int = 0
  generation_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
  max_turns: int = 10
  max_response_length: int | None = None
  target_policy_version: int = 0

  @property
  def traj_id(self) -> str:
    """Standardized trajectory identifier: traj_{prompt_id}_g{group_index}."""
    return format_traj_id(self.prompt_id, self.group_index)


@dataclasses.dataclass(kw_only=True)
class TokenSegment:
  """One contiguous span of the conversation token stream representing a single turn.

  Each segment corresponds to a single turn's response from either the assistant
  or the environment.

  Attributes:
    source: Origin of the span, e.g. "assistant" (model-emitted) or "env".
    tokens: Array of token ids for this span.
    loss_mask: Array of ints, 1 where the token is model-emitted (trainable).
    logps: Array of per-token log-probabilities under the sampling distribution,
      or None for spans the model did not emit (e.g. env tokens).
    routed_experts: `[len(tokens), num_layers, top_k]` raw `int16`/`int32` MoE
      expert ids this span was routed through (`UNSET_ROUTED_EXPERT == -1` where
      unset), so training can replay the routing the rollout actually used.
      After episode completion, `TrajectoryCollectorEngine` may compact-encode
      the concatenated episode routing on `TrajectoryItem.routed_experts` via
      `packing.encode_routed_experts`. None for dense models, spans the model
      did not emit, or when the sampler was not asked to capture routing.
  """

  source: str
  tokens: np.ndarray
  loss_mask: np.ndarray
  logps: np.ndarray | None = None
  routed_experts: np.ndarray | None = None

  def __post_init__(self):
    if self.loss_mask.shape != self.tokens.shape:
      raise ValueError(
          f"loss_mask shape {self.loss_mask.shape} != tokens shape"
          f" {self.tokens.shape}"
      )
    if self.logps is not None and self.logps.shape != self.tokens.shape:
      raise ValueError(
          f"logps shape {self.logps.shape} != tokens shape {self.tokens.shape}"
      )
    if self.routed_experts is not None:
      # The trailing axes are [num_layers, top_k] and are model-dependent, so
      # only the rank and the leading (per-token) axis are checked.
      if self.routed_experts.ndim != 3:
        raise ValueError(
            "routed_experts must be [length, num_layers, top_k]; got shape"
            f" {self.routed_experts.shape}"
        )
      if self.routed_experts.shape[0] != self.tokens.shape[0]:
        raise ValueError(
            f"routed_experts shape {self.routed_experts.shape} does not cover"
            f" tokens shape {self.tokens.shape} along axis 0"
        )


@dataclasses.dataclass(kw_only=True)
class RolloutResponse(Response):
  """Serializable result of a rollout generation request carrying a TrajectoryItem payload.

  Attributes:
    status: Terminal status name (e.g. "COMPLETED", "ERROR", "TIMEOUT",
      "CANCELLED").
    payload: TrajectoryItem carrying episode trajectory, token arrays, masks,
      and metadata.
  """

  status: str = "COMPLETED"
  payload: TrajectoryItem | None = None


##### Weight Sync DTOs #####


@dataclasses.dataclass(kw_only=True)
class WeightSyncRequest(Request):
  """Configuration and routing metadata for synchronizing policy model weights.

  Attributes:
    controller_id: Optional identifier for transport controllers (e.g., TPU
      Raiden).
    policy_version: Target policy version identifier of the weights to sync.
    weights: Optional source weights payload for non-Raiden / fallback sync.
    source_metadata: Optional transport/layout metadata describing source
      weights.
    extra_config: Optional backend-specific configuration parameters.
  """

  controller_id: str = ""
  policy_version: int = 0
  weights: Any = None
  source_metadata: Any = None
  extra_config: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class WeightSyncMetadata:
  """Metadata passed by Orchestrator during Phase 3 decentralized weight synchronization.

  Used to coordinate peer-to-peer (P2P) or broadcast weight transfers between
  TPU Trainer slices and RolloutWorker inference pods without routing large
  model weights through the central orchestrator.

  Attributes:
    worker_ips: List of target rollout worker IP addresses participating in
      sync.
    data_ports: Corresponding list of high-speed data transfer ports for each
      worker.
    ctrl_ports: List of control plane ports for coordination.
    mesh: Optional device mesh object for sharded tensor transfer.
    layout: Optional tensor memory sharding layout dimensions.
    new_policy_version: The updated policy weight version index being
      synchronized.
    transfer_mode: Transport mechanism option for decentralized weight sync: -
      "p2p" (default): Direct peer-to-peer transfer (e.g., direct RPC, Ray, or
      Raiden KV/weight streaming between source Trainer pods and target
      RolloutWorker pods using P2P DMA). - "broadcast" (Not implemented):
      Collective/tree broadcast (e.g., distributing weights across a pod mesh
      via collective communication like NCCL or Gloo over DMA links). - "fs"
      (Not implemented): Shared filesystem checkpoint transfer (where trainers
      save checkpoint weights to a shared filesystem like CNS/POSIX and rollout
      workers reload from disk).
    source_endpoints: Host/port endpoints of the source trainer pods serving the
      weights.
    sharding_topology: Optional device mesh sharding layout (e.g., {"mesh": [2,
      2]}).
  """

  worker_ips: list[str] = dataclasses.field(default_factory=list)
  data_ports: list[str] = dataclasses.field(default_factory=list)
  ctrl_ports: list[str] = dataclasses.field(default_factory=list)
  mesh: Any = None
  layout: list[int] = dataclasses.field(default_factory=list)
  new_policy_version: int = 0
  transfer_mode: str = "p2p"
  source_endpoints: list[str] = dataclasses.field(default_factory=list)
  sharding_topology: Dict[str, Any] = dataclasses.field(default_factory=dict)

  def __post_init__(self):
    if self.transfer_mode in ("broadcast", "fs"):
      raise NotImplementedError(
          f"transfer_mode='{self.transfer_mode}' is not implemented yet."
          " Currently only 'p2p' is supported."
      )
    elif self.transfer_mode != "p2p":
      raise ValueError(f"Unknown transfer_mode: '{self.transfer_mode}'")


##### Training DTOs #####
#
# Trainer payloads are plain frozen dataclasses so this module stays free of
# any ML framework: the orchestrator builds them from numpy arrays (the wire
# contract, see `rpc_utils.validate_wire_safe`) and ships them unchanged.
# Framework-specific container behaviour is registered next to the trainer
# that needs it; for JAX, importing `tunix.dist_rl.train.jax_pytrees`
# registers them as pytrees, after which a trainer may hold device arrays in
# the same fields. Use `dataclasses.replace` to derive modified copies.

# Marks a payload field as static configuration rather than array data. Array
# fields are the per-batch tensors a trainer traces / shards / moves between
# devices; static fields describe the batch (e.g. a packing bound) and must
# stay Python values. `jax.tree_util.register_dataclass` reads this exact
# metadata key to pick `meta_fields`, so JAX needs no separate field list.
STATIC_FIELD_METADATA = {"static": True}


@dataclasses.dataclass(frozen=True, kw_only=True)
class TrainerPayload:
  """Base abstract class for generic trainer payloads."""


@dataclasses.dataclass(frozen=True, kw_only=True)
class SFTTrainerPayload(TrainerPayload):
  """Supervised Fine-Tuning (SFT) trainer payload.

  Attributes:
    token_ids: [B, T] token IDs for a batched trainer payload. By default, each
      row is structured as left-padded prompt tokens concatenated with
      right-padded completion tokens.
    token_mask: [B, T] token mask to differentiate padding tokens from valid
      tokens.
    segment_ids: Optional [B, T] packing segment ids.
    segment_positions: Optional [B, T] position indices within each segment.
  """

  token_ids: np.ndarray
  token_mask: np.ndarray
  segment_ids: np.ndarray | None = None
  segment_positions: np.ndarray | None = None


# TODO(tunix-dev): Introduce PPOTrainerPayload to replace generic
# RLTrainerPayload when PPO specific fields are needed.
@dataclasses.dataclass(frozen=True, kw_only=True)
class RLTrainerPayload(TrainerPayload):
  """RL training payload.

  Attributes:
    advantages: [B] or [B, C] advantages.
    prompt_ids: Optional prompt token ids for GRPO-style losses. Unbatched
      payloads may carry 1D unpadded rows; batch assembly pads them to [B, P].
    prompt_mask: Optional [B, P] prompt mask.
    completion_ids: Optional completion token ids. Unbatched payloads may carry
      1D unpadded rows; batch assembly pads them to [B, C].
    completion_mask: Optional [B, C] completion/action mask.
    segment_ids: Optional [B, T] or [B, C] packing segment ids.
    segment_positions: Optional [B, T] or [B, C] position indices within each
      segment.
    ref_per_token_logps: Optional [B, C] reference model log-probabilities.
    old_per_token_logps: Optional [B, C] behavior policy log-probabilities.
    sampler_is_weights: Optional [B, C] importance sampling weights.
    routed_experts: Optional MoE expert ids captured during rollout. May be
      encoded in one of three wire representations: (1) raw `int16` of shape
      `[T, num_layers, top_k]` (unbatched) or `[B, T, num_layers, top_k]`
      (batched) where `UNSET_ROUTED_EXPERT == -1` marks a padded/unused slot;
      (2) `+1`-shifted `uint8` of shape `[T, num_layers, top_k]` or `[B, T,
      num_layers, top_k]` where `0` marks an unset slot and `1..255` represent
      expert ids `0..254`; or (3) `+1`-shifted `uint10` packed 4 experts per 5
      `uint8` bytes of shape `[T, num_layers, top_k // 4, 5]` (unbatched) or
      `[B, T, num_layers, top_k // 4, 5]` (batched) where `0` marks an unset
      slot and `1..1023` represent expert ids `0..1022`. Readers must decode via
      `tunix.rl.packing.decode_routed_experts` (on CPU) or
      `tunix.rl.common.decode_routed_experts_jax` (on device/XLA) rather than
      indexing raw bytes directly.
    returns: Optional [B, C] value baseline returns (for PPO / Critic).
    old_values: Optional [B, C] critic value estimates (for PPO / Critic).
    num_segments: Optional static upper bound on number of segments in packed
      rows (static field, see `STATIC_FIELD_METADATA`).
    sampler_agreement_applied: Whether the orchestrator already scored the batch
      against the trainer (static field).
    metadata: Extra payload metadata dictionary (static field).
  """

  prompt_ids: np.ndarray
  prompt_mask: np.ndarray
  completion_ids: np.ndarray
  completion_mask: np.ndarray
  advantages: np.ndarray
  segment_ids: np.ndarray | None = None
  segment_positions: np.ndarray | None = None
  ref_per_token_logps: np.ndarray | None = None
  old_per_token_logps: np.ndarray | None = None
  sampler_is_weights: np.ndarray | None = None
  routed_experts: np.ndarray | None = None
  returns: np.ndarray | None = None
  old_values: np.ndarray | None = None
  num_segments: int | None = dataclasses.field(
      default=None, metadata=STATIC_FIELD_METADATA
  )
  sampler_agreement_applied: bool = dataclasses.field(
      default=False, metadata=STATIC_FIELD_METADATA
  )
  metadata: dict[str, Any] = dataclasses.field(
      default_factory=dict, metadata=STATIC_FIELD_METADATA
  )
  # TODO(tunix-dev): add ppo specific fields in PPORLTrainerPayload.


@dataclasses.dataclass(frozen=True, kw_only=True)
class LogprobsScoringPayload(TrainerPayload):
  """Read-only payload for actor-side per-token log-probability scoring.

  Attributes:
    prompt_ids: [B, P] prompt token IDs.
    completion_ids: [B, C] completion (or packed sequence) token IDs.
    pad_id: Padding token ID (static field, see `STATIC_FIELD_METADATA`).
    eos_id: End-of-sequence token ID (static field, see
      `STATIC_FIELD_METADATA`).
    temperature: Softmax temperature used when scoring log-probs (static field,
      see `STATIC_FIELD_METADATA`).
    segment_ids: Optional [B, T] packing segment ids.
    segment_positions: Optional [B, T] position indices within each segment.
    routed_experts: Optional MoE expert ids captured during rollout.
  """

  prompt_ids: np.ndarray
  completion_ids: np.ndarray
  pad_id: int = dataclasses.field(metadata=STATIC_FIELD_METADATA)
  eos_id: int = dataclasses.field(metadata=STATIC_FIELD_METADATA)
  temperature: float = dataclasses.field(metadata=STATIC_FIELD_METADATA)
  segment_ids: np.ndarray | None = None
  segment_positions: np.ndarray | None = None
  routed_experts: np.ndarray | None = None


@dataclasses.dataclass(kw_only=True)
class TrainRequest(Request):
  """Request to execute training (forward/backward or eval) on a TrainerWorker.

  Attributes:
    payload: The TrainerPayload containing model inputs, masks, etc.
    target_policy_version: Version of the policy weights to train.
  """

  payload: TrainerPayload
  target_policy_version: int = 0


class WeightedMetricLike(Protocol):
  """An unreduced weighted metric transported across the wire.

  Trainers accumulate some metrics as a running `unreduced_sum` and a
  `denominator` so the orchestrator can reduce them correctly across
  micro-batches and workers. This is the structural wire contract; readers match
  on these read-only attributes rather than a concrete class so the wire format
  stays framework-neutral. `tunix.sft.utils.WeightedMetric` is the on-device JAX
  counterpart whose `jax.Array` fields must be converted to host numpy/float
  values before crossing the wire (`rpc_utils.validate_wire_safe` rejects
  `jax.Array`).
  """

  @property
  def unreduced_sum(self) -> np.ndarray | np.number | float:
    ...

  @property
  def denominator(self) -> np.ndarray | np.number | float:
    ...

  @property
  def eps(self) -> float | None:
    ...

  @property
  def min_denom(self) -> float | None:
    ...


@dataclasses.dataclass(frozen=True, kw_only=True)
class MetricsBuffer:
  """Step metrics returned by a trainer worker to the orchestrator.

  Carries one update's metrics unreduced so the orchestrator can apply the
  right reduction per metric; values must already be host (numpy/float) scalars
  when the buffer crosses the wire.

  Attributes:
    id: The trainer step (or any caller-chosen identifier) the metrics belong
      to.
    weighted_metrics: Metrics that reduce as `unreduced_sum / denominator`.
    scalar_metrics: Metrics that reduce with a plain aggregation function.
    aggregation_fns: Optional per-metric host-side aggregation overrides for
      `scalar_metrics`, keyed by metric name.
    mode: The training mode the metrics were collected in (`"train"` /
      `"eval"`).
  """

  id: int | str
  weighted_metrics: Mapping[str, WeightedMetricLike] = dataclasses.field(
      default_factory=dict
  )
  scalar_metrics: Mapping[str, np.ndarray | np.number | float] = (
      dataclasses.field(default_factory=dict)
  )
  aggregation_fns: Mapping[
      str, Callable[[np.ndarray | np.number | float], Any]
  ] = dataclasses.field(default_factory=dict)
  mode: str = "train"


@dataclasses.dataclass(kw_only=True)
class LogprobsRequest(Request):
  """Request to score per-token log-probabilities under a frozen model.

  Attributes:
    prompt_tokens: [B, P] token ids, already LEFT-padded by the caller.
    completion_tokens: [B, C] token ids, already RIGHT-padded by the caller; the
      result aligns to these completion columns.
    temperature: Softmax temperature to score under. Mandatory: it must match
      the temperature the tokens were sampled at, or the log-probs are biased.
    model_role: Which hosted model to score against (v1: "reference").
    pad_id: Pad token id. Used by the trainer/actor scoring path; the reference
      path leaves it unset and relies on the worker's own id.
    eos_id: End-of-sequence token id, used by the trainer/actor scoring path.
    segment_ids: Optional packing segment ids (sequence packing).
    segment_positions: Optional packing local position indices (sequence
      packing).
    routed_experts: Optional MoE expert ids captured during rollout, laid out
      like `RLTrainerPayload.routed_experts` (`4D` `int16` `[B, P + C,
      num_layers, top_k]` with `-1` unset, `4D` `+1`-shifted `uint8` `[B, P + C,
      num_layers, top_k]` with `0` unset, or `5D` `+1`-shifted `uint10` `[B, P +
      C, num_layers, top_k // 4, 5]` with `0` unset; decoded via
      `packing.decode_routed_experts(..., batched=True)` or
      `common.decode_routed_experts_jax`). When set, the scorer replays this
      routing so its log-probs match the training forward pass; trainer path
      only.
  """

  prompt_tokens: np.ndarray
  completion_tokens: np.ndarray
  temperature: float
  model_role: str = "reference"
  pad_id: int | None = None
  eos_id: int | None = None
  segment_ids: np.ndarray | None = None
  segment_positions: np.ndarray | None = None
  routed_experts: np.ndarray | None = None


##### Inference DTOs #####


@dataclasses.dataclass(kw_only=True)
class LogprobsResponse(Response):
  """Per-token log-probabilities for a LogprobsRequest.

  Attributes:
    per_token_logps: [B, C], aligned to the request's completion columns.
    model_version: Version of the scoring weights (constant for a frozen model).
    error: Failure details when the request did not succeed, else None.
  """

  per_token_logps: np.ndarray
  model_version: int = 0


@dataclasses.dataclass(kw_only=True)
class ScoreRequest(Request):
  """Request to score scalar rewards/values under a hosted model.

  Attributes:
    prompt_tokens: [B, P] token ids, already LEFT-padded by the caller.
    completion_tokens: [B, C] token ids, already RIGHT-padded by the caller.
    model_role: Which hosted model to score against (e.g. "reward").
  """

  prompt_tokens: np.ndarray
  completion_tokens: np.ndarray
  model_role: str = "reward"


@dataclasses.dataclass(kw_only=True)
class ScoreResponse(Response):
  """Scalar scores for a ScoreRequest.

  Attributes:
    scores: [B], one scalar per row.
    model_version: Version of the scoring weights (constant for a frozen model).
    error: Failure details when the request did not succeed, else None.
  """

  scores: np.ndarray
  model_version: int = 0
