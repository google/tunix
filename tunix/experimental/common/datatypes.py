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

import dataclasses
import enum
import time
from typing import Any, Dict
import flax
import uuid
from jax.typing import ArrayLike  # pylint: disable=g-importing-member
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

# Marks a router-replay slot the trainer must not replay, so the model falls
# back to its own gate there. Matches what MaxText's replay path expects.
UNSET_ROUTED_EXPERT = -1


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


##### Rollout DTOs #####


@dataclasses.dataclass(frozen=True, kw_only=True)
class GenerationArgs:
  """Typed generation arguments used by the orchestrator generate API."""
  max_generation_steps: int | None = None
  max_response_length: int | None = None
  temperature: float | None = None
  top_p: float | None = None
  top_k: int | None = None
  seed: int | None = None
  return_logprobs: bool | None = None

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
    generation_kwargs: Additional keyword arguments for generation (e.g.
      sampling parameters like max_tokens and temperature).
    max_turns: Maximum number of conversation turns for environment interaction.
    target_policy_version: Policy model version identifier to use for rollout
      generation.
  """

  prompt: Any = ""
  prompt_id: str = ""
  group_index: int = 0
  generation_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
  max_turns: int = 10
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
    routed_experts: `[len(tokens), num_layers, top_k]` MoE expert ids this span
      was routed through, so training can replay the routing the rollout
      actually used. None for dense models, spans the model did not emit, or
      when the sampler was not asked to capture routing.
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
    status: Terminal status name (e.g. "COMPLETED", "ERROR", "TIMEOUT", "CANCELLED").
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


@flax.struct.dataclass(frozen=True, kw_only=True)
class TrainerPayload:
  """Base abstract class for generic trainer payloads. """


@flax.struct.dataclass(frozen=True, kw_only=True)
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

  token_ids: ArrayLike
  token_mask: ArrayLike
  segment_ids: ArrayLike | None = None
  segment_positions: ArrayLike | None = None


# TODO(tunix-dev): Introduce PPOTrainerPayload to replace generic
# RLTrainerPayload when PPO specific fields are needed.
@flax.struct.dataclass(frozen=True, kw_only=True)
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
    routed_experts: Optional `[B, T, num_layers, top_k]` MoE expert ids captured
      during rollout. When set, a training engine that supports router replay
      forces these experts instead of re-running its own gate, so the training
      forward pass matches the routing the rollout actually used. `-1` marks a
      padded or unused slot.
    returns: Optional [B, C] value baseline returns (for PPO / Critic).
    old_values: Optional [B, C] critic value estimates (for PPO / Critic).
    num_segments: Optional static upper bound on number of segments in packed
      rows.
    metadata: Extra payload metadata dictionary.
  """

  prompt_ids: ArrayLike
  prompt_mask: ArrayLike
  completion_ids: ArrayLike
  completion_mask: ArrayLike
  advantages: ArrayLike
  segment_ids: ArrayLike | None = None
  segment_positions: ArrayLike | None = None
  ref_per_token_logps: ArrayLike | None = None
  old_per_token_logps: ArrayLike | None = None
  sampler_is_weights: ArrayLike | None = None
  routed_experts: ArrayLike | None = None
  returns: ArrayLike | None = None
  old_values: ArrayLike | None = None
  num_segments: int | None = flax.struct.field(default=None, pytree_node=False)
  metadata: dict[str, Any] = flax.struct.field(
      default_factory=dict, pytree_node=False
  )
  # TODO(tunix-dev): add ppo specific fields in PPORLTrainerPayload.


@dataclasses.dataclass(kw_only=True)
class TrainRequest(Request):
  """Request to execute training (forward/backward or eval) on a TrainerWorker.

  Attributes:
    payload: The TrainerPayload containing model inputs, masks, etc.
    target_policy_version: Version of the policy weights to train.
  """

  payload: TrainerPayload
  target_policy_version: int = 0


@dataclasses.dataclass(kw_only=True)
class LogprobsRequest(Request):
  # TODO(tunix-dev): add router replay support to LogprobsRequest.
  """Request to score per-token log-probabilities under a frozen model.

  Attributes:
    prompt_tokens: [B, P] token ids, already LEFT-padded by the caller.
    completion_tokens: [B, C] token ids, already RIGHT-padded by the caller;
      the result aligns to these completion columns.
    temperature: Softmax temperature to score under. Mandatory: it must match
      the temperature the tokens were sampled at, or the log-probs are biased.
    model_role: Which hosted model to score against (v1: "reference").
    pad_id: Pad token id. Used by the trainer/actor scoring path; the reference
      path leaves it unset and relies on the worker's own id.
    eos_id: End-of-sequence token id, used by the trainer/actor scoring path.
    segment_ids: Optional 1D packing segment ids (sequence packing); trainer path
      only.
    segment_positions: Optional 1D packing local position indices (sequence
      packing); trainer path only.
  """

  prompt_tokens: ArrayLike
  completion_tokens: ArrayLike
  temperature: float
  model_role: str = "reference"
  pad_id: int | None = None
  eos_id: int | None = None
  segment_ids: ArrayLike | None = None
  segment_positions: ArrayLike | None = None


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
