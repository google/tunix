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
from jax.typing import ArrayLike  # pylint: disable=g-importing-member
import numpy as np
from tunix.rl.agentic.agents import agent_types

##### Worker-internal datatypes #####


# Worker-internal episode representation produced during rollout.
Trajectory = agent_types.Trajectory
Step = agent_types.Step
TrajectoryStatus = agent_types.TrajectoryStatus


# TODO: Unify this extended TrajectoryItem back into agent_types.TrajectoryItem
# so that all agentic workflows share the same strict token array fields.
@dataclasses.dataclass(kw_only=True)
class TrajectoryItem(agent_types.TrajectoryItem):
  """Extended TrajectoryItem for Orchestrator with token arrays."""
  prompt_tokens: np.ndarray | None = None
  completion_tokens: np.ndarray | None = None
  action_mask: np.ndarray | None = None
  policy_version: int = 0

class Role(str, enum.Enum):
  """Orchestrator worker roles."""
  ACTOR = "actor"
  CRITIC = "critic"
  ROLLOUT = "rollout"
  REFERENCE = "reference"
  REWARD = "reward"


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


@dataclasses.dataclass(kw_only=True)
class Request:
  """Standard base for generic RPC requests.

  Attributes:
    request_id: Unique identifier for this request, echoed back on the
      corresponding response so callers can correlate responses.
    metadata: Optional free-form data attached to the request.
  """

  request_id: str = ""
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
    group_offset_id: Optional identifier for grouping related rollout requests
      (e.g. for GRPO).
    generation_kwargs: Additional keyword arguments for generation (e.g.
      sampling parameters like max_tokens and temperature).
    max_turns: Maximum number of conversation turns for environment interaction.
    target_policy_version: Policy model version identifier to use for rollout
      generation.
  """

  prompt: Any = ""
  prompt_id: str = "default_prompt"
  group_offset_id: str = ""
  generation_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
  max_turns: int = 10
  target_policy_version: int = 0

  @property
  def traj_id(self) -> str:
    """Standardized semantic trajectory identifier computed from prompt_id and group_offset_id."""
    return (
        f"traj_{self.prompt_id}_{self.group_offset_id}"
        if self.group_offset_id
        else f"traj_{self.prompt_id}"
    )


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
  """

  source: str
  tokens: np.ndarray
  loss_mask: np.ndarray
  logps: np.ndarray | None = None

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


@dataclasses.dataclass(kw_only=True)
class RolloutResponse(Response):
  """Serializable result of a generation request.

  This is the wire-facing counterpart to RolloutRequest (and to the
  worker-internal Trajectory): it carries only primitives and numpy
  arrays, so it can cross a process boundary. A failed request is reported as a
  result with `error` set and a non-success `status`, never as a dropped
  response.

  Attributes:
    prompt_id: Unique identifier for this prompt within a task or dataset.
    status: Terminal status name (e.g. a rollout trajectory status, or
      "CANCELLED").
    prompt_tokens: Array of prompt token ids, unpadded, as tokenized by the
      worker.
    segments: Ordered conversation turns (segments) from the assistant (model
      call) and environment; concatenated they form the full generated stream.
    env_reward: Scalar environment reward for the trajectory.
    policy_version: Weight version used to generate the trajectory.
    error: Failure details when the request did not succeed, else None.
  """

  prompt_id: str = ""
  status: str
  prompt_tokens: np.ndarray = dataclasses.field(
      default_factory=lambda: np.zeros(0, dtype=np.int32)
  )
  segments: list[TokenSegment] = dataclasses.field(default_factory=list)
  env_reward: float = 0.0
  policy_version: int = 0
  # TODO(b/532722981): capture rollout metrics, e.g., env time.

  @classmethod
  def from_trajectory(
      cls,
      request_id: str,
      traj: Trajectory,
      prompt_tokens: np.ndarray,
      policy_version: int,
  ) -> "RolloutResponse":
    """Constructs a wire-safe RolloutResponse from an internal Trajectory.

    Extracts only the required arrays (tokens, masks, logprobs) from the
    semantic steps, discarding string metadata and unpicklable objects.

    Args:
      request_id: The ID of the original rollout request.
      traj: The internal trajectory to convert.
      prompt_tokens: Array of prompt token ids.
      policy_version: Weight version used to generate the trajectory.

    Returns:
      A wire-safe RolloutResponse.
    """
    def _get_step_attr(step, attr):
      val = getattr(step, attr, None)
      if val is not None:
        return val
      extra = getattr(step, "extra", None)
      if isinstance(extra, dict):
        return extra.get(attr)
      return None

    segments = []
    for step in traj.steps:
      assistant_tokens = _get_step_attr(step, "assistant_tokens")
      if assistant_tokens is not None:
        segments.append(
            TokenSegment(
                source="assistant",
                tokens=assistant_tokens,
                loss_mask=_get_step_attr(step, "assistant_masks"),
                logps=_get_step_attr(step, "logprobs"),
            )
        )
      env_tokens = _get_step_attr(step, "env_tokens")
      if env_tokens is not None:
        segments.append(
            TokenSegment(
                source="env",
                tokens=env_tokens,
                loss_mask=_get_step_attr(step, "env_masks"),
                logps=None,
            )
        )
    if hasattr(traj, "status") and traj.status is not None:
      status_val = getattr(traj.status, "name", str(traj.status))
    else:
      status_val = "COMPLETED"
    return cls(
        request_id=request_id,
        status=status_val,
        prompt_tokens=prompt_tokens,
        segments=segments,
        env_reward=getattr(traj, "reward", 0.0) or 0.0,
        policy_version=policy_version,
    )


##### Weight Sync DTOs #####


@dataclasses.dataclass(kw_only=True)
class WeightSyncRequest(Request):
  """Configuration and routing metadata for synchronizing policy model weights.

  Attributes:
    controller_id: Optional identifier for transport controllers (e.g., TPU
      Raiden).
    policy_version: Target policy version identifier of the weights to sync.
    source_metadata: Optional transport/layout metadata describing source
      weights.
    extra_config: Optional backend-specific configuration parameters.
  """

  controller_id: str = ""
  policy_version: int = 0
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


@dataclasses.dataclass(kw_only=True)
class TrainerPayload:
  """Generic trainer payload.

  Attributes:
    token_ids: [B, T] token IDs for a batched trainer payload. By default,
      each row is structured as left-padded prompt tokens concatenated with
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


# TODO: Introduce PPOTrainerPayload to replace generic RLTrainerPayload when PPO specific fields are needed.
@dataclasses.dataclass(kw_only=True)
class RLTrainerPayload(TrainerPayload):
  """RL training payload.

  Attributes:
    advantages: [B] or [B, C] advantages.
    loss_mask: [B, T], 1 where the position contributes to the loss.
    action_mask: Optional [B, T] or [B, C] mask of policy actions.
    prompt_ids: Optional prompt token ids for GRPO-style losses. Unbatched
      payloads may carry 1D unpadded rows; batch assembly pads them to [B, P].
    prompt_mask: Optional [B, P] prompt mask.
    completion_ids: Optional completion token ids. Unbatched payloads may carry
      1D unpadded rows; batch assembly pads them to [B, C].
    completion_mask: Optional [B, C] completion/action mask.
    ref_per_token_logps: Optional [B, C] reference model log-probabilities.
    old_per_token_logps: Optional [B, C] behavior policy log-probabilities.
    sampler_is_weights: Optional [B, C] importance sampling weights.
    returns: Optional [B, C] value baseline returns (for PPO / Critic).
    old_values: Optional [B, C] critic value estimates (for PPO / Critic).
    metadata: Extra payload metadata dictionary.
  """

  advantages: ArrayLike
  loss_mask: ArrayLike
  action_mask: ArrayLike | None = None
  prompt_ids: ArrayLike | None = None
  prompt_mask: ArrayLike | None = None
  completion_ids: ArrayLike | None = None
  completion_mask: ArrayLike | None = None
  ref_per_token_logps: ArrayLike | None = None
  old_per_token_logps: ArrayLike | None = None
  sampler_is_weights: ArrayLike | None = None
  returns: ArrayLike | None = None
  old_values: ArrayLike | None = None
  metadata: dict[str, Any] = dataclasses.field(default_factory=dict)
  # TODO: add ppo sepcific fields in a PPO specific fields in  PPORLTrainerPayload


@dataclasses.dataclass(kw_only=True)
class LogprobsRequest(Request):
  """Request to score per-token log-probabilities under a frozen model.

  Attributes:
    prompt_tokens: [B, P] token ids, already LEFT-padded by the caller.
    completion_tokens: [B, C] token ids, already RIGHT-padded by the caller;
      the result aligns to these completion columns.
    temperature: Softmax temperature to score under. Mandatory: it must match
      the temperature the tokens were sampled at, or the log-probs are biased.
    model_role: Which hosted model to score against (v1: "reference").
  """

  prompt_tokens: np.ndarray
  completion_tokens: np.ndarray
  temperature: float
  model_role: str = "reference"


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
