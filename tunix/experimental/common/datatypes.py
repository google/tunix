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
from typing import Any

from jax.typing import ArrayLike  # pylint: disable=g-importing-member
import numpy as np
from tunix.rl.agentic.agents import agent_types

##### Worker-internal datatypes #####

# Worker-internal episode representation produced during rollout.
Trajectory = agent_types.Trajectory
Step = agent_types.Step
TrajectoryStatus = agent_types.TrajectoryStatus


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


@dataclasses.dataclass(kw_only=True)
class RolloutRequest(Request):
  """Request to generate a rollout from a given prompt.

  Attributes:
    prompt: The prompt to generate from (e.g. formatted string, token array, or
      chat dictionary).
    prompt_id: Unique identifier for this prompt within a task or dataset.
    group_id: Optional identifier for grouping related rollout requests (e.g.
      for GRPO).
    generation_kwargs: Additional keyword arguments for generation (e.g.
      sampling parameters like max_tokens and temperature).
    max_turns: Maximum number of conversation turns for environment interaction.
    target_policy_version: Policy model version identifier to use for rollout
      generation.
  """

  prompt: Any = ""
  prompt_id: str = "default_prompt"
  group_id: str = ""
  generation_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
  max_turns: int = 10
  target_policy_version: int = 0


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
    segments = []
    for step in traj.steps:
      if step.assistant_tokens is not None:
        segments.append(
            TokenSegment(
                source="assistant",
                tokens=step.assistant_tokens,
                loss_mask=step.assistant_masks,
                logps=step.logprobs,
            )
        )
      if step.env_tokens is not None:
        segments.append(
            TokenSegment(
                source="env",
                tokens=step.env_tokens,
                loss_mask=step.env_masks,
                logps=None,
            )
        )
    return cls(
        request_id=request_id,
        status=traj.status.name,
        prompt_tokens=prompt_tokens,
        segments=segments,
        env_reward=traj.reward,
        policy_version=policy_version,
    )


@dataclasses.dataclass(kw_only=True)
class TrainerPayload:
  """Generic trainer payload.

  Attributes:
    token_ids: [B, T] token IDs. By default, structured as left-padded prompt
      tokens concatenated with right-padded completion tokens.
    token_mask: [B, T] token mask to differentiate padding tokens from valid
      tokens.
    segment_ids: Optional [B, T] packing segment ids.
  """

  token_ids: ArrayLike
  token_mask: ArrayLike
  segment_ids: ArrayLike | None = None


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


##### Training DTOs #####


@dataclasses.dataclass(kw_only=True)
class RLTrainerPayload(TrainerPayload):
  """RL training payload.

  Attributes:
    advantages: [B] or [B, C] advantages.
    loss_mask: [B, T], 1 where the position contributes to the loss.
    ref_per_token_logps: Optional [B, C] reference model log-probabilities.
    old_per_token_logps: Optional [B, C] behavior policy log-probabilities.
    sampler_is_weights: Optional [B, C] importance sampling weights.
  """

  advantages: ArrayLike
  loss_mask: ArrayLike
  ref_per_token_logps: ArrayLike | None = None
  old_per_token_logps: ArrayLike | None = None
  sampler_is_weights: ArrayLike | None = None


@dataclasses.dataclass(kw_only=True)
class LogprobsRequest(Request):
  """Request to score per-token log-probabilities under a frozen model.

  Attributes:
    prompt_tokens: [B, P], LEFT-padded.
    completion_tokens: [B, C], RIGHT-padded; the result aligns to these
      completion columns.
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
    prompt_tokens: [B, P], LEFT-padded.
    completion_tokens: [B, C], RIGHT-padded.
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
