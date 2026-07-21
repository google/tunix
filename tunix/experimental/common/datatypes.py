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

from jax.typing import ArrayLike  # pylint: disable=g-importing-member
import numpy as np
from tunix.rl.agentic.agents import agent_types

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
class SamplingParams:
  """Engine-neutral sampling configuration for a generation request.

  Attributes:
    max_tokens: Maximum number of tokens to generate.
    temperature: Softmax temperature applied while sampling. Kept explicit so it
      can be carried through to any later log-probability scoring, ensuring the
      sampling distribution and the scoring distribution match.
  """

  max_tokens: int
  temperature: float = 1.0


@dataclasses.dataclass
class RolloutRequest:
  """Request to generate a rollout from a given prompt.

  Attributes:
    request_id: Unique identifier for this request, echoed back on the
      corresponding result so callers can correlate and de-duplicate responses.
    prompt_id: Identifier of the source prompt (e.g. a dataset row). Provenance
      only; not unique across requests that reuse the same prompt.
    prompt_text: The prompt to generate from.
    sampling_params: Sampling configuration for the generation.
  """

  request_id: str
  prompt_id: str
  prompt_text: str
  sampling_params: SamplingParams


@dataclasses.dataclass(kw_only=True)
class TokenSegment:
  """One contiguous span of the conversation token stream representing a single turn.

  Each segment corresponds to a single turn's response from either the sampler
  or the environment.

  Attributes:
    source: Origin of the span, e.g. "sampler" (model-emitted) or "env".
    tokens: Array of token ids for this span.
    loss_mask: Array of ints, 1 where the token is model-emitted (trainable).
    logprobs: Array of per-token log-probabilities under the sampling
      distribution, or None for spans the model did not emit (e.g. env tokens).
  """

  source: str
  tokens: np.ndarray
  loss_mask: np.ndarray
  logprobs: np.ndarray | None = None

  def __post_init__(self):
    if self.loss_mask.shape != self.tokens.shape:
      raise ValueError(
          f"loss_mask shape {self.loss_mask.shape} != tokens shape"
          f" {self.tokens.shape}"
      )
    if self.logprobs is not None and self.logprobs.shape != self.tokens.shape:
      raise ValueError(
          f"logprobs shape {self.logprobs.shape} != tokens shape"
          f" {self.tokens.shape}"
      )


@dataclasses.dataclass(kw_only=True)
class RolloutResult:
  """Serializable result of a generation request.

  This is the wire-facing counterpart to RolloutRequest (and to the
  worker-internal Trajectory): it carries only primitives and numpy
  arrays, so it can cross a process boundary. A failed request is reported as a
  result with `error` set and a non-success `status`, never as a dropped
  response.

  Attributes:
    request_id: Echoes the originating request, for correlation/de-duplication.
    prompt_id: Echoes the source prompt id.
    status: Terminal status name (e.g. a rollout trajectory status, or
      "CANCELLED").
    prompt_tokens: Array of prompt token ids, unpadded, as tokenized by the
      worker.
    segments: Ordered conversation turns (segments) from the sampler (model
      call) and environment; concatenated they form the full generated stream.
    env_reward: Scalar environment reward for the trajectory.
    policy_version: Weight version used to generate the trajectory.
    error: Failure details when the request did not succeed, else None.
  """

  request_id: str
  prompt_id: str
  status: str
  prompt_tokens: np.ndarray = dataclasses.field(
      default_factory=lambda: np.zeros(0, dtype=np.int32)
  )
  segments: list[TokenSegment] = dataclasses.field(default_factory=list)
  env_reward: float = 0.0
  policy_version: int = 0
  error: ErrorInfo | None = None
  # TODO(b/532722981): capture rollout metrics, e.g., env time.


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


##### Worker-internal datatypes #####

# Worker-internal episode representation produced during rollout.
Trajectory = agent_types.Trajectory
