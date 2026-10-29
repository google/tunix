# Copyright 2026 The Tunix Authors.
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

"""Runtime state for requests being processed by the engine.

`RequestState` is the mutable, engine-internal record of a single request. The
engine's components (e.g. the scheduler and KV cache manager) share and update
it across engine steps.

Lifecycle:
  1. A `RequestState` is created when a request is submitted to the engine.
  2. It is updated on each step the request is active.
  3. It is released once the request finishes or is aborted.
"""

import dataclasses
import enum
from typing import Literal, SupportsFloat

import numpy as np


class RequestStatus(enum.Enum):
  """Lifecycle status of a request."""

  # Waiting to be scheduled.
  PENDING = enum.auto()
  # Scheduled, and holds KV cache pages.
  RUNNING = enum.auto()
  # Sampled one of the engine's EOS tokens.
  FINISHED_EOS = enum.auto()
  # Generated `max_tokens` tokens.
  FINISHED_LENGTH = enum.auto()
  # Aborted by the caller.
  ABORTED = enum.auto()


_DONE_STATUSES = frozenset({
    RequestStatus.FINISHED_EOS,
    RequestStatus.FINISHED_LENGTH,
    RequestStatus.ABORTED,
})


FinishReason = Literal['stop', 'length']


@dataclasses.dataclass(kw_only=True)
class SamplingParams:
  """Engine-neutral sampling configuration for a generation request.

  Attributes:
    max_tokens: Maximum number of tokens to generate.
    temperature: Softmax temperature applied while sampling. Kept explicit so it
      can be carried through to any later log-probability scoring, ensuring the
      sampling distribution and the scoring distribution match.
    top_p: Nucleus sampling probability threshold.
    top_k: Top-k sampling candidate count threshold.
    seed: Random seed for reproducible generation.
    return_logprobs: Whether to record per-token log probabilities.
    return_logits: Whether to record per-token output logits.
    return_routed_experts: Whether to record the MoE expert ids each token was
      routed through, so training can replay them instead of re-routing. Only
      meaningful for MoE models on a backend that supports capture; the backend
      must also be configured to capture, since it is an engine-level setting.
    routed_experts_prompt_start: Token index in prompt from which to start
      returning routing data. In multi-turn rollouts, set to the cumulative
      length of previously processed tokens to return only newly prefilled
      environment tokens and completion.
    beam_size: Beam width for beam search decoding.
    cache_salt: Optional salt string for isolating KV prefix cache entries
      (e.g., pinned to Turn-1 policy version during partial rollout).
  """

  max_tokens: int = 64
  temperature: float = 1.0
  top_p: float | None = None
  top_k: int | None = None
  seed: int | None = None
  return_logprobs: bool = False
  return_logits: bool = False
  return_routed_experts: bool = False
  routed_experts_prompt_start: int = 0
  beam_size: int | None = None
  cache_salt: str | None = None


@dataclasses.dataclass(frozen=True, kw_only=True)
class EngineRequest:
  """A request to submit to the engine."""

  # Unique identifier of the request.
  request_id: str
  # The prompt tokens.
  prompt_token_ids: list[int]
  # Sampling parameters for the request.
  sampling_params: SamplingParams


@dataclasses.dataclass(frozen=True, kw_only=True)
class EngineResponse:
  """The result of a finished request."""

  # The id of the request this is the result of.
  request_id: str
  # The prompt tokens, as the engine tokenized them.
  prompt_token_ids: np.ndarray
  # The generated tokens. A sampled EOS token is kept.
  token_ids: np.ndarray
  # The log probability of each generated token, if the engine returns them.
  logprobs: np.ndarray | None
  # The logits each generated token was sampled from, if the engine returns
  # them.
  logits: np.ndarray | None
  # Why the request finished.
  finish_reason: FinishReason


# TODO(yatlas): Rename RequestState to TaskState.
class RequestState:
  """Mutable runtime state for a single request."""

  def __init__(
      self,
      req_id: str,
      prompt_token_ids: list[int],
      sampling_params: SamplingParams,
  ):
    # Unique identifier of the request.
    self._request_id = req_id
    # Number of tokens in the prompt.
    self.prompt_length = len(prompt_token_ids)
    # Sampling parameters for the request.
    self.sampling_params = sampling_params

    # The prompt tokens followed by any tokens generated so far.
    self.token_ids = list(prompt_token_ids)
    # Token log probabilities, if requested.
    self.logprobs: list[SupportsFloat] = []
    # Token logits, if requested.
    self.logits: list[np.ndarray] = []

    # Number of tokens whose KV cache is computed. This includes tokens that
    # were speculatively scheduled and are in flight.
    self.num_computed_tokens = 0
    # Number of tokens being sampled by an in-flight model runner step.
    self.num_in_flight_tokens = 0
    # Number of tokens an in-flight step was sampling when the request was
    # preempted. They are still appended once that step's output arrives, but
    # until then the request must not sample: it would sample the same
    # positions a second time.
    self.num_stale_output_tokens = 0
    # Whether the request is scheduled as a decode step.
    self.is_decode = False
    # Whether the request is scheduled as a chunked prefill step.
    self.is_chunked_prefill = False

    self.status = RequestStatus.PENDING

  @property
  def is_done(self) -> bool:
    """Whether the request has finished or been aborted."""
    return self.status in _DONE_STATUSES

  @property
  def num_total_tokens(self) -> int:
    """Total tokens including any tokens being sampled in flight."""
    return len(self.token_ids) + self.num_in_flight_tokens

  @property
  def request_id(self) -> str:
    return self._request_id
