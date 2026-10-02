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
from tunix.experimental.rollout import sampler as sampler_lib


class RequestStatus(enum.Enum):
  """Lifecycle status of a request."""

  # Waiting to be scheduled.
  PENDING = enum.auto()
  # Scheduled, and holds KV cache pages.
  RUNNING = enum.auto()
  # Sampled one of the scheduler's EOS tokens.
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


# Why a request finished: "stop" if it sampled an EOS token, "length" if it
# generated `max_tokens` tokens.
FinishReason = Literal['stop', 'length']


@dataclasses.dataclass(frozen=True, kw_only=True)
class RequestOutput:
  """The result of a finished request."""

  # The id of the request this is the result of.
  request_id: str
  # The generated tokens, detokenized.
  text: str
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


class RequestState:
  """Mutable runtime state for a single request."""

  def __init__(
      self,
      req_id: str,
      prompt_token_ids: list[int],
      sampling_params: sampler_lib.SamplingParams,
  ):
    # Unique identifier of the request.
    self._request_id = req_id
    self.prompt_length = len(prompt_token_ids)
    self.sampling_params = sampling_params

    # The prompt tokens followed by any tokens generated so far.
    self.token_ids = list(prompt_token_ids)
    self.logprobs: list[SupportsFloat] = []
    self.logits: list[SupportsFloat] = []

    # Number of leading tokens in `token_ids` whose KV cache entries have been
    # computed.
    self.num_completed_tokens = 0
    # Number of tokens scheduled as input to the model runner step.
    self.num_tokens_scheduled = 0
    # Number of tokens being sampled by an in-flight model runner step.
    self.num_in_flight_tokens = 0

    # Kernel category for the current step. Only meaningful while
    # `num_tokens_scheduled > 0`.
    self.is_decode = False
    self.is_chunked_prefill = False

    self.status = RequestStatus.PENDING

  @property
  def is_done(self) -> bool:
    """Whether the request has finished or been aborted."""
    return self.status in _DONE_STATUSES

  @property
  def num_processed_tokens(self) -> int:
    return self.num_tokens_scheduled + self.num_completed_tokens

  @property
  def num_total_tokens(self) -> int:
    """Total tokens including any tokens being sampled in flight."""
    return len(self.token_ids) + self.num_in_flight_tokens

  @property
  def request_id(self) -> str:
    return self._request_id
