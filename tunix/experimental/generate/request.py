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

import enum


class RequestStatus(enum.Enum):
  """Lifecycle status of a request."""

  # Waiting to be scheduled.
  PENDING = enum.auto()
  # Scheduled, and holds KV cache pages.
  RUNNING = enum.auto()
  # Aborted by the caller.
  ABORTED = enum.auto()


_DONE_STATUSES = frozenset({RequestStatus.ABORTED})


class RequestState:
  """Mutable runtime state for a single request."""

  def __init__(self, req_id: str, prompt_token_ids: list[int]):
    # Unique identifier of the request.
    self._request_id = req_id
    # The prompt tokens followed by any tokens generated so far.
    self.token_ids = list(prompt_token_ids)
    # Number of tokens whose KV cache is computed. This includes tokens that
    # were speculatively scheduled and are in flight.
    self.num_computed_tokens = 0
    # Number of tokens being sampled by an in-flight model runner step.
    self.num_in_flight_tokens = 0
    # Whether the request is scheduled as a decode step.
    self.is_decode = False
    # Whether the request is scheduled as a chunked prefill step.
    self.is_chunked_prefill = False

    self.status = RequestStatus.PENDING

  @property
  def is_done(self) -> bool:
    """Whether the request has been aborted."""
    return self.status in _DONE_STATUSES

  @property
  def num_total_tokens(self) -> int:
    """Total tokens including any tokens being sampled in flight."""
    return len(self.token_ids) + self.num_in_flight_tokens

  @property
  def request_id(self) -> str:
    return self._request_id
