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

"""Request definition."""

import dataclasses
from typing import SupportsFloat


@dataclasses.dataclass(frozen=True, kw_only=True)
class SamplingParams:
  """Parameters that control how a request's completion is generated."""

  # The maximum number of tokens to generate.
  max_tokens_to_generate: int
  # Generation stops once any of these tokens is sampled.
  eos_token_ids: frozenset[int] = frozenset()


class Request:
  """A request to generate a completion for a prompt."""

  def __init__(
      self,
      req_id: str,
      prompt_token_ids: list[int],
      sampling_params: SamplingParams,
  ):
    self._request_id = req_id
    self.prompt_length = len(prompt_token_ids)
    self.sampling_params = sampling_params

    self.token_ids = prompt_token_ids
    self.logprobs: list[SupportsFloat] = []
    self.logits: list[SupportsFloat] = []

    self.num_completed_tokens = 0
    self.num_in_flight_tokens = 0

    self.is_decode = False
    self.is_chunked_prefill = False
    self.is_aborted = False

  @property
  def num_processed_tokens(self) -> int:
    return self.num_in_flight_tokens + self.num_completed_tokens

  @property
  def request_id(self) -> str:
    return self._request_id
