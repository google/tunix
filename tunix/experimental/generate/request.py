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


class Request:
  """A request to generate a completion for a prompt."""

  def __init__(self, req_id: str, prompt_token_ids: list[int]):
    self._request_id = req_id
    self.token_ids = prompt_token_ids

    self.num_completed_tokens = 0

  @property
  def request_id(self) -> str:
    return self._request_id
