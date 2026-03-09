# Copyright 2025 Google LLC
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

"""Rollout traffic router interface."""

import abc
import dataclasses

ABC = abc.ABC
abstractmethod = abc.abstractmethod


@dataclasses.dataclass(kw_only=True)
class RolloutTrafficRoutingParams:
  """Parameters for rollout traffic routing."""

  # trajectory id, used for per-trajectory routing.
  trajectory_id: int


class RolloutTrafficRouter:

  @abstractmethod
  def route(self, params: RolloutTrafficRoutingParams) -> int:
    """returns the rollout engine id."""
