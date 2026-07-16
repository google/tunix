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

This module centralizes type aliases and dataclasses used for routing
data and commands between the orchestrator and distributed RL workers.
"""

import dataclasses
from typing import Any, Dict, List, Optional

from tunix.rl import common
from tunix.rl.agentic.agents import agent_types


@dataclasses.dataclass
class TrajectoryRequest:
  """Request to generate a trajectory from a given prompt."""

  prompt_id: str
  prompt_text: str
  max_turns: int = 10
  sampling_params: Optional[Dict[str, Any]] = None


# Generated Trajectory from a rollout
Trajectory = agent_types.Trajectory


# Train example used for policy optimization
TrainExample = common.TrainExample
