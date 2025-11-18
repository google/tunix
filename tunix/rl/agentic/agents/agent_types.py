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

"""Agent Data Types.

This module defines the core data structures used throughout the agent system.
These types provide standardized containers for actions, interaction steps,
and complete episode trajectories.
"""

from collections.abc import Hashable
import dataclasses
from typing import Any, Dict, Optional

field = dataclasses.field
dataclass = dataclasses.dataclass
asdict = dataclasses.asdict


@dataclass
class Action:
  """Container for structured actions that can be executed by an environment.

  The action content is environment-specific and can be any type of data
  structure (dict, string, custom object, etc.) that the target environment
  can interpret and execute.

  Attributes:
    action: The action payload, format depends on the environment.
  """

  action: Any = None


@dataclass
class Step:
  """Represents a single interaction step in an agent-environment conversation.

  Each Step captures the complete context of one turn: the input to the LLM,
  the model's response and reasoning, the parsed action, the environment's
  response, and associated metadata for tracking and analysis.

  Attributes:
    chat_completions: Messages sent to LLM (OpenAI Chat API format).
    thought: Agent's reasoning or chain-of-thought for this step.
    action: Parsed structured action from LLM response.
    observation: Environment's response after executing the action.
    model_response: Raw text output from the language model.
    info: Additional metadata (timestamps, debug info, trace IDs, etc.).
    reward: Immediate reward signal from environment for this step.
    done: Terminal state flag - True if episode has ended.
    mc_return: Monte Carlo return from this step to episode end.
  """

  chat_completions: list[dict[str, str]] = field(default_factory=list)
  thought: str = ""
  action: Optional[Action] = None
  observation: Any = None
  model_response: str = ""
  info: dict[str, Any] = field(default_factory=dict)
  reward: float = 0.0
  done: bool = False
  mc_return: float = 0.0


@dataclass
class Trajectory:
  """Represents a complete episode or task execution trace.

  A Trajectory contains the full sequence of Steps taken to complete a task,
  along with the task description and overall performance metrics. This is
  the primary data structure for episode storage, analysis, and replay.

  Attributes:
    task: Task description, initial prompt, or episode specification.
    steps: Chronologically ordered sequence of interaction steps.
    reward: Total episode reward (cumulative or final environment score).
  """

  task: Any = None
  steps: list[Step] = field(default_factory=list)
  reward: float = 0.0

  def to_dict(self) -> dict[str, Any]:
    """Convert trajectory to dictionary format for serialization.

    Useful for logging, storage, or transmission over APIs. All Step objects
    are recursively converted to dictionaries using dataclass serialization.

    Returns:
      dict: Serializable dictionary representation of the trajectory.
    """
    return {
        "task": self.task,
        "steps": [asdict(step) for step in self.steps],
        "reward": float(self.reward),
    }


@dataclass
class TrajectoryItem:
  """Represents an item within a Trajectory, potentially for pairing or grouping.

  Attributes:
    pair_index: Index for pairing.
    group_id: Identifier for grouping trajectories.
    episode_id: Unique identifier for the episode.
    start_step: The starting step index within the full trajectory.
    traj: The Trajectory object itself.
    metadata: Additional metadata.
  """

  pair_index: int
  group_id: Hashable
  episode_id: int
  start_step: int
  traj: Trajectory
  metadata: Dict[str, Any] = field(default_factory=dict)
