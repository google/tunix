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

"""Common types used for reward computation and representation."""

import dataclasses
from typing import Any, Dict

field = dataclasses.field
dataclass = dataclasses.dataclass


@dataclass
class RewardOutput:
  """Standardized container for reward computation results.

  This class encapsulates both the numerical reward signal and associated
  metadata from reward function evaluations. It provides a consistent
  interface for reward systems across different environments and tasks.

  The reward value represents the core signal used for reinforcement learning,
  while metadata enables debugging, analysis, and detailed performance tracking
  without cluttering the main reward signal.

  Attributes:
      reward (float): The numerical reward signal for the agent's action. This
        is the primary value used by RL algorithms for learning. Positive values
        indicate good performance, negative values indicate poor performance,
        and zero typically represents neutral outcomes.
      metadata (Dict[str, Any]): Additional information about the reward
        computation. Can include intermediate scores, detailed breakdowns,
        confidence measures, error messages, timing information, or any
        diagnostic data useful for analysis and debugging. Default is an empty
        dictionary.

  Examples:

      Simple reward:
          reward_output = RewardOutput(reward=0.85)

      Detailed reward with metadata:
          reward_output = RewardOutput(
              reward=0.75,
              metadata={
                  'accuracy': 0.8,
                  'completeness': 0.7,
                  'confidence': 0.95,
                  'execution_time': 1.23
              }
          )
  """

  reward: float
  metadata: Dict[str, Any] = field(default_factory=dict)
