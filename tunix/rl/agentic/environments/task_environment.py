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

"""RL Environment for single-turn task-based agent interactions."""

import logging
from typing import Any, Dict

from tunix.rl.agentic.environments import base_environment
from tunix.rl.agentic.rewards import reward

BaseEnv = base_environment.BaseEnv
dummy_reward = reward.dummy_reward


class TaskEnvironment(BaseEnv):
  """Reinforcement learning environment for single-turn agent interactions.

  This environment is designed for tasks where the agent receives an
  initial observation (the task) and provides a single response, after
  which the episode terminates. It does not involve multi-step interactions
  or tool use.
  """

  def __init__(
      self,
      task: Dict[str, Any] | None = None,
      *,
      reward_fn=None,
      **kwargs,
  ):
    """Initialize the task environment.

    Args:
        task (Dict[str, Any] | None): Task specification containing problem
          description, ground truth, or other parameters.
        reward_fn: Reward function that takes (task, action) and returns
          RewardOutput. If None, defaults to dummy_reward.
        **kwargs: Ignores extra arguments like max_steps for compatibility.
    """
    super().__init__()
    if reward_fn is None:
      logging.warning("No reward_fn provided, defaulting to dummy_reward().")
      reward_fn = dummy_reward
    self.reward_fn = reward_fn
    self.task = task or {}

  def reset(self) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Reset the environment and return the task as the initial observation."""
    return self.task, {}

  def step(self, action: Any) -> tuple[Any, float, bool, Dict[str, Any]]:
    """Process the agent's action, calculate reward, and terminate."""
    # In a single-turn environment, any action terminates the episode.
    # We assume 'action' is the agent's final response string.
    r_out = self.reward_fn(task=self.task, action=action)
    return (
        {},
        r_out.reward,
        True,
        {"response": action, "metadata": r_out.metadata},
    )

  @staticmethod
  def from_dict(env_args: Dict[str, Any]) -> "TaskEnvironment":
    """Create TaskEnvironment instance from configuration dictionary."""
    reward_fn = env_args.pop("reward_fn", None)
    task = env_args
    return TaskEnvironment(task=task, reward_fn=reward_fn)
