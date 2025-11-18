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

"""Base classes for Large Language Model powered agents.

This module defines:

* `LLMBaseAgent`: the minimal abstract base class that provides a standard
  interface for agents interacting with LLMs and environments.

* `ConversationAgentBase`: a higher-level base class for chat-style agents
  that maintain conversation history and trajectories. Most concrete agents
  (single-turn, tool-using, gaming, etc.) should subclass this instead of
  `LLMBaseAgent` directly.
"""

import abc
import asyncio
from typing import Any, Dict

from tunix.rl.agentic.agents import agent_types

ABC = abc.ABC
Trajectory = agent_types.Trajectory
Step = agent_types.Step
Action = agent_types.Action
abstractmethod = abc.abstractmethod


class LLMBaseAgent(ABC):
  """Abstract base class for Large Language Model powered agents."""

  # ──────────────────────────────────────────────────────────────
  # State Access Properties
  # ──────────────────────────────────────────────────────────────

  @property
  @abc.abstractmethod
  def chat_completions(self) -> list[dict[str, str]]:
    """Get the current conversation context for the LLM."""
    raise NotImplementedError

  @property
  @abc.abstractmethod
  def trajectory(self) -> Trajectory:
    """Get the complete trajectory for the current task/episode."""
    raise NotImplementedError

  # ──────────────────────────────────────────────────────────────
  # Environment Interaction Interface
  # ──────────────────────────────────────────────────────────────

  @abstractmethod
  def update_from_env(
      self,
      observation: Any,
      reward: float,
      done: bool,
      info: Dict[str, Any] | None = None,
      **kwargs,
  ) -> None:
    """Process feedback from environment after action execution."""
    raise NotImplementedError("update_from_env is not implemented.")

  async def update_from_env_async(self, *args, **kwargs) -> None:
    """Asynchronous version of update_from_env."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, self.update_from_env, *args, **kwargs
    )

  # ──────────────────────────────────────────────────────────────
  # Model Interaction Interface
  # ──────────────────────────────────────────────────────────────

  @abstractmethod
  def update_from_model(self, response: str, **kwargs) -> Action:
    """Process LLM response and extract structured action."""
    raise NotImplementedError("update_from_model is not implemented.")

  async def update_from_model_async(self, *args, **kwargs) -> Action:
    """Asynchronous version of update_from_model."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, self.update_from_model, *args, **kwargs
    )

  # ──────────────────────────────────────────────────────────────
  # Lifecycle Management
  # ──────────────────────────────────────────────────────────────

  @abstractmethod
  def reset(self) -> None:
    """Reset agent state for a new episode."""
    ...

  # ──────────────────────────────────────────────────────────────
  # Debugging and Introspection
  # ──────────────────────────────────────────────────────────────

  def get_current_state(self) -> Step | None:
    """Get the most recent step for debugging and introspection."""
    if not self.trajectory.steps:
      return None
    return self.trajectory.steps[-1]


class ConversationAgentBase(LLMBaseAgent):
  """Base class for chat-style LLM agents with trajectory support.

  This class implements common functionality for agents that:
  * Maintain a list of chat messages (`_messages`) to send to the LLM.
  * Maintain a `Trajectory` of `Step` objects for RL training.
  * Cache the last environment observation for step recording.

  Subclasses are expected to:
  * Provide a system prompt via constructor.
  * Implement `_observation_to_messages()` to convert environment observations
    into chat messages.
  * Implement `update_from_model()` to parse LLM responses into `Action`s and
    append new `Step`s to the trajectory.
  """

  def __init__(self, system_prompt: str):
    self.system_prompt = system_prompt
    self._trajectory = Trajectory()
    self._messages: list[dict[str, Any]] = []
    self._obs_cache: Any = None
    self._init_messages(system_prompt)

  # ---------- Internal helpers ----------

  def _init_messages(self, system_prompt: str) -> None:
    """Initialize conversation history with a system prompt.

    Subclasses may override this to inject additional content (e.g., tool
    documentation) into the initial system message.

    Args:
      system_prompt: The system prompt to use.
    """
    self._messages = [{"role": "system", "content": system_prompt}]

  def _observation_to_messages(self, observation: Any) -> None:
    """Convert environment observation into chat messages.

    Default behavior:
    * If observation is a dict containing "question", use it as user content.
    * If observation is a string, append as a user message.
    * Otherwise, do nothing.

    Subclasses can override this to handle richer observation formats.

    Args:
      observation: The observation from the environment.
    """
    if isinstance(observation, dict) and "question" in observation:
      self._messages.append(
          {"role": "user", "content": observation["question"]}
      )
    elif isinstance(observation, str):
      self._messages.append({"role": "user", "content": observation})

  # ---------- Properties ----------

  @property
  def chat_completions(self) -> list[dict[str, str]]:
    return self._messages

  @property
  def trajectory(self) -> Trajectory:
    return self._trajectory

  # ---------- Public interface implementations ----------

  def update_from_env(
      self,
      observation: Any,
      reward: float,
      done: bool,
      info: Dict[str, Any] | None = None,
      **kwargs,
  ) -> None:
    """Update current step with environment feedback and extend conversation."""
    step = self.get_current_state()
    if step:
      step.observation = observation
      step.reward = reward
      step.done = done
      step.info = info or {}

    # Cache observation for use when constructing the next Step.
    self._obs_cache = observation

    # Let subclass / default handler convert observation into messages.
    if observation is not None:
      self._observation_to_messages(observation)

  def reset(self) -> None:
    """Reset trajectory, cache, and conversation history."""
    self._trajectory = Trajectory()
    self._obs_cache = None
    self._init_messages(self.system_prompt)
