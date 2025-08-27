"""Base class for Large Language Model powered agents.

This module defines the abstract base class `LLMBaseAgent`, which provides a
standard interface for agents interacting with LLMs and environments.
"""

import abc
import asyncio
from typing import Any, Dict

from tunix.rl.experimental.agentic.agents import agent_types

ABC = abc.ABC
Trajectory = agent_types.Trajectory
Step = agent_types.Step
Action = agent_types.Action
abstractmethod = abc.abstractmethod


class LLMBaseAgent(ABC):
  """Abstract base class for Large Language Model powered agents.

  This class defines the standard interface for agents that can:
  - Maintain conversation history and state
  - Generate structured actions from LLM responses
  - Process environment feedback
  - Support both synchronous and asynchronous operation

  Subclasses must implement the core interaction methods while this base
  class provides common utilities and enforces the interface contract.
  """

  # ──────────────────────────────────────────────────────────────
  # State Access Properties
  # ──────────────────────────────────────────────────────────────

  @property
  def chat_completions(self) -> list[dict[str, str]]:
    """Get the current conversation context for the LLM.

    Returns the list of messages that should be sent to the language model
    for the next inference call. Subclasses typically construct this from
    internal state including conversation history, system prompts, and
    tool call contexts.

    Returns:
        list[dict[str, str]]: Messages in OpenAI Chat Completions format
            Each dict should have 'role' and 'content' keys at minimum
    """
    return []

  @property
  def trajectory(self) -> Trajectory:
    """Get the complete trajectory for the current task/episode.

    The trajectory object serves as the primary data structure that the
    execution engine uses to read agent state and append new Steps.
    Subclasses should maintain this as their canonical episode record.

    Returns:
        Trajectory: Complete episode trace including all steps and metadata
    """
    return Trajectory()

  # ──────────────────────────────────────────────────────────────
  # Environment Interaction Interface
  # ──────────────────────────────────────────────────────────────

  @abstractmethod
  def update_from_env(
      self,
      observation: Any,
      reward: float,
      done: bool,
      info: Dict[str, Any],
      **kwargs,
  ) -> None:
    """Process feedback from environment after action execution.

    Called by the execution engine after the environment processes an action.
    The agent should use this information to:
    1. Update the current Step with observation and reward data
    2. Update any internal state (memory, caches, belief state, etc.)
    3. Prepare for the next interaction cycle if episode continues

    Args:
        observation (Any): Environment's response to the executed action Can be
          text, structured data, images, or any environment-specific format
        reward (float): Numerical reward signal for the action taken
        done (bool): Whether the episode has terminated (success, failure, or
          timeout)
        info (dict): Additional environment metadata (debug info, metrics, etc.)
        **kwargs: Additional parameters for extensibility
    """
    ...

  async def update_from_env_async(self, *args, **kwargs) -> None:
    """Asynchronous version of update_from_env.

    Wraps the synchronous update_from_env method in an executor to avoid
    blocking the event loop. Override this method if the agent needs
    to perform actual async I/O operations during environment updates.

    Args:
        *args: Positional arguments passed to update_from_env
        **kwargs: Keyword arguments passed to update_from_env

    Returns:
        None: This method does not return a value.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, self.update_from_env, *args, **kwargs
    )

  # ──────────────────────────────────────────────────────────────
  # Model Interaction Interface
  # ──────────────────────────────────────────────────────────────

  @abstractmethod
  def update_from_model(self, response: str, **kwargs) -> Action:
    """Process LLM response and extract structured action.

    Called by the execution engine after the language model generates a
    response.
    The agent should:
    1. Parse the raw response text to extract reasoning and action components
    2. Create a structured Action object that the environment can execute
    3. Update the current Step with response text and extracted thought process

    Args:
        response (str): Raw text output from the language model
        **kwargs: Additional parameters (model metadata, generation settings,
          etc.)

    Returns:
        Action: Structured action object ready for environment execution
            The action content should match the expected format for the target
            environment
    """
    ...

  async def update_from_model_async(self, *args, **kwargs) -> Action:
    """Asynchronous version of update_from_model.

    Wraps the synchronous update_from_model method in an executor to avoid
    blocking the event loop. Override this method if the agent needs
    to perform actual async I/O operations during response processing.

    Args:
        *args: Positional arguments passed to update_from_model
        **kwargs: Keyword arguments passed to update_from_model

    Returns:
        Action: Structured action object from update_from_model
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, self.update_from_model, *args, **kwargs
    )

  # ──────────────────────────────────────────────────────────────
  # Lifecycle Management
  # ──────────────────────────────────────────────────────────────

  @abstractmethod
  def reset(self) -> None:
    """Reset agent state for a new episode.

    Called at the beginning of each new task/episode to ensure clean state.
    The agent should:
    1. Clear or reset the trajectory object
    2. Reset any internal caches, memory, or accumulated state
    3. Prepare for a fresh interaction sequence

    This method is critical for multi-episode training or evaluation to
    prevent information leakage between different tasks.
    """
    ...

  # ──────────────────────────────────────────────────────────────
  # Debugging and Introspection
  # ──────────────────────────────────────────────────────────────

  def get_current_state(self) -> Step | None:
    """Get the most recent step for debugging and introspection.

    Provides direct access to the latest Step object in the trajectory
    for debugging, logging, or real-time monitoring purposes. Useful
    for understanding agent state during execution or post-mortem analysis.

    Returns:
        Step | None: The most recently created Step object, or None if
            no steps have been generated yet in the current episode
    """
    if not self.trajectory.steps:
      return None
    return self.trajectory.steps[-1]
