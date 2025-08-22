"""Base class for reinforcement learning environments.

This module defines the `BaseEnv` abstract base class, which provides a
standard interface for environments used in reinforcement learning tasks.
Subclasses must implement the `reset`, `step`, and `from_dict` methods.
"""

import abc
import asyncio
from typing import Any, Dict

ABC = abc.ABC
abstractmethod = abc.abstractmethod


class BaseEnv(ABC):
  """Abstract base class for reinforcement learning environments.

  This class defines the standard interface for environments used in both
  single-turn and multi-turn RL tasks. All custom environments should inherit
  from this class and implement the required abstract methods to ensure
  compatibility with the RL training framework.

  The environment follows the standard RL interface pattern with reset() and
  step() methods, while also supporting batched rollouts through indexing
  and asynchronous operations for improved performance.
  """

  def __init__(self):
    """Initialize the base environment.

    Sets up the environment index as None, which can be assigned later
    for coordination in multi-environment batched rollout scenarios.
    """
    # Environment index for batched rollout coordination and identification
    self._idx = None

  @property
  def idx(self) -> Any:
    """Get the environment's assigned index.

    The index is used to identify this environment instance within a batch
    of environments during parallel rollouts or distributed training.

    Returns:
        Any: The environment's index, can be int, string, or any identifier type
    """
    return self._idx

  @idx.setter
  def idx(self, value: Any):
    """Set the environment's index.

    This setter allows external systems (like rollout managers) to assign
    an identifier to this environment instance for tracking and coordination.

    Args:
        value (Any): The index value to assign to this environment
    """
    self._idx = value

  @abstractmethod
  def reset(self) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reset the environment to its initial state for a new episode.

    This method should restore the environment to a clean starting condition,
    clear any episode-specific state, and return the initial observation
    that the agent will use to begin interaction.

    Returns:
        tuple[dict[str, Any], dict[str, Any]]: A tuple containing:
            - initial_observation (dict): The starting state/observation for the
            agent
            - info_dict (dict): Additional metadata about the reset (debug info,
            metrics, etc.)
    """
    pass

  @abstractmethod
  def step(self, action: Any) -> tuple[Any, float, bool, dict[str, Any]]:
    """Execute one interaction step in the environment.

    Takes an action from the agent (typically a model response or tool call),
    processes it within the environment's dynamics, and returns the resulting
    state transition information following the standard RL interface.

    Args:
        action (Any): The agent's action to execute. Can be a string response,
          structured tool call, or any action format specific to the environment

    Returns:
        tuple[Any, float, bool, dict]: A tuple containing:
            - next_observation (Any): The resulting state/observation after the
            action
            - reward (float): Numerical reward signal for the action taken
            - done (bool): Whether the episode has terminated (success, failure,
            or limit reached)
            - info (dict): Additional step information (metrics, debug data,
            intermediate results)
    """
    pass

  async def step_async(
      self, action: Any
  ) -> tuple[Any, float, bool, Dict[str, Any]]:
    """Asynchronous version of the step method.

    Provides non-blocking execution of environment steps by wrapping the
    synchronous step() method in an executor. This is useful for environments
    that perform I/O operations or when running multiple environments
    concurrently.

    Args:
        action (Any): The agent's action to execute, same format as step()

    Returns:
        tuple[Any, float, bool, dict]: Same return format as step() method
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, self.step, action)

  def close(self):
    """Clean up environment resources.

    Override this method in subclasses to perform any necessary cleanup
    such as closing files, network connections, releasing memory, or
    shutting down external processes. The base implementation does nothing.
    """
    pass

  @staticmethod
  @abstractmethod
  def from_dict(env_args: Dict[str, Any]) -> "BaseEnv":
    """Create an environment instance from a configuration dictionary.

    This factory method enables environment creation from configuration files
    (YAML, JSON) and supports dynamic instantiation in distributed or
    parallelized training scenarios where environments need to be created
    from serialized parameters.

    Args:
        env_args (dict): Dictionary containing all necessary parameters for
          environment initialization. The structure and required keys are
          specific to each environment subclass.

    Returns:
        BaseEnv: A new instance of the environment subclass, fully initialized
            with the provided parameters

    Raises:
        NotImplementedError: If the subclass hasn't implemented this method
    """
    raise NotImplementedError("Subclasses must implement from_dict")
