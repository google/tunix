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

"""Base classes for reinforcement learning environments.

This module defines:

* `BaseEnv`: the minimal abstract base class that provides a standard
  interface for environments used in reinforcement learning tasks.

* `BaseTaskEnv`: a higher-level template environment that implements
  common episode management logic for task-based environments, including
  `max_steps` and step counting. Most single-turn and multi-turn task
  environments should subclass this instead of `BaseEnv` directly.
"""

import abc
import asyncio
import dataclasses
from typing import Any, Callable, Dict


ABC = abc.ABC
abstractmethod = abc.abstractmethod


@dataclasses.dataclass
class EnvStepResult:
  """Container for the result of a single environment step.

  This is used internally by `BaseTaskEnv`'s `_step_impl` template method,
  so that concrete environments only need to implement their own per-step
  logic, without worrying about step counters or max_steps truncation.

  Attributes:
    observation: The resulting observation after this step.
    reward: Numerical reward signal for this step.
    done: Whether the episode has terminated at this step.
    info: Additional metadata, debug info, or metrics.
  """

  observation: Any
  reward: float
  done: bool
  info: Dict[str, Any]


class BaseEnv(ABC):
  """Abstract base class for reinforcement learning environments.

  This class defines the minimal standard interface for environments used in
  both single-turn and multi-turn RL tasks. All environments must implement
  the `reset` and `step` methods, and may optionally override `step_async`,
  `close`, and `from_dict`.

  Higher-level templates such as `BaseTaskEnv` can build on top of this
  to provide shared logic for common patterns (task + reward_fn + max_steps).
  """

  def __init__(self):
    """Initialize the base environment.

    Sets up the environment index as None, which can be assigned later
    for coordination in multi-environment batched rollout scenarios.
    """
    # Environment index for batched rollout coordination and identification.
    self._idx = None

  @property
  def idx(self) -> Any:
    """Get the environment's assigned index.

    The index is used to identify this environment instance within a batch
    of environments during parallel rollouts or distributed training.

    Returns:
      Any: The environment's index, can be int, string, or any identifier type.
    """
    return self._idx

  @idx.setter
  def idx(self, value: Any):
    """Set the environment's index.

    This setter allows external systems (like rollout managers) to assign
    an identifier to this environment instance for tracking and coordination.

    Args:
      value (Any): The index value to assign to this environment.
    """
    self._idx = value

  @abstractmethod
  def reset(self) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reset the environment to its initial state for a new episode.

    This method should restore the environment to a clean starting condition,
    clear any episode-specific state, and return the initial observation that
    the agent will use to begin interaction.

    Returns:
      tuple[dict[str, Any], dict[str, Any]]: A tuple containing:
        - initial_observation (dict): The starting state/observation for
          the agent.
        - info_dict (dict): Additional metadata about the reset (debug info,
          metrics, etc.).
    """
    raise NotImplementedError

  @abstractmethod
  def step(self, action: Any) -> tuple[Any, float, bool, Dict[str, Any]]:
    """Execute one interaction step in the environment.

    Takes an action from the agent (typically a model response or tool call),
    processes it within the environment's dynamics, and returns the resulting
    state transition information following the standard RL interface.

    Args:
      action (Any): The agent's action to execute. Can be a string response,
        structured tool call, or any action format specific to the environment.

    Returns:
      tuple[Any, float, bool, dict]: A tuple containing:
        - next_observation (Any): The resulting state/observation after the
          action.
        - reward (float): Numerical reward signal for the action taken.
        - done (bool): Whether the episode has terminated (success, failure,
          or limit reached).
        - info (dict): Additional step information (metrics, debug data,
          intermediate results).
    """
    raise NotImplementedError

  async def step_async(
      self, action: Any
  ) -> tuple[Any, float, bool, Dict[str, Any]]:
    """Asynchronous version of the step method.

    Provides non-blocking execution of environment steps by wrapping the
    synchronous step() method in an executor. This is useful for environments
    that perform I/O operations or when running multiple environments
    concurrently.

    Args:
      action (Any): The agent's action to execute, same format as step().

    Returns:
      tuple[Any, float, bool, dict]: Same return format as step() method.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, self.step, action)

  def close(self):
    """Clean up environment resources.

    Override this method in subclasses to perform any necessary cleanup
    such as closing files, network connections, releasing memory, or
    shutting down external processes. The base implementation does nothing.
    """
    # Default implementation: nothing to clean up.
    return

  @classmethod
  def from_dict(cls, env_args: Dict[str, Any]) -> "BaseEnv":
    """Create an environment instance from a configuration dictionary.

    This factory method enables environment creation from configuration files
    (YAML, JSON) and supports dynamic instantiation in distributed or
    parallelized training scenarios where environments need to be created
    from serialized parameters.

    The default implementation assumes that the environment's constructor
    accepts the configuration keys as keyword arguments:

      env = MyEnv.from_dict({"foo": 1, "bar": 2})
      # is equivalent to:
      env = MyEnv(foo=1, bar=2)

    Subclasses that need custom behavior (e.g., splitting task vs env
    arguments) should override this method.

    Args:
      env_args (dict): Dictionary containing all necessary parameters for
        environment initialization. The structure and required keys are specific
        to each environment subclass.

    Returns:
      BaseEnv: A new instance of the environment subclass, fully initialized
        with the provided parameters.
    """
    return cls(**env_args)


class BaseTaskEnv(BaseEnv):
  """Template base class for task-based RL environments.

  This class extends `BaseEnv` with common episode management logic for
  task-based environments, including:

  * A `task` dictionary holding problem specification and metadata.
  * An optional `reward_fn` that can be used by subclasses.
  * `max_steps` and `step_count` to control and track episode length.

  Subclasses should implement `_initial_observation()` and `_step_impl()`
  instead of overriding `reset()` and `step()` directly. This provides a
  structured pattern for implementing both single-turn and multi-turn
  environments.

  Typical usage:

    class MyEnv(BaseTaskEnv):
      def _initial_observation(self):
        ...

      def _step_impl(self, action):
        # compute observation, reward, done, info
        return EnvStepResult(...)

  The outer `step()` implementation in `BaseTaskEnv` will handle step
  counting and max_steps truncation.
  """

  def __init__(
      self,
      task: Dict[str, Any] | None = None,
      *,
      reward_fn: Callable[..., Any] | None = None,
      max_steps: int = 1,
      **kwargs,
  ):
    """Initialize the task-based environment.

    Args:
      task: Task specification containing problem description, ground truth, or
        other task-specific parameters. If None, defaults to empty dict.
      reward_fn: Optional reward function that subclasses may use to compute
        rewards. The exact signature is defined by the environment.
      max_steps: Maximum number of interaction steps before forced termination.
        Prevents infinite loops and controls episode length.
      **kwargs: Additional arguments that subclasses may use and store.
    """
    super().__init__()
    self.task = task or {}
    self.reward_fn = reward_fn
    self.max_steps = max_steps
    self.step_count = 0
    # Allow subclasses to store any extra initialization arguments if needed.
    self._extra_kwargs = kwargs

  # ---------- Template methods to be implemented by subclasses ----------

  def _initial_observation(self) -> Any:
    """Return the initial observation for a new episode.

    Subclasses must override this to provide the observation returned
    by `reset()`.

    Returns:
      Any: The observation for the agent at the beginning of the episode.
    """
    raise NotImplementedError

  def _step_impl(self, action: Any) -> EnvStepResult:
    """Per-step environment logic.

    Subclasses must override this to implement a single logical step given
    an action from the agent. This method should NOT manage `step_count`
    or `max_steps`; that is handled by `BaseTaskEnv.step()`.

    Args:
      action: The action from the agent.

    Returns:
      EnvStepResult: The outcome of this logical step, including observation,
        reward, done flag, and info.
    """
    raise NotImplementedError

  # --------------------- Public BaseEnv interface -----------------------

  def reset(self) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Reset the environment and return the initial observation.

    This implementation resets the internal step counter and delegates
    construction of the initial observation to `_initial_observation()`.

    Returns:
      A tuple of (observation, info_dict).
    """
    self.step_count = 0
    obs = self._initial_observation()
    return obs, {}

  def step(self, action: Any) -> tuple[Any, float, bool, Dict[str, Any]]:
    """Execute one interaction step in the environment.

    This implementation increments the step counter, delegates the per-step
    logic to `_step_impl()`, and enforces the `max_steps` limit by truncating
    the episode if necessary.

    Args:
      action: The agent's action to execute.

    Returns:
      A tuple of (observation, reward, done, info).
    """
    self.step_count += 1
    result = self._step_impl(action)

    # Enforce maximum episode length; if _step_impl has already finished
    # the episode, keep done=True.
    done = result.done or (self.step_count >= self.max_steps)
    return result.observation, result.reward, done, result.info

  @classmethod
  def from_dict(cls, env_args: Dict[str, Any]) -> "BaseTaskEnv":
    """Create a task-based environment from a configuration dictionary.

    The default implementation assumes that `env_args` can be passed directly
    to the constructor:

      env = MyTaskEnv.from_dict({"task": {...}, "max_steps": 5})
      # is equivalent to:
      env = MyTaskEnv(task={...}, max_steps=5)

    Subclasses that need to split parameters (e.g., separating `task` and
    environment configuration) should override this method.

    Args:
      env_args: Configuration dictionary for this environment.

    Returns:
      BaseTaskEnv: A new environment instance.
    """
    return cls(**env_args)
