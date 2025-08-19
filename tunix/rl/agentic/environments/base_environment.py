from abc import ABC, abstractmethod
from typing import Any
import asyncio

class BaseEnv(ABC):
    """
    Abstract base class for all environments used in multi-turn or single-turn RL tasks.

    Any custom environment should inherit from this class and implement its abstract methods.
    """

    def __init__(self):
        # Optional identifier for multi-env rollout coordination
        self._idx = None

    @property
    def idx(self) -> Any:
        """Return the environment's index (used in batched rollout)."""
        return self._idx

    @idx.setter
    def idx(self, value: Any):
        """Set the environment's index externally."""
        self._idx = value

    @abstractmethod
    def reset(self) -> tuple[dict, dict]:
        """
        Reset the environment to its initial state.

        Returns:
            A tuple (initial_observation, info_dict)
        """
        pass

    @abstractmethod
    def step(self, action: Any) -> tuple[Any, float, bool, dict]:
        """
        Take one step in the environment.

        Args:
            action: The model's response or tool call(s)

        Returns:
            A tuple (next_observation, reward, done, info)
        """
        pass

    async def step_async(self, action: Any) -> tuple[Any, float, bool, dict]:
        """
        Async wrapper for step().
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.step, action)

    def close(self):
        """Clean up any resources (optional override)."""
        pass

    @staticmethod
    @abstractmethod
    def from_dict(env_args: dict) -> "BaseEnv":
        """
        Create an environment from a dictionary.

        Used to support YAML-based config or parallel instantiation.

        Args:
            env_args: Dictionary of environment initialization parameters

        Returns:
            An instance of a subclass of BaseEnv
        """
        raise NotImplementedError("Subclasses must implement from_dict")
