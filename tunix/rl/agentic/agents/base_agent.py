from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any
import asyncio

# ────────────────────────────────────────────────────────────────
# Basic Data Structures
# ────────────────────────────────────────────────────────────────

@dataclass
class Step:
    # The conversational context sent to the LLM, equivalent to OpenAI Chat API's `messages`
    chat_completions: list[dict[str, str]] = field(default_factory=list)

    # The reasoning or chain-of-thought notes inferred by the Agent at this step
    thought: str = ""

    # Structured action parsed from LLM output (e.g., tool invocation parameters)
    action: Any = None

    # Observation returned by the environment after executing the action (could be text, JSON, image, etc.)
    observation: Any = None

    # Raw text response from the LLM
    model_response: str = ""

    # Additional metadata: timestamp, debugging info, trace id, etc.
    info: dict = field(default_factory=dict)

    # Immediate reward for this step, calculated by the environment
    reward: float = 0.0

    # Whether the task is terminated (True means the episode is over)
    done: bool = False

    # Discounted return from this step to the end (Monte Carlo return), filled in by the engine
    mc_return: float = 0.0


@dataclass
class Action:
    # Container for structured action; content depends on the specific environment
    action: Any = None


@dataclass
class Trajectory:
    # Description of the current task or episode (question, initial prompt, etc.)
    task: Any = None

    # List of Steps stacked in temporal order under this task
    steps: list[Step] = field(default_factory=list)

    # Total reward of the entire episode (either accumulated or provided at once by the environment)
    reward: float = 0.0

    def to_dict(self):
        """Convert to dictionary format for serialization or logging"""
        return {
            "steps": [asdict(step) for step in self.steps],
            "reward": float(self.reward),
        }


# ────────────────────────────────────────────────────────────────
# Abstract Base Class for Agent
# ────────────────────────────────────────────────────────────────

class BaseAgent(ABC):
    # —— Property Interface ————————————————————————————————

    @property
    def chat_completions(self) -> list[dict[str, str]]:
        """
        Returns the list of messages to send to the LLM.
        Subclasses typically construct this from internal state (e.g., history, tool calls).
        """
        return []

    @property
    def trajectory(self) -> Trajectory:
        """
        Returns the full trajectory object of the current task.
        The engine uses this object to read/write Steps.
        """
        return Trajectory()

    # —— Interaction with Environment ————————————————————————————————

    @abstractmethod
    def update_from_env(
        self,
        observation: Any,
        reward: float,
        done: bool,
        info: dict,
        **kwargs,
    ):
        """
        Called after one step of environment execution, used to:
          1. Write observation / reward into the latest Step
          2. Update internal state (e.g., memory, tool cache)
        """
        ...

    async def update_from_env_async(self, *args, **kwargs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.update_from_env, *args, **kwargs)


    # —— Interaction with Model ————————————————————————————————

    @abstractmethod
    def update_from_model(self, response: str, **kwargs) -> Action:
        """
        Called after the LLM generates a response:
          1. Parse response → structured Action
          2. Write response / thought into the latest Step
        The returned Action will be executed in the environment.
        """
        ...

    async def update_from_model_async(self, *args, **kwargs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.update_from_model, *args, **kwargs)

    # —— Lifecycle Control ————————————————————————————————

    @abstractmethod
    def reset(self):
        """
        Called at the beginning of a new episode to clear historical state:
          - Reset trajectory
          - Reset internal caches
        """
        ...

    # —— Debugging Helper ————————————————————————————————

    def get_current_state(self) -> Step | None:
        """
        For debugging: directly access the latest Step.
        Returns None if no Step has been generated yet.
        """
        if not self.trajectory.steps:
            return None
        return self.trajectory.steps[-1]
