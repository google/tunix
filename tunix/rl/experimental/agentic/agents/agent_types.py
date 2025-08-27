"""Agent Data Types.

This module defines the core data structures used throughout the agent system.
These types provide standardized containers for actions, interaction steps,
and complete episode trajectories.
"""

import dataclasses
from typing import Any

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
      action (Any): The action payload, format depends on the environment
  """

  action: Any = None


@dataclass
class Step:
  """Represents a single interaction step in an agent-environment conversation.

  Each Step captures the complete context of one turn: the input to the LLM,
  the model's response and reasoning, the parsed action, the environment's
  response, and associated metadata for tracking and analysis.

  Attributes:
      chat_completions (list[dict[str, str]]): Messages sent to LLM (OpenAI Chat
        API format)
      thought (str): Agent's reasoning or chain-of-thought for this step
      action (Action): Parsed structured action from LLM response
      observation (Any): Environment's response after executing the action
      model_response (str): Raw text output from the language model
      info (dict): Additional metadata (timestamps, debug info, trace IDs, etc.)
      reward (float): Immediate reward signal from environment for this step
      done (bool): Terminal state flag - True if episode has ended
      mc_return (float): Monte Carlo return from this step to episode end
  """

  chat_completions: list[dict[str, str]] = field(default_factory=list)
  thought: str = ""
  action: Action = None
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
      task (Any): Task description, initial prompt, or episode specification
      steps (list[Step]): Chronologically ordered sequence of interaction steps
      reward (float): Total episode reward (cumulative or final environment
        score)
  """

  task: Any = None
  steps: list[Step] = field(default_factory=list)
  reward: float = 0.0

  def to_dict(self) -> dict[str, Any]:
    """Convert trajectory to dictionary format for serialization.

    Useful for logging, storage, or transmission over APIs. All Step objects
    are recursively converted to dictionaries using dataclass serialization.

    Returns:
        dict: Serializable dictionary representation of the trajectory
    """
    return {
        "steps": [asdict(step) for step in self.steps],
        "reward": float(self.reward),
    }
