"""Inference client for agentic models."""

from __future__ import annotations

import abc
import dataclasses
from typing import Any, Dict, List, Optional

import transformers


PreTrainedTokenizerBase = transformers.PreTrainedTokenizerBase
dataclass = dataclasses.dataclass


@dataclass
class GenerateResult:
  text: str
  finish_reason: str = "stop"
  usage: Optional[Dict[str, Any]] = None
  raw: Optional[Any] = None


class ModelServer(abc.ABC):
  """Base class for model servers."""

  @abc.abstractmethod
  def generate(
      self,
      messages: List[Dict[str, str]],
  ) -> GenerateResult:
    """Generate text from the model."""
    ...


class InferenceModelLocal:
  """Inference client for a local RL cluster."""

  def __init__(
      self,
      rl_cluster: Any,
      rollout_micro_batch_size: int,
      grpo_config: Any,
      tokenizer: PreTrainedTokenizerBase,
  ):
    self.rl_cluster = rl_cluster
    self._rollout_micro_batch_size = rollout_micro_batch_size
    self.grpo_config = grpo_config
    self.tokenizer = tokenizer


  def generate(
      self,
      messages: List[Dict[str, str]],
      mode: Any = "train",
  ) -> Any:
    """Generates output from the local RL cluster.

    Args:
      messages: A list of message dictionaries, each containing 'role' and
        'content'.
      mode: The mode for generation, defaults to "train".

    Returns:
      The output from the RL cluster's generate method.
    """
    formatted_prompt = self.tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )

    messages = [formatted_prompt]
    rollout_output = self.rl_cluster.generate(
        prompts=messages,
        mode=mode,
        micro_batch_size=1,
    )

    return rollout_output.text[0]
