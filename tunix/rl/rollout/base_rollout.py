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

"""Base rollout worker interface."""

import abc
import dataclasses
from typing import Any, List, Optional, Tuple

import jax
from jax import numpy as jnp
import jaxtyping
import numpy as np
from tunix.common import configs

CacheConfig = configs.CacheConfig
RolloutConfig = configs.RolloutConfig
from tunix.generate import mappings

ABC = abc.ABC
abstractmethod = abc.abstractmethod


@dataclasses.dataclass
class RolloutOutput:
  """Output of the rollout worker."""

  # Generated samples from the model.
  text: list[str]

  # Unpadded per-step logits used during sampling.
  # TODO(tsbao): consider enforcing this to be np.ndarray as well,
  # but let's solve it as part of the IS effort.
  logits: list[jax.Array] | None

  # Unpadded tokens corresponding to the generated samples.
  # Since tokens need to be transfered to RAM for decoding, we use numpy array
  # here.
  tokens: list[np.ndarray]

  # Left padded prompt tokens.
  # TODO(tsbao): Reconcile with vLLM output and see if we should remove this
  # field, or add prompt + generated as extra.
  left_padded_prompt_tokens: np.ndarray

  # The log probs from sampler generations.
  logprobs: list[np.ndarray] | None

  # Per-generation MoE routing decisions, each `[length, num_layers, top_k]`
  # covering prompt + completion, so training can replay the experts the
  # rollout actually routed through instead of re-running the router. None
  # unless the backend was asked to capture them, or for dense models.
  routed_experts: list[np.ndarray | None] | None = None


class BaseRollout(ABC):
  """Base RolloutWorker."""

  @abstractmethod
  def __init__(self, **kwargs):
    """Initializes the rollout worker."""

  @abstractmethod
  def generate(
      self,
      prompts: list[str],
      rollout_config: RolloutConfig,
      **kwargs,
  ) -> RolloutOutput:
    """Generates samples from the model."""

  @abstractmethod
  def get_per_token_logps(
      self,
      prompt_tokens: jax.Array,
      completion_tokens: jax.Array,
  ) -> jax.Array:
    """Returns per-token log probabilities from the model."""

  @abstractmethod
  def update_params(
      self,
      params: jaxtyping.PyTree,
      filter_types: Optional[Tuple[Any, ...]] = None,
  ) -> None:
    """Updates the rollout model parameters."""

  @abstractmethod
  def pad_id(self) -> int:
    """Returns the pad id."""

  @abstractmethod
  def eos_id(self) -> int:
    """Returns the eos id."""

  @abstractmethod
  def model(self) -> Any:
    """Returns the rollout model."""
