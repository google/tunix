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
from typing import Any, Optional, Tuple

import jax
import jaxtyping


@dataclasses.dataclass(frozen=True)
class CacheConfig:
  """Configuration for the KV cache."""

  cache_size: int
  num_layers: int
  num_kv_heads: int
  head_dim: int


@dataclasses.dataclass
class RolloutOutput:
  """Output of the rollout worker."""

  # Generated samples from the model.
  text: list[str]

  # Per-step logits used during sampling.
  logits: jax.Array

  # Tokens corresponding to the generated samples.
  tokens: jax.Array

  # Left padded prompt tokens.
  # TODO(tsbao): Reconcile with vLLM output and see if we should remove this
  # field, or add prompt + generated as extra.
  left_padded_prompt_tokens: jax.Array

  # The log probs from sampler generations.
  logprobs: list[float] | None


@dataclasses.dataclass
class RolloutConfig:
  """Configuration for the rollout worker.

  Fields should be mapped to a subset of vLLM sampling knobs
  https://docs.vllm.ai/en/v0.6.4/dev/sampling_params.html
  """

  # Maximum number of tokens to generate per output sequence
  max_tokens_to_generate: int = 64

  # Float that controls the randomness of the sampling.
  # Lower values make the model more deterministic, while higher values make the
  # model more random. Zero means greedy sampling.
  temperature: float = 0.9

  # Float that controls the cumulative probability of the top tokens to
  # consider. Must be in (0, 1]. Set to 1 to consider all tokens.
  top_p: float | None = 1.0

  # Integer that controls the number of top tokens to consider. Set to -1 to
  # consider all tokens.
  top_k: int | None = None

  # Random seed to use for the generation.
  seed: jax.Array | None = None

  # Maximum length of the prompt. The prompt will be padded/truncated to this
  # length.
  max_prompt_length: int = 64

  # Only used for vanilla rollout engine.
  kv_cache_size: int = 1024  # Only used for vanilla rollout engine.


class BaseRollout(abc.ABC):
  """Base RolloutWorker."""

  @abc.abstractmethod
  def generate(
      self,
      prompts: list[str],
      rollout_config: RolloutConfig,
      **kwargs,
  ) -> RolloutOutput:
    """Generates samples from the model."""

  @abc.abstractmethod
  def get_per_token_logps(
      self,
      prompt_tokens: jax.Array,
      completion_tokens: jax.Array,
      completion_mask: jax.Array | None = None,
  ) -> jax.Array:
    """Returns per-token log probabilities from the model."""

  @abc.abstractmethod
  def update_params(
      self,
      params: jaxtyping.PyTree,
      filter_types: Optional[Tuple[Any, ...]] = None,
  ) -> None:
    """Updates the rollout model parameters."""

  @abc.abstractmethod
  def pad_id(self) -> int:
    """Returns the pad id."""

  @abc.abstractmethod
  def eos_id(self) -> int:
    """Returns the eos id."""

  @abc.abstractmethod
  def model(self) -> Any:
    """Returns the rollout model."""
