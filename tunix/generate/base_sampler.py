# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Base class for samplers."""
import abc
import dataclasses
from typing import List, Optional

from flax import nnx
from flax.nnx import statelib
import jax

ABC = abc.ABC
abstractmethod = abc.abstractmethod


@dataclasses.dataclass
class SamplerOutput:
  """Output of the sampler."""

  # Decoded samples from the model.
  text: list[str]

  # Per-step logits used during sampling.
  logits: Optional[list[jax.Array] | jax.Array]

  # Tokens corresponding to the generated samples.
  tokens: list[jax.Array] | jax.Array

  # Left padded prompt tokens.
  padded_prompt_tokens: jax.Array

  logprobs: Optional[list[float]]


class BaseSampler(ABC):
  """Base class for samplers."""

  @property
  @abstractmethod
  def transformer(self) -> nnx.Module:
    """Returns the transformer module used by the sampler."""

  @property
  @abstractmethod
  def transformer_state(self) -> statelib.State:
    """Returns the transformer state used by the sampler."""

  @abstractmethod
  def __call__(
      self,
      input_strings: List[str],
      max_generation_steps,
      max_prompt_length=None,
      temperature=0.0,
      top_p=None,
      top_k=None,
      beam_size=None,
      seed=None,
      multi_sampling: int = 1,
      return_logits: bool = True,
      echo: bool = False,
      pad_output: bool = False,
  ):
    """Returns a list of generated samples for the input strings."""

  @abstractmethod
  def tokenize(self, input_string: str) -> jax.Array:
    """Returns the tokenized the input string."""
