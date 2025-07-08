import abc
import jax
from typing import List, Optional
import dataclasses
from flax.nnx import statelib
from flax.nnx import Module

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


class BaseSampler(abc.ABC):
    """Base class for samplers."""

    @property
    @abc.abstractmethod
    def transformer(self) -> Module:
        """Returns the transformer module used by the sampler."""

    @property
    @abc.abstractmethod
    def transformer_state(self) -> statelib.State:
      """Returns the transformer state used by the sampler."""

    @abc.abstractmethod
    def __call__(
          self,
          input_strings: List[str],
          total_generation_steps,
          max_prompt_length=None,
          temperature=0.0,
          top_p=None,
          top_k=None,
          beam_size=None,
          seed=None,
          multi_sampling: int=1,
          return_logits: bool=True,
          echo:bool = False,
          pad_output: bool = False,
      ):
        """Returns a list of generated samples for the input strings."""

    @abc.abstractmethod
    def tokenize(self, input_string: str) -> jax.Array:
        """Returns the tokenized the input string."""


