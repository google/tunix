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

"""Vanilla rollout worker with Tunix sampler."""

import dataclasses
import itertools
from typing import Any, Optional, Tuple

from flax import nnx
import jax
import jaxtyping
from tunix.generate import sampler
from tunix.rl import common
from tunix.rl.rollout import base_rollout


class VanillaRollout(base_rollout.BaseRollout):
  """Vanilla rollout worker."""

  def __init__(
      self,
      model: nnx.Module,
      tokenizer: Any,
      cache_config_or_size: base_rollout.CacheConfig,
  ):
    self._sampler = sampler.Sampler(
        model,
        tokenizer,
        sampler.CacheConfig(**dataclasses.asdict(cache_config_or_size)),
    )
    # Counts generate() calls so that every call gets its own PRNG key.
    self._generate_calls = itertools.count()

  def _next_seed(self, seed: int | jax.Array | None) -> jax.Array:
    """Returns a fresh PRNG key for this generate() call.

    The sampler maps `seed=None` to `PRNGKey(0)`, so passing the config seed
    through unchanged gives every call the same key. Agentic RL calls
    generate() once per trajectory turn with a single prompt, so all
    `num_generations` rollouts of a group sampled identically. Folding a
    per-call counter into the key keeps runs reproducible for a fixed seed
    while making successive calls differ.

    Args:
      seed: The seed from the rollout config: an int, a PRNG key, or None
        (treated as 0).

    Returns:
      A PRNG key unique to this call.
    """
    if seed is None:
      seed = 0
    key = jax.random.PRNGKey(seed) if isinstance(seed, int) else seed
    return jax.random.fold_in(key, next(self._generate_calls))

  def generate(
      self,
      prompts: list[str] | None,
      rollout_config: base_rollout.RolloutConfig,
      *,
      prompt_token_ids=None,
      **kwargs,
  ) -> base_rollout.RolloutOutput:
    """Generates samples from the model."""
    output = self._sampler(
        input_strings=prompts,
        max_generation_steps=rollout_config.max_tokens_to_generate,
        max_prompt_length=rollout_config.max_prompt_length,
        echo=False,
        temperature=rollout_config.temperature,
        top_p=rollout_config.top_p,
        top_k=rollout_config.top_k,
        seed=self._next_seed(rollout_config.seed),
        pad_output=False,
        eos_tokens=rollout_config.eos_tokens,
        return_logprobs=rollout_config.return_logprobs,
        prompt_token_ids=prompt_token_ids,
    )
    return base_rollout.RolloutOutput(
        text=output.text,
        logits=output.logits,  # pyrefly: ignore[bad-argument-type]
        tokens=output.tokens,  # pyrefly: ignore[bad-argument-type]
        left_padded_prompt_tokens=output.padded_prompt_tokens,
        logprobs=output.logprobs,  # pyrefly: ignore[bad-argument-type]
        prompt_lengths=output.prompt_lengths,
    )

  def get_per_token_logps(
      self,
      prompt_tokens: jax.Array,
      completion_tokens: jax.Array,
  ) -> jax.Array:
    """Returns per-token log probabilities from the rollout policy."""
    graphdef, state = self._sampler.model_def_and_state()
    return common.compute_per_token_logps(
        graphdef,
        state,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        pad_id=self.pad_id(),
        eos_id=self.eos_id(),
        stop_gradient=True,
    )

  def update_params(
      self,
      params: jaxtyping.PyTree,
      filter_types: Optional[Tuple[Any, ...]] = None,
  ) -> None:
    self._sampler.update_params(params, filter_types)

  def pad_id(self) -> int:
    return self._sampler.tokenizer.pad_id()

  def eos_id(self) -> int:
    return self._sampler.tokenizer.eos_id()

  def model(self) -> nnx.Module:
    return self._sampler.transformer
