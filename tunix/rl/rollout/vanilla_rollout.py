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
import functools
import operator
from typing import Any
from flax import nnx
import jax
import jaxtyping
from tunix.generate import sampler
from tunix.rl import common
from tunix.rl import utils
from tunix.rl.rollout import base_rollout


@dataclasses.dataclass(frozen=True)
class CacheConfig:
  """Configuration for the KV cache."""

  cache_size: int
  num_layers: int
  num_kv_heads: int
  head_dim: int


class VanillaRollout(base_rollout.BaseRollout):
  """Vanilla rollout worker."""

  def __init__(
      self, model: nnx.Module, tokenizer: Any, cache_config: CacheConfig
  ):
    self._sampler = sampler.Sampler(
        model,
        tokenizer,
        sampler.CacheConfig(**dataclasses.asdict(cache_config)),
    )

  def generate(
      self,
      prompts: list[str],
      rollout_config: base_rollout.RolloutConfig,
      **kwargs,
  ) -> base_rollout.RolloutOutput:
    """Generates samples from the model."""
    output = self._sampler(
        input_strings=prompts,
        total_generation_steps=rollout_config.max_tokens_to_generate,
        max_prompt_length=rollout_config.max_prompt_length,
        echo=False,
        temperature=rollout_config.temperature,
        top_p=rollout_config.top_p,
        top_k=rollout_config.top_k,
        seed=rollout_config.seed,
        pad_output=True,
    )
    return base_rollout.RolloutOutput(
        text=output.text,
        logits=output.logits,
        tokens=output.tokens,
        left_padded_prompt_tokens=output.padded_prompt_tokens,
    )

  def get_per_token_logps(
      self,
      prompt_tokens: jax.Array,
      completion_tokens: jax.Array,
  ) -> jax.Array:
    """Returns per-token log probabilities from the rollout policy."""
    return common.compute_per_token_logps(
        self.model(),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        pad_id=self.pad_id(),
        eos_id=self.eos_id(),
    )

  def update_params(self, params: jaxtyping.PyTree) -> None:
    flat_new_params, _ = utils.to_flat_dict(params)
    flat_old_params, tree_def = utils.to_flat_dict(
        self._sampler.transformer_state
    )
    merged_params = functools.reduce(
        operator.ior, [flat_old_params, flat_new_params], {}
    )
    merged_params = jax.tree.unflatten(tree_def, merged_params.values())
    new_model = nnx.merge(self._sampler._transformer_graphdef, merged_params)  # pylint: disable=protected-access
    self._sampler.transformer_state = nnx.variables(new_model, nnx.Param)

  def pad_id(self) -> int:
    return self._sampler.tokenizer.pad_id()

  def eos_id(self) -> int:
    return self._sampler.tokenizer.eos_id()

  def model(self) -> nnx.Module:
    return self._sampler.transformer
