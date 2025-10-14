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

"""sglang jax rollout worker with Tunix sampler."""

from typing import Any, Dict, Optional, Tuple

from flax import nnx
import jax
import jaxtyping
from tunix.generate import mappings
from tunix.generate import sglang_jax_sampler
from tunix.rl import common
from tunix.rl.rollout import base_rollout


class SglangJaxRollout(base_rollout.BaseRollout):
  """sglang jax rollout worker."""

  def __init__(
      self,
      model: Any,
      tokenizer: Any,
      mesh: jax.sharding.Mesh,
      model_version: str,
      context_length: int,
      mem_fraction_static: float,
      init_with_random_weights: bool,
      disable_radix_cache: bool,
      enable_deterministic_sampling: bool,
      mapping_config: Optional[mappings.MappingConfig] = None,
      rollout_engine: str = "sglang_jax",
  ):
    self.mesh = mesh
    mapping_config = mappings.MappingConfig.build(
        mapping_obj=mapping_config, model=model, backend=rollout_engine
    )
    self._sampler = sglang_jax_sampler.SglangJaxSampler(
        tokenizer=tokenizer,
        config=sglang_jax_sampler.SglangJaxConfig(
            mesh=mesh,
            context_length=context_length,
            model_version=model_version,
            mem_fraction_static=mem_fraction_static,
            init_with_random_weights=init_with_random_weights,
            disable_radix_cache=disable_radix_cache,
            enable_deterministic_sampling=enable_deterministic_sampling,
            mapping_config=mapping_config,
        ),
    )
    state = nnx.state(model)
    self._sampler.load_checkpoint(state)

  def generate(
      self,
      prompts: list[str],
      rollout_config: base_rollout.RolloutConfig,
      **kwargs,
  ) -> base_rollout.RolloutOutput:
    """Generates samples from the model."""
    self.output = self._sampler(
        input_strings=prompts,
        max_generation_steps=rollout_config.max_tokens_to_generate,
        max_prompt_length=rollout_config.max_prompt_length,
        temperature=rollout_config.temperature,
        top_p=rollout_config.top_p,
        top_k=rollout_config.top_k,
        seed=rollout_config.seed,
        echo=False,
        pad_output=True,
    )

    return base_rollout.RolloutOutput(
        text=self.output.text,
        logits=None,
        tokens=self.output.tokens,
        left_padded_prompt_tokens=self.output.padded_prompt_tokens,
        logprobs=self.output.logprobs,
    )

  def get_per_token_logps(
      self,
      prompt_tokens: jax.Array,
      completion_tokens: jax.Array,
      completion_mask: jax.Array | None = None,
  ) -> jax.Array:
    """Returns per-token log probabilities from the rollout policy."""
    return common.compute_per_token_logps(
        self.model(),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        pad_id=self.pad_id(),
        eos_id=self.eos_id(),
        completion_mask=completion_mask,
    )[0]

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
