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
from typing import Any, Optional, Tuple

from flax import nnx
import jax
import jaxtyping
from tunix.generate import sampler
from tunix.rl import common
from tunix.rl import reshard
from tunix.rl import utils
from tunix.rl.rollout import base_rollout
import numpy as np


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

  def generate(
      self,
      prompts: list[str],
      rollout_config: base_rollout.RolloutConfig,
      **kwargs,
  ) -> base_rollout.RolloutOutput:
    """Generates samples from the model."""
    print("begin rollout generate")
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
        logprobs=None,
    )

#   def _aggregate_outputs(
#       self,
#       outputs: list[Any]
#   ) -> base_rollout.RolloutOutput:
#       """Aggregates the outputs from multiple micro-batches into a single one."""
#       if not outputs:
#           return base_rollout.RolloutOutput(
#               text=[], logits=np.array([]), tokens=np.array([]),
#               left_padded_prompt_tokens=np.array([]), logprobs=None
#           )

#       # Use np.concatenate for array-like outputs (logits, tokens)
#       # and list extension for simple lists (text).
#       aggregated_text = [item for output in outputs for item in output.text]
#       aggregated_logits = np.concatenate([output.logits for output in outputs], axis=0)
#       aggregated_tokens = np.concatenate([output.tokens for output in outputs], axis=0)
#       aggregated_padded_tokens = np.concatenate(
#           [output.padded_prompt_tokens for output in outputs], axis=0
#       )

#       return base_rollout.RolloutOutput(
#           text=aggregated_text,
#           logits=aggregated_logits,
#           tokens=aggregated_tokens,
#           left_padded_prompt_tokens=aggregated_padded_tokens,
#           logprobs=None,
#       )

#   def generate(
#       self,
#       prompts: list[str],
#       rollout_config: base_rollout.RolloutConfig,
#       **kwargs,
#   ) -> base_rollout.RolloutOutput:
#       """
#       Generates samples from the model using micro-batching to prevent OOM errors.
#       """
#       print("Begin rollout generate with micro-batching.")
      
#       # Default to a batch size of 1 if not specified or invalid
#       micro_batch_size = 8
#       if micro_batch_size <= 0:
#           micro_batch_size = len(prompts)
          
#       all_outputs = []
      
#       # Process prompts in micro-batches
#       for i in range(0, len(prompts), micro_batch_size):
#           batch_prompts = prompts[i:i + micro_batch_size]
#           print(f"Processing batch {i // micro_batch_size + 1} with {len(batch_prompts)} prompts...")
          
#           output = self._sampler(
#               input_strings=batch_prompts,
#               total_generation_steps=rollout_config.max_tokens_to_generate,
#               max_prompt_length=rollout_config.max_prompt_length,
#               echo=False,
#               temperature=rollout_config.temperature,
#               top_p=rollout_config.top_p,
#               top_k=rollout_config.top_k,
#               seed=rollout_config.seed,
#               pad_output=True,
#               **kwargs # Pass along any other kwargs
#           )
#           all_outputs.append(output)

#       # Aggregate the results from all micro-batches
#       return self._aggregate_outputs(all_outputs)

  def get_per_token_logps(
      self,
      prompt_tokens: jax.Array,
      completion_tokens: jax.Array,
  ) -> jax.Array:
    """Returns per-token log probabilities from the rollout policy."""
    print("begin rollout logps compute")
    return common.compute_per_token_logps(
        self.model(),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        pad_id=self.pad_id(),
        eos_id=self.eos_id(),
    )

  def update_params(
      self,
      params: jaxtyping.PyTree,
      filter_types: Optional[Tuple[Any, ...]] = None,
  ) -> None:
    if filter_types is not None:
      dst_params = nnx.state(self.model(), filter_types)
      resharded_params = reshard.reshard_pytree(params, dst_params)
    else:
      resharded_params = params
    flat_new_params, _ = utils.to_flat_dict(resharded_params)
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
