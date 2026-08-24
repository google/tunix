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
from tunix.generate import engine
from tunix.generate import sampler_v2
from tunix.rl import common
from tunix.rl import reshard
from tunix.rl import utils
from tunix.rl.rollout import base_rollout


class VanillaRollout(base_rollout.BaseRollout):
  """Vanilla rollout worker."""

  def __init__(
      self,
      model: nnx.Module,
      tokenizer: Any,
      cache_config_or_size: base_rollout.CacheConfig,
  ):
    engine_cache_config = sampler_v2.CacheConfig()
    engine_cache_config.max_num_seqs = max(256, getattr(cache_config_or_size, "cache_size", 256))
    
    self.engine = engine.LLMEngine(
        transformer=model,
        tokenizer=tokenizer,
        cache_config=engine_cache_config,
    )

  def generate(
      self,
      prompts: list[str],
      rollout_config: base_rollout.RolloutConfig,
      **kwargs,
  ) -> base_rollout.RolloutOutput:
    """Generates samples from the model seamlessly via LLMEngine."""
    req_ids = []
    padded_prompts = []
    
    tokenizer = self.engine.sampler.tokenizer
    bos_tok = [tokenizer.bos_id()] if hasattr(tokenizer, 'bos_id') and tokenizer.bos_id() else []
    
    for i, prompt in enumerate(prompts):
        req_id = f"rl_rollout_{i}_{id(self)}"
        input_ids = tokenizer.encode(prompt)
        if hasattr(tokenizer, 'dedup_bos_ids'):
            input_ids = tokenizer.dedup_bos_ids(bos_tok + input_ids)
        else:
            input_ids = bos_tok + input_ids
            
        padded_prompts.append(input_ids)
        self.engine.add_request(req_id, input_ids)
        req_ids.append(req_id)
        
    while self.engine.has_unfinished_requests():
        self.engine.step(
            temperature=rollout_config.temperature,
            top_p=rollout_config.top_p,
            top_k=rollout_config.top_k,
            return_logprobs=rollout_config.return_logprobs,
            eos_tokens=rollout_config.eos_tokens,
        )

    out_tokens = []
    decoded_texts = []
    for req_id in req_ids:
        gen_tokens = self.engine.generated_tokens[req_id]
        out_tokens.append(gen_tokens)
        
        if hasattr(tokenizer, "decode"):
             decoded_texts.append(tokenizer.decode(gen_tokens))
        else:
             decoded_texts.append("".join(str(t) for t in gen_tokens))

    return base_rollout.RolloutOutput(
        text=decoded_texts,
        logits=[],
        tokens=out_tokens,
        left_padded_prompt_tokens=padded_prompts,
        logprobs=None,
    )

  def get_per_token_logps(
      self,
      prompt_tokens: jax.Array,
      completion_tokens: jax.Array,
  ) -> jax.Array:
    """Returns per-token log probabilities from the rollout policy."""
    graphdef, state = self.engine.sampler.model_def_and_state()
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
    if filter_types is not None:
      dst_params = nnx.state(self.model(), filter_types)
      resharded_params = reshard.reshard_pytree(params, dst_params)
    else:
      resharded_params = params
    flat_new_params, _ = utils.to_flat_dict(resharded_params)
    # TODO(linchai): Cast on rollout devices when from lower precision to
    # higher precision.
    new_params_precision = jax.tree.leaves(flat_new_params)[0].dtype
    rollout_precision = jax.tree.leaves(self.engine.sampler.transformer_state)[
        0
    ].dtype
    if new_params_precision != rollout_precision:
      flat_new_params = jax.tree.map(
          lambda x: x.astype(rollout_precision), flat_new_params
      )
    flat_old_params, tree_def = utils.to_flat_dict(
        self.engine.sampler.transformer_state
    )
    merged_params = functools.reduce(
        operator.ior, [flat_old_params, flat_new_params], {}
    )
    merged_params = jax.tree.unflatten(tree_def, merged_params.values())
    new_model = nnx.merge(self.engine.sampler._transformer_graphdef, merged_params)  # pylint: disable=protected-access  # pyrefly: ignore[no-matching-overload]
    self.engine.sampler.transformer_state = nnx.variables(new_model, nnx.Param)

  def pad_id(self) -> int:
    return self.engine.sampler.tokenizer.pad_id()

  def eos_id(self) -> int:
    return self.engine.sampler.tokenizer.eos_id()

  def model(self) -> nnx.Module:
    return self.engine.sampler.transformer
