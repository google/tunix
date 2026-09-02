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
import numpy as np
import jax
import jaxtyping
from tunix.experimental.generate import engine
from tunix.experimental.generate import sampler as sampler_lib
from tunix.experimental.generate import utils as generate_utils 
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
    
    engine_cache_config = sampler_lib.CacheConfig()
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
    # --- INJECT PROFILING START LOGIC ---
    if not hasattr(self, '_rollout_count'):
        self._rollout_count = 0
    self._rollout_count += 1

    if self._rollout_count == 1:
      options = jax.profiler.ProfileOptions()
      options.python_tracer_level = 1  # Traces every Python function call
      options.host_tracer_level = 2    # Traces CPU XLA/JAX host operations
      # jax.profiler.start_trace("/tmp/xprof_traces", profiler_options=options)
      jax.profiler.start_trace("gs://yatlas-xprof-traces", profiler_options=options)

    
    req_ids = []
    padded_prompts = []
    
    tokenizer = self.engine.tokenizer
    bos_tok = [tokenizer.bos_id()] if hasattr(tokenizer, 'bos_id') and tokenizer.bos_id() else []

        
    for i, prompt in enumerate(prompts):
        req_id = f"rl_rollout_{i}_{id(self)}"
        """
        input_ids = tokenizer.encode(prompt)
        if hasattr(tokenizer, 'dedup_bos_ids'):
            input_ids = tokenizer.dedup_bos_ids(bos_tok + input_ids)
        else:
            input_ids = bos_tok + input_ids
            
        padded_prompts.append(input_ids)
        """
        self.engine.add_request(req_id, prompt)
        req_ids.append(req_id)
    
    res = []  
    sampling_config = sampler_lib.SamplingConfig(
      temperature=rollout_config.temperature,
      top_p=rollout_config.top_p,
      top_k=rollout_config.top_k,
      return_logprobs=rollout_config.return_logprobs,
      # eos_tokens=rollout_config.eos_tokens,
    )

      
    while self.engine.has_unfinished_requests():
        completed = self.engine.step(
          sampling_config,
          # eos_tokens=rollout_config.eos_tokens
        )

        res.extend(completed)

    out_tokens = []
    decoded_texts = []
    max_prompt_len = rollout_config.max_prompt_length
    if max_prompt_len % 2 != 0:
      max_prompt_len = generate_utils.next_power_of_2(max_prompt_len)

    for req in res:
        token_ids = np.array(req.token_ids)
        prompt_tokens = token_ids[:req.prompt_length]
        gen_tokens = token_ids[req.prompt_length:]
        # gen_tokens = self.engine.generated_tokens[req_id]
        padded_prompts.append(
            generate_utils.pad_to_length(
                np.array(prompt_tokens, dtype=np.int32),
                target_length=max_prompt_len,
                pad_value=tokenizer.pad_id(),
                left=True,
            )
        )
        out_tokens.append(gen_tokens)
        
        if hasattr(tokenizer, "decode"):
             decoded_texts.append(tokenizer.decode(gen_tokens))
        else:
             decoded_texts.append("".join(str(t) for t in gen_tokens))

    # Ensure all device operations finish before we count the step as complete
    # --- INJECT PROFILING STOP LOGIC ---
    if self._rollout_count == 3:
        # Stop capturing after the 3rd rollout completes
        jax.profiler.stop_trace()
    # -----------------------------------


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
    return self.engine.tokenizer.pad_id()

  def eos_id(self) -> int:
    return self.engine.tokenizer.eos_id()

  def model(self) -> nnx.Module:
    return self.engine.sampler.transformer
