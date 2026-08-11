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
import numpy as np
from tunix.generate import continuous_async_driver
from tunix.generate import continuous_sampler
from tunix.generate import sampler
from tunix.rl import common
from tunix.rl import reshard
from tunix.rl import utils
from tunix.rl.rollout import base_rollout

@dataclasses.dataclass(frozen=True)
class CacheConfig:
  """Serving & execution config (decoupled from ModelConfig)."""
  # Paged memory allocation
  page_size: int = 128 		 
  max_num_seqs: int = 32
  max_prompt_length: int = 4096
  max_tokens_to_generate: int = 1024
  hbm_cache_max_bytes: int = 20 * 1024 **3 # 20 GiB
  # Host-RAM Prefix Cache
  host_cache_max_bytes: int = 300 * 1024**3  # 300 GiB

class VanillaRollout(base_rollout.BaseRollout):
  """Vanilla rollout worker with continuous sampling support."""

  def __init__(
      self,
      model: nnx.Module,
      tokenizer: Any,
      cache_config: CacheConfig,
      use_continuous_sampling: bool = True,
      server_mode: bool = False,
  ):
    self._model = model
    self._tokenizer = tokenizer
    self.cache_config = cache_config
    self.use_continuous_sampling = use_continuous_sampling
    self.server_mode = server_mode

    if self.use_continuous_sampling:
      self._continuous_sampler = continuous_sampler.VanillaSampler(
          model,
          tokenizer,
          cache_config,
      )
      self._driver = None
      if self.server_mode:
        self._driver = continuous_async_driver.VanillaInProcessDriver(
            sampler=self._continuous_sampler,
            sampling_config=continuous_sampler.SamplingConfig(
                max_generation_steps=800,
                max_prompt_length=256
            ),
        )
        self._driver.start()
    else:
      self._sampler = sampler.Sampler(
          model,
          tokenizer,
          sampler.CacheConfig(
              cache_size=cache_config.max_num_seqs,
              num_layers=model.config.num_layers,
              num_kv_heads=model.config.num_kv_heads,
              head_dim=model.config.head_dim,
              max_seq_len=cache_config.max_prompt_length + cache_config.max_tokens_to_generate,
              page_size=cache_config.page_size,
          ),
      )

  def generate(
      self,
      prompts: list[str],
      rollout_config: base_rollout.RolloutConfig,
      **kwargs,
  ) -> base_rollout.RolloutOutput:
    """Generates samples from the model."""
    if self.use_continuous_sampling:
      if self.server_mode:
        return self._generate_server_mode(prompts, rollout_config, **kwargs)
      return self._generate_continuous(prompts, rollout_config, **kwargs)

    output = self._sampler(
        input_strings=prompts,
        max_generation_steps=rollout_config.max_tokens_to_generate,
        max_prompt_length=rollout_config.max_prompt_length,
        echo=False,
        temperature=rollout_config.temperature,
        top_p=rollout_config.top_p,
        top_k=rollout_config.top_k,
        seed=rollout_config.seed,  # pyrefly: ignore[bad-argument-type]
        pad_output=False,
        eos_tokens=rollout_config.eos_tokens,
        return_logprobs=rollout_config.return_logprobs,
    )
    return base_rollout.RolloutOutput(
        text=output.text,
        logits=output.logits,  # pyrefly: ignore[bad-argument-type]
        tokens=output.tokens,  # pyrefly: ignore[bad-argument-type]
        left_padded_prompt_tokens=output.padded_prompt_tokens,
        logprobs=output.logprobs,  # pyrefly: ignore[bad-argument-type]
    )

  def _generate_continuous(
      self,
      prompts: list[str],
      rollout_config: base_rollout.RolloutConfig,
      **kwargs,
  ) -> base_rollout.RolloutOutput:
    sampling_config = continuous_sampler.SamplingConfig(
        max_generation_steps=rollout_config.max_tokens_to_generate,
        temperature=rollout_config.temperature,
        top_p=rollout_config.top_p,
        top_k=rollout_config.top_k,
        seed=rollout_config.seed,
        eos_tokens=rollout_config.eos_tokens,
    )
    sampling_state = self._continuous_sampler.init_sample_state(sampling_config)

    req_dicts = [
        {
            "id": f"sync_{i}",
            "prompt": prompt,
            "max_new_tokens": rollout_config.max_tokens_to_generate,
            "eos_tokens": rollout_config.eos_tokens,
        }
        for i, prompt in enumerate(prompts)
    ]

    completed: dict[str, continuous_sampler.RequestOutput] = {}
    step_reqs = req_dicts
    while len(completed) < len(prompts):
      sampling_state, outputs = self._continuous_sampler._sample_step(sampling_state, step_reqs)
      step_reqs = []
      completed.update(outputs)
      print(f"Completions: {len(completed)} / {len(prompts)}")
    
    print("DONE :)")
    del sampling_state

    results = [completed[f"sync_{i}"] for i in range(len(prompts))]
    return base_rollout.RolloutOutput(
        text=[r.text for r in results],
        logits=np.array([r.logits for r in results]),
        tokens=[r.tokens for r in results],
        left_padded_prompt_tokens=np.array([r.padded_tokens for r in results]),
        logprobs=np.array([r.logprobs for r in results]),
    )

  def _generate_server_mode(
      self,
      prompts: list[str],
      rollout_config: base_rollout.RolloutConfig,
      **kwargs,
  ) -> base_rollout.RolloutOutput:
    req_dicts = [
        {"id": f"req_{i}_{id(prompt)}", "prompt": prompt, "max_new_tokens": rollout_config.max_tokens_to_generate}
        for i, prompt in enumerate(prompts)
    ]
    futures = self._driver.submit_requests(req_dicts)
    results = [fut.result() for fut in futures]

    return base_rollout.RolloutOutput(
        text=[r.text for r in results],
        logits=None,
        tokens=[r.tokens for r in results],
        left_padded_prompt_tokens=[],
        logprobs=None,
    )

  def get_per_token_logps(
      self,
      prompt_tokens: jax.Array,
      completion_tokens: jax.Array,
  ) -> jax.Array:
    """Returns per-token log probabilities from the rollout policy."""
    if self.use_continuous_sampling:
      graphdef, state = nnx.split(self._model)
      return common.compute_per_token_logps(
          graphdef,
          state,
          prompt_tokens=prompt_tokens,
          completion_tokens=completion_tokens,
          pad_id=self.pad_id(),
          eos_id=self.eos_id(),
          stop_gradient=True,
      )
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
    """
    if self.use_continuous_sampling:
      self._continuous_sampler.update_params(params)
      return
    """

    if filter_types is not None:
      dst_params = nnx.state(self.model(), filter_types)
      resharded_params = reshard.reshard_pytree(params, dst_params)
    else:
      resharded_params = params
    flat_new_params, _ = utils.to_flat_dict(resharded_params)
    # TODO(linchai): Cast on rollout devices when from lower precision to
    # higher precision.
    new_params_precision = jax.tree.leaves(flat_new_params)[0].dtype
    rollout_precision = jax.tree.leaves(self._sampler.transformer_state)[
        0
    ].dtype
    if new_params_precision != rollout_precision:
      flat_new_params = jax.tree.map(
          lambda x: x.astype(rollout_precision), flat_new_params
      )
    flat_old_params, tree_def = utils.to_flat_dict(
        self._sampler.transformer_state
    )
    merged_params = functools.reduce(
        operator.ior, [flat_old_params, flat_new_params], {}
    )
    merged_params = jax.tree.unflatten(tree_def, merged_params.values())
    new_model = nnx.merge(self._sampler._transformer_graphdef, merged_params)  # pylint: disable=protected-access  # pyrefly: ignore[no-matching-overload]
    self._sampler.transformer_state = nnx.variables(new_model, nnx.Param)

  def pad_id(self) -> int:
    val = getattr(self._tokenizer, "pad_id", 0)
    return val() if callable(val) else val

  def eos_id(self) -> int:
    val = getattr(self._tokenizer, "eos_id", 1)
    return val() if callable(val) else val

  def model(self) -> nnx.Module:
    if self.use_continuous_sampling:
      return self._model
    return self._sampler.transformer
