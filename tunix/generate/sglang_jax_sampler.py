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

"""Sampler for sglang-jax-style autoregressive decoding using JAX and NNX models."""

import dataclasses
import math
import os
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from absl import logging
from flax import nnx
import jax
import jax.numpy as jnp
import jaxtyping
from sgl_jax.srt.entrypoints.engine import Engine
from tunix.generate import base_sampler
from tunix.generate import mappings
from tunix.generate import utils
import tunix.generate.tokenizer_adapter as tok_adapter
from tunix.rl import reshard


@dataclasses.dataclass
class SglangJaxConfig:
  model_version: str
  context_length: int
  mesh: jax.sharding.Mesh
  mem_fraction_static: float
  init_with_random_weights: bool
  disable_radix_cache: bool
  enable_deterministic_sampling: bool
  mapping_config: mappings.MappingConfig


class SglangJaxSampler(base_sampler.BaseSampler):  # pylint: disable=invalid-name
  """A sampler for sglang-jax-style autoregressive decoding using JAX and NNX models.

  This class wraps an NNX model and tokenizer for performing inference
  with optimized KV cache allocation based on available HBM memory.

  Inherits from:
      base_sampler.BaseSampler
  """

  def __init__(
      self,
      tokenizer: Any,
      config: SglangJaxConfig,
  ):
    """Initializes the SglangJaxSampler.

    Args:
        tokenizer (Any): A tokenizer compatible with the model.
        config: The sglang-jax related configurations
    """
    self.tokenizer = tok_adapter.TokenizerAdapter(tokenizer)
    self.args = self._sglang_jax_config(config)
    self.engine = Engine(**self.args)

    self.mappings = config.mapping_config.to_hf_mappings
    self.to_hf_transpose_keys = config.mapping_config.to_hf_transpose_keys
    self.to_hf_hook_fns = config.mapping_config.to_hf_hook_fns

  # TODO(b/434969743): Optimize weight sharing between trainer and sglang-jax sampler.
  # TODO(b/434975493): Consider Release KV cache on the fly
  def update_params(
      self,
      updated_weights: jaxtyping.PyTree,
      filter_types: Optional[Tuple[Any, ...]] = None,
  ):
    del filter_types
    new_state = utils.transfer_state_with_mappings(
        src_state=updated_weights,
        dst_state=self.transformer_state,
        key_mappings=self.mappings,
        transpose_keys=self.to_hf_transpose_keys,
        reshard_fn=reshard.reshard_pytree,
    )
    nnx.update(self._model_runner.model, new_state)
    self._model_runner.initialize_jit()  ## need to run initialize_jit to make it effective

  def load_checkpoint(self, path_or_weights: str | jaxtyping.PyTree):
    # TODO(b/434741253): Consider support orbax checkpoint loading
    if isinstance(path_or_weights, jaxtyping.PyTree):
      self.update_params(updated_weights=path_or_weights, filter_types=None)
    else:
      raise NotImplementedError("Only support in memory weight sync as of now.")

  def _find_tp_size(self, mesh: jax.sharding.Mesh) -> int:
    """Finds the tensor parallel size from the mesh."""
    # since sglang-jax doesn't support DP yet, simply return the total rank size.
    return math.prod(mesh.shape.values())

  def _sglang_jax_config(self, config: SglangJaxConfig):
    args = {}
    args["model_path"] = config.model_version
    args["precompile_bs_paddings"] = [1, 64]
    args["precompile_token_paddings"] = [8192]
    args["page_size"] = 64
    args["context_length"] = config.context_length
    args["tp_size"] = self._find_tp_size(config.mesh)
    args["mem_fraction_static"] = config.mem_fraction_static
    args["enable_single_process"] = True
    if config.disable_radix_cache:
      args["disable_radix_cache"] = True
    if config.enable_deterministic_sampling:
      args["enable_deterministic_sampling"] = True
    if config.init_with_random_weights:
      args["load_format"] = "dummy"
    return args

  @property
  def _model_runner(self):
    if "scheduler" in self.engine.scheduler_info:
      return self.engine.scheduler_info[
          "scheduler"
      ].tp_worker.worker.model_runner
    else:
      return None

  @property
  def transformer(self):
    # sglang-jax doesn't expose the underlying model
    return None

  @property
  def transformer_state(self):
    return nnx.split(self._model_runner.model)[1]

  def tokenize(self, input_string: str) -> List[int]:
    """Tokenizes the input string."""
    input_ids = self.tokenizer.encode(input_string)
    bos_tok = (
        [self.tokenizer.bos_id()]
        if (self.tokenizer.bos_id() and input_ids[0] != self.tokenizer.bos_id())
        else []
    )
    eos_tok = (
        [self.tokenizer.eos_id()]
        if input_ids[-1] != self.tokenizer.eos_id()
        else []
    )
    return bos_tok + input_ids + eos_tok

  def __call__(
      self,
      input_strings: List[str],
      max_generation_steps: int,
      max_prompt_length: int = None,
      temperature: float = 0.0,
      top_p: float = None,
      top_k: int = None,
      beam_size: int = None,
      seed: Optional[Union[List[int], int]] = None,
      multi_sampling: int = 1,
      return_logits: bool = True,
      echo: bool = False,
      pad_output: bool = False,
  ) -> base_sampler.SamplerOutput:
    # max_generation_steps: maximum number of tokens to generate
    if max_generation_steps > self.args["context_length"]:
      raise ValueError(
          "`max_generation_steps` must be less than or equal to "
          "`context_length`. Received:  `max_generation_steps`="
          f"{max_generation_steps} and `max_model_len`="
          f"{self.args['context_length']}."
      )

    self.sampling_params = self.engine.get_default_sampling_params()
    self.sampling_params.max_new_tokens = max_generation_steps
    self.sampling_params.n = multi_sampling
    self.sampling_params.temperature = temperature
    self.sampling_params.stop_token_ids = [self.tokenizer.eos_id()]
    self.sampling_params.skip_special_tokens = True

    if top_p is not None:
      self.sampling_params.top_p = top_p
    if top_k is not None:
      self.sampling_params.top_k = top_k
    sampling_params = [
        self.sampling_params.convert_to_dict() for _ in input_strings
    ]
    if seed is not None:
      if type(seed) is List:
        assert len(seed) == len(
            input_strings
        ), "seed and input_strings must have same length"
        for i, seed_i in enumerate(seed):
          sampling_params[i]["sampling_seed"] = seed_i
      else:
        for i, _ in enumerate(input_strings):
          sampling_params[i]["sampling_seed"] = seed

    prompt_ids = [self.tokenize(x) for x in input_strings]
    outputs = self.engine.generate(
        input_ids=[ids for ids in prompt_ids],
        sampling_params=sampling_params,
    )

    max_tokens_length = max(len(x) for x in prompt_ids)

    if max_prompt_length is None or max_prompt_length < max_tokens_length:
      max_prompt_length = utils.next_power_of_2(max_tokens_length)
    all_input_ids = [
        utils.pad_to_length(
            jnp.array(x),
            target_length=max_prompt_length,
            pad_value=self.tokenizer.pad_id(),
            left=True,
        )
        for x in prompt_ids
    ]
    all_input_ids = jnp.array(all_input_ids)

    all_output_ids = [
        utils.pad_to_length(
            jnp.array(x["output_ids"]),
            target_length=max_generation_steps,
            pad_value=self.tokenizer.pad_id(),
            left=False,
        )
        for x in outputs
    ]
    all_output_ids = jnp.array(all_output_ids)
    output_texts = [o["text"] for o in outputs]
    # To support multisampling, just return the whole list of SamplerOutput
    return base_sampler.SamplerOutput(
        text=output_texts,
        logits=None,
        tokens=all_output_ids,
        padded_prompt_tokens=all_input_ids,
        logprobs=None,
    )
