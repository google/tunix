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

"""Sampler for vLLM-style autoregressive decoding using JAX and NNX models."""

import dataclasses
import math
import os
from typing import Any, Dict, List, Optional, Tuple

from absl import logging
import jax
import jax.numpy as jnp
import jaxtyping
from tunix.generate import base_sampler
from tunix.generate import utils
import tunix.generate.tokenizer_adapter as tok_adapter
from tunix.rl import reshard
from vllm import LLM
from vllm.inputs import TokensPrompt
from vllm.outputs import RequestOutput

# Colocate vllm engine and worker in the main process
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"


@dataclasses.dataclass
class MappingConfig:
  # Mappings for parameter names
  to_hf_mappings: Optional[Dict[str, str]]
  lora_to_hf_mappings: Optional[Dict[str, str]]
  # Mapping for weight transformations (e.g. rescaling)
  to_hf_hook_fns: Optional[Dict[str, callable]]
  # Parameters that need to be transposed
  to_hf_transpose_keys: Optional[Dict[str, Tuple[int, ...]]]
  lora_config: Optional[Dict[str, Any]]


@dataclasses.dataclass
class VllmConfig:
  model_version: str
  max_model_len: int
  mesh: jax.sharding.Mesh
  hbm_utilization: float
  init_with_random_weights: bool
  tpu_backend_type: str
  mapping_config: MappingConfig


class VllmSampler(base_sampler.BaseSampler):  # pylint: disable=invalid-name
  """A sampler for vLLM-style autoregressive decoding using JAX and NNX models.

  This class wraps an NNX model and tokenizer for performing inference
  with optimized KV cache allocation based on available HBM memory.

  Inherits from:
      base_sampler.BaseSampler
  """

  def __init__(
      self,
      tokenizer: Any,
      config: VllmConfig,
  ):
    """Initializes the VllmSampler.

    Args:
        tokenizer (Any): A tokenizer compatible with the model.
        config: The vllm related configurations
    """

    # Select vllm TPU backend type, there are jax, torchax and torchxla
    if config.tpu_backend_type:
      os.environ["TPU_BACKEND_TYPE"] = config.tpu_backend_type
    # Init vLLM model with random weights to speed up bootstrap time, because
    # model weights are synced from trainer later on
    if config.init_with_random_weights:
      os.environ["JAX_RANDOM_WEIGHTS"] = "True"

    self.tokenizer = tok_adapter.TokenizerAdapter(tokenizer)
    self.args = self._vllm_config(config)
    self.llm = LLM(**self.args)

    self.mappings = config.mapping_config.to_hf_mappings
    self.to_hf_transpose_keys = config.mapping_config.to_hf_transpose_keys
    self.to_hf_hook_fns = config.mapping_config.to_hf_hook_fns

    # TODO(b/434959964) It's not taking effect until vLLM Jax backend support
    # lora.
    if (
        config.mapping_config.lora_config is not None
        and config.mapping_config.lora_to_hf_mappings is not None
    ):
      self.mappings |= config.mapping_config.lora_to_hf_mappings

  # TODO(b/434969743): Optimize weight sharing between trainer and vllm sampler.
  # TODO(b/434975493): Consider Release KV cache on the fly
  def update_params(
      self,
      updated_weights: jaxtyping.PyTree,
      filter_types: Optional[Tuple[Any, ...]] = None,
  ):
    del filter_types
    utils.transfer_state_with_mappings(
        src_state=updated_weights,
        dst_state=self.transformer_state,
        key_mappings=self.mappings,
        key_mapping_hook_fns=self.to_hf_hook_fns,
        transpose_keys=self.to_hf_transpose_keys,
        reshard_fn=reshard.reshard_pytree,
    )

  def load_checkpoint(self, path_or_weights: str | jaxtyping.PyTree):
    # TODO(b/434741253): Consider support orbax checkpoint loading
    if isinstance(path_or_weights, jaxtyping.PyTree):
      self.update_params(updated_weights=path_or_weights, filter_types=None)
    else:
      raise NotImplementedError("Only support in memory weight sync as of now.")

  def _find_tp_size(self, mesh: jax.sharding.Mesh) -> int:
    """Finds the tensor parallel size from the mesh."""
    # since vllm doesn't support DP yet, simply return the total rank size.
    return math.prod(mesh.shape.values())

  def _vllm_config(self, config: VllmConfig):
    args = {}
    args["additional_config"] = {}
    args["model"] = config.model_version
    args["max_model_len"] = config.max_model_len
    args["tensor_parallel_size"] = self._find_tp_size(config.mesh)
    args["gpu_memory_utilization"] = config.hbm_utilization
    if config.mapping_config.lora_config is not None:
      args["additional_config"][
          "lora_config"
      ] = config.mapping_config.lora_config
    return args

  @property
  def _model_runner(self):
    return self.llm.llm_engine.model_executor.driver_worker.model_runner

  @property
  def transformer(self):
    # vLLM doesn't expose the underlying model
    return None

  @property
  def transformer_state(self):
    if hasattr(self._model_runner, "state"):
      return self._model_runner.state
    else:
      raise AttributeError("vLLM model runner doesn't have state.")

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

  def detokenize(
      self, input_strings: List[str], request_outputs: List[RequestOutput]
  ) -> Tuple[List[str], List[float], List[int]]:
    generations = len(request_outputs[0].outputs)
    decoded_outputs = [[] for _ in range(generations)]
    out_logprobs = [[] for _ in range(generations)]
    out_tokens = [[] for _ in range(generations)]
    for input_string, multi_sampling_output in zip(
        input_strings, request_outputs
    ):
      for idx, single_output in enumerate(multi_sampling_output.outputs):
        # vLLM still returns 1 eos id even if we ask it to stop at eos.
        if single_output.token_ids[-1] == self.tokenizer.eos_id():
          single_output.token_ids = single_output.token_ids[:-1]
          single_output.logprobs = single_output.logprobs[:-1]

        out_tokens[idx].append(single_output.token_ids)
        decoded_outputs[idx].append(
            self.tokenizer.decode(single_output.token_ids)
        )
        logprobs = utils.get_logprobs_from_vllm_output(
            single_output.token_ids, single_output.logprobs
        )
        out_logprobs[idx].append(logprobs)
        logging.debug(
            "Prompt: %r\n\nGenerated text: %r\n\n ",
            input_string,
            decoded_outputs[idx][-1],
        )
    return decoded_outputs, out_logprobs, out_tokens

  def __call__(
      self,
      input_strings: List[str],
      max_generation_steps: int,
      max_prompt_length: int = None,
      temperature: float = 0.0,
      top_p: float = None,
      top_k: int = None,
      beam_size: int = None,
      seed: int = None,  # vLLM Jax backend doesn't support per request seed.
      multi_sampling: int = 1,
      return_logits: bool = True,
      echo: bool = False,
      pad_output: bool = False,
  ) -> base_sampler.SamplerOutput:
    # max_tokens: maximum number of tokens to generate
    if max_generation_steps > self.args["max_model_len"]:
      raise ValueError(
          "`max_generation_steps` must be less than or equal to "
          "`max_model_len`. Received:  `max_generation_steps`="
          f"{max_generation_steps} and `max_model_len`="
          f"{self.args['max_model_len']}."
      )
    if beam_size is not None:
      self.sampling_params = self.llm.sampling_params.BeamSearchParams(
          beam_width=beam_size,
          max_tokens=max_generation_steps,
          ignore_eos=False,
          temperature=temperature,
      )
    else:
      self.sampling_params = self.llm.get_default_sampling_params()
      self.sampling_params.detokenize = False
      self.sampling_params.max_tokens = max_generation_steps
      self.sampling_params.n = multi_sampling
      self.sampling_params.temperature = temperature
      self.sampling_params.logprobs = 1  # b/428730696
      self.sampling_params.prompt_logprobs = 1  # b/428730696
      self.sampling_params.stop_token_ids = [self.tokenizer.eos_id()]
      self.sampling_params.skip_special_tokens = True

      if top_p is not None:
        self.sampling_params.top_p = top_p
      if top_k is not None:
        self.sampling_params.top_k = top_k

    prompt_ids = [self.tokenize(x) for x in input_strings]
    outputs = self.llm.generate(
        prompts=[TokensPrompt(prompt_token_ids=ids) for ids in prompt_ids],
        sampling_params=self.sampling_params,
        use_tqdm=True,
    )
    decoded_outputs, out_logprobs, out_tokens = self.detokenize(
        input_strings, outputs
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
            jnp.array(x),
            target_length=max_generation_steps,
            pad_value=self.tokenizer.pad_id(),
            left=False,
        )
        for x in out_tokens[0]
    ]
    all_output_ids = jnp.array(all_output_ids)
    # To support multisampling, just return the whole list of SamplerOutput
    return base_sampler.SamplerOutput(
        text=decoded_outputs[0],
        logits=None,
        tokens=all_output_ids,
        padded_prompt_tokens=all_input_ids,
        logprobs=out_logprobs[0],
    )
