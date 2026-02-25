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

import atexit
import dataclasses
from itertools import count
import os
from typing import Any, Dict, List, Optional, Tuple, Union

from absl import logging
import jax
import jaxtyping
import numpy as np
from tunix.generate import base_sampler
from tunix.generate import tokenizer_adapter as tok_adapter
from tunix.generate import utils
from tunix.generate.mappings import MappingConfig
from tunix.generate.vllm_async_driver import VLLMInProcessDriver
from tunix.rl import reshard
from vllm import LLM
from vllm.engine.arg_utils import EngineArgs
from vllm.inputs import TokensPrompt
from vllm.outputs import RequestOutput
from vllm.sampling_params import BeamSearchParams
from vllm.sampling_params import SamplingParams

# Colocate vllm engine and worker in the main process
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"


@dataclasses.dataclass
class VllmConfig:
  """Vllm rollout configuations."""

  # Sampler related
  server_mode: bool = False
  mapping_config: MappingConfig = dataclasses.field(
      default_factory=MappingConfig
  )

  # vLLM Env vars
  init_with_random_weights: bool = True
  tpu_backend_type: str = "jax"

  # vLLM engine arg related, requires additional processing before passing into engine
  additional_config: Optional[Dict[str, Any]] = None
  enable_dp_attention: bool = False
  hbm_utilization: float = 0.5
  lora_config: Optional[Dict[str, Any]] = None
  mesh: jax.sharding.Mesh = None
  data_parallel_size: int = -1
  tensor_parallel_size: int = -1
  expert_parallel_size: int = 1

  # vLLM engine args that can be directly passed in without additional processing, e.g. max_model_len, async_scheduling, etc.
  engine_kwargs: dataclasses.InitVar[Optional[Dict[str, Any]]] = None
  _processed_engine_kwargs: Dict[str, Any] = dataclasses.field(
      init=False, default_factory=dict
  )

  def __post_init__(self, engine_kwargs: Optional[Dict[str, Any]]):
    engine_kwargs = engine_kwargs or {}
    self._processed_engine_kwargs = engine_kwargs
    if engine_kwargs:
      for key, value in engine_kwargs.items():
        logging.info(
            "Engine kwargs setting key '%s' with value '%s'.", key, value
        )
        setattr(self, key, value)


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

    # vLLM DP only works with the new model design
    if config.data_parallel_size > 1:
      os.environ["NEW_MODEL_DESIGN"] = "1"

    # tpu-inference backend recently removed this environment variable, however
    # still set it here for backward compatibility.
    if config.init_with_random_weights:
      os.environ["JAX_RANDOM_WEIGHTS"] = "1"

    self.tokenizer = tok_adapter.TokenizerAdapter(tokenizer)
    self.config = config
    self.args = self._vllm_config(config)
    self._driver: VLLMInProcessDriver | None = None
    self.llm: LLM | None = None
    self._request_counter = count()

    if config.server_mode:
      self._driver = self._create_driver()
      atexit.register(self.stop)
    else:
      self.llm = LLM(**self.args)

    self.to_hf_key_mappings = dict(config.mapping_config.to_hf_mappings or {})
    self.to_hf_transpose_keys = config.mapping_config.to_hf_transpose_keys
    self.to_hf_hook_fns = config.mapping_config.to_hf_hook_fns

    # TODO(b/434959964) It's not taking effect until vLLM Jax backend support
    # lora.
    if config.lora_config and config.mapping_config.lora_to_hf_mappings:
      self.to_hf_key_mappings |= config.mapping_config.lora_to_hf_mappings

  @property
  def mesh(self) -> jax.sharding.Mesh:
    if hasattr(self._model_runner, "mesh") and isinstance(
        self._model_runner.mesh, jax.sharding.Mesh
    ):
      return self._model_runner.mesh
    else:
      raise AttributeError(
          "vLLM model runner doesn't have mesh or mesh is not a"
          " jax.sharding.Mesh."
      )

  # TODO(b/434969743): Optimize weight sharing between trainer and vllm sampler.
  # TODO(b/434975493): Consider Release KV cache on the fly
  def update_params(
      self,
      updated_weights: jaxtyping.PyTree,
      filter_types: Optional[Tuple[Any, ...]] = None,
  ):
    del filter_types

    if self.to_hf_key_mappings:
      # Mapped Weight Sync (e.g. Vanilla -> vLLM)
      utils.transfer_state_with_mappings(
          src_state=updated_weights,
          dst_state=self.transformer_state,
          key_mappings=self.to_hf_key_mappings,
          key_mapping_hook_fns=self.to_hf_hook_fns,
          transpose_keys=self.to_hf_transpose_keys,
          reshard_fn=reshard.reshard_pytree,
      )
    else:
      # Direct Weight Sync (e.g. MaxText -> MaxText)
      logging.debug(
          "No key mappings configuration found. Proceeding with direct"
          " structural weight synchronization (assuming matching source/target"
          " structures)."
      )

      additional_config = self.config.additional_config or {}
      if "maxtext_config" not in additional_config:
        raise ValueError(
            "Direct weight synchronization is currently supported only for "
            "MaxText models. The required 'maxtext_config' key is missing "
            "from 'additional_config'."
        )

      utils.transfer_state_directly(
          src_state=updated_weights,
          dst_state=self.transformer_state,
          reshard_fn=reshard.reshard_pytree,
      )

  def load_checkpoint(self, path_or_weights: str | jaxtyping.PyTree):
    # TODO(b/434741253): Consider support orbax checkpoint loading
    if isinstance(path_or_weights, jaxtyping.PyTree):
      self.update_params(updated_weights=path_or_weights, filter_types=None)
    else:
      raise NotImplementedError("Only support in memory weight sync as of now.")

  def _vllm_config(self, config: VllmConfig):
    """Setup vllm config from Tunix Vllm config."""
    args = config._processed_engine_kwargs.copy()

    # Init vLLM model with random weights to speed up bootstrap time, because
    # model weights are synced from trainer later on
    if config.init_with_random_weights:
      args["load_format"] = "dummy"

    args["gpu_memory_utilization"] = config.hbm_utilization

    args["additional_config"] = config.additional_config or {}

    if config.lora_config is not None:
      args["additional_config"]["lora_config"] = config.lora_config

    tp, dp, ep = utils.resolve_parallelism_sizes(
        mesh=config.mesh,
        tensor_parallel_size=config.tensor_parallel_size,
        data_parallel_size=config.data_parallel_size,
        expert_parallel_size=config.expert_parallel_size,
    )
    args["tensor_parallel_size"] = tp
    args["data_parallel_size"] = dp

    device_indexes = config.mesh.device_ids.flatten().tolist()
    args["additional_config"]["sharding"] = {
        "sharding_strategy": {
            "expert_parallelism": ep,
            "device_indexes": device_indexes,
            "enable_dp_attention": config.enable_dp_attention,
        }
    }

    return args

  def _build_engine_args(self) -> EngineArgs:
    engine_kwargs = dict(self.args)
    engine_kwargs.setdefault("disable_log_stats", True)
    return EngineArgs(**engine_kwargs)

  def _create_driver(self) -> VLLMInProcessDriver:
    engine_args = self._build_engine_args()
    return VLLMInProcessDriver.from_engine_args(
        engine_args,
    )

  def stop(self):
    logging.debug("Shutting down VLLMInProcessDriver.")
    if self._driver is not None:
      self._driver.shutdown()
      self._driver = None

  @property
  def _model_runner(self):
    if self.llm is not None:
      return self.llm.llm_engine.model_executor.driver_worker.model_runner
    if self._driver is not None:
      return self._driver.llm_engine.model_executor.driver_worker.model_runner
    raise RuntimeError("vLLM engine is not initialized.")

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

  def tokenize(self, input_string: str) -> jax.Array | list[int]:
    """Tokenizes the input string."""
    input_ids = self.tokenizer.encode(input_string)
    bos_tok = [self.tokenizer.bos_id()] if self.tokenizer.bos_id() else []
    return self.tokenizer.dedup_bos_ids(bos_tok + input_ids)

  def detokenize(
      self, input_strings: List[str], request_outputs: List[RequestOutput]
  ) -> Tuple[List[str], List[float], List[int]]:
    """Detokenize the vllm outputs."""
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

        out_tokens[idx].append(
            np.array(single_output.token_ids, dtype=np.int32)
        )
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

  def _generate_server_mode(
      self,
      prompts: List[TokensPrompt],
      sampling_params: Union[SamplingParams, BeamSearchParams],
  ) -> List[RequestOutput]:
    """Generate the response in server mode."""
    if self._driver is None:
      raise RuntimeError("vLLM in-process driver is not initialized.")

    futures = []
    for idx, prompt in enumerate(prompts):
      request_id = str(next(self._request_counter))
      params = sampling_params
      if idx > 0 and hasattr(sampling_params, "clone"):
        params = sampling_params.clone()
      future = self._driver.submit_request(
          request_id=request_id,
          prompt=prompt,
          params=params,
      )
      futures.append(future)

    outputs: List[RequestOutput] = []
    for future in futures:
      result = future.result()
      if not isinstance(result, RequestOutput):
        raise TypeError(
            f"Expected RequestOutput from driver, received {type(result)}."
        )
      outputs.append(result)
    return outputs

  def __call__(
      self,
      input_strings: str | List[str],
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
      **kwargs,
  ) -> base_sampler.SamplerOutput:
    """The entry point API for vLLM Sampler"""
    if isinstance(input_strings, str):
      input_strings = [input_strings]

    # max_tokens: maximum number of tokens to generate
    if max_generation_steps > self.args["max_model_len"]:
      raise ValueError(
          "`max_generation_steps` must be less than or equal to "
          "`max_model_len`. Received:  `max_generation_steps`="
          f"{max_generation_steps} and `max_model_len`="
          f"{self.args['max_model_len']}."
      )
    if beam_size is not None:
      self.sampling_params = BeamSearchParams(
          beam_width=beam_size,
          max_tokens=max_generation_steps,
          ignore_eos=False,
          temperature=temperature,
      )
    else:
      if self._driver is not None:
        diff_params = (
            self._driver.llm_engine.model_config.get_diff_sampling_param()
        )
        if diff_params:
          sampling_params = SamplingParams.from_optional(**diff_params)
        else:
          sampling_params = SamplingParams()
      else:
        sampling_params = self.llm.get_default_sampling_params()
      sampling_params.detokenize = False
      sampling_params.max_tokens = max_generation_steps
      sampling_params.n = multi_sampling
      sampling_params.temperature = temperature
      sampling_params.logprobs = 1  # b/428730696
      sampling_params.prompt_logprobs = 1  # b/428730696
      sampling_params.stop_token_ids = [self.tokenizer.eos_id()]
      sampling_params.skip_special_tokens = True

      if top_p is not None:
        sampling_params.top_p = top_p
      if top_k is not None:
        sampling_params.top_k = top_k
      if seed is not None:
        sampling_params.seed = seed

      if kwargs:
        try:
          sampling_params.update(**kwargs)
          logging.log_first_n(
              logging.INFO,
              "Received additional kwargs that are not explicitly defined in"
              f" the method signature: {kwargs}. These will be forwarded to the"
              " underlying sampler, but please ensure that they are valid.",
              1,
          )
        except Exception as e:
          logging.log_first_n(
              logging.INFO,
              f"Failed to update sampling_params with kwargs: {kwargs}."
              f" Error: {e}",
              1,
          )

      self.sampling_params = sampling_params

    prompt_ids = [self.tokenize(x) for x in input_strings]
    prompt_objects = [TokensPrompt(prompt_token_ids=ids) for ids in prompt_ids]
    if self._driver is not None:
      outputs = self._generate_server_mode(prompt_objects, self.sampling_params)
    else:
      outputs = self.llm.generate(
          prompts=prompt_objects,
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
            np.array(x, dtype=np.int32),
            target_length=max_prompt_length,
            pad_value=self.tokenizer.pad_id(),
            left=True,
        )
        for x in prompt_ids
    ]
    all_input_ids = np.array(all_input_ids, dtype=np.int32)

    # To support multisampling, just return the whole list of SamplerOutput
    return base_sampler.SamplerOutput(
        text=decoded_outputs[0],
        logits=None,
        tokens=out_tokens[0],
        padded_prompt_tokens=all_input_ids,
        logprobs=out_logprobs[0],
    )
