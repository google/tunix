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
import logging
import math
import re
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from flax import nnx
import jax
import jax.numpy as jnp
import jaxtyping
import numpy as np
from sgl_jax.srt.entrypoints.engine import Engine
from sgl_jax.srt.utils.common_utils import SUPPORTED_LORA_TARGET_MODULES
from tunix.generate import base_sampler
from tunix.generate import mappings
from tunix.generate import utils
import tunix.generate.tokenizer_adapter as tok_adapter
from tunix.rl import reshard


def update_hf_key_mappings_with_lora(
    mappings: Optional[Dict[str, Any]] = None,
    enable_static_lora: bool = False,
    lora_target_modules: Optional[List] = None,
):
  """Update LoRA key_mapping into hf_key_mapping."""
  if mappings is None or not enable_static_lora or not lora_target_modules:
    return mappings

  # Note: SGLangJax implements the LoRA through wraping the base_layer, so the value in mappings needs to be updated.
  # From 'model.layers.*.mlp.gate_proj.weight' to 'model.layers.*.mlp.gate_proj.base_layer.weight'
  for module in lora_target_modules:
    for src_path, tgt_params in mappings.items():
      if module in src_path:
        tgt_path, sharding = tgt_params
        keys = tgt_path.split(".")
        new_tgt_path = ".".join(keys[:-1]) + ".base_layer." + keys[-1]
        mappings[src_path] = (new_tgt_path, sharding)
        break
  return mappings


@dataclasses.dataclass
class SglangJaxConfig:
  mesh: jax.sharding.Mesh
  mapping_config: mappings.MappingConfig

  model_version: str
  context_length: int

  mem_fraction_static: float = 0.2
  init_with_random_weights: bool = True
  disable_radix_cache: bool = True
  enable_deterministic_sampling: bool = False
  # Note: use_sort_for_toppk_minp may be removed in the future. It depends on SGLang-Jax.
  use_sort_for_toppk_minp: bool = True
  enable_static_lora: bool = False
  enable_single_process: bool = (
      True  # Note: this is required when you run it in pathways.
  )

  lora_target_modules: Optional[List[str]] = None
  max_lora_rank: Optional[int] = None
  lora_scaling: Optional[float] = None

  precompile_token_paddings: Optional[List[int]] = None
  precompile_bs_paddings: Optional[List[int]] = None
  chunked_prefill_size: Optional[int] = -1
  page_size: int = 64
  load_format: str = "auto"
  max_running_requests: int = None
  log_level: str = "info"


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

    self.to_hf_key_mappings = update_hf_key_mappings_with_lora(
        config.mapping_config.to_hf_mappings,
        self.args["enable_static_lora"],
        self.args["lora_target_modules"],
    )
    self.to_hf_transpose_keys = config.mapping_config.to_hf_transpose_keys
    self.to_hf_hook_fns = config.mapping_config.to_hf_hook_fns

    if config.mapping_config.lora_to_hf_mappings:
      self.to_hf_key_mappings |= config.mapping_config.lora_to_hf_mappings

    if config.mapping_config.lora_to_hf_transpose_keys:
      self.to_hf_transpose_keys |= (
          config.mapping_config.lora_to_hf_transpose_keys
      )

    self._logger = logging.getLogger(self.__class__.__name__)

    self._logger.debug(f"{self.to_hf_key_mappings=}")

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
        key_mappings=self.to_hf_key_mappings,
        transpose_keys=self.to_hf_transpose_keys,
        reshard_fn=reshard.reshard_pytree,
        rollout_engine="sglang_jax",
    )
    new_model_state_leaves, _ = jax.tree_util.tree_flatten(new_state)
    self._model_runner.model_state_leaves = new_model_state_leaves

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
    self._validate_config(config)

    args = {}
    args["model_path"] = config.model_version
    args["context_length"] = config.context_length
    args["mem_fraction_static"] = config.mem_fraction_static
    if config.init_with_random_weights:
      args["load_format"] = "dummy"
    args["disable_radix_cache"] = config.disable_radix_cache
    args["enable_deterministic_sampling"] = config.enable_deterministic_sampling
    args["use_sort_for_toppk_minp"] = config.use_sort_for_toppk_minp
    args["enable_static_lora"] = config.enable_static_lora
    args["enable_single_process"] = config.enable_single_process
    if config.lora_target_modules == ["all"]:
      args["lora_target_modules"] = SUPPORTED_LORA_TARGET_MODULES
    else:
      args["lora_target_modules"] = config.lora_target_modules
    args["max_lora_rank"] = config.max_lora_rank
    args["max_loras_per_batch"] = 1
    args["lora_scaling"] = config.lora_scaling
    args["precompile_token_paddings"] = config.precompile_token_paddings
    args["precompile_bs_paddings"] = config.precompile_bs_paddings
    args["chunked_prefill_size"] = config.chunked_prefill_size
    args["page_size"] = config.page_size
    args["tp_size"] = self._find_tp_size(config.mesh)
    args["device_indexes"] = config.mesh.device_ids.flatten().tolist()
    args["load_format"] = config.load_format
    args["max_running_requests"] = config.max_running_requests
    args["enable_engine_loop_run_forever_daemon"] = True
    args["log_level"] = config.log_level

    return args

  def _validate_config(self, config: SglangJaxConfig):
    if config.precompile_token_paddings is not None:
      assert isinstance(config.precompile_token_paddings, List)

    if config.precompile_bs_paddings is not None:
      assert isinstance(config.precompile_bs_paddings, List)

    if config.enable_static_lora:
      assert (
          config.lora_target_modules is not None
          and config.max_lora_rank is not None
          and config.lora_scaling is not None
      )
      # check whether the lora_target_modules are valid
      if config.lora_target_modules != ["all"]:
        for module in config.lora_target_modules:
          if module not in SUPPORTED_LORA_TARGET_MODULES:
            raise ValueError(
                f"{module} in lora_target_modules does not exist in"
                f" {SUPPORTED_LORA_TARGET_MODULES}"
            )

  @property
  def _model_runner(self):
    if "scheduler" in self.engine.scheduler_info:
      return self.engine.scheduler_info[
          "scheduler"
      ].tp_worker.worker.model_runner
    else:
      return None

  @property
  def mesh(self) -> jax.sharding.Mesh:
    if hasattr(self._model_runner, "mesh") and isinstance(
        self._model_runner.mesh, jax.sharding.Mesh
    ):
      return self._model_runner.mesh
    else:
      raise AttributeError(
          "SGLang model runner doesn't have mesh or mesh is not a"
          " jax.sharding.Mesh."
      )

  @property
  def transformer(self):
    # sglang-jax doesn't expose the underlying model
    return None

  @property
  def transformer_state(self):
    return nnx.split(self._model_runner.model)[1]

  def tokenize(self, input_string: str) -> jax.Array | list[int]:
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
      input_strings: str | List[str],
      max_generation_steps: int,
      max_prompt_length: int | None = None,
      temperature: float = 0.0,
      top_p: float | None = None,
      top_k: int | None = None,
      beam_size: int | None = None,
      seed: Optional[Union[List[int], int]] = None,
      multi_sampling: int = 1,
      return_logits: bool = True,
      echo: bool = False,
      pad_output: bool = False,
  ) -> base_sampler.SamplerOutput:
    # max_generation_steps: maximum number of tokens to generate
    if (
        self.args["context_length"] is not None
        and max_generation_steps > self.args["context_length"]
    ):
      raise ValueError(
          "`max_generation_steps` must be less than or equal to "
          "`context_length`. Received:  `max_generation_steps`="
          f"{max_generation_steps} and `max_model_len`="
          f"{self.args['context_length']}."
      )

    if isinstance(input_strings, str):
      input_strings = [input_strings]

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
    outputs = self._generate_with_loop_guard(
        input_ids=[ids for ids in prompt_ids],
        sampling_params=sampling_params,
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

    all_output_ids = [
        utils.pad_to_length(
            np.array(x["output_ids"], dtype=np.int32),
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

  def _generate_with_loop_guard(
      self,
      *,
      input_ids: List[List[int]],
      sampling_params: List[dict],
  ):
    coro = self.engine.async_generate(
        input_ids=input_ids,
        sampling_params=sampling_params,
        stream=False,
    )
    loop = get_or_create_event_loop()

    if not loop.is_running():
      res = loop.run_until_complete(coro)
      return res

    import concurrent

    def wrap_generate():
      loop = get_or_create_event_loop()
      return loop.run_until_complete(coro)

    with concurrent.futures.ThreadPoolExecutor() as executor:
      future = executor.submit(wrap_generate)
      return future.result()


import asyncio


def get_or_create_event_loop():
  try:
    loop = asyncio.get_running_loop()
  except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
  finally:
    return loop
