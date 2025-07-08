import dataclasses
import jax
import jaxtyping
import jax.numpy as jnp
import tunix.generate.tokenizer_adapter as tok_adapter
import os

from typing import Dict, Optional, List, Any, Tuple
from flax import nnx
from vllm.entrypoints.llm import LLM, EngineArgs
from vllm.outputs import RequestOutput
from absl import logging
from tunix.generate import utils
from tunix.generate import base_sampler


os.environ["TPU_BACKEND_TYPE"] = "jax"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"


@dataclasses.dataclass
class MappingConfig:
  to_hf_mappings: Optional[Dict[str, str]]
  lora_to_hf_mappings: Optional[Dict[str, str]]
  to_hf_transpose_keys: Optional[Dict[str, Tuple[int]]]
  lora_config: Optional[Dict[str, Any]]


class vLLMSampler(base_sampler.BaseSampler):
  """
  A sampler for vLLM-style autoregressive decoding using JAX and NNX models.

  This class wraps an NNX model and tokenizer for performing inference
  with optimized KV cache allocation based on available HBM memory.

  Args:
      model (nnx.module): The NNX model to transfer the weights to vLLM engine and log probs calculation.
      tokenizer (Any): A tokenizer compatible with the model.
      mesh (jax.sharding.Mesh): The JAX mesh for parallel execution.
      max_model_len (int): Maximum sequence (prompt + generation) length supported by vLLM.
      model_version (Optional[str]): The model version identifier.
          This is currently required for vLLM compatibility, but will be removed in the future.
      to_hf_mappings: The weight name mappings from external model to vLLM model
      lora_to_hf_mappings: The lora weight name mappings from external model to vLLM model
      to_hf_transpose_keys: The mapping from the partial or full key name to the tranpose indexes
      lora_config (Optional[Dict[str, Any]], optional): Optional LoRA configuration for fine-tuning.
      hbm_utilization (Optional[float], optional): Fraction of HBM memory to utilize for vLLM. Mainly for KV cache size tuning.
          Defaults to 0.3.

  Inherits from:
      base_sampler.BaseSampler
  """
  def __init__(
        self,
        tokenizer: Any,
        mesh: jax.sharding.Mesh,
        max_model_len: int,
        model_version: str,
        mapping_config: MappingConfig,
        hbm_utilization: Optional[float] = 0.3,
    ):
    self.tokenizer = tok_adapter.TokenizerAdapter(tokenizer)
    self.model_version = model_version
    self.max_model_len = max_model_len
    self.mesh = mesh
    self.lora_config = mapping_config.lora_config
    self.hbm_utilization = hbm_utilization

    self.args = self._vllm_config()
    self.llm = LLM(**self.args)

    self.mappings = mapping_config.to_hf_mappings
    self.to_hf_transpose_keys  = mapping_config.to_hf_transpose_keys

    # TODO(b/434959964) It's not taking effect until vLLM Jax backend support lora.
    if mapping_config.lora_config is not None \
      and mapping_config.lora_to_hf_mappings is not None:
        self.mappings |= mapping_config.lora_to_hf_mappings

  # TODO(b/434969743): Optimize weight sharing between trainer and vllm sampler.
  # TODO(b/434975493): Consider Release KV cache on the fly
  def update_params(self, updated_weights: jaxtyping.PyTree):
    self.llm.llm_engine.model_executor.collective_rpc(
      "sync_weights",
      args=(updated_weights, self.mappings, self.to_hf_transpose_keys))

  def load_checkpoint(self, path_or_weights: str | jaxtyping.PyTree):
    # TODO(b/434741253): Consider support orbax checkpoint loading
    if isinstance(path_or_weights, jaxtyping.PyTree):
      self.update_params(updated_weights=path_or_weights)
    else:
      raise NotImplementedError("Only support in memory weight sync as of now.")

  def _vllm_config(self):
    args = {}
    args["additional_config"] = {}
    args["model"] = self.model_version
    args["max_model_len"] = self.max_model_len
    args["tensor_parallel_size"] = self.mesh.shape["tp"]
    args["gpu_memory_utilization"] = self.hbm_utilization
    if self.lora_config is not None:
      args["additional_config"]["lora_config"] = self.lora_config
    return args

  @property
  def _model_runner(self):
    return self.llm.llm_engine.model_executor.driver_worker.model_runner

  @property
  def transformer(self):
    raise NotImplementedError("vLLM doesn't expose the underlying model!")

  @property
  def transformer_state(self):
    return self._model_runner.state

  def tokenize(self, input_string: str) -> List[int]:
    """Tokenizes the input string."""
    input_ids = self.tokenizer.encode(input_string)
    bos_tok = [self.tokenizer.bos_id()] \
      if (self.tokenizer.bos_id()
          and input_ids[0] !=  self.tokenizer.bos_id()) \
      else []
    return bos_tok + input_ids

  def _get_logprobs_from_vllm_output(self, logprobs: List[Optional[Dict[int, Any]]]) -> List[float]:
      # Below is to get the log probs from vLLM output
      assert logprobs[0] is not None, f"Logprobs are missing"
      assert len(logprobs[0]) == 1, f"The log probs contains more than 1 ({len(logprobs[0])} token per position"
      return [list(logprob_dict.values())[0].logprob for logprob_dict in logprobs]

  def detokenize(
      self,
      input_strings: List[str],
      request_outputs: List[RequestOutput]) -> Tuple[List[str],
                                                     List[float],
                                                     List[int]]:
    generations = len(request_outputs[0].outputs)
    decoded_outputs = [[] for _ in range(generations)]
    out_logprobs = [[] for _ in range(generations)]
    out_tokens = [[] for _ in range(generations)]
    for input_string, multi_sampling_output in zip(input_strings, request_outputs):
      for idx, single_output in enumerate(multi_sampling_output.outputs):
        # vLLM still returns 1 eos id even if we ask it to stop at eos.
        if single_output.token_ids[-1] == self.tokenizer.eos_id():
          single_output.token_ids = single_output.token_ids[:-1]
          single_output.logprobs =  single_output.logprobs[:-1]

        out_tokens[idx].append(single_output.token_ids)
        decoded_outputs[idx].append(self.tokenizer.decode(single_output.token_ids))
        logprobs = self._get_logprobs_from_vllm_output(single_output.logprobs)
        out_logprobs[idx].append(logprobs)
        logging.debug(f"Prompt: {input_string!r}\n\n"
                      f"Generated text: {decoded_outputs[idx][-1]!r}\n\n ")
    return decoded_outputs, out_logprobs, out_tokens

  def __call__(
        self,
        input_strings: List[str],
        total_generation_steps: int,
        max_prompt_length: int=None,
        temperature: float=0.0,
        top_p: float=None,
        top_k: int=None,
        beam_size: int=None,
        seed: int=None,  # vLLM Jax backend doesn't support per request seed.
        multi_sampling: int=1,
        return_logits: bool=True,
        echo:bool = False,
        pad_output: bool = False,
    ) -> base_sampler.SamplerOutput:
    # max_tokens: maximum number of tokens to generate
    assert total_generation_steps <= self.args["max_model_len"], \
      f"{total_generation_steps} > {self.args['max_model_len']}"
    if beam_size is not None:
      self.sampling_params = self.llm.sampling_params.BeamSearchParams(
        beam_width=beam_size,
        max_tokens=total_generation_steps,
        ignore_eos=False,
        temperature=temperature,
      )
    else:
      self.sampling_params = self.llm.get_default_sampling_params()
      self.sampling_params.detokenize = False
      self.sampling_params.max_tokens = total_generation_steps
      self.sampling_params.n = multi_sampling
      self.sampling_params.temperature = temperature
      self.sampling_params.logprobs = 1        # b/428730696
      self.sampling_params.prompt_logprobs = 1 # b/428730696
      self.sampling_params.stop_token_ids=[self.tokenizer.eos_id()]
      self.sampling_params.skip_special_tokens = True

      if top_p is not None:
          self.sampling_params.top_p = top_p
      if top_k is not None:
          self.sampling_params.top_k = top_k

    prompt_ids = [self.tokenize(x) for x in input_strings]
    outputs = self.llm.generate(
       prompts=None,
       prompt_token_ids=prompt_ids,
       sampling_params=self.sampling_params,
       use_tqdm=True,
       )
    decoded_outputs, out_logprobs, out_tokens = self.detokenize(input_strings, outputs)

    max_tokens_length = max(len(x) for x in prompt_ids)

    if max_prompt_length is None or max_prompt_length < max_tokens_length:
      max_prompt_length = utils.next_power_of_2(max_tokens_length)
    all_input_ids  = [
        utils.pad_to_length(
            jnp.array(x),
            target_length=max_prompt_length,
            pad_value=self.tokenizer.pad_id(),
            left=True,
        )
        for x in prompt_ids
    ]
    all_input_ids = jnp.array(all_input_ids)

    all_output_ids  = [
        utils.pad_to_length(
            jnp.array(x),
            target_length=total_generation_steps,
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
