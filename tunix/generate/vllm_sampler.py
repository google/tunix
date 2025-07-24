from typing import Dict, Optional, List, Any
from flax import nnx
import tunix.generate.tokenizer_adapter as tok_adapter
from vllm.entrypoints.llm import LLM, EngineArgs
from vllm.outputs import RequestOutput
from absl import logging
import jax
import jax.numpy as jnp
from tunix.generate import utils
from tunix.generate import base_sampler
import os

class vLLMSampler(base_sampler.SamplerBase):
  def __init__(
        self,
        model: nnx.module,
        tokenizer: Any,
        mesh: jax.sharding.Mesh,
        max_model_len: int,
        lora_config: Optional[Dict[str, Any]],
        model_version: Optional[str], # We still need the model version for now, will remove it later
        gpu_memory_utilization: Optional[float] = 0.3,
    ):
    os.environ["TPU_BACKEND_TYPE"] = "jax"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    self.tokenizer = tok_adapter.TokenizerAdapter(tokenizer)
    self.model_version = model_version
    self.max_model_len = max_model_len
    self.mesh = mesh
    self.lora_config = lora_config
    self.gpu_memory_utilization = gpu_memory_utilization

    self.args = self._vllm_config()
    self.llm = LLM(**self.args)

    self.mappings = self.get_to_hf_mappings
    if lora_config is not None:
      self.mappings |= self.get_lora_to_hf_mappings

    # This is how we load the weights to vLLM now, consider load the orbax checkpoint in vLLM directly
    _, params = nnx.split(model)
    self.update_params(updated_weights=params)

  def update_params(self, updated_weights: nnx.Module):
    self.llm.llm_engine.model_executor.collective_rpc("sync_weights", args=(updated_weights, self.mappings, self.get_to_hf_transpose_keys))

  def _vllm_config(self):
    args = {}
    args["additional_config"] = {}
    args["model"] = self.model_version
    args["max_model_len"] = self.max_model_len
    args["tensor_parallel_size"] = self.mesh.shape["tp"]
    args["gpu_memory_utilization"] = self.gpu_memory_utilization
    if self.lora_config is not None:
      args["additional_config"]["lora_config"] = self.lora_config
    return args

  @property
  def _model_runner(self):
    return self.llm.llm_engine.model_executor.driver_worker.model_runner

  @property
  def transformer(self):
    return nnx.merge(self._model_runner.graph_def, self._model_runner.state)

  @property
  def transformer_state(self):
    return self._model_runner.state

  def tokenize(self, input_string: str) -> List[int]:
    """Tokenizes the input string."""
    input_ids = self.tokenizer.encode(input_string)
    bos_tok = [self.tokenizer.bos_id()] if self.tokenizer.bos_id() else []
    return bos_tok + input_ids

  def _get_logprobs_from_vllm_output(self, logprobs: List[Optional[Dict[int, Any]]]):
      logging.info("vLLM Jax backend doesn't support returnning log probs yet!")
      return []
      # Below is to get the log probs from vLLM output
      # assert logprobs[0] is not None, f"Logprobs are missing"
      # assert len(logprobs[0]) == 1, f"The log probs contains more than 1 ({len(logprobs[0])} token per position"
      # return [list(logprob_dict.values())[0].logprob for logprob_dict in logprobs]

  def detokenize(self, input_strings, return_logits, request_outputs: List[RequestOutput]):
    print("-" * 50)
    generations = len(request_outputs[0].outputs)
    decoded_outputs = [[] for _ in range(generations)]
    out_logits = [[] for _ in range(generations)]
    out_tokens = [[] for _ in range(generations)]
    for input_string, multi_sampling_output in zip(input_strings, request_outputs):
      prompt_logits = self._get_logprobs_from_vllm_output(multi_sampling_output.prompt_logprobs)
      for idx, single_output in enumerate(multi_sampling_output.outputs):
        out_tokens[idx].append(single_output.token_ids)
        decoded_outputs[idx].append(self.tokenizer.decode(single_output.token_ids))
        if return_logits:
          generation_logits = self._get_logprobs_from_vllm_output(single_output.logprobs)
          out_logits[idx].append(prompt_logits + generation_logits)
        print(f"Prompt: {input_string!r}\n\nGenerated text: {decoded_outputs[idx][-1]!r}\n\n Generated text length: {len(decoded_outputs[idx][-1])} {len(out_tokens[idx][-1])=}")
        print("-" * 50)
    return decoded_outputs, out_logits, out_tokens

  def __call__(
        self,
        input_strings: List[str],
        total_generation_steps,
        max_prompt_length=None,
        temperature=0.0,
        top_p=None,
        top_k=None,
        beam_size=None,
        seed=None,
        multi_sampling: int=1,
        return_logits: bool=True,
        echo:bool = False,
        pad_output: bool = False,
    ):
    # max_tokens: maximum number of tokens to generate
    assert total_generation_steps <= self.args["max_model_len"], f"{total_generation_steps} > {self.args['max_model_len']}"
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


      if top_p is not None:
          self.sampling_params.top_p = top_p
      if top_k is not None:
          self.sampling_params.top_k = top_k
      if seed is not None:
        self.sampling_params.seed = seed

    prompt_ids = [self.tokenize(x) for x in input_strings]
    outputs = self.llm.generate(
       prompts=None,
       prompt_token_ids=prompt_ids,
       sampling_params=self.sampling_params,
       use_tqdm=True,
       )

    decoded_outputs, out_logits, out_tokens = self.detokenize(input_strings, return_logits, outputs)

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
      logits=out_logits[0],
      tokens=all_output_ids,
      padded_prompt_tokens=all_input_ids,
    )

  # For now, we are still passing sharding to vLLM, consider removing it after switching to the reshard API
  @property
  def get_to_hf_mappings(self):
    return {
      "lm_head.w": ("lm_head.input_embedding_table_DV", (None, "model")),
      "embedder.input_embedding": ("embedder.input_embedding_table_DV", ("model", None)),
      "layers.*.input_layernorm.w":
      ("layers.*.pre_attention_norm.scale", (None, )),
      "layers.*.mlp.down_proj.kernel":
      ("layers.*.mlp.kernel_down_proj_FD", ("model", None)),
      "layers.*.mlp.gate_proj.kernel":
      ("layers.*.mlp.kernel_gating_DF", (None, "model")),
      "layers.*.mlp.up_proj.kernel":
      ("layers.*.mlp.kernel_up_proj_DF", (None, "model")),
      "layers.*.post_attention_layernorm.w":
      ("layers.*.pre_mlp_norm.scale", (None, )),
      "layers.*.attn.k_proj.w":
      ("layers.*.attn.kernel_k_proj_DKH", (None, "model", None)),
      "layers.*.attn.o_proj.w":
      ("layers.*.attn.kernel_o_proj_NHD", ("model", None, None)),
      "layers.*.attn.q_proj.w":
      ("layers.*.attn.kernel_q_proj_DNH", (None, "model", None)),
      "layers.*.attn.v_proj.w":
      ("layers.*.attn.kernel_v_proj_DKH", (None, "model", None)),
      "final_norm.w": ("final_norm.scale", (None, )),
    }

  @property
  def get_lora_to_hf_mappings(self):
    return {
        "layers.*.mlp.gate_proj.kernel_lora_a": ("layers.*.mlp.kernel_gating_DF_lora_a",(None, None)),
        "layers.*.mlp.gate_proj.kernel_lora_b": ("layers.*.mlp.kernel_gating_DF_lora_b",(None, "model")),
        "layers.*.mlp.up_proj.kernel_lora_a": ("layers.*.mlp.kernel_up_proj_DF_lora_a",(None, None)),
        "layers.*.mlp.up_proj.kernel_lora_b": ("layers.*.mlp.kernel_up_proj_DF_lora_b",(None, "model")),
        "layers.*.mlp.down_proj.kernel_lora_a": ("layers.*.mlp.kernel_down_proj_FD_lora_a",("model", None)),
        "layers.*.mlp.down_proj.kernel_lora_b": ("layers.*.mlp.kernel_down_proj_FD_lora_b",(None, None)),
        "layers.*.attn.q_proj.w_lora_a": ("layers.*.attn.kernel_q_proj_DNH_lora_a",("model", None)),
        "layers.*.attn.q_proj.w_lora_b": ("layers.*.attn.kernel_q_proj_DNH_lora_b",(None, None)),
        "layers.*.attn.k_proj.w_lora_a": ("layers.*.attn.kernel_k_proj_DKH_lora_a",("model", None)),
        "layers.*.attn.k_proj.w_lora_b": ("layers.*.attn.kernel_k_proj_DKH_lora_b",(None, None)),
        "layers.*.attn.v_proj.w_lora_a": ("layers.*.attn.kernel_v_proj_DKH_lora_a",("model", None)),
        "layers.*.attn.v_proj.w_lora_b": ("layers.*.attn.kernel_v_proj_DKH_lora_b",(None, None)),
        "layers.*.attn.o_proj.w_lora_a": ("layers.*.attn.kernel_o_proj_NHD_lora_a",("model", None)),
        "layers.*.attn.o_proj.w_lora_b": ("layers.*.attn.kernel_o_proj_NHD_lora_b",(None, None))
      }

  @property
  def get_to_hf_transpose_keys(self):
    return {
        "input_embedding": (1, 0),
    }
