from typing import Dict, Optional, List, Any
from flax import nnx
import tunix.generate.tokenizer_adapter as tok_adapter
from vllm import LLM, EngineArgs
from tunix.generate.sampler import SamplerOutput
from vllm.outputs import RequestOutput

class vLLMSampler():
  def __init__(
        self,
        tokenizer: Any,
        model: nnx.module,
        lora_config: Optional[Dict[str, Any]],
        model_version: Optional[str] = "meta-llama/Llama-3.1-8B", # TODO(lancewang): We still need the model version for now, will remove it later
        max_model_len: int = 1024):
    
    self.tokenizer = tok_adapter.TokenizerAdapter(tokenizer) if isinstance(tokenizer, str) else tokenizer
    self.args = {}
    self.args["additional_config"] = {}
    self.args["model"] = model_version
    if lora_config is None:
       lora_config = {
          "rank": 64,
          "alpha": 64.0,
          "module_path":
              ".*q_proj|.*k_proj|.*v_proj|.*o_proj|.*gate_proj|.*down_proj|.*up_proj",
        }
    self.args["additional_config"]["lora_config"] = lora_config
    self.args["max_model_len"] = max_model_len

    _, params = nnx.split(model)
    self.args["additional_config"]["custom_nnx_weights"] = params
    self.args["model"] = model_version

    self.llm = LLM(**self.args)
    self.sampling_params = self.llm.get_default_sampling_params()
    self.sampling_params.detokenize = False

  @property
  def transformer_state(self):
    return self.llm.llm_engine.model_executor.driver_worker.model_runner.transformer_state

  def tokenize(self, input_string: str) -> List[int]:
    """Tokenizes the input string."""
    input_ids = self.tokenizer.encode(input_string)
    bos_tok = [self.tokenizer.bos_id()] if self.tokenizer.bos_id() else []
    return bos_tok + input_ids

  def detokenize(self, prompts, return_logits, outputs: RequestOutput):
    print("-" * 50)
    decoded_outputs = []
    out_logits = []
    out_tokens = []
    for prompt, multi_sampling_output in zip(prompts, outputs):
      single_decoded_outputs = []
      single_out_logits = []
      single_out_tokens = []
      for single_output in multi_sampling_output:
        single_out_tokens.append(single_output[0].token_ids)
        single_decoded_outputs.append(self.tokenizer.decode(single_output[0].token_ids))
        if return_logits:
          single_out_logits.append(single_output[0].logprob)
        print(f"Prompt: {prompt!r}\nGenerated text: {single_decoded_outputs[-1]!r}")
        print("-" * 50)
      decoded_outputs.append(single_decoded_outputs)
      out_logits.append(single_out_logits)
      out_tokens.append(single_out_tokens)
    return decoded_outputs, out_logits, out_tokens

  def __call__(
        self,
        prompts: List[str]=None,
        max_generation_length=512,
        temperature=None,
        top_p=None,
        top_k=None,
        multi_sampling: int=1,
        return_logits: bool=False):
    # max_tokens: maximum number of tokens to generate
    assert max_generation_length <= self.args["max_model_len"], f"{max_generation_length} > {self.args["max_model_len"]}"

    self.sampling_params.max_tokens = max_generation_length
    self.sampling_params.n = multi_sampling

    if temperature is not None:
        self.sampling_params.temperature = temperature
    if top_p is not None:
        self.sampling_params.top_p = top_p
    if top_k is not None:
        self.sampling_params.top_k = top_k

    prompt_ids = [self.tokenize(x) for x in prompts]
    outputs = self.llm.generate(
       prompts=None,
       prompt_token_ids=prompt_ids,
       sampling_params=self.sampling_params,
       use_tqdm=True,
       )
    decoded_outputs, out_logits, out_tokens = self.detokenize(prompts, return_logits, outputs)

    # Only return the first sampled outputs
    return SamplerOutput(
      text=decoded_outputs[0],
      logits=out_logits[0],
      tokens=out_tokens[0],
      padded_prompt_tokens=prompt_ids, # No padding for vLLM
    )

