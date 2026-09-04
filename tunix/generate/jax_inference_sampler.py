# Copyright 2026 Google LLC
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

"""A high-throughput TPU sampler backend based on jax-inference."""

from __future__ import annotations

import dataclasses
import logging
import math
from typing import Any, List, Optional, Tuple, Union

from flax import nnx
import jax
import jax.numpy as jnp
import jaxtyping
import numpy as np

from tunix.generate import base_sampler
from tunix.generate import mappings
from tunix.generate import tokenizer_adapter as tok_adapter
from tunix.generate import utils
from tunix.rl import reshard

try:
  from jax_inference.engine import JaxInferenceEngine, SamplingConfig
except ImportError:
  JaxInferenceEngine = None  # pyrefly: ignore[assignment]
  SamplingConfig = None  # pyrefly: ignore[assignment]


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class JaxInferenceConfig:
  """Configuration for JaxInferenceSampler.

  Attributes:
    model_name: HuggingFace model repo id or local checkpoint path.
    mesh: JAX sharding mesh.
    tensor_parallel_size: Tensor parallel size. If None, inferred from mesh.
    num_blocks: Number of KV cache blocks. If None, calculated dynamically.
    block_size: Number of tokens per KV cache block.
    kv_cache_dtype: KV cache data type ("bf16", "fp8", etc.).
    chunk_prefill_size: Chunk size for prefill computation.
    enable_expert_parallel: Whether to enable expert parallelism for MoE.
    quantization: Quantization scheme (e.g., "fp8") or None.
    unroll_steps: Number of unroll steps for decode loop.
    dummy_weights: Initialize with dummy weights instead of loading checkpoint.
    init_with_random_weights: Alias/flag to avoid loading base weights if
      weights will be updated via update_params.
    mapping_config: MappingConfig describing how to map Tunix weights to HF/JAX
      format.
    additional_config: Extra configuration dictionary passed to vLLM config.
    max_model_len: Maximum model sequence length.
    delete_dst_buffers: Whether to free destination buffers during weight
      transfer.
    reshard_chunk_size: Chunk size for resharding parameters.
    monolithic: Whether to use monolithic generation execution.
  """

  model_name: str
  mesh: Optional[jax.sharding.Mesh] = None
  tensor_parallel_size: Optional[int] = None
  num_blocks: Optional[int] = None
  block_size: int = 256
  kv_cache_dtype: Union[str, jnp.dtype] = "bf16"
  chunk_prefill_size: int = 8192
  enable_expert_parallel: bool = False
  quantization: Optional[str] = None
  unroll_steps: int = 16
  dummy_weights: bool = False
  init_with_random_weights: bool = False
  mapping_config: Optional[mappings.MappingConfig] = None
  additional_config: Optional[dict[str, Any]] = None
  max_model_len: Optional[int] = None
  delete_dst_buffers: bool = False
  reshard_chunk_size: Optional[int] = None
  monolithic: bool = True


class JaxInferenceSampler(base_sampler.BaseSampler):
  """A high-throughput TPU sampler backend wrapping jax-inference."""

  def __init__(
      self,
      tokenizer: Any,
      config: JaxInferenceConfig,
      **kwargs,
  ):
    """Initializes JaxInferenceSampler.

    Args:
      tokenizer: HuggingFace tokenizer or TokenizerAdapter.
      config: JaxInferenceConfig instance.
      **kwargs: Additional keyword arguments passed to JaxInferenceEngine.
    """
    if JaxInferenceEngine is None:
      raise ImportError(
          "jax-inference is required for JaxInferenceSampler. "
          "Please ensure jax-inference is installed in your environment."
      )

    self.tokenizer = tokenizer
    if not isinstance(tokenizer, tok_adapter.TokenizerAdapter):
      self.tokenizer = tok_adapter.TokenizerAdapter(tokenizer)
    self.config = config

    # Derive tensor parallel size from mesh or devices if not explicitly specified.
    tp_size = config.tensor_parallel_size
    if tp_size is None and config.mesh is not None:
      if "tp" in config.mesh.shape:
        tp_size = config.mesh.shape["tp"]
      elif "model" in config.mesh.shape:
        tp_size = config.mesh.shape["model"]
      else:
        tp_size = math.prod(config.mesh.shape.values())
    if tp_size is None:
      tp_size = len(jax.devices())

    # Derive num_blocks if not specified.
    num_blocks = config.num_blocks
    if num_blocks is None:
      max_len = config.max_model_len or 4096
      num_blocks = max(
          512, 16 * ((max_len + config.block_size - 1) // config.block_size)
      )

    engine_kwargs = dict(
        model_name=config.model_name,
        tensor_parallel_size=tp_size,
        num_blocks=num_blocks,
        block_size=config.block_size,
        kv_cache_dtype=config.kv_cache_dtype,
        chunk_prefill_size=config.chunk_prefill_size,
        enable_expert_parallel=config.enable_expert_parallel,
        additional_config=config.additional_config,
        quantization=config.quantization,
        unroll_steps=config.unroll_steps,
        dummy_weights=config.dummy_weights or config.init_with_random_weights,
    )
    if kwargs:
      engine_kwargs.update(kwargs)

    self.engine = JaxInferenceEngine(**engine_kwargs)

    if not config.init_with_random_weights and not config.dummy_weights:
      self.engine.load_model_and_weights()

    if config.mapping_config is not None:
      self.to_hf_key_mappings = dict(
          config.mapping_config.to_hf_mappings or {}
      )
      self.to_hf_transpose_keys = config.mapping_config.to_hf_transpose_keys
      self.to_hf_hook_fns = config.mapping_config.to_hf_hook_fns
    else:
      self.to_hf_key_mappings = {}
      self.to_hf_transpose_keys = None
      self.to_hf_hook_fns = None

  @property
  def transformer(self) -> Optional[nnx.Module]:
    """Underlying Flax NNX model instance."""
    return self.engine.model

  @property
  def transformer_state(self) -> Optional[nnx.State]:
    """Underlying Flax NNX state."""
    if hasattr(self.engine, "state") and self.engine.state is not None:
      return self.engine.state
    if self.engine.model is not None:
      return nnx.state(self.engine.model)
    return None

  @property
  def mesh(self) -> jax.sharding.Mesh:
    """Device mesh used by the engine."""
    return self.engine.mesh

  def tokenize(self, input_string: str) -> np.ndarray | list[int]:
    """Tokenize a string using the tokenizer."""
    input_ids = self.tokenizer.encode(input_string)
    bos_tok = [self.tokenizer.bos_id()] if self.tokenizer.bos_id() else []
    return self.tokenizer.dedup_bos_ids(bos_tok + input_ids)

  def update_params(
      self,
      updated_weights: jaxtyping.PyTree,
      filter_types: Optional[Tuple[Any, ...]] = None,
  ) -> None:
    """Synchronize weights from Tunix trainer/model into jax-inference engine.

    Args:
      updated_weights: Tunix PyTree/State or HF-structured weights.
      filter_types: Unused filter types for compatibility with BaseSampler.
    """
    del filter_types
    if self.to_hf_key_mappings:
      preprocess_fn = (
          self.config.mapping_config.preprocess_src_state
          if self.config.mapping_config
          else None
      )
      if preprocess_fn:
        updated_weights = preprocess_fn(updated_weights)

      new_state = utils.transfer_state_with_mappings(
          src_state=updated_weights,
          dst_state=self.transformer_state,
          key_mappings=self.to_hf_key_mappings,
          key_mapping_hook_fns=self.to_hf_hook_fns,
          transpose_keys=self.to_hf_transpose_keys,
          reshard_fn=reshard.reshard_pytree,
          delete_dst_buffers=self.config.delete_dst_buffers,
          reshard_chunk_size=self.config.reshard_chunk_size,
      )
      self.engine.update_model_weights(new_state)
    else:
      self.engine.update_model_weights(updated_weights)

  def __call__(
      self,
      input_strings: str | List[str],
      max_generation_steps: int,
      max_prompt_length: Optional[int] = None,
      temperature: float = 0.0,
      top_p: Optional[float] = None,
      top_k: Optional[int] = None,
      beam_size: Optional[int] = None,
      seed: Optional[int] = None,
      multi_sampling: int = 1,
      return_logits: bool = True,
      echo: bool = False,
      pad_output: bool = False,
      **kwargs,
  ) -> base_sampler.SamplerOutput:
    """Generate completions for prompts using jax-inference.

    Args:
      input_strings: A prompt string or list of prompt strings.
      max_generation_steps: Maximum new tokens to generate.
      max_prompt_length: Maximum length for prompt padding.
      temperature: Sampling temperature (0.0 for greedy).
      top_p: Cumulative probability threshold for nucleus sampling.
      top_k: Top-k sampling threshold.
      beam_size: Unused; present for BaseSampler interface compatibility.
      seed: Random seed for sampling.
      multi_sampling: Number of samples per prompt.
      return_logits: Unused; present for BaseSampler interface compatibility.
      echo: Whether to include prompt in returned text.
      pad_output: Unused; present for BaseSampler interface compatibility.
      **kwargs: Additional parameters.

    Returns:
      SamplerOutput containing generated text and token arrays.
    """
    del beam_size, return_logits, pad_output, kwargs
    if isinstance(input_strings, str):
      input_strings = [input_strings]

    prompts_to_run = input_strings
    if multi_sampling > 1:
      prompts_to_run = [
          prompt for prompt in input_strings for _ in range(multi_sampling)
      ]

    sampling_cfg = SamplingConfig(
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        seed=seed,
    )

    raw_outputs = self.engine.generate(
        prompts=prompts_to_run,
        max_new_tokens=max_generation_steps,
        sampling_config=sampling_cfg,
        monolithic=self.config.monolithic,
    )

    if isinstance(raw_outputs, str):
      generated_texts = [raw_outputs]
    else:
      generated_texts = list(raw_outputs)

    if echo:
      generated_texts = [
          p + g for p, g in zip(prompts_to_run, generated_texts)
      ]

    prompt_ids = [self.tokenize(x) for x in input_strings]
    max_tokens_length = max(len(x) for x in prompt_ids) if prompt_ids else 0
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
    padded_prompts_np = np.array(all_input_ids, dtype=np.int32)

    out_tokens = [
        np.array(self.tokenizer.encode(text), dtype=np.int32)
        for text in generated_texts
    ]

    return base_sampler.SamplerOutput(
        text=generated_texts,
        logits=None,
        tokens=out_tokens,
        padded_prompt_tokens=padded_prompts_np,
        logprobs=None,
    )
