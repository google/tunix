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

"""Base rollout worker interface."""

import abc
import dataclasses
from typing import Any, List, Optional, Tuple

import jax
from jax import numpy as jnp
import jaxtyping
import numpy as np
from tunix.generate import mappings

ABC = abc.ABC
abstractmethod = abc.abstractmethod


@dataclasses.dataclass(frozen=True)
class CacheConfig:
  """Configuration for the KV cache."""

  cache_size: int
  num_layers: int
  num_kv_heads: int
  head_dim: int


@dataclasses.dataclass
class RolloutOutput:
  """Output of the rollout worker."""

  # Generated samples from the model.
  text: list[str]

  # Per-step logits used during sampling.
  # TODO(tsbao): consider enforcing this to be np.ndarray as well,
  # but let's solve it as part of the IS effort.
  logits: jax.Array

  # Tokens corresponding to the generated samples.
  # Since tokens need to be transfered to RAM for decoding, we use numpy array
  # here.
  tokens: np.ndarray

  # Left padded prompt tokens.
  # TODO(tsbao): Reconcile with vLLM output and see if we should remove this
  # field, or add prompt + generated as extra.
  left_padded_prompt_tokens: np.ndarray

  # The log probs from sampler generations.
  logprobs: list[float] | None


@dataclasses.dataclass
class RolloutConfig:
  """Configuration for the rollout worker.

  Fields should be mapped to a subset of vLLM sampling knobs
  https://docs.vllm.ai/en/v0.6.4/dev/sampling_params.html
  """

  # Maximum number of tokens to generate per output sequence
  max_tokens_to_generate: int = 64

  # Float that controls the randomness of the sampling.
  # Lower values make the model more deterministic, while higher values make the
  # model more random. Zero means greedy sampling.
  temperature: float = 0.9

  # Float that controls the cumulative probability of the top tokens to
  # consider. Must be in (0, 1]. Set to 1 to consider all tokens.
  top_p: float | None = 1.0

  # Integer that controls the number of top tokens to consider. Set to -1 to
  # consider all tokens.
  top_k: int | None = None

  # Random seed to use for the generation.
  seed: jax.Array | None = None

  # Maximum length of the prompt. The prompt will be padded/truncated to this
  # length.
  max_prompt_length: int = 64

  # Only used for vanilla rollout engine.
  kv_cache_size: int = 1024  # Only used for vanilla rollout engine.

  # data type of the rollout model.
  data_type: jnp.dtype | None = None

  # EOS tokens to stop the generation. If not defined, eos_id from tokenizer
  # will be used.
  eos_tokens: list[int] | None = None

  # Weights mapping config for the rollout model.
  rollout_mapping_config: mappings.MappingConfig | None = None

  # Parallelism configs.
  tensor_parallel_size: int = -1
  data_parallel_size: int = -1

  # vLLM specific rollout configs.

  # Whether to run rollout in vLLM server mode or batch inference mode.
  rollout_vllm_server_mode: bool = False

  # Model version for vLLM rollout engine.
  rollout_vllm_model_version: str = ""

  # LoRA config for vLLM rollout engine.
  rollout_vllm_lora_config: dict[str, Any] | None = None

  # Allocated HBM fraction for vLLM rollout engine.
  rollout_vllm_hbm_utilization: float = 0.2

  # Whether to initialize vLLM model with random weights or huggingface weights.
  rollout_vllm_init_with_random_weights: bool = True

  # TPU backend type for vLLM rollout engine, "jax" or "torchax", default to "jax".
  rollout_vllm_tpu_backend_type: str | None = None

  # Swap space size for vLLM rollout engine, in GiB.
  rollout_vllm_swap_space_size_gb: float = 4.0

  # Whether to enable asynchronous scheduling for vLLM rollout engine.
  rollout_vllm_async_scheduling: bool = False

  # SG-Lang JAX specific rollout configs.

  # Model version for SG-Lang JAX rollout engine.
  rollout_sglang_jax_model_version: str = ""

  # Context length for SG-Lang JAX rollout engine.
  rollout_sglang_jax_context_length: Optional[int] = None

  # Allocated HBM fraction for SG-Lang JAX rollout engine.
  rollout_sglang_jax_mem_fraction_static: float = 0.2

  # Whether to initialize SG-Lang JAX model with random weights.
  rollout_sglang_jax_init_with_random_weights: bool = True

  # Radix cache disabling flag for SG-Lang JAX rollout engine. Default to True for RL.
  rollout_sglang_jax_disable_radix_cache: bool = True

  # Whether to enable deterministic sampling for SG-Lang JAX rollout engine.
  rollout_sglang_jax_enable_deterministic_sampling: bool = False

  # List of token buckets for jax jit
  rollout_sglang_jax_precompile_token_paddings: Optional[List] = None

  # List of batch sizes buckets for jax jit
  rollout_sglang_jax_precompile_bs_paddings: Optional[List] = None

  # The maximum number of tokens in a chunk for the chunked prefill.
  # Setting this to -1 means disabling chunked prefill.
  rollout_sglang_jax_chunked_prefill_size: int = -1

  # The number of tokens in a page
  rollout_sglang_jax_page_size: int = 64


class BaseRollout(ABC):
  """Base RolloutWorker."""

  @abstractmethod
  def __init__(self, **kwargs):
    """Initializes the rollout worker."""

  @abstractmethod
  def generate(
      self,
      prompts: list[str],
      rollout_config: RolloutConfig,
      **kwargs,
  ) -> RolloutOutput:
    """Generates samples from the model."""

  @abstractmethod
  def get_per_token_logps(
      self,
      prompt_tokens: jax.Array,
      completion_tokens: jax.Array,
      completion_mask: jax.Array | None = None,
  ) -> jax.Array:
    """Returns per-token log probabilities from the model."""

  @abstractmethod
  def update_params(
      self,
      params: jaxtyping.PyTree,
      filter_types: Optional[Tuple[Any, ...]] = None,
  ) -> None:
    """Updates the rollout model parameters."""

  @abstractmethod
  def pad_id(self) -> int:
    """Returns the pad id."""

  @abstractmethod
  def eos_id(self) -> int:
    """Returns the eos id."""

  @abstractmethod
  def model(self) -> Any:
    """Returns the rollout model."""
