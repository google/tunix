# Copyright 2026 Google LLC
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

"""Common configuration classes for Tunix."""

import dataclasses
from typing import Any, List, Optional, Tuple

import flax
import jax
from jax import numpy as jnp
from jax.sharding import Mesh
import numpy as np
import optax
from tunix.common import datatypes
from tunix.generate import mappings
from tunix.perf import metrics as perf_metrics
from tunix.sft import checkpoint_options
from tunix.sft import metrics_logger as sft_metrics_logger
from tunix.sft import profiler

# For rl_utils calls inside RLTrainingConfig
# For base_rollout typing inside ClusterConfig

MetricsLoggerOptions = sft_metrics_logger.MetricsLoggerOptions
Mode = datatypes.Mode
Role = datatypes.Role


def _is_positive_integer(value: int | None, name: str):
  if value is None:
    return
  if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
    raise ValueError(f"{name} must be a positive integer. Got: {value}")
  if value <= 0:
    raise ValueError(f"{name} must be a positive integer. Got: {value}")


def _check_divisibility(small_size, big_size, small_size_name, big_size_name):
  if big_size % small_size != 0:
    raise ValueError(
        f"{big_size_name} must be a multiple of {small_size_name}."
    )


@dataclasses.dataclass(frozen=True)
class CacheConfig:
  """Configuration for the KV cache."""

  cache_size: int
  num_layers: int
  num_kv_heads: int
  head_dim: int


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
  expert_parallel_size: int = 1

  # Whether to return logprobs from the sampler.
  return_logprobs: bool = False

  # vLLM specific rollout configs.

  # Whether to run rollout in vLLM server mode or batch inference mode.
  rollout_vllm_server_mode: bool = False

  # Only drain the vLLM server-mode submission queue once at least this many
  # requests have accumulated. 0 disables the threshold.
  rollout_vllm_server_mode_submission_threshold: int = 0

  # Flush the vLLM server-mode submission queue after this many seconds have
  # elapsed since the first request of the current window arrived, even if the
  # submission threshold has not been reached. This bounds latency when fewer
  # than the threshold accumulate. 0 disables the timeout.
  rollout_vllm_server_mode_submission_timeout_s: float = 0.0

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

  # Whether to enable asynchronous scheduling for vLLM rollout engine.
  rollout_vllm_async_scheduling: bool = False

  # Mode for processing logprobs from vLLM.
  rollout_vllm_logprobs_mode: str = "processed_logprobs"

  # Configs for MaxText/Custom Model support in vLLM rollout engine.
  rollout_vllm_hf_config_path: str | None = None
  rollout_vllm_additional_config: dict[str, Any] | None = None

  # Whether to enable data parallel in attention for vLLM rollout engine.
  # The "attn_dp" mesh axis is used when the degree of tensor parallelism
  # specified is more than the number of KV heads in the model. Enabling this
  # allows for non-attention tensors to be sharded across "attn_dp" and "model"
  # axes, which can help reduce memory usage for large models with few KV heads.
  rollout_vllm_enable_dp_attention: bool = False

  # Whether to delete destination buffers when synchronizing weights between
  # trainer and vLLM model. Default to True to ensure old weights are deleted
  # to free up HBM memory.
  rollout_vllm_delete_dst_buffers: bool = True

  # Maximum number of batched tokens allowed in vLLM. This allows for pending prefill requests
  # to be batched along with decode requests if enough tokens are available. Only used when
  # chunked prefill is enabled.
  rollout_vllm_max_num_batched_tokens: Optional[int] = None

  # Maximum number of concurrent sequences allowed to be processed in vLLM.
  rollout_vllm_max_num_seqs: Optional[int] = None

  # Number of flat keys to reshard at a time when synchronizing weights between
  # trainer and vLLM model. None (default) reshards the whole model in one call.
  # Set to a smaller value to reduce peak HBM pressure on large models.
  rollout_vllm_reshard_chunk_size: Optional[int] = None

  # Additional keyword arguments forwarded directly to the vLLM engine constructor.
  rollout_vllm_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)

  # Additional keyword arguments forwarded directly to the vLLM sampling params.
  rollout_vllm_sampling_kwargs: dict[str, Any] = dataclasses.field(
      default_factory=dict
  )

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

  # Whether to use sort or mask implementation in sampler, sort has better evaluation result.
  rollout_sglang_jax_use_sort_for_toppk_minp: bool = True

  # Whether to use lora
  rollout_sglang_jax_enable_static_lora: bool = False

  # Whether to use single controller mode, single controller mode is required in pathways
  rollout_sglang_jax_enable_single_process: bool = True

  # Specify the modules which are required to use lora
  rollout_sglang_jax_lora_target_modules: Optional[List[str]] = None

  # Specify the lora RANK
  rollout_sglang_jax_max_lora_rank: Optional[int] = None

  rollout_sglang_jax_lora_scaling: Optional[float] = None

  # Specify the paddings for batch_size
  rollout_sglang_jax_precompile_bs_paddings: Optional[List[int]] = None

  # Specify the paddings for tokens which is used in prefll
  rollout_sglang_jax_precompile_token_paddings: Optional[List[int]] = None

  # Specify the the maximum number of tokens in a chunk for the chunked prefill
  rollout_sglang_jax_chunked_prefill_size: Optional[int] = -1

  # The number of tokens in a page
  rollout_sglang_jax_page_size: int = 128

  # The format of the model weights to load.
  rollout_sglang_jax_load_format: str = "auto"

  # The maximum number of running requests to accumulate batch
  rollout_sglang_jax_max_running_requests: Optional[int] = None

  # The log level of sglang_jax
  rollout_sglang_jax_log_level: Optional[str] = "info"

  # Additional keyword arguments forwarded directly to the SG-Lang JAX sampler/engine.
  rollout_sglang_jax_kwargs: dict[str, Any] = dataclasses.field(
      default_factory=dict
  )


@dataclasses.dataclass(slots=True, kw_only=True)
class TrainingConfig:
  """Configuration for the trainer."""

  eval_every_n_steps: int
  max_steps: int | None = None
  gradient_accumulation_steps: int | None = None

  # If set, the checkpoints will be saved to this path. Checkpoints
  # contains the model params and the train data iterator state.
  checkpoint_root_directory: str | None = None
  # Checkpoint configurations. If None, the default options will be used.
  checkpointing_options: checkpoint_options.CheckpointingOptions | None = None

  # Configs for the metrics logger.
  metrics_logging_options: MetricsLoggerOptions | None = None

  # Configs for the profiler.
  profiler_options: profiler.ProfilerOptions | None = None

  # Configs for performance metrics.
  perf_metrics_options: perf_metrics.PerfMetricsOptions | None = None

  data_sharding_axis: Tuple[str, ...] = ("fsdp",)

  # Controls how many train_steps can be scheduled ahead of time.
  max_inflight_computations: int = 2

  # Prefix for metric names for logging. Not sticking it in
  # `metrics_logging_options` because the latter is optional.
  metrics_prefix: str = ""

  # Progress bar description.
  pbar_description: str | None = "Training"

  # Sequence packing configuration.
  max_seq_token_per_tpu: int | None = None
  max_segments_per_packed_row: int | None = None

  def get_with_default(self, key: str, default: Any) -> Any:
    val = getattr(self, key)
    if val is None:
      return default
    return val


@dataclasses.dataclass(slots=True, kw_only=True)
class RLTrainingConfig(TrainingConfig):
  """RLTraining config.

  Attributes:
    actor_optimizer: Optimizer for the actor model.
    critic_optimizer: Optimizer for the critic model. If None, the critic model
      will be trained in the same optimizer as the actor model.
    mini_batch_size: The mini-batch size used for policy weight updates. One
      mini-batch corresponds to one optimizer update. `mini_batch_size` must be
      divisible by the global batch size.
    train_micro_batch_size: The micro-batch size used for gradient accumulation
      at training time. `train_micro_batch_size` must be divisible by
      `mini_batch_size`.
    rollout_micro_batch_size: The micro-batch size used for model rollouts.
    compute_logps_micro_batch_size: The micro-batch size used for computing log
      probabilities (e.g. for reference and old policy models).
    compute_logps_chunk_size: The chunk size used for computing log
      probabilities. Instead of using final logits from model, where size is [B,
      T, V], this will use the last hidden output with size [B, T, D] from model
      and compute logps in a chunked manner. Good values to pick are like 256,
      512, etc. When value is 0, it means this feature is disabled. This also
      requires model to support `skip_lm_head` in its `__call__` method and have
      a `compute_final_logits` method.
  """

  actor_optimizer: optax.GradientTransformation
  critic_optimizer: optax.GradientTransformation | None = None
  mini_batch_size: int | None = None
  train_micro_batch_size: int | None = None
  rollout_micro_batch_size: int | None = None
  compute_logps_micro_batch_size: int | None = None
  compute_logps_chunk_size: int = 0

  def __post_init__(self):
    """Validates the configuration after initialization."""
    for name in [
        "mini_batch_size",
        "train_micro_batch_size",
        "rollout_micro_batch_size",
        "compute_logps_micro_batch_size",
        "max_segments_per_packed_row",
    ]:
      _is_positive_integer(getattr(self, name, None), name)

    if self.gradient_accumulation_steps is not None:
      raise ValueError(
          "For RL training, gradient_accumulation_steps should be None. It is "
          "automatically derived from: "
          "`mini_batch_size // train_micro_batch_size`."
      )

    if self.train_micro_batch_size is not None:
      if self.mini_batch_size is None:
        raise ValueError(
            "For RL training, `mini_batch_size` must be set when"
            " `train_micro_batch_size` is set."
        )
      _check_divisibility(
          self.train_micro_batch_size,
          self.mini_batch_size,
          f"{self.train_micro_batch_size=}",
          f"{self.mini_batch_size=}",
      )
      self.gradient_accumulation_steps = (
          self.mini_batch_size // self.train_micro_batch_size
      )


@dataclasses.dataclass(kw_only=True, frozen=True)
class ClusterConfig:
  """Cluster config.

  Attributes:
    role_to_mesh: Mapping from model role to mesh. Key config for colocated vs
      disaggregated setup.
    role_to_logical_axis_rule: Mapping from model role to logical axis rule.
      This is used when models are sharded with logical axis and expects a
      logical to physical axis mapping at runtime.
    rollout_engine: Rollout engine to use. E.g. "vanilla", "vllm", "sglang_jax".
      Alternatively, if a subclass of `BaseRollout` is provided, it will be used
      as the rollout engine.
    offload_to_cpu: Whether to offload models to CPU at each step..
    training_config: RL training config.
    rollout_config: Rollout config. It may be different for different modes,
      e.g. TRAIN vs EVAL.
    rollout_vllm_model_version: Model version for vllm rollout engine.
    rollout_vllm_lora_config: LoRA config for vllm rollout engine.
    rollout_vllm_hbm_utilization: The percentage of TPU/GPU HBM allocated the
      vllm rollout engine.
    rollout_vllm_init_with_random_weights: Init the vllm TPU backend model with
      random weights instead of loading from the given path.
    rollout_vllm_tpu_backend_type: The TPU Jax backend type for vllm rollout
      engine, E.g. "jax", "torchax" or "pytorch_xla".
  """

  role_to_mesh: dict[Role, Mesh]
  role_to_logical_axis_rule: dict[Role, flax.typing.LogicalRules] | None = None
  rollout_engine: str | type[Any] = "vanilla"
  offload_to_cpu: bool = False

  training_config: RLTrainingConfig
  rollout_config: dict[Mode, RolloutConfig] | RolloutConfig
