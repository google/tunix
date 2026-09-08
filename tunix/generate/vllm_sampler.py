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
from concurrent.futures import TimeoutError as FutureTimeoutError
import dataclasses
import gc
import hashlib
from itertools import count
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast

from absl import logging
from flax import nnx
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


def _configure_logprob_request(
    sampling_params: Any, *, return_logprobs: bool
) -> None:
  """Configures vLLM logprob fields without using zero as an off sentinel.

  In vLLM, ``logprobs=0`` and ``prompt_logprobs=0`` still request the sampled
  token and prompt-token logprobs respectively.  Absence is represented by
  ``None``.  Keeping this distinction here prevents reward-only callers from
  silently entering gather/device-to-host logprob paths.
  """
  if return_logprobs:
    sampling_params.logprobs = 1  # b/428730696
    # None, not 0: on the TPU/JAX backend prompt_logprobs=0 still allocates
    # and computes full prompt-logprob structures (and feeds -1 placeholder
    # ids to the tokenizer when detokenize=True).  The rollout path never
    # consumes prompt logprobs; the B-arm rescore requests them explicitly
    # on its own path (vllm_rollout.get_prefill_rescore_logps).
    sampling_params.prompt_logprobs = None
  else:
    sampling_params.logprobs = None
    sampling_params.prompt_logprobs = None


def _prompt_token_ids(value: Any, *, field: str) -> np.ndarray:
  """Returns one canonical int32 prompt vector for witness hashing."""
  tokens = np.asarray(value)
  if (
      tokens.ndim != 1
      or tokens.dtype.kind not in "iu"
      or np.any(tokens < 0)
      or np.any(tokens > np.iinfo(np.int32).max)
  ):
    raise ValueError(f"{field} must be a 1-D nonnegative int32 token vector")
  return np.asarray(tokens, dtype="<i4")


def _prompt_token_sha256(tokens: np.ndarray) -> str:
  portable = np.ascontiguousarray(np.asarray(tokens, dtype="<i8"))
  return hashlib.sha256(portable.tobytes()).hexdigest()


def build_prompt_token_witnesses(
    submitted_prompts: Sequence[Sequence[int] | np.ndarray],
    request_outputs: Sequence[RequestOutput],
) -> list[base_sampler.PromptTokenWitness]:
  """Joins submit-side prompts to vLLM RequestOutput echoes by list position."""
  if len(submitted_prompts) != len(request_outputs):
    raise ValueError(
        "vLLM prompt witness count differs: "
        f"submitted={len(submitted_prompts)} outputs={len(request_outputs)}"
    )
  witnesses = []
  request_ids = set()
  for index, (submitted, output) in enumerate(
      zip(submitted_prompts, request_outputs, strict=True)
  ):
    request_id = getattr(output, "request_id", None)
    if not isinstance(request_id, str) or not request_id:
      raise ValueError(f"vLLM prompt witness {index} has no request ID")
    if request_id in request_ids:
      raise ValueError(f"duplicate vLLM prompt witness request ID: {request_id}")
    request_ids.add(request_id)
    engine_echo = getattr(output, "prompt_token_ids", None)
    if engine_echo is None:
      raise ValueError(
          f"vLLM prompt witness {request_id} has no RequestOutput prompt IDs"
      )
    submitted_tokens = _prompt_token_ids(
        submitted, field=f"vLLM submitted prompt {index}"
    )
    echoed_tokens = _prompt_token_ids(
        engine_echo, field=f"vLLM engine echo {request_id}"
    )
    witnesses.append(
        base_sampler.PromptTokenWitness(
            request_id=request_id,
            submitted_tokens=int(submitted_tokens.size),
            submitted_sha256=_prompt_token_sha256(submitted_tokens),
            engine_echo_tokens=int(echoed_tokens.size),
            engine_echo_sha256=_prompt_token_sha256(echoed_tokens),
            submitted_token_ids=tuple(int(token) for token in submitted_tokens),
            engine_echo_token_ids=tuple(int(token) for token in echoed_tokens),
        )
    )
  return witnesses


@dataclasses.dataclass
class VllmConfig:
  """Vllm rollout configuations."""

  # Sampler related
  server_mode: bool = False
  server_mode_submission_threshold: int = 0
  server_mode_submission_timeout_s: float = 0.0
  mapping_config: MappingConfig = dataclasses.field(
      default_factory=MappingConfig
  )
  return_logprobs: bool = False

  # vLLM Env vars
  init_with_random_weights: bool = True
  tpu_backend_type: str = "jax"

  # vLLM engine arg related, requires additional processing before passing into engine
  additional_config: Optional[Dict[str, Any]] = None
  enable_dp_attention: bool = False
  hbm_utilization: float = 0.5
  lora_config: Optional[Dict[str, Any]] = None
  mesh: Optional[jax.sharding.Mesh] = None
  data_parallel_size: int = -1
  tensor_parallel_size: int = -1
  expert_parallel_size: int = 1
  # Default to True to ensure old weights are deleted to free up HBM memory
  delete_dst_buffers: bool = True
  reshard_chunk_size: Optional[int] = None

  # vLLM engine args that can be directly passed in without additional processing, e.g. max_model_len, async_scheduling, etc.
  engine_kwargs: dataclasses.InitVar[Optional[Dict[str, Any]]] = None
  _processed_engine_kwargs: Dict[str, Any] = dataclasses.field(
      init=False, default_factory=dict
  )

  # VllmConfig fields that require special processing before being passed to
  # vLLM and must not be passed via engine_kwargs, which is a raw pass-through
  # to vLLM EngineArgs.
  _RESERVED_KEYS: frozenset[str] = dataclasses.field(
      default=frozenset(
          {"tensor_parallel_size", "data_parallel_size", "expert_parallel_size"}
      ),
      init=False,
      repr=False,
      compare=False,
  )
  # vLLM sampling args that can be directly passed in without additional processing, e.g. temperature, stop etc.
  sampling_kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

  def __post_init__(self, engine_kwargs: Optional[Dict[str, Any]]):
    engine_kwargs = engine_kwargs or {}
    illegal = self._RESERVED_KEYS & engine_kwargs.keys()
    if illegal:
      raise ValueError(
          "VllmConfig fields must be set directly on VllmConfig, not passed"
          f" via engine_kwargs: {sorted(illegal)}"
      )
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

    self.tokenizer = tokenizer
    if not isinstance(tokenizer, tok_adapter.TokenizerAdapter):
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
  def update_params(
      self,
      updated_weights: jaxtyping.PyTree,
      filter_types: Optional[Tuple[Any, ...]] = None,
  ):
    del filter_types

    if self.llm is not None:
      self.llm.reset_prefix_cache()
      self.llm.collective_rpc("delete_kv_cache") # will free hbm
    elif self._driver is not None:
      self._driver.llm_engine.reset_prefix_cache()
      self._driver.llm_engine.collective_rpc("delete_kv_cache")

    # Synchronization point before weight sync
    jax.effects_barrier()

    if self.to_hf_key_mappings:
      preprocess_fn = self.config.mapping_config.preprocess_src_state
      if preprocess_fn:
        updated_weights = preprocess_fn(updated_weights)

      utils.transfer_state_with_mappings(
          src_state=updated_weights,
          dst_state=self.transformer_state,
          key_mappings=self.to_hf_key_mappings,
          key_mapping_hook_fns=self.to_hf_hook_fns,
          transpose_keys=self.to_hf_transpose_keys,
          reshard_fn=reshard.reshard_pytree,
          delete_dst_buffers=self.config.delete_dst_buffers,
          reshard_chunk_size=self.config.reshard_chunk_size,
          num_kv_heads=(
              None
              if not self._model_runner
              else self._model_runner.model_config.get_total_num_kv_heads()
          ),
          head_dim=(
              None
              if not self._model_runner
              else self._model_runner.model_config.get_head_size()
          ),
          tp_size=self.args.get("tensor_parallel_size", 1),
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
          delete_dst_buffers=True,  # Ensure old weights are deleted to free up HBM memory
          reshard_chunk_size=self.config.reshard_chunk_size,
      )

    if hasattr(self._model_runner, "state_leaves"):
      if isinstance(self._model_runner.state, nnx.State):
        self._model_runner.state_leaves = tuple(
            jax.tree_util.tree_leaves(self._model_runner.state)
        )
      else:
        self._model_runner.state_leaves = self._model_runner.state

    if self.llm is not None:
      self.llm.collective_rpc("reinitialize_kv_cache")
    elif self._driver is not None:
      self._driver.llm_engine.collective_rpc("reinitialize_kv_cache")

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

    assert config.mesh is not None
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
        submission_threshold=self.config.server_mode_submission_threshold,
        submission_timeout_s=self.config.server_mode_submission_timeout_s,
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

  def tokenize(
      self, input_string: str | Sequence[int] | np.ndarray
  ) -> np.ndarray | list[int]:
    """Returns text tokenization or validates an exact pre-tokenized prompt."""
    if not isinstance(input_string, str):
      prompt_ids = np.asarray(input_string)
      if (
          prompt_ids.ndim != 1
          or prompt_ids.dtype.kind not in "iu"
          or prompt_ids.size == 0
          or np.any(prompt_ids < 0)
      ):
        raise ValueError(
            "pre-tokenized vLLM prompt must be a nonempty 1-D array of "
            "nonnegative integer token IDs"
        )
      return np.asarray(prompt_ids, dtype=np.int32)
    input_ids = self.tokenizer.encode(input_string)
    bos_tok = [self.tokenizer.bos_id()] if self.tokenizer.bos_id() else []
    return self.tokenizer.dedup_bos_ids(bos_tok + input_ids)

  def detokenize(
      self,
      input_strings: List[str | Sequence[int]],
      request_outputs: List[RequestOutput],
  ) -> Tuple[
      List[List[str]], List[List[List[float] | None]], List[List[np.ndarray]]
  ]:
    """Detokenize the vllm outputs."""
    generations = len(request_outputs[0].outputs)
    decoded_outputs = [[] for _ in range(generations)]
    out_logprobs = [[] for _ in range(generations)]
    out_tokens = [[] for _ in range(generations)]
    for input_string, multi_sampling_output in zip(
        input_strings, request_outputs
    ):
      for idx, single_output in enumerate(multi_sampling_output.outputs):
        # KEEP the eos token in the returned token_ids — needed so multi-turn
        # consumers (agentic engine) can reconstruct the exact sequence the
        # next turn's prompt was rendered from. Combined with
        # `include_stop_str_in_output=True`, vLLM emits one eos at the end of
        # each generation. Stripping it (the previous behavior) made
        # trainer-side concatenation miss `<|im_end|>` at every turn boundary
        # and produced 30+ nat sampler-trainer logp diffs.

        out_tokens[idx].append(
            np.array(single_output.token_ids, dtype=np.int32)
        )
        decoded_outputs[idx].append(
            self.tokenizer.decode(single_output.token_ids)
        )
        logprobs = None
        if self.config.return_logprobs:
          logprobs = utils.get_logprobs_from_vllm_output(
              list(single_output.token_ids), single_output.logprobs
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
      *,
      reset_prefix_cache: bool = False,
      reset_timeout_s: float = 300.0,
      request_timeout_s: float | None = None,
      verify_request_identity: bool = False,
  ) -> List[RequestOutput]:
    """Generate the response in server mode."""
    if self._driver is None:
      raise RuntimeError("vLLM in-process driver is not initialized.")
    if request_timeout_s is not None and request_timeout_s <= 0:
      raise ValueError("request_timeout_s must be positive when set")

    requests = []
    for idx, prompt in enumerate(prompts):
      request_id = str(next(self._request_counter))
      params = sampling_params
      if idx > 0 and hasattr(sampling_params, "clone"):
        params = sampling_params.clone()
      requests.append({
          "request_id": request_id,
          "prompt": prompt,
          "params": params,
      })

    if reset_prefix_cache:
      request_futures = (
          self._driver.submit_requests_after_idle_prefix_cache_reset(
              requests, timeout_s=reset_timeout_s
          )
      )
    else:
      request_futures = self._driver.submit_requests(requests)

    outputs: List[RequestOutput] = []
    deadline = (
        None
        if request_timeout_s is None
        else time.monotonic() + request_timeout_s
    )
    try:
      for request, future in zip(requests, request_futures, strict=True):
        remaining = None if deadline is None else deadline - time.monotonic()
        if remaining is not None and remaining <= 0:
          raise FutureTimeoutError()
        result = future.result(timeout=remaining)
        if not isinstance(result, RequestOutput):
          raise TypeError(
              f"Expected RequestOutput from driver, received {type(result)}."
          )
        if verify_request_identity and result.request_id != request["request_id"]:
          raise ValueError(
              "vLLM server-mode result order/request identity differs: "
              f"expected={request['request_id']} actual={result.request_id}"
          )
        outputs.append(result)
    except FutureTimeoutError as exc:
      if request_timeout_s is None:
        raise
      pending_ids = [
          request["request_id"]
          for request, future in zip(requests, request_futures)
          if not future.done()
      ]
      for request_id in pending_ids:
        self._driver.cancel(request_id)
      raise TimeoutError(
          "vLLM request deadline exceeded; aborted "
          f"{len(pending_ids)} unfinished request(s) after "
          f"{request_timeout_s:.1f}s"
      ) from exc
    except BaseException:
      for request, future in zip(requests, request_futures):
        if not future.done():
          self._driver.cancel(request["request_id"])
      raise
    return outputs

  def reset_prefix_cache_when_idle(self) -> bool:
    """Resets prefix cache without racing the continuous-batching driver."""
    if self.llm is not None:
      return self.llm.reset_prefix_cache()
    if self._driver is not None:
      return self._driver.reset_prefix_cache_when_idle()
    raise RuntimeError("vLLM engine is not initialized.")

  def generate_request_outputs(
      self,
      prompts: List[TokensPrompt],
      sampling_params: Union[SamplingParams, BeamSearchParams],
      *,
      reset_prefix_cache: bool = False,
      reset_timeout_s: float = 300.0,
  ) -> List[RequestOutput]:
    """Runs pre-tokenized requests through either supported in-process mode."""
    if self._driver is not None:
      return self._generate_server_mode(
          prompts,
          sampling_params,
          reset_prefix_cache=reset_prefix_cache,
          reset_timeout_s=reset_timeout_s,
      )
    if self.llm is not None:
      if reset_prefix_cache:
        self.llm.reset_prefix_cache()
      return self.llm.generate(
          prompts=prompts,
          sampling_params=sampling_params,
          use_tqdm=False,
      )
    raise RuntimeError("vLLM engine is not initialized.")

  def __call__(
      self,
      input_strings: str | List[str] | List[Sequence[int]],
      max_generation_steps: int,
      max_prompt_length: Optional[int] = None,
      temperature: float = 0.0,
      top_p: Optional[float] = None,
      top_k: Optional[int] = None,
      beam_size: Optional[int] = None,
      seed: Optional[
          int
      ] = None,  # vLLM Jax backend doesn't support per request seed.
      multi_sampling: int = 1,
      return_logits: bool = True,
      echo: bool = False,
      pad_output: bool = False,
      request_timeout_s: float | None = None,
      return_prompt_token_witnesses: bool = False,
      **kwargs,
  ) -> base_sampler.SamplerOutput:
    """The entry point API for vLLM Sampler"""
    if isinstance(input_strings, str):
      input_strings = [input_strings]
    if return_prompt_token_witnesses and self._driver is None:
      raise ValueError(
          "prompt-token witnesses require server mode so submitted request "
          "IDs can be bound to RequestOutput IDs"
      )

    # max_tokens: maximum number of tokens to generate
    if max_generation_steps > self.args["max_model_len"]:
      raise ValueError(
          "`max_generation_steps` must be less than or equal to "
          "`max_model_len`. Received:  `max_generation_steps`="
          f"{max_generation_steps} and `max_model_len`="
          f"{self.args['max_model_len']}."
      )
    if beam_size is not None:
      sampling_params = BeamSearchParams(
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
      _configure_logprob_request(
          sampling_params, return_logprobs=self.config.return_logprobs
      )
      if os.environ.get("CANON_APC_M15_TARGET_DEBUG", "") in ("off", "on"):
        if (
            sampling_params.prompt_logprobs is not None
            or sampling_params.logprobs != 1
            or sampling_params.skip_reading_prefix_cache is not False
        ):
          raise RuntimeError(
              "M15 APC A arm lost its production cache-readable request "
              "contract: "
              f"prompt_logprobs={sampling_params.prompt_logprobs!r} "
              f"logprobs={sampling_params.logprobs!r} "
              "skip_reading_prefix_cache="
              f"{sampling_params.skip_reading_prefix_cache!r}"
          )
        logging.log_first_n(
            logging.INFO,
            "[CAN" "ON_APC_M15_A_CONTRACT] prompt_logprobs=None logprobs=1 "
            "skip_reading_prefix_cache=False",
            1,
        )
      logging.log_first_n(
          logging.INFO,
          "[VLLM.LOGPROB_REQUEST] return_logprobs=%d sampled=%r prompt=%r "
          "host_extraction=%s",
          1,
          int(self.config.return_logprobs),
          sampling_params.logprobs,
          sampling_params.prompt_logprobs,
          "enabled" if self.config.return_logprobs else "skipped",
      )
      sampling_params.stop_token_ids = [self.tokenizer.eos_id()]
      sampling_params.skip_special_tokens = True
      # Keep the stop token in the returned ``token_ids`` so multi-turn
      # consumers can reconstruct the exact sequence the model was sampled
      # on. This makes the trainer-side concatenation align with what
      # ``apply_chat_template`` produces for the next turn's prompt; without
      # it, the trailing ``<|im_end|>`` (or equivalent eos token) is missing
      # at every turn boundary in the recorded sequence, biasing logp
      # recomputation against the model's actual sampling context.
      sampling_params.include_stop_str_in_output = True

      if top_p is not None:
        sampling_params.top_p = top_p
      if top_k is not None:
        sampling_params.top_k = top_k
      if seed is not None:
        sampling_params.seed = seed

      sampling_kwargs = self.config.sampling_kwargs.copy()
      sampling_kwargs.update(kwargs)
      if sampling_kwargs:
        try:
          logging.log_first_n(
              logging.INFO,
              "Received additional kwargs that are not explicitly defined in"
              f" the method signature: {sampling_kwargs}. These will be"
              " forwarded to the underlying sampler, but please ensure that"
              " they are valid.",
              1,
          )
          for key, value in sampling_kwargs.items():
            logging.log_first_n(
                logging.DEBUG,
                f"Sampler kwargs setting key {key} with value {value}.",
                len(sampling_kwargs),
            )
            setattr(sampling_params, key, value)
        except (AttributeError, TypeError) as e:
          logging.info(
              "Failed to update sampling_params with kwargs:"
              f" {sampling_kwargs}. Error: {e}",
          )

    prompt_ids = [self.tokenize(x) for x in input_strings]
    prompt_objects = cast(
        List[TokensPrompt],
        [{"prompt_token_ids": list(ids)} for ids in prompt_ids],
    )
    if self._driver is not None:
      outputs = self._generate_server_mode(
          prompt_objects,
          sampling_params,
          request_timeout_s=request_timeout_s,
          verify_request_identity=return_prompt_token_witnesses,
      )
    else:
      outputs = self.llm.generate(
          prompts=prompt_objects,
          sampling_params=sampling_params,
          use_tqdm=True,
      )
    decoded_outputs, out_logprobs, out_tokens = self.detokenize(
        input_strings, outputs
    )
    prompt_token_witnesses = None
    if return_prompt_token_witnesses:
      prompt_token_witnesses = build_prompt_token_witnesses(prompt_ids, outputs)
    if self.config.return_logprobs and (
        out_logprobs is None or out_logprobs[0] is None
    ):
      raise ValueError("Logprobs are not returned from the vLLM.")

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
        logprobs=out_logprobs[0] if self.config.return_logprobs else None,
        prompt_lengths=np.asarray(
            [len(prompt_id) for prompt_id in prompt_ids], dtype=np.int32
        ),
        prompt_token_witnesses=prompt_token_witnesses,
    )
