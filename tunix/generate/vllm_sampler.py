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
import concurrent.futures
import copy
import dataclasses
import gc
from itertools import count
import os
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast

from absl import logging
import jax
import jaxtyping
import numpy as np
import tqdm
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
from vllm.sampling_params import RequestOutputKind
from vllm.sampling_params import SamplingParams

# Colocate vllm engine and worker in the main process
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"


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
  # Capture the MoE expert ids the rollout actually routed through, so training
  # can replay them. Sets vLLM's `enable_return_routed_experts` engine arg.
  return_routed_experts: bool = False
  # Token ids that terminate a generation. Defaults to the tokenizer's single
  # `eos_id()`, which for a chat model is only the chat-turn terminator.
  eos_tokens: Optional[List[int]] = None

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
  # The weight sync materializes a second copy of the sampler weights on HBM
  # while the new values are resharded in; freeing the KV cache first makes
  # room for that copy. When the copy fits next to the KV pool anyway (small
  # model, large pool) set False to skip the two collective RPCs and the
  # re-allocation (~2 s per RL step on Qwen3-0.6B with a 57 GB pool).
  free_kv_cache_during_weight_sync: bool = True
  # Decode the text and extract the logprobs of each request as soon as it
  # finishes, in a thread pool, while the engine keeps decoding the rest of
  # the batch. Otherwise all of that runs serially after the last request
  # finishes. Outputs are identical either way; False keeps the plain
  # `LLM.generate` call (offline mode) / post-processing after all driver
  # futures resolved (server mode).
  overlap_postprocessing: bool = True
  # Threads decoding finished requests when `overlap_postprocessing` is on.
  postprocessing_threads: int = 4

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
    if self.postprocessing_threads < 1:
      raise ValueError(
          "postprocessing_threads must be >= 1, got"
          f" {self.postprocessing_threads}"
      )
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
    self._postprocess_pool: Optional[concurrent.futures.ThreadPoolExecutor] = (
        None
    )
    self._postprocess_pool_lock = threading.Lock()
    self._thread_local = threading.local()
    self._driver: VLLMInProcessDriver | None = None
    self.llm: LLM | None = None
    self._request_counter = count()

    if config.server_mode:
      self._driver = self._create_driver()
      utils.detach_incompatible_vllm_cleanup_finalizer(self._driver.llm_engine)
      atexit.register(self.stop)
    else:
      self.llm = LLM(**self.args)
      utils.detach_incompatible_vllm_cleanup_finalizer(self.llm.llm_engine)

    self.to_hf_key_mappings = dict(config.mapping_config.to_hf_mappings or {})
    self.to_hf_transpose_keys = config.mapping_config.to_hf_transpose_keys
    self.to_hf_hook_fns = config.mapping_config.to_hf_hook_fns

    # TODO(b/434959964) It's not taking effect until vLLM Jax backend support
    # lora.
    if config.lora_config and config.mapping_config.lora_to_hf_mappings:
      self.to_hf_key_mappings |= config.mapping_config.lora_to_hf_mappings

  @property
  def _postprocessed(self) -> Optional[Dict[str, list[Any]]]:
    return getattr(self._thread_local, "postprocessed", None)

  @_postprocessed.setter
  def _postprocessed(self, value: Optional[Dict[str, list[Any]]]) -> None:
    self._thread_local.postprocessed = value

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

  def delete_cache(self) -> None:
    if self.llm is not None:
      self.llm.reset_prefix_cache()
      self.llm.collective_rpc("delete_kv_cache")  # will free hbm
    elif self._driver is not None:
      self._driver.llm_engine.reset_prefix_cache()
      self._driver.llm_engine.collective_rpc("delete_kv_cache")

  def reset_prefix_cache(self) -> None:
    if self.llm is not None:
      self.llm.reset_prefix_cache()
    elif self._driver is not None:
      self._driver.llm_engine.reset_prefix_cache()

  def refresh_state_leaves(self) -> None:
    """Re-reads the runner's state leaves after its params were updated."""
    self._model_runner.state_leaves = tuple(
        jax.tree_util.tree_leaves(self._model_runner.state)
    )

  def reinitialize_cache(self) -> None:
    self.refresh_state_leaves()

    if self.llm is not None:
      self.llm.collective_rpc("reinitialize_kv_cache")
    elif self._driver is not None:
      self._driver.llm_engine.collective_rpc("reinitialize_kv_cache")

  # TODO(b/434969743): Optimize weight sharing between trainer and vllm sampler.
  def update_params(
      self,
      updated_weights: jaxtyping.PyTree,
      filter_types: Optional[Tuple[Any, ...]] = None,
  ):
    del filter_types

    if self.config.free_kv_cache_during_weight_sync:
      self.delete_cache()
    else:
      # Keep the KV pool allocated; only its (stale) prefix entries go.
      self.reset_prefix_cache()

    # Synchronization point before weight sync
    jax.effects_barrier()

    if self.to_hf_key_mappings:
      preprocess_fn = self.config.mapping_config.preprocess_src_state
      if preprocess_fn:
        updated_weights = preprocess_fn(updated_weights)

      if self._is_torchax_backend():
        self._update_params_torchax(updated_weights)
      else:
        self._update_params_jax(updated_weights)
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

    if self.config.free_kv_cache_during_weight_sync:
      self.reinitialize_cache()
    else:
      self.refresh_state_leaves()

  def _is_torchax_backend(self) -> bool:
    """True when tpu-inference runs the vLLM (torchax) model implementation.

    That path keeps the weights in a flat ``{name: jax.Array}`` dict in an
    internal, tp-dependent layout and exposes `load_canonical_weights` on the
    model wrapper to accept weights in vLLM's canonical (TP=1) layout.
    """
    runner = self._model_runner
    return isinstance(getattr(runner, "state", None), dict) and hasattr(
        getattr(runner, "model", None), "load_canonical_weights"
    )

  def _update_params_torchax(self, updated_weights: jaxtyping.PyTree) -> None:
    """Mapped weight sync into the tpu-inference torchax model.

    The mapping targets are the canonical vLLM parameter names/shapes
    (`canonical_weight_specs`), not the runner's internal layout, so the
    mapping stays independent of tensor parallelism and MoE backend. The
    resharding and layout processing is delegated to tpu-inference.
    """
    runner = self._model_runner
    specs = runner.model.canonical_weight_specs(runner.state)
    mapped = utils.transfer_state_with_mappings(
        src_state=updated_weights,
        dst_state=specs,
        key_mappings=self.to_hf_key_mappings,
        key_mapping_hook_fns=self.to_hf_hook_fns,
        transpose_keys=self.to_hf_transpose_keys,
        reshard_fn=None,
        num_kv_heads=runner.model_config.get_total_num_kv_heads(),
        head_dim=runner.model_config.get_head_size(),
        tp_size=self.args.get("tensor_parallel_size", 1),
    )
    # Untouched targets are still their ShapeDtypeStruct placeholders.
    canonical = {
        k: v
        for k, v in mapped.items()
        if not isinstance(v, jax.ShapeDtypeStruct)
    }
    runner.model.load_canonical_weights(canonical, runner.state)

  def _update_params_jax(self, updated_weights: jaxtyping.PyTree) -> None:
    """Mapped weight sync into the tpu-inference flax/nnx model state."""
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

    if config.return_routed_experts:
      args["enable_return_routed_experts"] = True

    args["additional_config"] = config.additional_config or {}

    if config.lora_config is not None:
      args["additional_config"]["lora_config"] = config.lora_config

    if config.mesh:
      tp, dp, ep = utils.resolve_parallelism_sizes(
          mesh=config.mesh,  # pyrefly: ignore[bad-argument-type]
          tensor_parallel_size=config.tensor_parallel_size,
          data_parallel_size=config.data_parallel_size,
          expert_parallel_size=config.expert_parallel_size,
      )
      args["tensor_parallel_size"] = tp
      args["data_parallel_size"] = dp

      assert config.mesh is not None
      device_indexes = config.mesh.device_ids.flatten().tolist()
      # Merge with any sharding settings the caller put in `additional_config`
      # (e.g. tpu-inference's `attn_dp_size`) instead of dropping them.
      sharding = dict(args["additional_config"].get("sharding") or {})
      strategy = dict(sharding.get("sharding_strategy") or {})
      strategy.setdefault("expert_parallelism", ep)
      strategy.setdefault("enable_dp_attention", config.enable_dp_attention)
      if config.enable_dp_attention:
        strategy["enable_dp_attention"] = True
      strategy["device_indexes"] = device_indexes
      sharding["sharding_strategy"] = strategy
      args["additional_config"] = dict(args["additional_config"])
      args["additional_config"]["sharding"] = sharding
    else:
      # In distributed setting, JAX backend is not initialized at this point, so
      # we can't use mesh to resolve parallelism sizes and device indexes.
      args["tensor_parallel_size"] = config.tensor_parallel_size
      args["data_parallel_size"] = config.data_parallel_size

      sharding = dict(args["additional_config"].get("sharding") or {})
      strategy = dict(sharding.get("sharding_strategy") or {})
      strategy.setdefault("expert_parallelism", config.expert_parallel_size)
      strategy.setdefault("enable_dp_attention", config.enable_dp_attention)
      if config.enable_dp_attention:
        strategy["enable_dp_attention"] = True
      sharding["sharding_strategy"] = strategy
      args["additional_config"] = dict(args["additional_config"])
      args["additional_config"]["sharding"] = sharding

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
    with self._postprocess_pool_lock:
      if self._postprocess_pool is not None:
        self._postprocess_pool.shutdown(wait=False)
        self._postprocess_pool = None

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

  def tokenize(self, input_string: str) -> np.ndarray | list[int]:
    """Tokenizes the input string."""
    input_ids = self.tokenizer.encode(input_string)
    bos_tok = [self.tokenizer.bos_id()] if self.tokenizer.bos_id() else []
    return self.tokenizer.dedup_bos_ids(bos_tok + input_ids)

  def detokenize(
      self, input_strings: List[str], request_outputs: List[RequestOutput]
  ) -> Tuple[
      List[List[str]],
      List[List[List[float] | None]],
      List[List[np.ndarray]],
      List[List[np.ndarray | None]],
  ]:
    """Detokenize the vllm outputs."""
    generations = len(request_outputs[0].outputs)
    decoded_outputs = [[] for _ in range(generations)]
    out_logprobs = [[] for _ in range(generations)]
    out_tokens = [[] for _ in range(generations)]
    out_routed_experts = [[] for _ in range(generations)]
    precomputed = self._postprocessed or {}
    self._postprocessed = None
    for input_string, multi_sampling_output in zip(
        input_strings, request_outputs
    ):
      ready = precomputed.get(multi_sampling_output.request_id)
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
        if ready is not None:
          text, logprobs = ready[idx]
        else:
          text, logprobs = self._decode_single_output(single_output)
        decoded_outputs[idx].append(text)
        out_logprobs[idx].append(logprobs)
        # `[length, num_layers, top_k]`, or None when capture is disabled.
        out_routed_experts[idx].append(
            getattr(single_output, "routed_experts", None)
        )
        logging.debug(
            "Prompt: %r\n\nGenerated text: %r\n\n ",
            input_string,
            decoded_outputs[idx][-1],
        )
    return decoded_outputs, out_logprobs, out_tokens, out_routed_experts

  def _decode_single_output(
      self, single_output: Any
  ) -> Tuple[str, List[float] | None]:
    """Text and per-token logprobs of one sampled completion."""
    text = self.tokenizer.decode(single_output.token_ids)  # pyrefly: ignore[bad-argument-type]
    logprobs = utils.get_logprobs_from_vllm_output(
        list(single_output.token_ids), single_output.logprobs  # pyrefly: ignore[bad-argument-type]
    )
    return text, logprobs

  def _postprocess_request_output(
      self, request_output: RequestOutput
  ) -> List[Tuple[str, List[float] | None]]:
    """Decodes every sample of a finished request (runs in the thread pool)."""
    return [self._decode_single_output(o) for o in request_output.outputs]

  def _get_postprocess_pool(self) -> concurrent.futures.ThreadPoolExecutor:
    if self._postprocess_pool is None:
      with self._postprocess_pool_lock:
        if self._postprocess_pool is None:
          self._postprocess_pool = concurrent.futures.ThreadPoolExecutor(
              max_workers=self.config.postprocessing_threads,
              thread_name_prefix="vllm-postprocess",
          )
    return self._postprocess_pool

  def _postprocess_as_completed(
      self, futures: List[concurrent.futures.Future[Any]]
  ) -> None:
    """Server mode: decodes each request as its driver future resolves."""
    pool = self._get_postprocess_pool()
    self._postprocessed = None
    pending: Dict[str, concurrent.futures.Future[Any]] = {}
    for future in concurrent.futures.as_completed(futures):
      result = future.result()
      if isinstance(result, RequestOutput):
        pending[result.request_id] = pool.submit(
            self._postprocess_request_output, result
        )
    self._postprocessed = {rid: f.result() for rid, f in pending.items()}

  def _generate_offline(
      self,
      prompts: List[TokensPrompt],
      sampling_params: Union[
          SamplingParams, BeamSearchParams, List[SamplingParams]
      ],
  ) -> List[RequestOutput]:
    """Offline generation; overlaps post-processing with decode when enabled.

    Beam search keeps the plain `LLM.generate` path: `BeamSearchParams` are
    expanded by vLLM itself and never reach the engine as one request.
    """
    if not self.config.overlap_postprocessing or isinstance(
        sampling_params, BeamSearchParams
    ):
      return self.llm.generate(  # pyrefly: ignore[missing-attribute]
          prompts=prompts,
          sampling_params=sampling_params,
          use_tqdm=True,
      )
    # Same loop as vllm's LLM.generate -> _run_engine, except that every
    # finished request is handed to the thread pool right away, so decoding
    # its text and extracting its logprobs overlap with the remaining decode
    # steps instead of running serially after the last one. Request ids come
    # from the LLM's own counter so they stay unique alongside any
    # `LLM.generate` call on the same engine.
    pool = self._get_postprocess_pool()
    self._postprocessed = None
    engine = self.llm.llm_engine  # pyrefly: ignore[missing-attribute]
    counter = self.llm.request_counter  # pyrefly: ignore[missing-attribute]
    for idx, prompt in enumerate(prompts):
      params = (
          sampling_params[idx]
          if isinstance(sampling_params, list)
          else sampling_params
      )
      params.output_kind = RequestOutputKind.FINAL_ONLY
      engine.add_request(str(next(counter)), prompt, params)
    outputs: List[RequestOutput] = []
    futures: Dict[str, concurrent.futures.Future[Any]] = {}
    progress = tqdm.tqdm(
        total=len(prompts), desc="Processed prompts", dynamic_ncols=True
    )
    try:
      while engine.has_unfinished_requests():
        for output in engine.step():
          if output.finished:
            outputs.append(output)
            futures[output.request_id] = pool.submit(
                self._postprocess_request_output, output
            )
            progress.update(1)
    finally:
      progress.close()
    self._postprocessed = {rid: f.result() for rid, f in futures.items()}
    # vLLM's LLM.generate returns outputs sorted by request id as well.
    return sorted(outputs, key=lambda o: int(o.request_id))

  def _generate_server_mode(
      self,
      prompts: List[TokensPrompt],
      sampling_params: Union[
          SamplingParams, BeamSearchParams, List[SamplingParams]
      ],
  ) -> List[RequestOutput]:
    """Generate the response in server mode."""
    if self._driver is None:
      raise RuntimeError("vLLM in-process driver is not initialized.")

    requests = []
    for idx, prompt in enumerate(prompts):
      request_id = str(next(self._request_counter))
      if isinstance(sampling_params, list):
        params = sampling_params[idx]
      else:
        params = sampling_params
        if idx > 0 and hasattr(sampling_params, "clone"):
          params = sampling_params.clone()
      requests.append({
          "request_id": request_id,
          "prompt": prompt,
          "params": params,
      })

    futures = self._driver.submit_requests(requests)
    if self.config.overlap_postprocessing:
      self._postprocess_as_completed(futures)

    outputs: List[RequestOutput] = []
    for future in futures:
      result = future.result()
      if not isinstance(result, RequestOutput):
        raise TypeError(
            f"Expected RequestOutput from driver, received {type(result)}."
        )
      outputs.append(result)
    return outputs

  def _eos_token_ids(self) -> List[int]:
    """Returns the token ids that terminate a generation.

    A tokenizer exposes a single `eos_id()`, but a model may declare several
    terminators and use a different one depending on how it was prompted. Qwen3,
    for instance, lists both `<|im_end|>` and `<|endoftext|>` in its generation
    config: a chat-formatted turn ends with the former, a raw completion with
    the latter, and the tokenizer only reports the former. Stopping solely on
    `eos_id()` therefore leaves completion-mode rollouts running until they hit
    `max_tokens`.

    Returns:
      The union of `VllmConfig.eos_tokens` and the tokenizer's single
      end-of-sequence id.
    """
    eos_ids = set(self.config.eos_tokens or [])
    if self.tokenizer is not None and self.tokenizer.eos_id() is not None:
      eos_ids.add(self.tokenizer.eos_id())
    return list(eos_ids)
  def __call__(
      self,
      input_strings: str | List[str] | None = None,
      max_generation_steps: int = 0,
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
      *,
      prompt_token_ids: Sequence[Sequence[int] | np.ndarray] | None = None,
      **kwargs,
  ) -> base_sampler.SamplerOutput:
    """The entry point API for vLLM Sampler"""
    exact_input = prompt_token_ids is not None
    prompt_ids = utils.resolve_prompt_tokens(
        input_strings,
        prompt_token_ids,
        self.tokenize,
        max_generation_steps=max_generation_steps,
        max_total_length=self.args["max_model_len"],
        max_length_name="max_model_len",
    )
    if exact_input:
      # TODO(b/399000000): Clean up detokenize() so dummy input_strings are not needed.
      input_strings = [""] * len(prompt_ids)
    else:
      assert input_strings is not None
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
    raw_prompt_start = None
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
        sampling_params = self.llm.get_default_sampling_params()  # pyrefly: ignore[missing-attribute]
      sampling_params.detokenize = False
      sampling_params.max_tokens = max_generation_steps
      sampling_params.n = multi_sampling
      sampling_params.temperature = temperature
      if self.config.return_logprobs:
        sampling_params.logprobs = 1  # b/428730696
        sampling_params.prompt_logprobs = None  # b/428730696
      else:
        sampling_params.logprobs = 0
        sampling_params.prompt_logprobs = None
      sampling_params.stop_token_ids = self._eos_token_ids()
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
      raw_prompt_start = sampling_kwargs.pop(
          "routed_experts_prompt_start", None
      )
      if raw_prompt_start is not None and not isinstance(
          raw_prompt_start, (list, tuple)
      ):
        setattr(
            sampling_params, "routed_experts_prompt_start", raw_prompt_start
        )
        raw_prompt_start = None

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

    if exact_input and (
        isinstance(sampling_params, BeamSearchParams)
        or getattr(sampling_params, "n", 1) != 1
        or getattr(sampling_params, "truncate_prompt_tokens", None) is not None
    ):
      # One sampled row per submitted row, no truncation: the echo check and
      # the recorded history assume the engine consumed exactly these ids.
      raise ValueError("prompt_token_ids requires exactly one output per row")
    prompt_objects = cast(
        List[TokensPrompt],
        [{"prompt_token_ids": list(ids)} for ids in prompt_ids],
    )
    target_sampling_params: Union[
        SamplingParams, BeamSearchParams, List[SamplingParams]
    ] = sampling_params
    if raw_prompt_start is not None and isinstance(
        raw_prompt_start, (list, tuple)
    ):
      assert len(raw_prompt_start) == len(prompt_objects), (
          f"Length of routed_experts_prompt_start ({len(raw_prompt_start)}) "
          f"does not match number of prompts ({len(prompt_objects)})."
      )
      prompt_params_list: List[SamplingParams] = []
      for offset in raw_prompt_start:
        p = cast(
            SamplingParams,
            sampling_params.clone()
            if hasattr(sampling_params, "clone")
            else copy.deepcopy(sampling_params),
        )
        setattr(p, "routed_experts_prompt_start", offset)
        prompt_params_list.append(p)
      target_sampling_params = prompt_params_list

    if self._driver is not None:
      outputs = self._generate_server_mode(
          prompt_objects, target_sampling_params
      )
    else:
      outputs = self._generate_offline(prompt_objects, target_sampling_params)
    if exact_input:
      utils.check_prompt_echo(prompt_ids, outputs, backend_name="vLLM")
    decoded_outputs, out_logprobs, out_tokens, out_routed_experts = (
        self.detokenize(input_strings, outputs)
    )
    if self.config.return_logprobs and (
        out_logprobs is None or out_logprobs[0] is None
    ):
      raise ValueError("Logprobs are not returned from the vLLM.")

    all_input_ids, prompt_lengths, max_prompt_length = (
        utils.left_pad_prompt_tokens(
            prompt_ids,
            max_prompt_length,
            self.tokenizer.pad_id(),
        )
    )

    # To support multisampling, just return the whole list of SamplerOutput
    return base_sampler.SamplerOutput(
        text=decoded_outputs[0],
        logits=None,
        tokens=out_tokens[0],
        padded_prompt_tokens=all_input_ids,
        logprobs=out_logprobs[0] if self.config.return_logprobs else None,  # pyrefly: ignore[bad-argument-type]
        routed_experts=(
            out_routed_experts[0] if self.config.return_routed_experts else None
        ),
        prompt_lengths=prompt_lengths,
    )
