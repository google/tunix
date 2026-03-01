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
import asyncio
import threading
import concurrent.futures
import queue as _queue
from itertools import count
import math
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
from vllm.engine.arg_utils import EngineArgs, AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.inputs import TokensPrompt
from vllm.outputs import RequestOutput
from vllm.sampling_params import BeamSearchParams
from vllm.sampling_params import SamplingParams
from concurrent.futures import ThreadPoolExecutor

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
  # Use AsyncLLM in-process (creates AsyncLLM with use_uniproc_engine_core=True)
  use_async_llm_inproc: bool = True

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
    self.llm: LLM | AsyncLLM | None = None
    self._request_counter = count()
    self.async_llm: AsyncLLM | None = None
    if config.server_mode:
      if self.config.use_async_llm_inproc:
        engine_args = AsyncEngineArgs(**self.args)
        self.async_llm = AsyncLLM.from_engine_args(
            engine_args, use_uniproc_engine_core=True
        )
        # Create a dedicated event loop running in a background thread
        # for driving AsyncLLM coroutines. This avoids calling
        # `asyncio.run` from multiple worker threads which can lead to
        # nested/competing event loops and deadlocks.
        self._async_loop: asyncio.AbstractEventLoop | None = None
        self._async_loop_thread: threading.Thread | None = None
        self._async_loop_q: _queue.Queue = _queue.Queue()

        def _start_async_loop(q: _queue.Queue):
          # Create and set a new event loop for this background thread.
          loop = asyncio.new_event_loop()
          loop.set_default_executor(
              ThreadPoolExecutor(max_workers=1024, thread_name_prefix="vLLM-AsyncLoop")
          )
          asyncio.set_event_loop(loop)
          logging.debug("AsyncLLM: created new event loop in thread %s", threading.current_thread().name)
          q.put(loop)
          logging.debug("AsyncLLM: event loop put on queue; entering run_forever")
          try:
            loop.run_forever()
          finally:
            logging.debug("AsyncLLM: event loop has exited run_forever")

        self._async_loop_thread = threading.Thread(
            target=_start_async_loop, args=(self._async_loop_q,), daemon=True
        )
        self._async_loop_thread.start()
        # Wait for loop to be ready and store it
        self._async_loop = self._async_loop_q.get()
        logging.debug(
          "AsyncLLM: obtained async loop=%s; thread_alive=%s",
          self._async_loop,
          bool(self._async_loop_thread and self._async_loop_thread.is_alive()),
        )
        atexit.register(self.stop)

        async def _log_async_tasks():
            import asyncio, logging
            while True:
                try:
                    tasks = asyncio.all_tasks(self._async_loop)
                    logging.debug("Async loop tasks count=%d", len(tasks))
                    for t in list(tasks):
                        logging.debug("task=%r state=%s coro=%r", t, getattr(t, "_state", None), t.get_coro())
                except Exception:
                    logging.exception("Error while logging async tasks")
                await asyncio.sleep(10)

        # schedule it on the created loop
        self._async_loop.create_task(_log_async_tasks())
      else:
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

  def _find_total_size(self, mesh: jax.sharding.Mesh) -> int:
    """Finds the tensor parallel size from the mesh."""
    # since vllm doesn't support DP yet, simply return the total rank size.
    return math.prod(mesh.shape.values())

  def _vllm_config(self, config: VllmConfig):
    """Setup vllm config from Tunix Vllm config."""
    args = config._processed_engine_kwargs.copy()

    tensor_parallel_size = config.tensor_parallel_size
    data_parallel_size = config.data_parallel_size
    total_mesh_devices = self._find_total_size(config.mesh)

    if config.tensor_parallel_size == -1 and config.data_parallel_size == -1:
      tensor_parallel_size = total_mesh_devices
      data_parallel_size = 1
    elif config.tensor_parallel_size == -1:
      tensor_parallel_size = total_mesh_devices // data_parallel_size
    elif config.data_parallel_size == -1:
      data_parallel_size = total_mesh_devices // tensor_parallel_size

    args["data_parallel_size"] = data_parallel_size
    args["tensor_parallel_size"] = tensor_parallel_size

    # Init vLLM model with random weights to speed up bootstrap time, because
    # model weights are synced from trainer later on
    if config.init_with_random_weights:
      args["load_format"] = "dummy"

    args["gpu_memory_utilization"] = config.hbm_utilization

    args["additional_config"] = config.additional_config or {}

    if config.lora_config is not None:
      args["additional_config"]["lora_config"] = config.lora_config

    device_indexes = config.mesh.device_ids.flatten().tolist()

    args["additional_config"]["sharding"] = {
        "sharding_strategy": {
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
    if self.async_llm is not None:
      try:
        self.async_llm.shutdown()
      except Exception:
        logging.exception("Error shutting down AsyncLLM.")
      self.async_llm = None
    # Stop async loop thread if it exists
    if getattr(self, "_async_loop", None) is not None:
      try:
        self._async_loop.call_soon_threadsafe(self._async_loop.stop)
      except Exception:
        logging.exception("Error stopping AsyncLLM event loop.")
      self._async_loop = None
    if getattr(self, "_async_loop_thread", None) is not None:
      try:
        # Thread is daemon; joining is optional. Attempt a short join.
        self._async_loop_thread.join(timeout=1.0)
      except Exception:
        pass
      self._async_loop_thread = None
    if self.llm is not None:
      try:
        self.llm.shutdown()
      except Exception:
        logging.exception("Error shutting down LLM.")
      self.llm = None

  @property
  def _model_runner(self):
    if self.llm is not None:
      return self.llm.llm_engine.model_executor.driver_worker.model_runner
    if self._driver is not None:
      return self._driver.llm_engine.model_executor.driver_worker.model_runner
    if self.async_llm is not None:
      return (
          self.async_llm.engine_core.engine_core.model_executor.driver_worker.worker.model_runner
      )
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

  def _generate_async_llm(
      self, prompts: List[TokensPrompt], sampling_params: SamplingParams
  ) -> List[RequestOutput]:
    """Generate using `AsyncLLM` in-process. Runs the async generator to
    completion and returns a list of final `RequestOutput` objects (one per
    prompt)."""
    if self.async_llm is None:
      raise RuntimeError("AsyncLLM not initialized")

    # async def _collect():
    #   async def collect_one(prompt: TokensPrompt) -> RequestOutput:
    #     request_id = str(next(self._request_counter))
    #     collected: List[RequestOutput] = []
    #     async for out in self.async_llm.generate(
    #         prompt, sampling_params, request_id=request_id
    #     ):
    #       collected.append(out)
    #     if not collected:
    #       raise RuntimeError(f"No output produced for request {request_id}")
    #     # Merge outputs into a single RequestOutput (aggregate token deltas)
    #     final_out = collected[0]
    #     for part in collected[1:]:
    #       final_out.add(part, aggregate=True)
    #     return final_out

    #   # Start all prompt collectors concurrently so the engine can batch them.
    #   tasks = [collect_one(p) for p in prompts]
    #   results = await asyncio.gather(*tasks)
    #   return list(results)

    # Run the async collector and return results. Use the dedicated async
    # event loop thread created at initialization to drive AsyncLLM
    # coroutines. This avoids creating many ephemeral event loops which can
    # cause deadlocks or resource exhaustion.
    # if getattr(self, "_async_loop", None) is None:
    #   # Fallback to creating a temporary loop if the loop thread isn't
    #   # available for some reason.
    #   return asyncio.run(_collect())

    # Schedule using the centralized submit helper to keep scheduling logic
    # in one place and return the synchronous result for compatibility.
    logging.debug("AsyncLLM: scheduling async generation for %d prompts", len(prompts))
    fut: concurrent.futures.Future = self.submit_async_generation(
      prompts, sampling_params
    )
    logging.debug("AsyncLLM: submitted coroutine, waiting for result")
    return fut.result()

  def _generate_async_llm_coroutine(
      self, prompts: List[TokensPrompt], sampling_params: SamplingParams
  ):
    """Return coroutine performing async generation for use by the
    dedicated async loop. This allows callers to schedule the coroutine
    with `asyncio.run_coroutine_threadsafe` and await it without blocking
    shared executors."""
    async def _collect_wrapper():
      async def collect_one(prompt: TokensPrompt) -> RequestOutput:
        request_id = str(next(self._request_counter))
        print(f"YY {request_id=}")
        final_out: RequestOutput | None = None
        # Only keep the final output (when finished==True). Ignore intermediate
        # streaming outputs to avoid partial/unfinished results and reduce memory.
        async for out in self.async_llm.generate(
            prompt, sampling_params, request_id=request_id
        ):
          if getattr(out, "finished", False):
            final_out = out
            break

        if final_out is None:
          raise RuntimeError(f"No finished output produced for request {request_id}")
        return final_out

      tasks = [collect_one(p) for p in prompts]
      results = await asyncio.gather(*tasks)
      return list(results)

    return _collect_wrapper()

  def submit_async_generation(
      self, prompts: List[TokensPrompt], sampling_params: SamplingParams
  ) -> "concurrent.futures.Future":
    """Schedule async generation on the dedicated async loop and return
    a concurrent.futures.Future. Callers can await it using
    `asyncio.wrap_future` in their event loop to avoid blocking executor
    threads."""
    if getattr(self, "_async_loop", None) is None:
      logging.error(
        "submit_async_generation: async loop is None; _async_loop_thread=%s",
        getattr(self, "_async_loop_thread", None),
      )
      raise RuntimeError("Async loop not initialized for AsyncLLM")

    thread_alive = bool(getattr(self, "_async_loop_thread", None) and self._async_loop_thread.is_alive())
    logging.debug(
      "submit_async_generation: scheduling on loop=%s thread_alive=%s prompts=%d",
      self._async_loop,
      thread_alive,
      len(prompts),
    )

    fut = asyncio.run_coroutine_threadsafe(
      self._generate_async_llm_coroutine(prompts, sampling_params), self._async_loop
    )
    logging.debug("submit_async_generation: returned future %s", fut)


    return fut


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
      elif self.async_llm is not None:
        sampling_params = self.async_llm.get_default_sampling_params()
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
      if getattr(self, "async_llm", None) is not None:
        outputs = self._generate_async_llm(prompt_objects, self.sampling_params)
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

    for token in out_tokens[0]:
      logging.info("vLLM output length: %d", len(token))

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
    # print("Decoded outputs: ", decoded_outputs[0])
    # print(f"YY {decoded_outputs[0]=}", flush=True)
    return base_sampler.SamplerOutput(
        text=decoded_outputs[0],
        logits=None,
        tokens=out_tokens[0],
        padded_prompt_tokens=all_input_ids,
        logprobs=out_logprobs[0],
    )
