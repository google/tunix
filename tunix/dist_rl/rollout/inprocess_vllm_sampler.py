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

"""In-process vLLM rollout sampler."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from concurrent import futures
import functools
import os
from typing import Any, TYPE_CHECKING

from absl import logging
import jax
import jaxtyping
import numpy as np
import transformers
from tunix.common import configs
from tunix.dist_rl.rollout import sampler as base_sampler_lib
from tunix.dist_rl.rollout.raiden_weight_sync_mixin import (
    RaidenDestinationWeightSyncMixin,
)
from tunix.dist_rl.weight_sync import raiden_weight_sync_delegate
from tunix.dist_rl.weight_sync import weight_sync
from tunix.generate import mappings
from tunix.generate import tokenizer_adapter as tok_adapter
from tunix.generate import utils as generate_utils

import sentencepiece as spm

# TODO(yatlas): Clean up lazy vLLM imports so TYPE_CHECKING is not needed.
if TYPE_CHECKING:
  from tunix.generate import vllm_sampler as generate_vllm_lib  # pylint: disable=g-import-not-at-top

Sampler = base_sampler_lib.Sampler
# TODO(yatlas): Remove default hbm_utilization once colocated hbm fixes are
# landed.
_DEFAULT_HBM_UTILIZATION = 0.5


def _get_vllm_sampler_cls():
  """Lazy import of tunix.generate.vllm_sampler to avoid top-level vLLM import side-effects."""
  from tunix.generate import vllm_sampler as generate_vllm_lib  # pylint: disable=g-import-not-at-top

  return generate_vllm_lib


class InprocessVllmSampler(RaidenDestinationWeightSyncMixin, Sampler):
  """Rollout sampler that runs the vLLM engine inside the rollout process.

  This is the default rollout sampler for RL training with the Tunix trainer.
  Use `VllmSampler` instead to keep the engine in its own process, or with a
  trainer that emits weights in the rollout model's layout, such as MaxText.

  Because the rollout weights live in this process, every weight-sync path
  works: Raiden binds the live weights, `get_target_state()` returns the weight
  tree the trainer converts to, and the non-Raiden `update_params` fallback
  works. Multi-host TPU works through vLLM's Ray executor
  (`TPU_MULTIHOST_BACKEND=ray`).

  `get_load_info()` and `get_transfer_status()` return placeholders.
  """

  def __init__(
      self,
      server_id: str,
      tokenizer: (
          transformers.PreTrainedTokenizerBase | spm.SentencePieceProcessor
      ),
      config: configs.RolloutConfig,
      mesh: jax.sharding.Mesh | None = None,
      max_concurrency: int = 256,
      raiden_sync_delegate: (
          raiden_weight_sync_delegate.RaidenWeightSyncDelegate | None
      ) = None,
  ):
    """Initializes the sampler.

    Args:
      server_id: Unique identifier for this sampler server slice.
      tokenizer: HuggingFace or SentencePiece tokenizer of the rollout model.
      config: Rollout configuration. `initialize()` builds the vLLM engine from
        it.
      mesh: Device mesh for the vLLM engine, or None when vLLM places the model
        itself (the Ray multi-host executor).
      max_concurrency: Number of threads that submit `sample()` calls to the
        engine.
      raiden_sync_delegate: Raiden delegate to use instead of building one when
        `config.weight_sync_mode` is RAIDEN.
    """
    self.server_id = server_id
    self.tokenizer = tok_adapter.TokenizerAdapter(tokenizer)
    self.config: configs.RolloutConfig = config
    self._mesh = mesh
    self.vllm_sampler: generate_vllm_lib.VllmSampler | None = None
    self.max_concurrency = max_concurrency
    self._executor = futures.ThreadPoolExecutor(
        max_workers=self.max_concurrency,
        thread_name_prefix=f"{self.server_id}_vllm_worker",
    )
    self.weight_sync_mode = weight_sync.WeightSyncMode(
        self.config.weight_sync_mode
    )
    self.enable_raiden = (
        self.weight_sync_mode == weight_sync.WeightSyncMode.RAIDEN
    )

    if self.enable_raiden and raiden_sync_delegate is None:
      logging.info(
          "InprocessVllmSampler [%s] weight_sync: initializing Raiden delegate",
          self.server_id,
      )
      raiden_sync_delegate = (
          raiden_weight_sync_delegate.RaidenWeightSyncDelegate(
              server_id=self.server_id,
              partial_rollout=self.config.partial_rollout,
          )
      )

    if not self.enable_raiden and raiden_sync_delegate is not None:
      logging.warning(
          "InprocessVllmSampler [%s] raiden_sync_delegate is set but"
          " enable_raiden is False.",
          self.server_id,
      )
    self.raiden_sync_delegate = raiden_sync_delegate

  def initialize(self) -> None:
    """Builds the vLLM engine from `config` on the first call."""
    if self.vllm_sampler is not None:
      return
    vllm_lib = _get_vllm_sampler_cls()
    config = self.config
    if not config.rollout_vllm_server_mode:
      logging.warning(
          "InprocessVllmSampler [%s] requires server_mode=True; ignoring"
          " config.rollout_vllm_server_mode=False.",
          self.server_id,
      )
    if self._mesh is None and (
        config.tensor_parallel_size < 1 or config.data_parallel_size < 1
    ):
      raise ValueError(
          f"InprocessVllmSampler [{self.server_id}] requires positive"
          " config.tensor_parallel_size and config.data_parallel_size when"
          f" mesh is None, got {config.tensor_parallel_size} and"
          f" {config.data_parallel_size}."
      )
    mesh_shape = getattr(self._mesh, "shape", None)
    if isinstance(mesh_shape, dict) and "fsdp" in mesh_shape:
      effective_dp = int(mesh_shape["fsdp"])
    else:
      effective_dp = config.data_parallel_size
    if config.rollout_vllm_lora_config is not None and effective_dp > 1:
      raise ValueError(
          "LoRA is not supported with rollout data parallelism:"
          f" data_parallel_size={effective_dp} with"
          " config.rollout_vllm_lora_config. Set data_parallel_size=1 or drop"
          " LoRA."
      )
    extra_kwargs = dict(config.rollout_vllm_kwargs)
    extra_kwargs.pop("server_mode", None)
    gpu_memory_utilization = extra_kwargs.pop("gpu_memory_utilization", None)
    additional_config = generate_utils.merge_additional_config(
        config.rollout_vllm_additional_config,
        extra_kwargs.pop("additional_config", None),
    )
    # TODO(tunix-dev): Share this engine-kwargs construction with
    # `VllmSampler.__init__` in one helper so the two samplers can't drift.
    engine_kwargs: dict[str, Any] = {
        "model": config.rollout_vllm_model_version,
        "max_model_len": config.kv_cache_size,
        "async_scheduling": config.rollout_vllm_async_scheduling,
        "max_num_batched_tokens": config.rollout_vllm_max_num_batched_tokens,
        "max_num_seqs": config.rollout_vllm_max_num_seqs,
        "hf_config_path": config.rollout_vllm_hf_config_path,
        # Only the logprob of each sampled token is needed.
        "max_logprobs": 1,
        "logprobs_mode": config.rollout_vllm_logprobs_mode,
        "trust_remote_code": True,
    }
    multihost_backend = os.environ.get("TPU_MULTIHOST_BACKEND", "")
    if multihost_backend:
      engine_kwargs["distributed_executor_backend"] = multihost_backend
    engine_kwargs.update(extra_kwargs)
    if config.rollout_vllm_hbm_utilization is not None:
      hbm_utilization = config.rollout_vllm_hbm_utilization
    elif gpu_memory_utilization is not None:
      hbm_utilization = gpu_memory_utilization
    else:
      hbm_utilization = _DEFAULT_HBM_UTILIZATION
    vllm_config = vllm_lib.VllmConfig(
        # `sample()` dispatches concurrent requests across `self._executor`
        # worker threads. Force `server_mode=True` so `VllmSampler` uses
        # `VLLMInProcessDriver` (where a single background engine thread drains
        # a thread-safe request queue for continuous batching) instead of
        # `_generate_offline()`, which calls `engine.step()` directly from
        # caller threads and races on donated JAX KV-cache buffers (`Array has
        # been deleted`).
        server_mode=True,
        server_mode_submission_threshold=(
            config.rollout_vllm_server_mode_submission_threshold
        ),
        server_mode_submission_timeout_s=(
            config.rollout_vllm_server_mode_submission_timeout_s
        ),
        mapping_config=(
            config.rollout_mapping_config or mappings.MappingConfig()
        ),
        return_logprobs=config.return_logprobs,
        return_routed_experts=config.return_routed_experts,
        eos_tokens=config.eos_tokens,
        init_with_random_weights=config.rollout_vllm_init_with_random_weights,
        tpu_backend_type=config.rollout_vllm_tpu_backend_type,
        additional_config=additional_config,
        enable_dp_attention=config.rollout_vllm_enable_dp_attention,
        hbm_utilization=hbm_utilization,
        lora_config=config.rollout_vllm_lora_config,
        mesh=self._mesh,
        tensor_parallel_size=config.tensor_parallel_size,
        data_parallel_size=config.data_parallel_size,
        expert_parallel_size=config.expert_parallel_size,
        delete_dst_buffers=config.rollout_vllm_delete_dst_buffers,
        reshard_chunk_size=config.rollout_vllm_reshard_chunk_size,
        free_kv_cache_during_weight_sync=(
            config.rollout_vllm_free_kv_cache_during_weight_sync
        ),
        partial_rollout=config.partial_rollout,
        overlap_postprocessing=config.rollout_vllm_overlap_postprocessing,
        engine_kwargs=engine_kwargs,
        sampling_kwargs=config.rollout_vllm_sampling_kwargs,
    )
    self.vllm_sampler = vllm_lib.VllmSampler(
        tokenizer=self.tokenizer, config=vllm_config
    )

  def _get_underlying_sampler(
      self,
  ) -> generate_vllm_lib.VllmSampler | None:
    return self.vllm_sampler

  def _require_vllm_sampler(self) -> generate_vllm_lib.VllmSampler:
    if self.vllm_sampler is None:
      raise RuntimeError(
          f"InprocessVllmSampler [{self.server_id}] vllm_sampler is not"
          " initialized."
      )
    return self.vllm_sampler

  @property
  def partial_rollout(self) -> bool:
    """Whether in-flight weight updates (partial rollout) are enabled."""
    return self.config.partial_rollout

  @property
  def transformer_state(self) -> Any:
    """The underlying vLLM sampler's transformer state."""
    return self._require_vllm_sampler().transformer_state

  def update_params(
      self,
      updated_weights: jaxtyping.PyTree,
      filter_types: tuple[Any, ...] | None = None,
  ) -> None:
    """Updates model parameters on the underlying vLLM sampler."""
    self._require_vllm_sampler().update_params(
        updated_weights, filter_types=filter_types
    )

  def delete_cache(self) -> None:
    """Deletes KV cache on the underlying vLLM sampler before weight sync."""
    self._require_vllm_sampler().delete_cache()

  def reset_prefix_cache(self) -> None:
    """Resets prefix cache on the underlying vLLM sampler."""
    self._require_vllm_sampler().reset_prefix_cache()

  def reinitialize_cache(self) -> None:
    """Reallocates KV cache on the underlying vLLM sampler after weight sync."""
    self._require_vllm_sampler().reinitialize_cache()

  def refresh_state_leaves(self) -> None:
    """Refreshes cached state leaves on the underlying vLLM sampler."""
    self._require_vllm_sampler().refresh_state_leaves()

  def _prompt_tokens_from_request(
      self,
      req: base_sampler_lib.SamplingRequest,
      fallback_padded_tokens: Any,
      prompt_length: int | None = None,
  ) -> np.ndarray:
    """Returns request token ids directly when available, else sampler output."""
    if generate_utils.is_token_id_sequence(req.prompt):
      return np.asarray(req.prompt, dtype=np.int32).reshape(-1)
    return generate_utils.unpad_prompt_tokens(
        fallback_padded_tokens,
        pad_id=self.tokenizer.pad_id(),
        prompt_length=prompt_length,
    )

  def _prompt_to_input_string(self, prompt: Any) -> Any:
    """Renders chat-message prompts to strings for Tunix VllmSampler."""
    if isinstance(prompt, str):
      return prompt
    if isinstance(prompt, (list, tuple)) and all(
        isinstance(message, dict) for message in prompt
    ):
      return self.tokenizer.apply_chat_template(
          list(prompt), tokenize=False, add_generation_prompt=True
      )
    return prompt

  # --- Lifecycle & Topology ---
  async def start(self, **kwargs) -> str | None | Any:
    """Starts the sampling engine or local loop."""
    del kwargs
    return True

  async def stop(self, **kwargs) -> str | None | Any:
    """Terminates sampler execution and closes local connections."""
    del kwargs
    if self._executor is not None:
      self._executor.shutdown(wait=False, cancel_futures=True)
    if self.vllm_sampler is not None:
      self.vllm_sampler.stop()
    return True

  async def pause(
      self,
      mode: str = "keep",
      *,
      clear_cache: bool = False,
      **kwargs,
  ) -> str | None | Any:
    """Pauses inference processing on this worker slice."""
    del kwargs
    if self.vllm_sampler is not None:
      await asyncio.to_thread(
          self.vllm_sampler.pause, mode=mode, clear_cache=clear_cache
      )
    return True

  async def resume(self, **kwargs) -> str | None | Any:
    """Resumes inference processing on this worker slice."""
    del kwargs
    if self.vllm_sampler is not None:
      await asyncio.to_thread(self.vllm_sampler.resume)
    return True

  async def get_mesh(self, **kwargs) -> Any:
    """Returns the underlying device mesh topology."""
    del kwargs
    if self.vllm_sampler is not None:
      return self.vllm_sampler.mesh
    return None

  def routed_experts_shape(self) -> tuple[int, int] | None:
    """Returns `(num_layers, top_k)` when MoE routing capture is enabled, else None."""
    if not self.config.return_routed_experts or self.vllm_sampler is None:
      return None
    return self.vllm_sampler.routed_experts_shape()

  # --- Inference ---
  async def sample(
      self,
      sampling_requests: (
          base_sampler_lib.SamplingRequest
          | Sequence[base_sampler_lib.SamplingRequest]
      ),
  ) -> (
      base_sampler_lib.SamplingResponse
      | Sequence[base_sampler_lib.SamplingResponse]
  ):
    """Generates completions using underlying Tunix VllmSampler."""
    if not self.vllm_sampler:
      raise RuntimeError(
          f"InprocessVllmSampler [{self.server_id}] vllm_sampler is not"
          " initialized."
      )

    if isinstance(sampling_requests, base_sampler_lib.SamplingRequest):
      requests: list[base_sampler_lib.SamplingRequest] = [sampling_requests]
      is_sequence = False
    else:
      requests = list(sampling_requests)
      is_sequence = True

    prompts = []
    prompt_token_ids_batch = []
    has_token_prompts = False
    max_gen_steps_list = []
    routed_experts_prompt_start_list = []
    cache_salt_list = []

    for req in requests:
      prompt = req.prompt
      if generate_utils.is_token_id_sequence(prompt):
        has_token_prompts = True
        prompt_token_ids_batch.append(
            np.asarray(prompt, dtype=np.int32).reshape(-1)
        )
      else:
        prompts.append(self._prompt_to_input_string(prompt))

      max_gen_steps_list.append(
          req.max_tokens
          if req.max_tokens is not None
          else self.config.max_tokens_to_generate
      )
      routed_experts_prompt_start_list.append(req.routed_experts_prompt_start)
      cache_salt_list.append(req.cache_salt)

    max_generation_steps = (
        max(max_gen_steps_list)
        if max_gen_steps_list
        else self.config.max_tokens_to_generate
    )
    routed_experts_prompt_start = (
        routed_experts_prompt_start_list
        if routed_experts_prompt_start_list
        else 0
    )

    loop = asyncio.get_running_loop()
    # TODO(tunix-dev): RolloutConfig.seed is `int | jax.Array | None`, but
    # VllmSampler.__call__ takes `int | None`; `dict[str, Any]` hides the
    # mismatch.
    sampler_call_kwargs: dict[str, Any] = dict(
        max_generation_steps=max_generation_steps,
        temperature=self.config.temperature,
        top_p=self.config.top_p,
        top_k=self.config.top_k,
        seed=self.config.seed,
        return_logprobs=self.config.return_logprobs,
        routed_experts_prompt_start=routed_experts_prompt_start,
    )
    if any(s is not None for s in cache_salt_list):
      sampler_call_kwargs["cache_salt"] = (
          cache_salt_list[0] if len(cache_salt_list) == 1 else cache_salt_list
      )
    if has_token_prompts:
      sampler_call_kwargs["input_strings"] = None
      sampler_call_kwargs["prompt_token_ids"] = prompt_token_ids_batch
    else:
      sampler_call_kwargs["input_strings"] = prompts

    sampler_output = await loop.run_in_executor(
        self._executor,
        functools.partial(self.vllm_sampler, **sampler_call_kwargs),
    )

    prompt_lengths = sampler_output.prompt_lengths
    responses = []
    for i, req in enumerate(requests):
      txt = (
          sampler_output.text[i]
          if isinstance(sampler_output.text, list)
          else sampler_output.text
      )
      toks = (
          sampler_output.tokens[i]
          if isinstance(sampler_output.tokens, list)
          else sampler_output.tokens
      )
      lps = None
      if sampler_output.logprobs and isinstance(sampler_output.logprobs, list):
        lps = sampler_output.logprobs[i]

      routed_list = sampler_output.routed_experts or []
      routed = routed_list[i] if i < len(routed_list) else None

      tok_ids = (
          np.array(toks, dtype=np.int32)
          if toks is not None
          else np.zeros(0, dtype=np.int32)
      )
      prompt_len = (
          int(prompt_lengths[i])
          if prompt_lengths is not None and i < len(prompt_lengths)
          else None
      )
      prompt_token_ids = self._prompt_tokens_from_request(
          req, sampler_output.padded_prompt_tokens[i], prompt_length=prompt_len
      )
      log_ps = np.array(lps, dtype=np.float32) if lps is not None else None

      responses.append(
          base_sampler_lib.SamplingResponse(
              request_id=req.request_id,
              text=txt,
              prompt_token_ids=prompt_token_ids,
              token_ids=tok_ids,
              logprobs=log_ps,
              routed_experts=routed,
              finish_reason="stop",
          )
      )

    if is_sequence:
      return responses
    return responses[0]

  async def get_transfer_status(self, req_id: str | Any, **kwargs) -> str | Any:
    """Queries status of an ongoing weight transfer or KV-cache migration."""
    del req_id, kwargs
    return "SUCCESS"

  async def migrate_kv_cache(
      self,
      source_server_id: str,
      target_server_id: str,
      token_ids: list[int],
      **kwargs,
  ) -> bool:
    """Triggers KV-cache transfer across TPU slices."""
    del source_server_id, target_server_id, token_ids, kwargs
    return True

  async def get_load_info(self, **kwargs) -> base_sampler_lib.LoadInfo:
    """Returns best-effort vLLM queue/cache load information."""
    del kwargs
    return base_sampler_lib.LoadInfo()
