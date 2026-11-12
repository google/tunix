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
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project

"""Asynchronous vLLM Sampler engine implementation for tpu-inference.

This module provides a standalone vLLM serving and rollout sampler
(`VllmSampler`) tailored for Reinforcement Learning (RL) workloads on TPUs.
Satisfies the open-source Tunix `Sampler` and `WeightSyncDestination` Protocols.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
import logging
import os
import time
from typing import Any, TYPE_CHECKING

import numpy as np
from tunix.common import configs
from tunix.dist_rl.common import datatypes
from tunix.dist_rl.rollout import sampler as base_sampler_lib
from tunix.dist_rl.weight_sync import raiden_weight_sync_delegate
from tunix.dist_rl.weight_sync import weight_sync
from tunix.dist_rl.weight_sync import weight_sync_coordinator
from tunix.generate import utils as generate_utils

# TODO(yatlas): Clean up lazy vLLM imports so TYPE_CHECKING is not needed.
if TYPE_CHECKING:
  from vllm.engine.arg_utils import AsyncEngineArgs  # pylint: disable=g-import-not-at-top
  from vllm.engine.async_llm_engine import AsyncLLMEngine  # pylint: disable=g-import-not-at-top
  from vllm.sampling_params import SamplingParams as VllmSamplingParams  # pylint: disable=g-import-not-at-top

Sampler = base_sampler_lib.Sampler
logger = logging.getLogger(__name__)


def _get_val(obj: Any, key: str, default: Any = None) -> Any:
  """Helper to extract an attribute or dict key seamlessly from any duck-typed request.

  Falls back to `default` both when `key` is absent and when it's present
  but explicitly `None` -- request dataclasses across callers (e.g. tunix's
  rollout requests) commonly default optional sampling fields to `None`
  rather than omitting them, and vLLM's `SamplingParams` rejects `None` for
  fields like `top_k` that it compares numerically.
  """
  if obj is None:
    return default
  if isinstance(obj, dict):
    val = obj.get(key, default)
  else:
    val = getattr(obj, key, default)
  return default if val is None else val


def _sync_request_id(
    sync_request: base_sampler_lib.WeightSyncRequest | None,
) -> Any | None:
  """Extracts the transfer request ID from `sync_request`, if present."""
  if sync_request is None:
    return None
  if sync_request.extra_config:
    rid = sync_request.extra_config.get("req_id")
    if rid:
      return rid
  return sync_request.request_id


async def _flush_engine_stats_periodically(
    engine: AsyncLLMEngine, interval_s: float
) -> None:
  """Logs vLLM engine stats every `interval_s` seconds until cancelled.

  `AsyncLLM` records engine stats (throughput, running/waiting requests, KV
  cache usage, prefix cache hit rate) but never logs them by itself:
  `do_log_stats()` has no internal caller and is normally driven by
  `lifespan()` in vllm/entrypoints/serve/utils/server_utils.py. `VllmSampler`
  binds the engine directly without that server, so it drives the flush
  itself. Unlike the server, this runs regardless of `disable_log_stats`: with
  stats disabled (and no stat-logger plugins installed) the engine has no stat
  loggers and `do_log_stats()` is a no-op.

  Args:
    engine: The running vLLM engine.
    interval_s: Seconds between flushes (`VLLM_LOG_STATS_INTERVAL`).
  """
  while True:
    await asyncio.sleep(interval_s)
    try:
      await engine.do_log_stats()
    except Exception:  # pylint: disable=broad-exception-caught
      # Stats are observability-only: log the failure and keep flushing.
      logger.exception("Failed to flush vLLM engine stats.")


class VllmSampler(Sampler, weight_sync.WeightSyncDestination):
  """Rollout sampler backed by vLLM's asynchronous engine (`AsyncLLMEngine`).

  Use this sampler to keep the engine in its own process, or with a trainer
  that emits weights in the rollout model's layout, such as MaxText. Prefer
  `InprocessVllmSampler` for RL training with the Tunix trainer.

  Raiden is the only way to update this sampler's weights. There is no
  `update_params` fallback, and `get_target_state()` returns None, so the
  trainer gets no destination weight tree to convert to.

  `get_load_info()` reports real engine load, and `get_transfer_status()`
  tracks each weight transfer.
  """

  def __init__(
      self,
      server_id: str,
      config: configs.RolloutConfig,
      worker_index: int = 0,
      parallelism: int = 4,
  ):
    """Initializes the sampler and builds its vLLM engine arguments.

    Args:
      server_id: Unique identifier for this rollout replica.
      config: Rollout configuration. The engine arguments come from its
        `rollout_vllm_model_version`, `tensor_parallel_size`,
        `data_parallel_size`, `kv_cache_size` (as `max_model_len`),
        `rollout_vllm_hbm_utilization` (as `gpu_memory_utilization`),
        `rollout_vllm_lora_config`, `rollout_vllm_hf_config_path`,
        `rollout_vllm_async_scheduling`, `rollout_vllm_max_num_seqs`,
        `rollout_vllm_max_num_batched_tokens`, `rollout_vllm_logprobs_mode`,
        `return_routed_experts` and `rollout_vllm_additional_config` fields,
        then `rollout_vllm_kwargs`. `sample()` reads the sampling settings from
        it and `rollout_vllm_sampling_kwargs`. Its other `rollout_vllm_*` fields
        are not used.
      worker_index: Host index within this replica.
      parallelism: Number of hosts in this replica.
    """
    self.config: configs.RolloutConfig = config
    self.server_id = server_id
    if not config.rollout_vllm_server_mode:
      logger.warning(
          "VllmSampler [%s] always runs in server mode; ignoring"
          " config.rollout_vllm_server_mode=False.",
          self.server_id,
      )
    lora_config = config.rollout_vllm_lora_config
    # TODO(tunix-dev): Share this engine-kwargs construction with
    # `InprocessVllmSampler.initialize` in one helper so the two samplers can't
    # drift.
    engine_kwargs: dict[str, Any] = {
        "model": config.rollout_vllm_model_version,
        "tensor_parallel_size": config.tensor_parallel_size,
        "data_parallel_size": config.data_parallel_size,
        "max_model_len": config.kv_cache_size,
        "async_scheduling": config.rollout_vllm_async_scheduling,
        "max_num_seqs": config.rollout_vllm_max_num_seqs,
        "max_num_batched_tokens": config.rollout_vllm_max_num_batched_tokens,
        # Only the logprob of each sampled token is needed.
        "max_logprobs": 1,
        "logprobs_mode": config.rollout_vllm_logprobs_mode,
        "enable_return_routed_experts": config.return_routed_experts,
        "trust_remote_code": True,
        "enable_lora": lora_config is not None,
    }
    if lora_config is not None:
      max_lora_rank = lora_config.get("max_lora_rank")
      if max_lora_rank is not None:
        engine_kwargs["max_lora_rank"] = max_lora_rank
      max_loras = lora_config.get("max_loras", 1)
      if max_loras is not None:
        engine_kwargs["max_loras"] = max_loras
    # TODO(yatlas): Remove default hbm_utilization once colocated hbm fixes are
    # landed.
    if config.rollout_vllm_hbm_utilization is not None:
      engine_kwargs["gpu_memory_utilization"] = (
          config.rollout_vllm_hbm_utilization
      )
    if config.rollout_vllm_hf_config_path is not None:
      engine_kwargs["hf_config_path"] = config.rollout_vllm_hf_config_path
    extra_kwargs = dict(config.rollout_vllm_kwargs)
    additional_config = generate_utils.merge_additional_config(
        config.rollout_vllm_additional_config,
        extra_kwargs.pop("additional_config", None),
    )
    if additional_config is not None:
      engine_kwargs["additional_config"] = additional_config
    # Applied last, so these override the values above.
    engine_kwargs.update(extra_kwargs)
    tp_size = engine_kwargs["tensor_parallel_size"]
    dp_size = engine_kwargs["data_parallel_size"]
    if tp_size < 1 or dp_size < 1:
      raise ValueError(
          f"VllmSampler [{server_id}] requires positive"
          " config.tensor_parallel_size and config.data_parallel_size, got"
          f" {tp_size} and {dp_size}."
      )
    if (lora_config is not None or engine_kwargs.get("enable_lora")) and (
        dp_size > 1
    ):
      raise ValueError(
          "LoRA is not supported with rollout data parallelism:"
          f" config.data_parallel_size={dp_size} with"
          " config.rollout_vllm_lora_config. Set data_parallel_size=1 or drop"
          " LoRA."
      )
    self._engine_kwargs: dict[str, Any] = engine_kwargs
    self.engine_args: AsyncEngineArgs | None = None
    self.worker_index = worker_index
    # Raiden treats units sharing a job_name as hosts of ONE job and splits the
    # weights across them (`num_dst_physical_hosts` in raiden_controller), so
    # independent rollout replicas each need their own job_name to be sent a
    # full copy. server_id already has that granularity; worker_index stays the
    # host index *within* one replica.
    self.raiden_job_name = f"replica_{self.server_id}"
    self._parallelism = parallelism

    self.weight_sync_mode = weight_sync.WeightSyncMode(
        self.config.weight_sync_mode
    )
    self.enable_raiden = (
        self.weight_sync_mode == weight_sync.WeightSyncMode.RAIDEN
    )
    if not self.enable_raiden:
      logger.info(
          "VllmSampler [%s] weight_sync_mode=%s; Raiden weight sync is"
          " disabled.",
          self.server_id,
          self.weight_sync_mode,
      )

    self._engine: AsyncLLMEngine | None = None
    self._is_running: bool = False
    self._is_paused: bool = False
    self._paused_generation: bool = False
    self._cache_valid: bool = True
    self._mesh: Any | None = None
    self._transfer_statuses: dict[str, str] = {}
    self._tracker = weight_sync_coordinator.WorkerRoundTracker()
    self._sync_lock = asyncio.Lock()
    self._policy_version: int = 0
    self._kv_cache_freed: bool = False
    self._log_stats_task: asyncio.Task[None] | None = None
    self._routed_experts_shape: tuple[int, int] | None = None

  def initialize(self) -> None:
    """Builds the vLLM `AsyncEngineArgs` from the rollout configuration."""
    if self.engine_args is not None:
      return
    from vllm.engine.arg_utils import AsyncEngineArgs  # pylint: disable=g-import-not-at-top

    self.engine_args = AsyncEngineArgs(**self._engine_kwargs)

  # ---------------------------------------------------------------------------
  # Lifecycle & Inference Methods
  # ---------------------------------------------------------------------------

  async def start(self, **kwargs: Any) -> None:
    """Initializes the vLLM engine and execution environment."""
    del kwargs
    if self._is_running:
      logger.warning("VllmSampler is already running.")
      return
    self.initialize()
    assert self.engine_args is not None

    logger.info(
        "Initializing VllmSampler with model: %s", self.engine_args.model
    )
    from vllm import envs as vllm_envs  # pylint: disable=g-import-not-at-top
    from vllm.engine.async_llm_engine import AsyncLLMEngine  # pylint: disable=g-import-not-at-top

    log_stats_interval_s = float(vllm_envs.VLLM_LOG_STATS_INTERVAL)
    engine = AsyncLLMEngine.from_engine_args(self.engine_args)
    self._engine = engine
    self._is_running = True
    self._log_stats_task = asyncio.create_task(
        _flush_engine_stats_periodically(engine, log_stats_interval_s)
    )
    logger.info("VllmSampler started successfully.")

  async def _ensure_started(self) -> None:
    """Brings the engine up if nothing has needed it yet.

    VllmSampler builds its AsyncLLM lazily and `sample()` is the only caller
    of `start()`. Weight sync needs the engine too -- it owns the TPU worker,
    and therefore the Raiden binding -- and the first sync lands before the
    first sample, because `prepare_rollout_policy` syncs ahead of dispatch.
    Without this the round finds no worker, reports an empty destination
    manifest, and deadlocks: the engine waits for a sample that dispatch is
    waiting on the sync to allow. Guarded on `_is_running` rather than calling
    `start()` unconditionally, because `start()` warns when the engine is
    already up and this runs on every sync round.

    TODO(tunix-dev): drop this once the orchestrator owns rollout-worker
    lifecycle and can guarantee the engine is up before it issues any phase
    call; the ordering belongs there, not in a guard on each entry point.
    """
    if not self._is_running:
      logger.info(
          "VllmSampler [%s] starting engine for weight sync (no sample has"
          " forced it up yet).",
          self.server_id,
      )
      await self.start()

  async def stop(self, **kwargs: Any) -> None:
    """Stops the sampler and releases resources."""
    del kwargs
    if not self._is_running:
      return
    logger.info("Stopping VllmSampler...")
    self._is_paused = True
    self._paused_generation = False
    if self._log_stats_task is not None:
      self._log_stats_task.cancel()
      await asyncio.wait([self._log_stats_task])
      self._log_stats_task = None
    self._engine = None
    self._is_running = False
    logger.info("VllmSampler stopped.")

  def _check_partial_rollout_kwarg(self, kwargs: Mapping[str, Any]) -> None:
    """Rejects a per-call `partial_rollout` kwarg that disagrees with config."""
    # TODO(yatlas): Stop passing `partial_rollout` from
    # `RolloutManager.pre_weight_sync` and remove `**kwargs` on `pause` /
    # `pre_weight_sync`.
    partial_rollout = kwargs.get("partial_rollout")
    if (
        partial_rollout is not None
        and bool(partial_rollout) != self.config.partial_rollout
    ):
      raise ValueError(
          f"VllmSampler [{self.server_id}] received"
          f" partial_rollout={partial_rollout!r}, which disagrees with"
          f" config.partial_rollout={self.config.partial_rollout}."
      )

  async def pause(self, **kwargs: Any) -> None:
    """Pauses request intake and drains or freezes active batch iterations during weight updates."""
    self._check_partial_rollout_kwarg(kwargs)
    if self._is_paused:
      return
    self._is_paused = True
    if self.config.partial_rollout and self._engine is not None:
      logger.info(
          "Pausing VllmSampler generation in-place (mode='keep',"
          " clear_cache=False)..."
      )
      await self._engine.pause_generation(mode="keep", clear_cache=False)
      self._paused_generation = True
    else:
      # TODO(tunix-dev): Without partial_rollout this neither drains nor
      # freezes the engine; RolloutManager drains in-flight trajectories before
      # pre_weight_sync. Drain here or correct the docstring.
      self._paused_generation = False
    await asyncio.sleep(0.01)

  async def resume(self, **kwargs: Any) -> None:
    """Resumes inference processing after weight sync completion."""
    del kwargs
    if not self._is_paused:
      return
    if self._paused_generation and self._engine is not None:
      logger.info("Resuming VllmSampler generation...")
      await self._engine.resume_generation()
      self._paused_generation = False
    self._is_paused = False

  async def get_mesh(self, **kwargs: Any) -> Any | None:
    """Returns the JAX device mesh."""
    del kwargs
    return self._mesh

  def routed_experts_shape(self) -> tuple[int, int] | None:
    """Returns `(num_layers, top_k)` when MoE routing capture is enabled, else None."""
    if not self.config.return_routed_experts:
      return None
    if self._routed_experts_shape is not None:
      return self._routed_experts_shape
    if self._engine is not None:
      model_cfg = self._engine.vllm_config.model_config
    elif self.engine_args is not None:
      model_cfg = self.engine_args.create_model_config()
    else:
      return None
    num_layers = int(model_cfg.get_total_num_hidden_layers())
    top_k = int(model_cfg.get_num_experts_per_tok())
    if num_layers <= 0 or top_k <= 0:
      return None
    self._routed_experts_shape = (num_layers, top_k)
    return self._routed_experts_shape

  # ---------------------------------------------------------------------------
  # Helper Methods for Sample Stream Processing
  # ---------------------------------------------------------------------------

  def _build_vllm_params(
      self,
      req: Any,
      kwargs: dict[str, Any],
  ) -> VllmSamplingParams:
    """Builds a vLLM SamplingParams object from a SamplingRequest or duck-typed request."""
    from vllm.sampling_params import SamplingParams as vllm_sampling_params_cls  # pylint: disable=g-import-not-at-top

    sparams = _get_val(req, "sampling_params")
    stop_token_ids = _get_val(
        sparams,
        "stop_token_ids",
        kwargs.get("stop_token_ids")
        or kwargs.get("eos_tokens")
        or self.config.eos_tokens,
    )
    cache_salt = _get_val(sparams, "cache_salt", kwargs.get("cache_salt"))
    routed_experts_prompt_start = _get_val(
        sparams,
        "routed_experts_prompt_start",
        _get_val(kwargs, "routed_experts_prompt_start"),
    )
    # TODO(tunix-dev): Read sampling fields directly from the typed request and
    # RolloutConfig instead of chained `_get_val` fallbacks (one well-lit path).
    # `kwargs` goes through `_get_val` too: callers pass unset fields as
    # explicit `None`, so `.get(key, default)` returns `None` rather than
    # the default, defeating it before `_get_val` can coalesce.
    sampling_defaults = self.config.rollout_vllm_sampling_kwargs
    vllm_kwargs: dict[str, Any] = dict(sampling_defaults)
    vllm_kwargs.update(
        temperature=_get_val(
            sparams,
            "temperature",
            _get_val(
                kwargs,
                "temperature",
                sampling_defaults.get("temperature", self.config.temperature),
            ),
        ),
        top_p=_get_val(
            sparams,
            "top_p",
            _get_val(
                kwargs,
                "top_p",
                sampling_defaults.get(
                    "top_p",
                    self.config.top_p
                    if self.config.top_p is not None
                    else 0.95,
                ),
            ),
        ),
        top_k=_get_val(
            sparams,
            "top_k",
            _get_val(
                kwargs,
                "top_k",
                sampling_defaults.get(
                    "top_k",
                    self.config.top_k if self.config.top_k is not None else -1,
                ),
            ),
        ),
        max_tokens=_get_val(
            sparams,
            "max_tokens",
            _get_val(
                kwargs,
                "max_tokens",
                sampling_defaults.get(
                    "max_tokens", self.config.max_tokens_to_generate
                ),
            ),
        ),
        stop=_get_val(sparams, "stop_sequences")
        or _get_val(sparams, "stop")
        or kwargs.get("stop")
        or sampling_defaults.get("stop"),
        stop_token_ids=(
            list(stop_token_ids)
            if stop_token_ids is not None
            else sampling_defaults.get("stop_token_ids")
        ),
        include_stop_str_in_output=bool(
            _get_val(
                sparams,
                "include_stop_str_in_output",
                _get_val(
                    kwargs,
                    "include_stop_str_in_output",
                    sampling_defaults.get("include_stop_str_in_output", True),
                ),
            )
        ),
        # Detokenize by default so callers that read `SamplingResponse.text`
        # (e.g. agentic rollouts) receive populated text, matching vLLM's
        # default. Token-only callers can opt out via
        # `config.rollout_vllm_sampling_kwargs` or per-request
        # `detokenize=False`.
        # TODO(yatlas): Drop per-request `detokenize` overrides.
        detokenize=bool(
            _get_val(
                sparams,
                "detokenize",
                _get_val(
                    kwargs,
                    "detokenize",
                    sampling_defaults.get("detokenize", True),
                ),
            )
        ),
        logprobs=1
        if _get_val(
            sparams,
            "return_logprobs",
            kwargs.get("return_logprobs", self.config.return_logprobs),
        )
        else None,
    )
    struct_fields = getattr(vllm_sampling_params_cls, "__struct_fields__", ())
    if cache_salt is not None and "cache_salt" in struct_fields:
      vllm_kwargs["cache_salt"] = str(cache_salt)
    if (
        routed_experts_prompt_start is not None
        and "routed_experts_prompt_start" in struct_fields
    ):
      vllm_kwargs["routed_experts_prompt_start"] = int(
          routed_experts_prompt_start
      )
    params = vllm_sampling_params_cls(**vllm_kwargs)
    if cache_salt is not None:
      setattr(params, "cache_salt", str(cache_salt))
    if routed_experts_prompt_start is not None:
      setattr(
          params,
          "routed_experts_prompt_start",
          int(routed_experts_prompt_start),
      )
    return params

  async def _process_request_output(
      self,
      req_id: str,
      task: Any,
      expected_prompt_token_ids: list[int] | None = None,
  ) -> base_sampler_lib.SamplingResponse:
    """Consumes an AsyncLLMEngine output stream and formats a SamplingResponse."""
    try:
      final_output = None
      async for step_output in task:
        final_output = step_output

      if not (final_output and final_output.outputs):
        err_obj = datatypes.ErrorInfo(
            error_type="EmptyOutput",
            message="No output generated by vLLM",
            retryable=False,
        )
        return base_sampler_lib.SamplingResponse(
            request_id=req_id,
            text="",
            token_ids=np.zeros(0, dtype=np.int32),
            prompt_token_ids=np.zeros(0, dtype=np.int32),
            logprobs=None,
            routed_experts=None,
            finish_reason="stop",
            error=err_obj,
        )

      output_choice = final_output.outputs[0]
      text = output_choice.text
      token_ids_arr = np.array(output_choice.token_ids, dtype=np.int32)
      # On the RequestOutput, not the CompletionOutput. Tunix feeds
      # this into np.asarray(..., dtype=np.int32), so omitting it
      # surfaces as "int() argument must be ... not 'NoneType'".
      raw_prompt_token_ids = getattr(final_output, "prompt_token_ids", None)
      if expected_prompt_token_ids is not None:
        expected_arr = np.asarray(expected_prompt_token_ids, dtype=np.int32)
        if raw_prompt_token_ids is not None:
          actual_arr = np.asarray(raw_prompt_token_ids, dtype=np.int32)
          if not np.array_equal(actual_arr, expected_arr):
            raise ValueError(
                f"vLLM prompt token echo mismatch for req_id={req_id}: "
                f"expected {expected_arr.tolist()}, got {actual_arr.tolist()}"
            )
        prompt_token_ids_arr = expected_arr
      else:
        prompt_token_ids_arr = np.array(
            raw_prompt_token_ids or [], dtype=np.int32
        )

      logprobs_arr = None
      if getattr(output_choice, "logprobs", None):
        logprob_vals = [
            lp_dict[next(iter(lp_dict.keys()))].logprob if lp_dict else 0.0
            for lp_dict in output_choice.logprobs
        ]
        logprobs_arr = np.array(logprob_vals, dtype=np.float32)

      routed_experts = getattr(output_choice, "routed_experts", None)
      if routed_experts is not None and not isinstance(
          routed_experts, np.ndarray
      ):
        routed_experts = np.array(routed_experts)

      finish_reason = getattr(output_choice, "finish_reason", "stop") or "stop"
    except Exception as e:  # pylint: disable=broad-exception-caught
      logger.exception("Error generating sampling result for req_id=%s", req_id)
      err_obj = datatypes.ErrorInfo(
          error_type=type(e).__name__,
          message=str(e),
          retryable=True,
      )
      return base_sampler_lib.SamplingResponse(
          request_id=req_id,
          text="",
          token_ids=np.zeros(0, dtype=np.int32),
          prompt_token_ids=np.zeros(0, dtype=np.int32),
          logprobs=None,
          routed_experts=None,
          finish_reason="stop",
          error=err_obj,
      )

    # Built outside the `try` so that a malformed response (for example a
    # logprobs/token_ids shape mismatch rejected by
    # `SamplingResponse.__post_init__`) raises instead of being converted into
    # a silent empty completion.
    return base_sampler_lib.SamplingResponse(
        request_id=req_id,
        text=text,
        token_ids=token_ids_arr,
        prompt_token_ids=prompt_token_ids_arr,
        logprobs=logprobs_arr,
        routed_experts=routed_experts,
        finish_reason=finish_reason,
        error=None,
    )

  # ---------------------------------------------------------------------------
  # Dynamic Batch Sampling
  # ---------------------------------------------------------------------------

  async def sample(
      self,
      sampling_requests: (
          base_sampler_lib.SamplingRequest
          | Sequence[base_sampler_lib.SamplingRequest]
          | Any
      ),
      **kwargs: Any,
  ) -> (
      base_sampler_lib.SamplingResponse
      | list[base_sampler_lib.SamplingResponse]
      | Any
  ):
    """Generates completions for a batch of sampling requests or raw prompts.

    Accepts raw string lists, dictionaries, or duck-typed objects from callers
    (such as `tunix.dist_rl.common.datatypes.SamplingRequest`).
    """
    if sampling_requests is None:
      raise ValueError("sampling_requests cannot be None.")

    while self._is_paused:
      await asyncio.sleep(0.05)

    if not self._is_running or self._engine is None:
      await self.start()
    assert self._engine is not None

    raw_input_mode = False
    is_sequence = isinstance(sampling_requests, (list, tuple))
    if isinstance(sampling_requests, (str, list)) and (
        isinstance(sampling_requests, str)
        or not sampling_requests
        or isinstance(sampling_requests[0], str)
        or (
            isinstance(sampling_requests[0], list)
            and not _get_val(sampling_requests[0], "prompt")
        )
    ):
      raw_input_mode = True
      items = (
          sampling_requests
          if isinstance(sampling_requests, list)
          else [sampling_requests]
      )
      req_list = [
          {
              "prompt": item,
              "request_id": f"req_{idx}_{time.time_ns()}",
          }
          for idx, item in enumerate(items)
      ]
    elif (
        isinstance(sampling_requests, base_sampler_lib.SamplingRequest)
        or _get_val(sampling_requests, "prompt") is not None
    ):
      req_list = [sampling_requests]
    else:
      req_list = list(sampling_requests)

    from vllm.inputs import TextPrompt, TokensPrompt  # pylint: disable=g-import-not-at-top,g-multiple-import

    pending_tasks = []
    for idx, req in enumerate(req_list):
      vllm_params = self._build_vllm_params(req, kwargs)
      req_id = _get_val(req, "request_id") or f"req_{time.time_ns()}_{idx}"
      prompt_val = _get_val(req, "prompt")
      param_cache_salt = getattr(vllm_params, "cache_salt", None)
      engine_prompt: TextPrompt | TokensPrompt | str
      expected_prompt_ids: list[int] | None
      if generate_utils.is_token_id_sequence(prompt_val):
        token_ids: list[int] = [
            int(x)
            for x in np.asarray(prompt_val, dtype=np.int32).reshape(-1).tolist()
        ]
        expected_prompt_ids = token_ids
        if param_cache_salt is not None:
          engine_prompt = {
              "prompt_token_ids": token_ids,
              "cache_salt": str(param_cache_salt),
          }
        else:
          engine_prompt = {"prompt_token_ids": token_ids}
      else:
        expected_prompt_ids = None
        prompt_str = (
            prompt_val if isinstance(prompt_val, str) else str(prompt_val)
        )
        if param_cache_salt is not None:
          engine_prompt = {
              "prompt": prompt_str,
              "cache_salt": str(param_cache_salt),
          }
        else:
          engine_prompt = prompt_str

      task_gen = self._engine.generate(
          engine_prompt, vllm_params, request_id=req_id
      )
      pending_tasks.append((req_id, task_gen, expected_prompt_ids))

    results = list(
        await asyncio.gather(*[
            self._process_request_output(
                req_id, task_gen, expected_prompt_token_ids=expected_ids
            )
            for req_id, task_gen, expected_ids in pending_tasks
        ])
    )

    if raw_input_mode:
      return [r.text for r in results]
    if is_sequence:
      return results
    return results[0] if results else base_sampler_lib.SamplingResponse()

  # ---------------------------------------------------------------------------
  # Cache Management
  # ---------------------------------------------------------------------------

  async def _clear_prefix_cache(self) -> None:
    """Purges prefix caching blocks from HBM."""
    logger.info("Clearing vLLM RadixAttention prefix cache...")
    self._cache_valid = False
    if self._engine is not None:
      await self._engine.reset_prefix_cache()

  # ---------------------------------------------------------------------------
  # Weight Synchronization (Integrated with TPUWorker)
  # ---------------------------------------------------------------------------

  async def _call_worker_method(
      self, method_name: str, *args: Any, **kwargs: Any
  ) -> list[Any]:
    """Dispatches a method call across TPU workers via collective_rpc.

    `AsyncLLMEngine` (an alias of `vllm.v1.engine.async_llm.AsyncLLM`)
    always exposes an async `collective_rpc`.
    """
    if self._engine is None:
      return []
    return await self._engine.collective_rpc(
        method_name, args=args, kwargs=kwargs
    )

  async def get_weights_state(self) -> list[Any]:
    """Returns the PyTree of weights or state from active TPU workers."""
    return await self._call_worker_method("get_weights_state")

  async def bind_raiden_sync(
      self,
      worker_index: int = 0,
      parallelism: int = 4,
      job_name: str = "rollout",
  ) -> None:
    """Binds Raiden to each TPU worker's live weights, in-process.

    `get_weights_state()` cannot back a rollout-side weight sync: the
    live `nnx.State` it returns has to be dispatched back across
    `collective_rpc` into this (parent) process to be useful, but vLLM's
    RPC transport can't serialize `nnx.State`/live TPU arrays, and even a
    pickle-based fallback would bind Raiden to a disconnected host copy
    rather than the device buffers actually being served. Binding must
    happen in the worker subprocess instead -- see
    `tpu_worker.TPUWorker.bind_raiden_sync`.
    """
    await self._call_worker_method(
        "bind_raiden_sync", worker_index, parallelism, job_name
    )

  async def refresh_model_state_leaves(self) -> None:
    """Re-points each worker's dispatch view after an h2d weight update."""
    await self._call_worker_method("refresh_model_state_leaves")

  async def get_raiden_metadata(self) -> list[dict[str, Any]]:
    """Wire-safe registration metadata for each worker's current Raiden binding."""
    return await self._call_worker_method("get_raiden_metadata")

  async def raiden_h2d(self, uuid: int | None = None) -> list[dict[str, Any]]:
    """Blocks each worker until its just-landed transfer is visible on-device.

    Args:
        uuid: Identifies the transfer generation to wait for.

    Returns each worker's checksums dict (empty unless VERIFY_WEIGHTS=true).
    """
    return await self._call_worker_method("raiden_h2d", uuid=uuid)

  async def raiden_metrics(self) -> list[dict[str, Any]]:
    """Returns Raiden transfer metrics from each worker."""
    return await self._call_worker_method("raiden_metrics")

  # ---------------------------------------------------------------------------
  # WeightSyncDestination Protocol Implementation
  # ---------------------------------------------------------------------------

  async def bind_weight_sync(
      self,
      sync_request: base_sampler_lib.WeightSyncRequest | Any = None,
      **kwargs: Any,
  ) -> Any:
    """Idempotent transport binding called while the worker is STILL SERVING."""
    del sync_request, kwargs
    if not self.enable_raiden:
      return None
    await self._ensure_started()
    return await self.bind_raiden_sync(
        worker_index=self.worker_index,
        parallelism=self._parallelism,
        job_name=self.raiden_job_name,
    )

  async def get_weight_sync_metadata(
      self,
      **kwargs: Any,
  ) -> Sequence[weight_sync.WorkUnitMetadata] | Any:
    """Returns transport metadata with Raiden endpoints and TensorMetadata."""
    del kwargs
    if not self.enable_raiden:
      raise NotImplementedError(
          f"VllmSampler [{self.server_id}] does not support"
          " get_weight_sync_metadata when Raiden is disabled"
          f" (weight_sync_mode={self.config.weight_sync_mode})."
      )
    await self._ensure_started()
    meta = await self.get_raiden_metadata()
    return [
        weight_sync.WorkUnitMetadata.from_dict(
            raiden_weight_sync_delegate.canonicalize_variable_names(m)
        )
        for m in meta or []
    ]

  async def pre_weight_sync(
      self,
      sync_request: base_sampler_lib.WeightSyncRequest | None = None,
      free_kv_cache: bool = False,
      **kwargs: Any,
  ) -> Any:
    """Phase 1: Pauses intake, invalidates prefix cache, and optionally drops KV cache.

    `free_kv_cache` defaults to False because Raiden weight sync streams updates
    directly into the existing live TPU weight buffers in-place without
    materializing a second copy of the weights in HBM. Invalidating the
    scheduler's prefix cache (`_clear_prefix_cache`) is sufficient to prevent
    serving stale KV activations computed under old weights while avoiding the
    per-step latency overhead of deallocating (`delete_kv_cache`) and
    reallocating (`reinitialize_kv_cache`) the HBM KV buffer pool.
    """
    self._check_partial_rollout_kwarg(kwargs)
    if not self.enable_raiden:
      return True
    async with self._sync_lock:
      if not self._tracker.admit(sync_request, "prepared"):
        return True

      use_partial = self.config.partial_rollout
      use_free_kv = bool(free_kv_cache)
      logger.info(
          "Executing pre_weight_sync for server_id=%s (free_kv_cache=%s,"
          " partial_rollout=%s)",
          self.server_id,
          use_free_kv,
          use_partial,
      )

      rid = _sync_request_id(sync_request)
      if rid:
        self._transfer_statuses[str(rid)] = "IN_PROGRESS"

      await self.pause()
      if not (use_partial and not use_free_kv):
        await self._clear_prefix_cache()

      # `AsyncLLMEngine.start_weight_update()` takes no arguments, so it
      # can't forward `free_kv_cache` to the worker -- go through
      # collective_rpc directly instead of the engine-level wrapper.
      await self._call_worker_method(
          "start_weight_update", free_kv_cache=use_free_kv
      )
      self._kv_cache_freed = True

      self._tracker.complete(sync_request, "prepared")
      return True

  async def weight_sync(
      self,
      sync_request: base_sampler_lib.WeightSyncRequest | None = None,
      **kwargs: Any,
  ) -> Any:
    """Phase 2: Flushes/awaits H2D transfers across TPU workers and refreshes state_leaves."""
    del kwargs
    if not self.enable_raiden:
      # VllmSampler owns its weight buffers, so there is no host-side
      # update_params fallback like InprocessVllmSampler's.
      raise RuntimeError(
          f"VllmSampler [{self.server_id}] supports Raiden weight sync only;"
          " no fallback path exists"
          f" (weight_sync_mode={self.config.weight_sync_mode})."
      )
    async with self._sync_lock:
      if not self._tracker.admit(sync_request, "h2d_done"):
        return True

      logger.info("Executing weight_sync barrier on Raiden synchronizers...")
      checksums = await self.raiden_h2d(
          uuid=raiden_weight_sync_delegate.round_uuid(sync_request)
      )
      if checksums:
        logger.info("Destination weights checksums: %s", checksums)

      # Never silently skip this: the sampler's runner dispatches through a
      # `state_leaves` view derived from `state` at load time, so without the
      # refresh it keeps serving pre-sync weights and the only symptom is
      # garbage completions.
      await self.refresh_model_state_leaves()

      self._tracker.complete(sync_request, "h2d_done")
      return True

  async def post_weight_sync(
      self,
      sync_request: base_sampler_lib.WeightSyncRequest | None = None,
      **kwargs: Any,
  ) -> Any:
    """Phase 3: Closes weight update session (reinitializes KV cache if freed), restores request intake, and bumps active policy version."""
    del kwargs
    if not self.enable_raiden:
      return True
    async with self._sync_lock:
      if not self._tracker.admit(sync_request, "committed"):
        return True

      rid = _sync_request_id(sync_request)
      logger.info("Executing post_weight_sync (req_id=%s)...", rid)
      if self._kv_cache_freed:
        await self._call_worker_method("finish_weight_update")
        self._kv_cache_freed = False
      self._cache_valid = True
      if rid:
        self._transfer_statuses[str(rid)] = "SUCCESS"
      await self.resume()

      if sync_request is not None and sync_request.policy_version is not None:
        self._policy_version = sync_request.policy_version
      else:
        self._policy_version += 1

      if os.environ.get("VERIFY_WEIGHTS", "").lower() == "true":
        logger.info("Raiden transfer metrics: %s", await self.raiden_metrics())

      self._tracker.complete(sync_request, "committed")
      return self._policy_version

  async def abort_weight_sync(
      self,
      sync_request: base_sampler_lib.WeightSyncRequest | None = None,
      **kwargs: Any,
  ) -> Any:
    """Safely rolls back to serving previous policy version without publishing staging."""
    del kwargs
    if not self.enable_raiden:
      return True
    async with self._sync_lock:
      if not self._tracker.admit(sync_request, "aborted"):
        return False

      logger.warning(
          "Aborting weight sync round: rolling back to policy_version=%d",
          self._policy_version,
      )
      # VllmSampler has no dedicated abort path; finish_weight_update + resume
      # does the same recovery (reinitialize KV cache + resume).
      if self._kv_cache_freed:
        await self._call_worker_method("finish_weight_update")
        self._kv_cache_freed = False
      await self.resume()

      self._tracker.complete(sync_request, "aborted")
      return True

  async def get_weight_sync_status(self) -> Mapping[str, Any]:
    """Reports worker-side round status for coordinator recovery checks."""
    return self._tracker.report()

  async def get_transfer_status(self, req_id: Any, **kwargs: Any) -> str:
    """Queries status of an ongoing weight transfer or KV-cache migration."""
    del kwargs
    return self._transfer_statuses.get(str(req_id), "MISSING")

  def get_target_state(self) -> Any:
    """Returns target state shape/dtype pytree for weight conversion."""
    return None

  async def get_load_info(self, **kwargs: Any) -> base_sampler_lib.LoadInfo:
    """Returns current load info (num_requests_waiting, num_requests_running, kv_cache_usage_perc)."""
    del kwargs
    num_waiting = 0
    num_running = 0
    kv_cache_usage_perc = 0.0

    if self._engine:
      engine_obj = getattr(self._engine, "engine", self._engine)

      # If running under DPScheduler, get aggregated stats across DP ranks
      if hasattr(engine_obj, "make_stats"):
        stats = engine_obj.make_stats()
        if stats:
          num_waiting = getattr(stats, "num_waiting_reqs", 0)
          num_running = getattr(stats, "num_running_reqs", 0)
          kv_usage = getattr(stats, "kv_cache_usage", 0.0)
          kv_cache_usage_perc = (
              (kv_usage * 100.0) if kv_usage <= 1.0 else float(kv_usage)
          )
      # Otherwise, read directly from standard vLLM scheduler(s)
      elif getattr(engine_obj, "scheduler", None):
        schedulers = engine_obj.scheduler
        sched_list = (
            schedulers if isinstance(schedulers, list) else [schedulers]
        )
        for sched in sched_list:
          num_waiting += len(getattr(sched, "waiting", []))
          num_running += len(getattr(sched, "running", []))

          block_manager = getattr(sched, "block_manager", None)
          if block_manager:
            get_free = getattr(block_manager, "get_num_free_gpu_blocks", None)
            get_total = getattr(block_manager, "get_num_total_gpu_blocks", None)
            if get_free and get_total:
              total_blocks = get_total()
              free_blocks = get_free()
              if total_blocks > 0:
                used_ratio = 1.0 - (free_blocks / total_blocks)
                kv_cache_usage_perc = max(
                    kv_cache_usage_perc, used_ratio * 100.0
                )

    return base_sampler_lib.LoadInfo(
        num_requests_waiting=num_waiting,
        num_requests_running=num_running,
        kv_cache_usage_perc=kv_cache_usage_perc,
    )

  async def migrate_kv_cache(
      self,
      source_server_id: str,
      target_server_id: str,
      token_ids: list[int],
      **kwargs: Any,
  ) -> bool:
    """Triggers Raiden P2P KV-cache block transfer from source to target sampler slice."""
    del source_server_id, target_server_id, token_ids, kwargs
    raise NotImplementedError(
        "P2P KV-cache migration is pending live TPU Raiden controller"
        " integration."
    )
