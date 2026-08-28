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

"""vLLM Sampler adapter implementing Tunix WeightSyncDestination via Raiden."""

from __future__ import annotations

import asyncio
import importlib
import logging
import os
from typing import Any, List, Mapping, Sequence

import numpy as np
from tunix.experimental.rollout import sampler as base_sampler_lib
from tunix.experimental.weight_sync import weight_sync
from tunix.experimental.weight_sync import weight_sync_coordinator

Sampler = base_sampler_lib.Sampler
logger = logging.getLogger(__name__)


def _get_rl_vllm_sampler_cls():
  """Lazy import of tpu_inference.rl.RLVllmSampler.

  Resolved through importlib so static analyzers do not try to follow the
  tpu-inference dependency, which is not available in every environment.
  """
  try:
    return getattr(importlib.import_module("tpu_inference.rl"), "RLVllmSampler")
  except (ImportError, AttributeError) as e:
    raise ImportError(
        "tpu_inference.rl.RLVllmSampler is not available. Please ensure"
        " tpu-inference is installed."
    ) from e


# Hooks RLVllmSampler must expose for Raiden weight sync; verified once at
# init rather than on every call.
_REQUIRED_RAIDEN_METHODS = (
    "bind_raiden_sync",
    "get_raiden_metadata",
    "pre_weight_sync",
    "raiden_h2d",
    "post_weight_sync",
)


def _format_sampling_response(r: Any) -> base_sampler_lib.SamplingResponse:
  """Formats raw sampler output into a standardized SamplingResponse."""
  if isinstance(r, base_sampler_lib.SamplingResponse):
    return r
  tok_ids = getattr(r, "token_ids", np.zeros(0, dtype=np.int32))
  if not isinstance(tok_ids, np.ndarray):
    tok_ids = np.array(tok_ids, dtype=np.int32)
  prompt_ids = getattr(r, "prompt_token_ids", None)
  if prompt_ids is None:
    prompt_ids = np.zeros(0, dtype=np.int32)
  elif not isinstance(prompt_ids, np.ndarray):
    prompt_ids = np.array(prompt_ids, dtype=np.int32)
  lps = getattr(r, "logprobs", None)
  if lps is not None and not isinstance(lps, np.ndarray):
    lps = np.array(lps, dtype=np.float32)
  return base_sampler_lib.SamplingResponse(
      request_id=getattr(r, "request_id", ""),
      text=getattr(r, "text", ""),
      token_ids=tok_ids,
      logprobs=lps,
      prompt_token_ids=prompt_ids,
      finish_reason=getattr(r, "finish_reason", "stop"),
      routed_experts=getattr(r, "routed_experts", None),
      error=getattr(r, "error", None),
  )


class VllmSamplerAdapter(Sampler, weight_sync.WeightSyncDestination):
  """Sampler adapter wrapping tpu-inference RLVllmSampler with full Raiden weight sync."""

  def __init__(
      self,
      server_id: str = "vllm-rollout-0",
      engine_args: Any = None,
      model_name: str = "",
      sampler_instance: Any = None,
      worker_index: int = 0,
      parallelism: int = 4,
      weight_sync_mode: weight_sync.WeightSyncMode | str | None = None,
      **kwargs,
  ):
    self.server_id = server_id
    self.engine_args = engine_args
    self.model_name = model_name or (engine_args.model if engine_args else "")
    self.sampler = sampler_instance
    self.worker_index = worker_index
    self._parallelism = parallelism

    # Defaults to RAIDEN when unspecified: RLVllmSampler drives weight sync
    # through its own native Raiden hooks, so callers that construct the
    # adapter directly keep the historical always-Raiden behaviour.
    if isinstance(weight_sync_mode, weight_sync.WeightSyncMode):
      self.weight_sync_mode = weight_sync_mode
    elif isinstance(weight_sync_mode, str):
      self.weight_sync_mode = weight_sync.WeightSyncMode(weight_sync_mode)
    else:
      self.weight_sync_mode = weight_sync.WeightSyncMode.RAIDEN
    self.enable_raiden = (
        self.weight_sync_mode == weight_sync.WeightSyncMode.RAIDEN
    )
    if not self.enable_raiden:
      logger.info(
          "VllmSamplerAdapter [%s] weight_sync_mode=%s; Raiden weight sync is"
          " disabled.",
          self.server_id,
          self.weight_sync_mode.value,
      )

    self._tracker = weight_sync_coordinator.WorkerRoundTracker()
    self._sync_lock = asyncio.Lock()
    self._policy_version = 0
    self._kv_cache_freed = False

    if self.sampler is None and self.engine_args is not None:
      sampler_cls = _get_rl_vllm_sampler_cls()
      self.sampler = sampler_cls(engine_args=self.engine_args)
    self._verify_sampler_protocol()

  def initialize(self) -> None:
    """Initializes RLVllmSampler if not already initialized."""
    if self.sampler is None:
      if self.engine_args is None and self.model_name:
        from vllm.engine.arg_utils import AsyncEngineArgs  # pylint: disable=g-import-not-at-top

        self.engine_args = AsyncEngineArgs(model=self.model_name)
      if self.engine_args is not None:
        sampler_cls = _get_rl_vllm_sampler_cls()
        self.sampler = sampler_cls(engine_args=self.engine_args)
    if self.sampler is None:
      raise RuntimeError(
          f"VllmSamplerAdapter [{self.server_id}] requires valid"
          " engine_args or model_name."
      )
    self._verify_sampler_protocol()

  def _verify_sampler_protocol(self) -> None:
    """Fails fast when a Raiden-enabled sampler lacks the required hooks."""
    if not self.enable_raiden or self.sampler is None:
      return
    missing = [
        name
        for name in _REQUIRED_RAIDEN_METHODS
        if not hasattr(self.sampler, name)
    ]
    if missing:
      raise RuntimeError(
          f"VllmSamplerAdapter [{self.server_id}] sampler"
          f" {type(self.sampler).__name__} is missing required Raiden"
          f" methods: {', '.join(missing)}."
      )

  def _require_sampler(self) -> Any:
    """Returns the sampler, failing if it has not been initialized."""
    if self.sampler is None:
      raise RuntimeError(
          f"VllmSamplerAdapter [{self.server_id}] is not initialized."
      )
    return self.sampler

  # ---------------------------------------------------------------------------
  # Lifecycle & Inference Methods
  # ---------------------------------------------------------------------------

  async def start(self, **kwargs) -> Any:
    """Starts the underlying sampler engine."""
    return await self._require_sampler().start(**kwargs)

  async def _ensure_started(self) -> None:
    """Brings the engine up if nothing has needed it yet.

    RLVllmSampler builds its AsyncLLM lazily, and `sample()` is the only thing
    that calls `start()`. Weight sync needs the engine too -- it owns the TPU
    worker, and therefore the Raiden binding -- and it can legitimately run
    before the first sample, because the orchestrator pushes the trainer's
    starting weights out before dispatching any rollouts. Without this the
    round finds no worker, reports an empty destination manifest, and the run
    deadlocks: the engine is waiting for a sample that dispatch is waiting on
    the sync to allow.

    Keys off the sampler's own `_is_running` rather than calling `start()`
    unconditionally, since `start()` logs a warning when already running and
    this runs on every sync round.
    """
    if self.sampler is None:
      self.initialize()
    if not getattr(self.sampler, "_is_running", False):
      logger.info(
          "VllmSamplerAdapter [%s] starting engine for weight sync (no"
          " sample has forced it up yet).",
          self.server_id,
      )
      await self.sampler.start()

  async def stop(self, **kwargs) -> Any:
    """Stops the underlying sampler engine."""
    return await self._require_sampler().stop(**kwargs)

  async def pause(self, **kwargs) -> Any:
    """Pauses inference processing on this worker slice."""
    return await self._require_sampler().pause(**kwargs)

  async def resume(self, **kwargs) -> Any:
    """Resumes inference processing on this worker slice."""
    return await self._require_sampler().resume(**kwargs)

  async def get_mesh(self, **kwargs) -> Any:
    """Returns the underlying device mesh topology."""
    return await self._require_sampler().get_mesh(**kwargs)

  async def sample(
      self,
      sampling_requests: (
          base_sampler_lib.SamplingRequest
          | Sequence[base_sampler_lib.SamplingRequest]
          | Any
      ),
      **kwargs,
  ) -> (
      base_sampler_lib.SamplingResponse
      | List[base_sampler_lib.SamplingResponse]
      | Any
  ):
    """Generates completions using underlying tpu-inference RLVllmSampler."""
    if sampling_requests is None:
      raise ValueError("sampling_requests cannot be None.")

    is_sequence = isinstance(sampling_requests, (list, tuple))
    raw_responses = await self._require_sampler().sample(
        sampling_requests, **kwargs
    )

    if isinstance(raw_responses, (list, tuple)):
      formatted = [_format_sampling_response(r) for r in raw_responses]
      if is_sequence:
        return formatted
      return formatted[0] if formatted else base_sampler_lib.SamplingResponse()

    return _format_sampling_response(raw_responses)

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
    return await self._require_sampler().bind_raiden_sync(
        worker_index=self.worker_index, parallelism=self._parallelism
    )

  async def get_weight_sync_metadata(
      self,
      **kwargs: Any,
  ) -> Sequence[weight_sync.WorkUnitMetadata] | Any:
    """Returns transport metadata with Raiden endpoints and TensorMetadata."""
    del kwargs
    if not self.enable_raiden:
      raise NotImplementedError(
          f"VllmSamplerAdapter [{self.server_id}] does not support"
          " get_weight_sync_metadata when Raiden is disabled"
          f" (weight_sync_mode={self.weight_sync_mode.value})."
      )
    await self._ensure_started()
    meta = await self._require_sampler().get_raiden_metadata()
    return [weight_sync.WorkUnitMetadata.from_dict(m) for m in meta or []]

  async def pre_weight_sync(
      self, sync_request: Any = None, **kwargs: Any
  ) -> Any:
    """Quiesces intake, drains pending requests, resets prefix cache, and drops KV cache."""
    if not self.enable_raiden:
      return True
    sampler = self._require_sampler()
    async with self._sync_lock:
      if not self._tracker.admit(sync_request, "prepared"):
        return True

      logger.info("Executing pre_weight_sync for server_id=%s", self.server_id)

      # delegate to RLVllmSampler's native pause + clear + free-kv-cache
      await sampler.pre_weight_sync(free_kv_cache=True)
      self._kv_cache_freed = True

      self._tracker.complete(sync_request, "prepared")
      return True

  async def weight_sync(self, sync_request: Any = None, **kwargs: Any) -> Any:
    """Flushes/awaits H2D transfers and refreshes state_leaves."""
    if not self.enable_raiden:
      # RLVllmSampler owns its weight buffers, so there is no host-side
      # update_params fallback equivalent to the in-process adapter's.
      raise RuntimeError(
          f"VllmSamplerAdapter [{self.server_id}] supports Raiden weight sync"
          " only; no fallback path exists"
          f" (weight_sync_mode={self.weight_sync_mode.value})."
      )
    sampler = self._require_sampler()
    async with self._sync_lock:
      if not self._tracker.admit(sync_request, "h2d_done"):
        return True

      logger.info("Executing weight_sync barrier on Raiden synchronizers...")
      checksums = await sampler.raiden_h2d()
      if checksums:
        logger.info("Destination weights checksums: %s", checksums)

      if hasattr(sampler, "refresh_model_state_leaves"):
        result = sampler.refresh_model_state_leaves()
        if asyncio.iscoroutine(result):
          await result

      self._tracker.complete(sync_request, "h2d_done")
      return True

  async def post_weight_sync(
      self, sync_request: Any = None, **kwargs: Any
  ) -> Any:
    """Reinitializes KV cache, restores request intake, and bumps active policy version."""
    if not self.enable_raiden:
      return True
    sampler = self._require_sampler()
    async with self._sync_lock:
      if not self._tracker.admit(sync_request, "committed"):
        return True

      logger.info("Executing post_weight_sync: restoring serving state...")
      # delegate to RLVllmSampler's native reinitialize-kv-cache + resume
      if self._kv_cache_freed:
        await sampler.post_weight_sync(sync_request)
        self._kv_cache_freed = False
      else:
        await self.resume()

      version = getattr(sync_request, "policy_version", None)
      if version is not None:
        self._policy_version = version
      else:
        self._policy_version += 1

      if os.environ.get("VERIFY_WEIGHTS", "").lower() == "true" and hasattr(
          sampler, "raiden_metrics"
      ):
        logger.info(
            "Raiden transfer metrics: %s", await sampler.raiden_metrics()
        )

      self._tracker.complete(sync_request, "committed")
      return self._policy_version

  async def abort_weight_sync(
      self, sync_request: Any = None, **kwargs: Any
  ) -> Any:
    """Safely rolls back to serving previous policy version without publishing staging."""
    if not self.enable_raiden:
      return True
    sampler = self._require_sampler()
    async with self._sync_lock:
      if not self._tracker.admit(sync_request, "aborted"):
        return False

      logger.warning(
          "Aborting weight sync round: rolling back to policy_version=%d",
          self._policy_version,
      )
      # RLVllmSampler has no dedicated abort path; post_weight_sync does
      # the same recovery (reinitialize KV cache + resume).
      if self._kv_cache_freed:
        await sampler.post_weight_sync(sync_request)
        self._kv_cache_freed = False
      else:
        await self.resume()

      self._tracker.complete(sync_request, "aborted")
      return True

  async def get_weight_sync_status(self) -> Mapping[str, Any]:
    """Reports worker-side round status for coordinator recovery checks."""
    return self._tracker.report()

  async def get_transfer_status(self, req_id: Any, **kwargs) -> Any:
    """Queries status of an ongoing weight transfer or KV-cache migration."""
    sampler = self._require_sampler()
    if hasattr(sampler, "get_transfer_status"):
      return await sampler.get_transfer_status(req_id, **kwargs)
    return "UNKNOWN"

  async def get_load_info(self, **kwargs) -> base_sampler_lib.LoadInfo:
    """Returns load information from the underlying engine."""
    info = await self._require_sampler().get_load_info(**kwargs)
    return base_sampler_lib.LoadInfo(
        num_requests_waiting=getattr(info, "num_requests_waiting", 0),
        num_requests_running=getattr(info, "num_requests_running", 0),
        kv_cache_usage_perc=getattr(info, "kv_cache_usage_perc", 0.0),
    )

  async def migrate_kv_cache(
      self,
      source_server_id: str,
      target_server_id: str,
      token_ids: List[int],
      **kwargs,
  ) -> bool:
    """Triggers KV-cache transfer across TPU slices."""
    sampler = self._require_sampler()
    if hasattr(sampler, "migrate_kv_cache"):
      return await sampler.migrate_kv_cache(
          route_key=kwargs.get("route_key", ""),
          source_server_id=source_server_id,
          target_server_id=target_server_id,
          token_ids=token_ids,
      )
    return False
