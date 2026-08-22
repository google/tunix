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
import logging
import os
from typing import Any, List, Mapping, Optional, Sequence

import numpy as np
from tunix.experimental.orchestrator import weight_sync
from tunix.experimental.orchestrator import weight_sync_coordinator
from tunix.experimental.rollout import sampler as base_sampler_lib
from tunix.experimental.worker import raiden_synchronizer

Sampler = base_sampler_lib.Sampler
logger = logging.getLogger(__name__)


def _get_rl_vllm_sampler_cls():
  """Lazy import of tpu_inference.rl.vllm_sampler."""
  try:
    from tpu_inference.rl import RLVllmSampler  # pylint: disable=g-import-not-at-top
    return RLVllmSampler
  except ImportError:
    from tpu_inference.rl.vllm_sampler import RLVllmSampler  # pylint: disable=g-import-not-at-top
    return RLVllmSampler


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
      **kwargs,
  ):
    self.server_id = server_id
    self.engine_args = engine_args
    self.model_name = model_name or (engine_args.model if engine_args else "")
    self.sampler = sampler_instance
    self.worker_index = worker_index
    self._parallelism = parallelism

    self._synchronizers: List[raiden_synchronizer.RaidenSynchronizer] = []
    self._tracker = weight_sync_coordinator.WorkerRoundTracker()
    self._sync_lock = asyncio.Lock()
    self._policy_version = 0
    self._kv_cache_freed = False

    if self.sampler is None and self.engine_args is not None:
      sampler_cls = _get_rl_vllm_sampler_cls()
      self.sampler = sampler_cls(engine_args=self.engine_args)

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

  # ---------------------------------------------------------------------------
  # Lifecycle & Inference Methods
  # ---------------------------------------------------------------------------

  async def start(self, **kwargs) -> Any:
    """Starts the underlying sampler engine."""
    if self.sampler is None:
      self.initialize()
    return await self.sampler.start(**kwargs)

  async def stop(self, **kwargs) -> Any:
    """Stops the underlying sampler engine."""
    if self.sampler is None:
      raise RuntimeError(
          f"VllmSamplerAdapter [{self.server_id}] is not initialized."
      )
    return await self.sampler.stop(**kwargs)

  async def pause(self, **kwargs) -> Any:
    """Pauses inference processing on this worker slice."""
    if self.sampler is None:
      raise RuntimeError(
          f"VllmSamplerAdapter [{self.server_id}] is not initialized."
      )
    return await self.sampler.pause(**kwargs)

  async def resume(self, **kwargs) -> Any:
    """Resumes inference processing on this worker slice."""
    if self.sampler is None:
      raise RuntimeError(
          f"VllmSamplerAdapter [{self.server_id}] is not initialized."
      )
    return await self.sampler.resume(**kwargs)

  async def get_mesh(self, **kwargs) -> Any:
    """Returns the underlying device mesh topology."""
    if self.sampler is None:
      raise RuntimeError(
          f"VllmSamplerAdapter [{self.server_id}] is not initialized."
      )
    return await self.sampler.get_mesh(**kwargs)

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

    if self.sampler is None:
      raise RuntimeError(
          f"VllmSamplerAdapter [{self.server_id}] is not initialized."
      )

    is_sequence = isinstance(sampling_requests, (list, tuple))
    raw_responses = await self.sampler.sample(sampling_requests, **kwargs)

    if isinstance(raw_responses, (list, tuple)):
      formatted = []
      for r in raw_responses:
        if isinstance(r, base_sampler_lib.SamplingResponse):
          formatted.append(r)
        else:
          tok_ids = getattr(r, "token_ids", np.zeros(0, dtype=np.int32))
          if not isinstance(tok_ids, np.ndarray):
            tok_ids = np.array(tok_ids, dtype=np.int32)
          lps = getattr(r, "logprobs", None)
          if lps is not None and not isinstance(lps, np.ndarray):
            lps = np.array(lps, dtype=np.float32)
          formatted.append(
              base_sampler_lib.SamplingResponse(
                  request_id=getattr(r, "request_id", ""),
                  text=getattr(r, "text", ""),
                  token_ids=tok_ids,
                  logprobs=lps,
                  prompt_token_ids=getattr(r, "prompt_token_ids", None),
                  finish_reason=getattr(r, "finish_reason", "stop"),
                  routed_experts=getattr(r, "routed_experts", None),
                  error=getattr(r, "error", None),
              )
          )
      if is_sequence:
        return formatted
      return formatted[0] if formatted else base_sampler_lib.SamplingResponse()

    if isinstance(raw_responses, base_sampler_lib.SamplingResponse):
      return raw_responses

    tok_ids = getattr(raw_responses, "token_ids", np.zeros(0, dtype=np.int32))
    if not isinstance(tok_ids, np.ndarray):
      tok_ids = np.array(tok_ids, dtype=np.int32)
    lps = getattr(raw_responses, "logprobs", None)
    if lps is not None and not isinstance(lps, np.ndarray):
      lps = np.array(lps, dtype=np.float32)
    return base_sampler_lib.SamplingResponse(
        request_id=getattr(raw_responses, "request_id", ""),
        text=getattr(raw_responses, "text", ""),
        token_ids=tok_ids,
        logprobs=lps,
        prompt_token_ids=getattr(raw_responses, "prompt_token_ids", None),
        finish_reason=getattr(raw_responses, "finish_reason", "stop"),
        routed_experts=getattr(raw_responses, "routed_experts", None),
        error=getattr(raw_responses, "error", None),
    )

  # ---------------------------------------------------------------------------
  # WeightSyncDestination Protocol Implementation
  # ---------------------------------------------------------------------------

  async def _ensure_synchronizers_bound(self) -> List[raiden_synchronizer.RaidenSynchronizer]:
    """Binds RaidenSynchronizer instances to underlying TPU worker model weights."""
    if self.sampler is None:
      self.initialize()

    if not self._synchronizers:
      states = []
      if hasattr(self.sampler, "get_weights_state"):
        states = await self.sampler.get_weights_state()
      else:
        workers = getattr(self.sampler, "_get_tpu_workers", lambda: [])()
        for w in workers:
          runner = getattr(w, "model_runner", None)
          if runner is not None:
            states.append(getattr(runner, "state", getattr(runner, "state_leaves", None)))

      if not states:
        raise RuntimeError("No TPU model weight states found to bind RaidenSynchronizer.")

      logging.info(
          "[VllmSamplerAdapter] _ensure_synchronizers_bound: found %d weight states to bind RaidenSynchronizer",
          len(states),
      )
      for idx, state in enumerate(states):
        w_idx = self.worker_index + idx
        sync = raiden_synchronizer.RaidenSynchronizer(
            job_name="rollout",
            worker_index=w_idx,
            auto_h2d=True,
            parallelism=self._parallelism,
        )
        sync.bind(state)
        self._synchronizers.append(sync)
    else:
      for sync in self._synchronizers:
        if not sync.bound:
          states = (
              await self.sampler.get_weights_state()
              if hasattr(self.sampler, "get_weights_state")
              else []
          )
          if states:
            sync.bind(states[0])

    return self._synchronizers

  async def bind_weight_sync(self) -> None:
    """Idempotent transport binding called while the worker is STILL SERVING."""
    if hasattr(self.sampler, "bind_raiden_sync"):
      await self.sampler.bind_raiden_sync(
          worker_index=self.worker_index, parallelism=self._parallelism
      )
    else:
      await self._ensure_synchronizers_bound()

  async def get_weight_sync_metadata(self) -> Sequence[weight_sync.WorkUnitMetadata]:
    """Returns transport metadata with Raiden endpoints and TensorMetadata."""
    if hasattr(self.sampler, "get_raiden_metadata"):
      meta = await self.sampler.get_raiden_metadata()
      if meta:
        out = []
        for m in meta:
          if isinstance(m, (list, tuple)):
            for sub_m in m:
              out.append(weight_sync.dict_to_metadata(sub_m))
          else:
            out.append(weight_sync.dict_to_metadata(m))
        return out
    synchronizers = await self._ensure_synchronizers_bound()
    return [s.work_unit_metadata() for s in synchronizers]

  async def pre_weight_sync(self, sync_request: Any = None, **kwargs: Any) -> Any:
    """Quiesces intake, drains pending requests, resets prefix cache, and drops KV cache."""
    if self.sampler is None:
      self.initialize()
    async with self._sync_lock:
      if not self._tracker.admit(sync_request, "prepared"):
        return True

      logger.info("Executing pre_weight_sync for server_id=%s", self.server_id)
      await self.pause()

      if hasattr(self.sampler, "_clear_prefix_cache"):
        await self.sampler._clear_prefix_cache()

      if hasattr(self.sampler, "free_kv_cache"):
        await self.sampler.free_kv_cache()
      elif hasattr(self.sampler, "_call_worker_method"):
        # RLVllmSampler has no free_kv_cache()/restore_kv_cache() wrapper;
        # dispatch straight to the real TPUWorker primitive its own
        # pre_weight_sync uses. (The _engine.start_weight_update() fallback
        # below is unreachable for RLVllmSampler -- kept for other sampler
        # implementations -- and was additionally broken: called
        # synchronously with a `free_kv_cache` kwarg the real
        # `async def start_weight_update(self) -> None` doesn't accept, and
        # dispatching to a `collective_rpc("start_weight_update")` TPUWorker
        # has no handler for anyway.)
        await self.sampler._call_worker_method("delete_kv_cache")
      elif hasattr(self.sampler, "_engine") and hasattr(self.sampler._engine, "start_weight_update"):
        await self.sampler._engine.start_weight_update()
      self._kv_cache_freed = True

      self._tracker.complete(sync_request, "prepared")
      return True

  async def weight_sync(self, sync_request: Any = None, **kwargs: Any) -> Any:
    """Flushes/awaits H2D transfers and refreshes state_leaves."""
    if self.sampler is None:
      self.initialize()
    async with self._sync_lock:
      if not self._tracker.admit(sync_request, "h2d_done"):
        return True

      logger.info("Executing weight_sync barrier on Raiden synchronizers...")
      if hasattr(self.sampler, "raiden_h2d"):
        checksums = await self.sampler.raiden_h2d()
        if checksums:
          logger.info("Destination weights checksums: %s", checksums)
      else:
        for sync in self._synchronizers:
          if not sync.bound:
            raise RuntimeError("bind_weight_sync must run before weight_sync.")
          sync.h2d()
          if os.environ.get("VERIFY_WEIGHTS", "").lower() == "true":
            logger.info("Destination weights checksums: %s", sync.checksums())

      if hasattr(self.sampler, "refresh_model_state_leaves"):
        result = self.sampler.refresh_model_state_leaves()
        if asyncio.iscoroutine(result):
          await result

      self._tracker.complete(sync_request, "h2d_done")
      return True

  async def post_weight_sync(self, sync_request: Any = None, **kwargs: Any) -> Any:
    """Reinitializes KV cache, restores request intake, and bumps active policy version."""
    if self.sampler is None:
      self.initialize()
    async with self._sync_lock:
      if not self._tracker.admit(sync_request, "committed"):
        return True

      logger.info("Executing post_weight_sync: restoring serving state...")
      if self._kv_cache_freed:
        if hasattr(self.sampler, "restore_kv_cache"):
          await self.sampler.restore_kv_cache()
        elif hasattr(self.sampler, "_call_worker_method"):
          # See pre_weight_sync's matching comment: dispatch straight to the
          # real TPUWorker primitive.
          await self.sampler._call_worker_method("reinitialize_kv_cache")
        elif hasattr(self.sampler, "_engine") and hasattr(self.sampler._engine, "finish_weight_update"):
          await self.sampler._engine.finish_weight_update()
        self._kv_cache_freed = False

      version = getattr(sync_request, "policy_version", None)
      if version is not None:
        self._policy_version = version
      else:
        self._policy_version += 1

      await self.resume()

      if os.environ.get("VERIFY_WEIGHTS", "").lower() == "true":
        if hasattr(self.sampler, "raiden_metrics"):
          logger.info("Raiden transfer metrics: %s", await self.sampler.raiden_metrics())
        else:
          for sync in self._synchronizers:
            logger.info("Raiden transfer metrics: %s", sync.metrics())

      self._tracker.complete(sync_request, "committed")
      return self._policy_version

  async def abort_weight_sync(self, sync_request: Any = None, **kwargs: Any) -> Any:
    """Safely rolls back to serving previous policy version without publishing staging."""
    if self.sampler is None:
      self.initialize()
    async with self._sync_lock:
      if not self._tracker.admit(sync_request, "aborted"):
        return False

      logger.warning(
          "Aborting weight sync round: rolling back to policy_version=%d",
          self._policy_version,
      )
      if self._kv_cache_freed:
        if hasattr(self.sampler, "restore_kv_cache"):
          await self.sampler.restore_kv_cache()
        elif hasattr(self.sampler, "_call_worker_method"):
          # See pre_weight_sync's matching comment: dispatch straight to the
          # real TPUWorker primitive.
          await self.sampler._call_worker_method("reinitialize_kv_cache")
        elif hasattr(self.sampler, "_engine") and hasattr(self.sampler._engine, "finish_weight_update"):
          await self.sampler._engine.finish_weight_update()
        self._kv_cache_freed = False

      await self.resume()

      self._tracker.complete(sync_request, "aborted")
      return True

  async def get_weight_sync_status(self) -> Mapping[str, Any]:
    """Reports worker-side round status for coordinator recovery checks."""
    return self._tracker.report()

  async def get_transfer_status(self, req_id: Any, **kwargs) -> Any:
    """Queries status of an ongoing weight transfer or KV-cache migration."""
    if self.sampler is None:
      raise RuntimeError(
          f"VllmSamplerAdapter [{self.server_id}] is not initialized."
      )
    if hasattr(self.sampler, "get_transfer_status"):
      return await self.sampler.get_transfer_status(req_id, **kwargs)
    return "UNKNOWN"

  async def get_load_info(self, **kwargs) -> base_sampler_lib.LoadInfo:
    """Returns load information from the underlying engine."""
    if self.sampler is None:
      raise RuntimeError(
          f"VllmSamplerAdapter [{self.server_id}] is not initialized."
      )
    info = await self.sampler.get_load_info(**kwargs)
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
    if self.sampler is None:
      raise RuntimeError(
          f"VllmSamplerAdapter [{self.server_id}] is not initialized."
      )
    if hasattr(self.sampler, "migrate_kv_cache"):
      return await self.sampler.migrate_kv_cache(
          route_key=kwargs.get("route_key", ""),
          source_server_id=source_server_id,
          target_server_id=target_server_id,
          token_ids=token_ids,
      )
    return False


# Compatibility alias
VllmInferenceSamplerAdapter = VllmSamplerAdapter
