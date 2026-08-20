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

"""vLLM Sampler adapter wrapping tpu-inference RLVllmSampler for Tunix."""

import logging
from typing import Any, List, Sequence

import numpy as np
from tunix.experimental.rollout import sampler as base_sampler_lib

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


class VllmSamplerAdapter(Sampler):
  """Sampler adapter wrapping tpu-inference RLVllmSampler."""

  def __init__(
      self,
      server_id: str = "vllm-rollout-0",
      engine_args: Any = None,
      model_name: str = "",
      sampler_instance: Any = None,
      **kwargs,
  ):
    self.server_id = server_id
    self.engine_args = engine_args
    self.model_name = model_name or (engine_args.model if engine_args else "")
    self.sampler = sampler_instance
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
        finish_reason=getattr(raw_responses, "finish_reason", "stop"),
        routed_experts=getattr(raw_responses, "routed_experts", None),
        error=getattr(raw_responses, "error", None),
    )

  async def get_weight_sync_metadata(self, **kwargs) -> Any:
    """Returns weight sharding specs and layout metadata."""
    if self.sampler is None:
      raise RuntimeError(
          f"VllmSamplerAdapter [{self.server_id}] is not initialized."
      )
    return await self.sampler.get_weight_sync_metadata(**kwargs)

  async def pre_weight_sync(self, sync_request: Any = None, **kwargs) -> Any:
    """Prepares staging handshake prior to policy weight update."""
    if self.sampler is None:
      raise RuntimeError(
          f"VllmSamplerAdapter [{self.server_id}] is not initialized."
      )
    return await self.sampler.pre_weight_sync(sync_request, **kwargs)

  async def weight_sync(self, sync_request: Any = None, **kwargs) -> Any:
    """Updates model weights in-place."""
    if self.sampler is None:
      raise RuntimeError(
          f"VllmSamplerAdapter [{self.server_id}] is not initialized."
      )
    return await self.sampler.weight_sync(sync_request, **kwargs)

  async def post_weight_sync(self, sync_request: Any = None, **kwargs) -> Any:
    """Finalizes and switches active policy weights."""
    if self.sampler is None:
      raise RuntimeError(
          f"VllmSamplerAdapter [{self.server_id}] is not initialized."
      )
    return await self.sampler.post_weight_sync(sync_request, **kwargs)

  async def get_transfer_status(self, req_id: Any, **kwargs) -> Any:
    """Queries status of an ongoing weight transfer or KV-cache migration."""
    if self.sampler is None:
      raise RuntimeError(
          f"VllmSamplerAdapter [{self.server_id}] is not initialized."
      )
    return await self.sampler.get_transfer_status(req_id, **kwargs)

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
    return await self.sampler.migrate_kv_cache(
        route_key=kwargs.get("route_key", ""),
        source_server_id=source_server_id,
        target_server_id=target_server_id,
        token_ids=token_ids,
    )


VllmInferenceSamplerAdapter = VllmSamplerAdapter
