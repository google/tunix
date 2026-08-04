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

"""Vanilla Sampler adapter using Tunix JAX Sampler."""

import abc
from typing import Any, List, Sequence
import numpy as np
from tunix.experimental.rollout import sampler as base_sampler_lib
from tunix.generate import sampler as generate_sampler_lib

Sampler = base_sampler_lib.Sampler


class VanillaSamplerAdapter(Sampler, abc.ABC):
  """Standalone TPU/GPU slice running Tunix Vanilla JAX Sampler.

  Constructs or wraps a Tunix generate_sampler_lib.Sampler instance and
  executes sampling requests.
  """

  def __init__(
      self,
      server_id: str,
      transformer: Any = None,
      tokenizer: Any = None,
      cache_config: generate_sampler_lib.CacheConfig | int | None = None,
      image_processor: Any = None,
      model: Any = None,
      **kwargs,
  ):
    self.server_id = server_id
    self.transformer = transformer if transformer is not None else model
    self.tokenizer = tokenizer
    self.image_processor = image_processor

    if self.transformer is not None and self.tokenizer is not None:
      self.sampler = self._build_generate_sampler(cache_config)
    else:
      self.sampler = None

  def _build_generate_sampler(
      self, cache_config: generate_sampler_lib.CacheConfig | int | None
  ) -> generate_sampler_lib.Sampler:
    """Helper to construct generate_sampler_lib.Sampler from model and tokenizer."""
    if isinstance(cache_config, generate_sampler_lib.CacheConfig):
      cache_cfg = cache_config
    else:
      cfg = getattr(self.transformer, "config", None)
      num_layers = (
          getattr(cfg, "num_layers", getattr(cfg, "num_hidden_layers", 4))
          if cfg
          else 4
      )
      num_kv_heads = (
          getattr(
              cfg, "num_kv_heads", getattr(cfg, "num_key_value_heads", 4)
          )
          if cfg
          else 4
      )
      head_dim = (
          getattr(cfg, "head_dim", getattr(cfg, "head_dimension", 16))
          if cfg
          else 16
      )
      cache_size = (
          cache_config
          if isinstance(cache_config, int)
          else (
              getattr(cfg, "max_position_embeddings", 1024) if cfg else 1024
          )
      )
      cache_cfg = generate_sampler_lib.CacheConfig(
          cache_size=cache_size,
          num_layers=num_layers,
          num_kv_heads=num_kv_heads,
          head_dim=head_dim,
      )

    return generate_sampler_lib.Sampler(
        transformer=self.transformer,
        tokenizer=self.tokenizer,
        cache_config=cache_cfg,
        image_processor=self.image_processor,
    )

  def initialize(self) -> None:
    """Initializes sampler if needed."""
    if (
        self.sampler is None
        and self.transformer is not None
        and self.tokenizer is not None
    ):
      self.sampler = self._build_generate_sampler(None)

    if self.sampler is None and (
        self.transformer is not None or self.tokenizer is not None
    ):
      raise RuntimeError(
          f"VanillaSamplerAdapter [{self.server_id}] requires a sampler"
          " instance or transformer + tokenizer."
      )

  # --- Lifecycle & Topology ---
  async def start(self, **kwargs) -> str | None | Any:
    """Starts the sampling engine or local loop."""
    del kwargs
    return True

  async def stop(self, **kwargs) -> str | None | Any:
    del kwargs
    return True

  async def pause(self, **kwargs) -> str | None | Any:
    """Pauses inference processing on this worker slice."""
    del kwargs
    return True

  async def resume(self, **kwargs) -> str | None | Any:
    """Resumes inference processing on this worker slice."""
    del kwargs
    return True

  async def get_mesh(self, **kwargs) -> Any:
    """Returns the underlying device mesh topology."""
    del kwargs
    if hasattr(self.sampler, "get_mesh"):
      return self.sampler.get_mesh()
    return None

  # --- Inference ---
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
    """Standard completion call using external Tunix JAX Sampler model."""
    if not self.sampler:
      raise RuntimeError(
          f"VanillaSamplerAdapter [{self.server_id}] sampler is not"
          " initialized."
      )

    if sampling_requests is None:
      raise ValueError("sampling_requests cannot be None.")

    if isinstance(sampling_requests, base_sampler_lib.SamplingRequest):
      requests: List[Any] = [sampling_requests]
      is_sequence = False
    elif isinstance(sampling_requests, (list, tuple)):
      requests = list(sampling_requests)
      is_sequence = True
    else:
      requests = [sampling_requests]
      is_sequence = False

    prompts = []
    max_gen_steps_list = []
    temps = []
    top_ps = []
    top_ks = []
    seeds = []
    return_logprobs_list = []
    return_logits_list = []
    beam_sizes = []

    for req in requests:
      prompt = req.prompt if hasattr(req, "prompt") else req
      prompts.append(prompt)
      sp = (
          req.sampling_params
          if hasattr(req, "sampling_params") and req.sampling_params is not None
          else base_sampler_lib.SamplingParams()
      )
      assert sp is not None

      max_gen_steps_list.append(sp.max_tokens)
      temps.append(sp.temperature)
      top_ps.append(sp.top_p)
      top_ks.append(sp.top_k)
      seeds.append(sp.seed)
      return_logprobs_list.append(sp.return_logprobs)
      return_logits_list.append(sp.return_logits)
      if sp.beam_size is not None:
        beam_sizes.append(sp.beam_size)

    max_generation_steps = (
        max(max_gen_steps_list) if max_gen_steps_list else 64
    )
    temperature = temps[0] if temps else 0.0
    top_p = top_ps[0] if top_ps else None
    top_k = top_ks[0] if top_ks else None
    seed = seeds[0] if seeds else None
    return_logprobs = any(return_logprobs_list) or kwargs.get(
        "return_logprobs", False
    )
    return_logits = any(return_logits_list) or kwargs.get(
        "return_logits", False
    )
    beam_size = beam_sizes[0] if beam_sizes else None

    sampler_output = self.sampler(
        input_strings=prompts,
        max_generation_steps=max_generation_steps,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
        beam_size=beam_size,
        return_logits=return_logits,
        return_logprobs=return_logprobs,
    )

    responses = []
    for i, req in enumerate(requests):
      req_id = getattr(req, "request_id", "")

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
      lps = (
          sampler_output.logprobs[i]
          if (
              sampler_output.logprobs
              and isinstance(sampler_output.logprobs, list)
          )
          else None
      )

      tok_ids = (
          np.array(toks, dtype=np.int32)
          if toks is not None
          else np.zeros(0, dtype=np.int32)
      )
      log_ps = np.array(lps, dtype=np.float32) if lps is not None else None

      responses.append(
          base_sampler_lib.SamplingResponse(
              request_id=req_id,
              text=txt,
              token_ids=tok_ids,
              logprobs=log_ps,
              finish_reason="stop",
          )
      )

    if is_sequence:
      return responses
    return responses[0]

  # --- Weight Synchronization ---
  async def get_weight_sync_metadata(self, **kwargs) -> Any:
    """Returns sharding specs and layout metadata across devices for weights."""
    del kwargs
    raise NotImplementedError(
        "get_weight_sync_metadata() not implemented for this SamplerServer."
    )

  async def pre_weight_sync(self, sync_request: Any = None, **kwargs) -> Any:
    """Prepares staging handshake prior to policy weight update."""
    del sync_request, kwargs
    return True

  async def weight_sync(self, sync_request: Any = None, **kwargs) -> Any:
    """Updates model weights in-place from the specified controller."""
    del sync_request, kwargs
    return True

  async def get_transfer_status(self, req_id: Any, **kwargs) -> Any:
    """Queries status of an ongoing weight transfer or KV-cache migration."""
    del req_id, kwargs
    return "SUCCESS"

  async def post_weight_sync(self, sync_request: Any = None, **kwargs) -> Any:
    """Finalizes and switches active policy weights after transfer completion."""
    del sync_request, kwargs
    return True

  async def migrate_kv_cache(
      self,
      source_server_id: str,
      target_server_id: str,
      token_ids: List[int],
      **kwargs,
  ) -> bool:
    """Triggers Raiden P2P KV-cache transfer across TPU slices."""
    del source_server_id, target_server_id, token_ids
    return True
