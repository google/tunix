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
import numbers
from typing import Any, List, Sequence
from absl import logging
import numpy as np
from tunix.experimental.rollout import sampler as base_sampler_lib
from tunix.experimental.weight_sync import weight_sync
from tunix.generate import sampler as generate_sampler_lib

Sampler = base_sampler_lib.Sampler


class VanillaSamplerAdapter(Sampler, abc.ABC):
  """Standalone TPU/GPU slice running Tunix Vanilla JAX Sampler.

  Constructs or wraps a Tunix generate_sampler_lib.Sampler instance and
  executes sampling requests.

  Supported Weight Synchronization Modes:
    1. Raiden Mode (`weight_sync_mode == RAIDEN`):
       Delegates high-performance decentralized P2P / DCN weight synchronization
       to `RaidenWeightSyncDelegate`. Binds destination memory buffers using
       `self.sampler.transformer_state` and executes phased synchronization
       lifecycle hooks (`bind`, `pre`, `sync`, `post`).
    2. Fallback Direct Mode (`weight_sync_mode == FALLBACK`):
       Synchronizes weights in-place without Raiden transport. When a weight
       payload is received in `sync_request.weights`, updates the underlying
       JAX sampler directly via `self.sampler.transformer_state`.
  """

  def __init__(
      self,
      server_id: str,
      transformer: Any = None,
      tokenizer: Any = None,
      cache_config: generate_sampler_lib.CacheConfig | int | None = None,
      image_processor: Any = None,
      model: Any = None,
      config: Any = None,
      raiden_sync_delegate: Any = None,
      **kwargs,
  ):
    self.server_id = server_id
    self.transformer = transformer if transformer is not None else model
    self.tokenizer = tokenizer
    self.image_processor = image_processor
    self.config = config
    self.raiden_sync_delegate = raiden_sync_delegate
    self.weight_sync_mode = getattr(
        config, "weight_sync_mode", weight_sync.WeightSyncMode.FALLBACK
    )
    self.enable_raiden = (
        self.weight_sync_mode == weight_sync.WeightSyncMode.RAIDEN
    )

    if self.enable_raiden and self.raiden_sync_delegate is None:
      from tunix.experimental.weight_sync import raiden_weight_sync_delegate  # pylint: disable=g-import-not-at-top

      self.raiden_sync_delegate = (
          raiden_weight_sync_delegate.RaidenWeightSyncDelegate()
      )

    if not self.enable_raiden and self.raiden_sync_delegate:
      logging.warning(
          "VanillaSamplerAdapter [%s] raiden_sync_delegate is set but"
          " enable_raiden is False.",
          self.server_id,
      )

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
      if cfg:
        num_layers = getattr(
            cfg, "num_layers", getattr(cfg, "num_hidden_layers", 4)
        )
        num_kv_heads = getattr(
            cfg, "num_kv_heads", getattr(cfg, "num_key_value_heads", 4)
        )
        head_dim = getattr(cfg, "head_dim", getattr(cfg, "head_dimension", 16))
        cache_size = (
            cache_config
            if isinstance(cache_config, int)
            else getattr(cfg, "max_position_embeddings", 1024)
        )
      else:
        num_layers = 4
        num_kv_heads = 4
        head_dim = 16
        cache_size = cache_config if isinstance(cache_config, int) else 1024

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

  def _unpadded_prompt_tokens(self, padded_tokens: Any) -> np.ndarray:
    """Returns sampler-tokenized prompt ids without backend left padding."""
    arr = np.asarray(padded_tokens, dtype=np.int32).reshape(-1)
    pad_id = getattr(self.tokenizer, "pad_token_id", None)
    if pad_id is None:
      pad_id = getattr(self.tokenizer, "eos_token_id", None)
    if not isinstance(pad_id, numbers.Integral):
      return arr
    non_pad = np.flatnonzero(arr != pad_id)
    if non_pad.size == 0:
      return np.zeros(0, dtype=np.int32)
    return arr[non_pad[0] :]

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
          | Sequence[Any]
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
      lps = None
      if sampler_output.logprobs and isinstance(sampler_output.logprobs, list):
        lps = sampler_output.logprobs[i]

      tok_ids = (
          np.array(toks, dtype=np.int32)
          if toks is not None
          else np.zeros(0, dtype=np.int32)
      )
      prompt_token_ids = self._unpadded_prompt_tokens(
          sampler_output.padded_prompt_tokens[i]
      )
      log_ps = np.array(lps, dtype=np.float32) if lps is not None else None

      responses.append(
          base_sampler_lib.SamplingResponse(
              request_id=req_id,
              text=txt,
              prompt_token_ids=prompt_token_ids,
              token_ids=tok_ids,
              logprobs=log_ps,
              finish_reason="stop",
          )
      )

    if is_sequence:
      return responses
    return responses[0]

  # --- Weight Synchronization ---
  async def get_transfer_status(self, req_id: str | Any, **kwargs) -> str | Any:
    """Queries status of an ongoing weight transfer or KV-cache migration."""
    del req_id, kwargs
    return "SUCCESS"

  async def get_load_info(self, **kwargs) -> base_sampler_lib.LoadInfo:
    """Returns best-effort local sampler load information."""
    del kwargs
    return base_sampler_lib.LoadInfo()

  # --- Weight Synchronization ---
  def _check_weight_sync_boundness(
      self,
  ):
    """Verifies that the Raiden delegate has been bound before executing sync phases."""
    if not self.enable_raiden:
      return

    if not self.raiden_sync_delegate.is_bounded():
      raise RuntimeError(
          f"VanillaSamplerAdapter [{self.server_id}] weight sync delegate"
          " is not bounded."
      )

  async def get_weight_sync_metadata(self, **kwargs) -> Any:
    """Returns sharding specs and layout metadata across devices for weights."""
    self._check_weight_sync_boundness()

    # In Raiden mode, retrieve transport endpoint and tensor shard layout
    # metadata.
    if self.enable_raiden:
      return await self.raiden_sync_delegate.get_weight_sync_metadata(**kwargs)

    # In Fallback mode, metadata query is not supported.
    raise NotImplementedError(
        f"VanillaSamplerAdapter [{self.server_id}] does not support"
        " get_weight_sync_metadata when Raiden is disabled."
    )

  async def bind_weight_sync(
      self,
      sync_request: base_sampler_lib.WeightSyncRequest | Any = None,
      **kwargs,
  ) -> Any:
    """Binds destination-side transport resources for weight transfer."""
    if self.enable_raiden:
      # In Raiden mode, register destination sampler transformer_state memory
      # buffers.
      if not hasattr(self.sampler, "transformer_state"):
        raise RuntimeError(
            f"VanillaSamplerAdapter [{self.server_id}] sampler does not expose"
            " transformer_state for Raiden weight sync."
        )

      if self.raiden_sync_delegate.is_bounded():
        raise RuntimeError(
            f"VanillaSamplerAdapter [{self.server_id}] weight sync delegate is"
            " already bounded before bind_weight_sync."
        )

      state = self.sampler.transformer_state
      return await self.raiden_sync_delegate.bind_weight_sync(
          sync_request=sync_request, state=state, **kwargs
      )
    # In Fallback mode, no transport binding is required.
    return None

  async def pre_weight_sync(
      self,
      sync_request: base_sampler_lib.WeightSyncRequest | Any = None,
      **kwargs,
  ) -> str | None | Any:
    """Prepares staging handshake prior to policy weight update."""
    self._check_weight_sync_boundness()

    # In Raiden mode, execute the pre-synchronization barrier via delegate.
    if self.enable_raiden:
      return await self.raiden_sync_delegate.pre_weight_sync(
          sync_request=sync_request, **kwargs
      )
    # In Fallback mode, acts as a no-op returning True.
    return True

  async def weight_sync(
      self,
      sync_request: base_sampler_lib.WeightSyncRequest | Any = None,
      **kwargs,
  ) -> str | None | Any:
    """Updates model weights in-place from the specified controller or request."""
    self._check_weight_sync_boundness()

    # Raiden mode: Invoke Raiden transport to stream weights into bound memory
    # buffers.
    if self.enable_raiden:
      return await self.raiden_sync_delegate.weight_sync(
          sync_request=sync_request, **kwargs
      )
    else:
      # Fallback mode: Directly assign source weights from sync_request.weights.
      if sync_request is None:
        raise ValueError(
            "VanillaSamplerAdapter Fallback mode [%s] weight_sync:"
            " sync_request is None."
            % self.server_id
        )
      if self.sampler and hasattr(self.sampler, "transformer_state"):
        weights = getattr(sync_request, "weights", None)
        if weights is None:
          raise ValueError(
              "VanillaSamplerAdapter [%s] weight_sync: weights not found"
              " in sync_request."
              % self.server_id
          )
        self.sampler.transformer_state = weights
      else:
        raise RuntimeError(
            f"VanillaSamplerAdapter [{self.server_id}] does not support"
            " Raiden weight sync, while the fallback path missing required"
            " components."
        )
      return True

  async def post_weight_sync(
      self,
      sync_request: base_sampler_lib.WeightSyncRequest | Any = None,
      **kwargs,
  ) -> str | None | Any:
    """Finalizes and switches active policy weights after transfer completion."""
    # Raiden mode: Commit newly transferred weights and execute post-sync
    # barrier.
    if self.enable_raiden:
      return await self.raiden_sync_delegate.post_weight_sync(
          sync_request=sync_request, **kwargs
      )
    # Fallback mode: acts as a no-op returning True.
    return True

  # --- KV-cache Migration ---
  async def migrate_kv_cache(
      self,
      source_server_id: str,
      target_server_id: str,
      token_ids: List[int],
      **kwargs,
  ) -> bool:
    """Triggers Raiden P2P KV-cache transfer across TPU slices."""
    del source_server_id, target_server_id, token_ids, kwargs
    return True
