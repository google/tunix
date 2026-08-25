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

"""In-process vLLM Sampler adapter integrating with Tunix VllmSampler."""

import abc
import numbers
from typing import Any, List, Sequence
from absl import logging
import numpy as np
from tunix.experimental.rollout import sampler as base_sampler_lib
from tunix.experimental.weight_sync import raiden_weight_sync_delegate
from tunix.experimental.weight_sync import weight_sync

Sampler = base_sampler_lib.Sampler


def _get_vllm_sampler_cls():
  """Lazy import of tunix.generate.vllm_sampler to avoid top-level vLLM import side-effects."""
  from tunix.generate import vllm_sampler as generate_vllm_lib  # pylint: disable=g-import-not-at-top
  return generate_vllm_lib


class InprocessVllmSamplerAdapter(Sampler, abc.ABC):
  """Sampler adapter wrapping Tunix VllmSampler."""

  def __init__(
      self,
      server_id: str,
      tokenizer: Any = None,
      config: Any = None,
      model_name: str = "",
      raiden_sync_delegate: Any = None,
      **kwargs,
  ):
    self.server_id = server_id
    self.tokenizer = tokenizer
    self.config = config
    self.model_name = model_name or kwargs.get("model", "")
    self.vllm_sampler = None
    self.raiden_sync_delegate = raiden_sync_delegate
    self.weight_sync_mode = getattr(
        config, "weight_sync_mode", weight_sync.WeightSyncMode.FALLBACK
    )
    self.enable_raiden = (
        self.weight_sync_mode == weight_sync.WeightSyncMode.RAIDEN
    )

    if self.enable_raiden:
      logging.info(
          "InprocessVllmSamplerAdapter [%s] weight_sync: initializing Raiden"
          " delegate",
          self.server_id,
      )

      if self.raiden_sync_delegate is None:
        self.raiden_sync_delegate = (
            raiden_weight_sync_delegate.RaidenWeightSyncDelegate()
        )

    if not self.enable_raiden and self.raiden_sync_delegate:
      logging.warning(
          "InprocessVllmSamplerAdapter [%s] raiden_sync_delegate is set but"
          " enable_raiden is False.",
          self.server_id,
      )

    if self.tokenizer is not None and self.config is not None:
      vllm_lib = _get_vllm_sampler_cls()
      self.vllm_sampler = vllm_lib.VllmSampler(
          tokenizer=self.tokenizer, config=self.config
      )

  def initialize(self) -> None:
    """Initializes vLLM sampler if needed."""
    if self.tokenizer is None and self.model_name:
      from transformers import AutoTokenizer  # pylint: disable=g-import-not-at-top
      from tunix.generate import vllm_sampler as tunix_vllm_sampler  # pylint: disable=g-import-not-at-top

      self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
      self.config = tunix_vllm_sampler.VllmConfig(
          engine_kwargs={"model": self.model_name}
      )

    if (
        self.vllm_sampler is None
        and self.tokenizer is not None
        and self.config is not None
    ):
      vllm_lib = _get_vllm_sampler_cls()
      self.vllm_sampler = vllm_lib.VllmSampler(
          tokenizer=self.tokenizer, config=self.config
      )

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

  def _prompt_tokens_from_request(
      self, req: Any, fallback_padded_tokens: Any
  ) -> np.ndarray:
    """Returns request token ids directly when available, else sampler output."""
    prompt = req.prompt if hasattr(req, "prompt") else req
    try:
      return np.asarray(prompt, dtype=np.int32).reshape(-1)
    except (TypeError, ValueError):
      pass
    return self._unpadded_prompt_tokens(fallback_padded_tokens)

  def _prompt_to_input_string(self, prompt: Any) -> Any:
    """Renders chat-message prompts to strings for Tunix VllmSampler."""
    if isinstance(prompt, str):
      return prompt
    if isinstance(prompt, (list, tuple)) and all(
        isinstance(message, dict) for message in prompt
    ):
      if hasattr(self.tokenizer, "apply_chat_template"):
        return self.tokenizer.apply_chat_template(
            list(prompt), tokenize=False, add_generation_prompt=True
        )
      return "\n".join(
          str(message.get("content", "")) for message in prompt
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
    if self.vllm_sampler and hasattr(self.vllm_sampler, "stop"):
      self.vllm_sampler.stop()
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
    if self.vllm_sampler and hasattr(self.vllm_sampler, "mesh"):
      return self.vllm_sampler.mesh
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
    """Generates completions using underlying Tunix VllmSampler."""
    if not self.vllm_sampler:
      raise RuntimeError(
          f"InprocessVllmSamplerAdapter [{self.server_id}] vllm_sampler is not"
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

    for req in requests:
      prompt = req.prompt if hasattr(req, "prompt") else req
      prompts.append(self._prompt_to_input_string(prompt))
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

    sampler_output = self.vllm_sampler(
        input_strings=prompts,
        max_generation_steps=max_generation_steps,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
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
      prompt_token_ids = self._prompt_tokens_from_request(
          req, sampler_output.padded_prompt_tokens[i]
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
  def _check_weight_sync_boundness(
      self,
  ):
    """Returns whether the weight sync delegate is bounded."""
    if not self.enable_raiden:
      return

    if not self.raiden_sync_delegate.is_bounded():
      raise RuntimeError(
          f"InprocessVllmSamplerAdapter [{self.server_id}] weight sync delegate"
          " is not bounded."
      )

  async def get_weight_sync_metadata(self, **kwargs) -> Any:
    """Returns sharding specs and layout metadata across devices for weights."""
    self._check_weight_sync_boundness()

    if self.enable_raiden:
      return await self.raiden_sync_delegate.get_weight_sync_metadata(**kwargs)
    raise NotImplementedError(
        f"InprocessVllmSamplerAdapter [{self.server_id}] does not support"
        " get_weight_sync_metadata when Raiden is disabled."
    )

  async def bind_weight_sync(
      self,
      sync_request: base_sampler_lib.WeightSyncRequest | Any = None,
      **kwargs,
  ) -> Any:
    """Binds destination-side transport resources."""
    if self.enable_raiden:
      if not hasattr(self.vllm_sampler, "transformer_state"):
        raise RuntimeError(
            f"InprocessVllmSamplerAdapter [{self.server_id}] does not expose"
            " transformer_state for Raiden weight sync."
        )

      if self.raiden_sync_delegate.is_bounded():
        raise RuntimeError(
            f"InprocessVllmSamplerAdapter [{self.server_id}] weight sync"
            " delegate is already bounded before bind_weight_sync."
        )

      state = self.vllm_sampler.transformer_state
      return await self.raiden_sync_delegate.bind_weight_sync(
          sync_request=sync_request, state=state, **kwargs
      )
    return None

  async def pre_weight_sync(
      self,
      sync_request: base_sampler_lib.WeightSyncRequest | Any = None,
      **kwargs,
  ) -> str | None | Any:
    """Prepares staging handshake prior to policy weight update."""
    self._check_weight_sync_boundness()

    if self.enable_raiden:
      return await self.raiden_sync_delegate.pre_weight_sync(
          sync_request=sync_request, **kwargs
      )
    return True

  async def weight_sync(
      self,
      sync_request: base_sampler_lib.WeightSyncRequest | Any = None,
      **kwargs,
  ) -> str | None | Any:
    """Updates model weights in-place from the specified controller."""
    self._check_weight_sync_boundness()

    if self.enable_raiden:
      return await self.raiden_sync_delegate.weight_sync(
          sync_request=sync_request, **kwargs
      )
    else:
      if sync_request is None:
        raise ValueError(
            f"InprocessVllmSamplerAdapter Fallback mode [{self.server_id}]"
            " weight_sync: sync_request is None."
        )
      if self.vllm_sampler and hasattr(self.vllm_sampler, "update_params"):
        weights = getattr(sync_request, "weights", None)
        if weights is None:
          raise ValueError(
              f"InprocessVllmSamplerAdapter [{self.server_id}] weight_sync:"
              " weights not found in sync_request."
          )

        self.vllm_sampler.update_params(weights)
      else:
        raise RuntimeError(
            f"InprocessVllmSamplerAdapter [{self.server_id}] does not support"
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
    if self.enable_raiden:
      return await self.raiden_sync_delegate.post_weight_sync(
          sync_request=sync_request, **kwargs
      )
    return True

  async def get_transfer_status(self, req_id: str | Any, **kwargs) -> str | Any:
    """Queries status of an ongoing weight transfer or KV-cache migration."""
    del req_id, kwargs
    return "SUCCESS"

  async def migrate_kv_cache(
      self,
      source_server_id: str,
      target_server_id: str,
      token_ids: List[int],
      **kwargs,
  ) -> bool:
    """Triggers KV-cache transfer across TPU slices."""
    del source_server_id, target_server_id, token_ids, kwargs
    return True

  async def get_load_info(self, **kwargs) -> base_sampler_lib.LoadInfo:
    """Returns best-effort vLLM queue/cache load information."""
    del kwargs
    return base_sampler_lib.LoadInfo()
