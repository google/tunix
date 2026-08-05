# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generic JAX in-process vLLM Rollout Worker Service wrapping LegacyVllmSamplerAdapter."""

from __future__ import annotations

import logging
from typing import Any
import numpy as np

from tunix.experimental.rollout import sampler as base_sampler_lib
from tunix.rl import rl_cluster as rl_engine_lib
from tunix.rl.rollout import base_rollout
from tunix.experimental.rollout import legacy_vllm_sampler_adapter


class VllmRolloutWorkerService:
  """Exposes in-process vLLM generation and parameter syncing over gRPC."""

  def __init__(self, sampler_adapter: legacy_vllm_sampler_adapter.LegacyVllmSamplerAdapter):
    self._sampler = sampler_adapter

  async def generate(
      self,
      prompts: list[str],
      apply_chat_template: bool = False,
      mode: Any = rl_engine_lib.Mode.TRAIN,
      max_generation_steps: int = 512,
      temperature: float = 1.0,
      top_p: float = 1.0,
      top_k: int | None = None,
      **kwargs,
  ) -> base_rollout.RolloutOutput:
    logging.info("[VllmRolloutWorker] Generating completions for %d prompts...", len(prompts))

    # 1. Map to vLLM legacy sampler requests
    requests = []
    for p in prompts:
      req = base_sampler_lib.SamplingRequest(
          prompt=p,
          sampling_params=base_sampler_lib.SamplingParams(
              max_tokens=max_generation_steps,
              temperature=temperature,
              top_p=top_p,
              top_k=top_k,
              return_logprobs=True,
          ),
      )
      requests.append(req)

    # 2. Run in-process vLLM sampling (runs async on TPU 2,3)
    responses = await self._sampler.sample(requests)

    # 3. Translate to Orchestrator's RolloutOutput format
    texts = [r.text for r in responses]
    tokens = [r.token_ids for r in responses]
    # logprobs are returned as numpy arrays
    logprobs = [r.logprobs for r in responses]

    logging.info("[VllmRolloutWorker] Generation completed.")
    return base_rollout.RolloutOutput(
        text=texts,
        logits=None,
        tokens=tokens,
        left_padded_prompt_tokens=np.zeros((len(prompts), 1), dtype=np.int32),  # Dummy left pad
        logprobs=logprobs,
    )

  async def update_params(self, lora_weights: Any) -> None:
    logging.info("[VllmRolloutWorker] Syncing LoRA weights from Trainer to vLLM JAX buffers...")
    # Wrap in WeightSyncRequest
    sync_req = base_sampler_lib.WeightSyncRequest(weights=lora_weights)
    await self._sampler.weight_sync(sync_req)
    logging.info("[VllmRolloutWorker] Weight sync completed.")
