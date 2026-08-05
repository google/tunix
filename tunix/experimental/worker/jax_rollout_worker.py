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

"""Generic JAX Rollout Worker Service wrapping VanillaRollout."""

from __future__ import annotations

import logging
from typing import Any

from flax import nnx
from tunix.rl import rl_cluster as rl_engine_lib
from tunix.rl.rollout import vanilla_rollout
from tunix.rl.rollout import base_rollout


class JaxRolloutWorkerService:
  """Exposes Rollout sampling and weight sync methods via gRPC."""

  def __init__(self, rollout_engine: vanilla_rollout.VanillaRollout, max_prompt_length: int = 512, kv_cache_size: int = 1024):
    self._rollout = rollout_engine
    self._max_prompt_length = max_prompt_length
    self._kv_cache_size = kv_cache_size

  def generate(
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
    logging.info("[JaxRolloutWorker] Generating completions for %d prompts...", len(prompts))
    
    # Configure local rollout configs dynamically
    self._rollout.config = base_rollout.RolloutConfig(
        max_prompt_length=self._max_prompt_length,
        max_tokens_to_generate=max_generation_steps,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        return_logprobs=True,
        kv_cache_size=self._kv_cache_size,
    )
    
    out = self._rollout.generate(
        prompts=prompts,
        apply_chat_template=apply_chat_template,
        mode=mode,
    )
    logging.info("[JaxRolloutWorker] Generation complete.")
    return out

  def update_params(self, lora_weights: Any) -> None:
    logging.info("[JaxRolloutWorker] Syncing new LoRA weights from Trainer...")
    # Update NNX Graph state for LoRAParam types
    self._rollout.update_params(lora_weights, (nnx.LoRAParam,))
    logging.info("[JaxRolloutWorker] Weight sync complete.")
