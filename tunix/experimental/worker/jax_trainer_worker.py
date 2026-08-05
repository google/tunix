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

"""Generic JAX Trainer Worker Service wrapping RLEngine."""

from __future__ import annotations

import logging
from typing import Any

from flax import nnx
from tunix.rl import rl_cluster as rl_engine_lib


class JaxTrainerWorkerService:
  """Exposes Trainer control loop and scoring methods via gRPC."""

  def __init__(self, rl_engine: rl_engine_lib.RLEngine):
    self._rl_engine = rl_engine

  def with_loss_fn(self, loss_fn: Any, has_aux: bool = False) -> None:
    logging.info("[JaxTrainerWorker] Registering custom Loss function.")
    self._rl_engine.actor_trainer.with_loss_fn(loss_fn, has_aux)

  def with_gen_model_input_fn(self, gen_model_input_fn: Any) -> None:
    logging.info("[JaxTrainerWorker] Registering Generator Model Input function.")
    self._rl_engine.actor_trainer.with_gen_model_input_fn(gen_model_input_fn)

  def train(self, chunks: list[Any], eval_ds: Any = None, skip_jit: bool = False) -> None:
    logging.info("[JaxTrainerWorker] Starting train step with %d chunks...", len(chunks))
    self._rl_engine.update_actor(chunks, eval_ds, skip_jit)
    logging.info("[JaxTrainerWorker] Train step complete.")

  def per_token_logps(
      self,
      prompt_ids: Any,
      completion_ids: Any,
      pad_id: int,
      eos_id: int,
  ) -> Any:
    logging.info("[JaxTrainerWorker] Computing Actor log probabilities...")
    return self._rl_engine.get_actor_per_token_logps(
        prompt_tokens=prompt_ids,
        completion_tokens=completion_ids,
        pad_id=pad_id,
        eos_id=eos_id,
    )

  def reference_logps(
      self,
      prompt_ids: Any,
      completion_ids: Any,
      pad_id: int,
      eos_id: int,
  ) -> Any:
    logging.info("[JaxTrainerWorker] Computing Reference log probabilities...")
    return self._rl_engine.get_ref_per_token_logps(
        prompt_tokens=prompt_ids,
        completion_tokens=completion_ids,
        pad_id=pad_id,
        eos_id=eos_id,
    )

  def get_lora_weights(self) -> Any:
    logging.info("[JaxTrainerWorker] Staging LoRA weights for synchronization...")
    # Extracts LoRAParam parameters from NNX Graph state
    lora_state = nnx.state(self._rl_engine.actor_trainer.model, nnx.LoRAParam)
    return lora_state
