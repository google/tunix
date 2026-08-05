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

"""Orchestrator-side client proxies for routing calls to remote gRPC workers."""

from __future__ import annotations

import logging
from typing import Any
from tunix.rl import rl_cluster as rl_engine_lib
from tunix.experimental.worker import remote_execution


class RemoteActorTrainerProxy:
  """Proxies configurations (like loss function) to the remote JAX Trainer Worker."""

  def __init__(self, handle: remote_execution.ActorHandle):
    self.handle = handle

  def with_loss_fn(self, loss_fn: Any, has_aux: bool = False) -> RemoteActorTrainerProxy:
    self.handle.submit("with_loss_fn", loss_fn, has_aux)
    return self

  def with_gen_model_input_fn(self, gen_model_input_fn: Any) -> RemoteActorTrainerProxy:
    self.handle.submit("with_gen_model_input_fn", gen_model_input_fn)
    return self


class GrpcRolloutWorkerProxy:
  """Proxies generate() and updates configs from local rollout config."""

  def __init__(self, handle: remote_execution.ActorHandle, cluster_config: Any):
    self.handle = handle
    self.cluster_config = cluster_config

  def generate(
      self,
      prompts: Any,
      apply_chat_template: bool = False,
      mode: Any = rl_engine_lib.Mode.TRAIN,
      **kwargs,
  ) -> Any:
    # Resolve configs from local base config
    rollout_cfg = self.cluster_config.rollout_config
    if isinstance(rollout_cfg, dict):
      rollout_cfg = rollout_cfg[mode]

    # Convert to wire format
    return self.handle.submit(
        "generate",
        prompts=prompts,
        apply_chat_template=apply_chat_template,
        mode=mode,
        max_generation_steps=rollout_cfg.max_tokens_to_generate,
        temperature=rollout_cfg.temperature,
        top_p=rollout_cfg.top_p,
        top_k=rollout_cfg.top_k,
        **kwargs,
    )


class GrpcWeightSyncProxy:
  """Orchestrates weight sync from Trainer to Rollout worker over gRPC."""

  def __init__(
      self,
      trainer_handle: remote_execution.ActorHandle,
      rollout_handle: remote_execution.ActorHandle,
  ):
    self.trainer_handle = trainer_handle
    self.rollout_handle = rollout_handle

  def sync(self) -> None:
    logging.info("[WeightSyncProxy] Pulling LoRA weights from Trainer...")
    lora_state = self.trainer_handle.submit("get_lora_weights")
    logging.info("[WeightSyncProxy] Pushing LoRA weights to Rollout worker...")
    self.rollout_handle.submit("update_params", lora_state)
    logging.info("[WeightSyncProxy] Sync completed.")


class GrpcTrainerWorkerProxy:
  """Proxies train() and per_token_logps() to remote JaxTrainerWorkerService."""

  def __init__(self, handle: remote_execution.ActorHandle):
    self.handle = handle

  def train(self, chunks: list[Any], eval_ds: Any = None, skip_jit: bool = False) -> None:
    self.handle.submit("train", chunks, eval_ds, skip_jit)

  def per_token_logps(
      self,
      prompt_ids: Any,
      completion_ids: Any,
      pad_id: int,
      eos_id: int,
  ) -> Any:
    return self.handle.submit(
        "per_token_logps",
        prompt_ids=prompt_ids,
        completion_ids=completion_ids,
        pad_id=pad_id,
        eos_id=eos_id,
    )


class GrpcInferenceWorkerProxy:
  """Proxies per_token_logps() to remote JaxTrainerWorkerService's reference_logps()."""

  def __init__(self, handle: remote_execution.ActorHandle):
    self.handle = handle

  def per_token_logps(
      self,
      prompt_ids: Any,
      completion_ids: Any,
      pad_id: int,
      eos_id: int,
  ) -> Any:
    # Routes to the reference model logprob calculation on Trainer worker
    return self.handle.submit(
        "reference_logps",
        prompt_ids=prompt_ids,
        completion_ids=completion_ids,
        pad_id=pad_id,
        eos_id=eos_id,
    )

